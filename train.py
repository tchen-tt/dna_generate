"""Training entry point."""
import argparse
import os
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from constants import SEQ_LEN
from dataloader import get_dataloader, get_dataset
from diffusion import Diffusion
from infer import generate_samples
from unet import UNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for training configuration."""
    p = argparse.ArgumentParser(description="Train DNA Diffusion model")

    # Data
    p.add_argument("--data-path", default="data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt")
    p.add_argument("--saved-data-path", default="data/encode_data.pkl")
    p.add_argument("--load-saved-data", action="store_true", default=True)
    p.add_argument("--debug", action="store_true", default=False,
                   help="Use a single sequence for fast iteration")

    # Model
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--dim-mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--resnet-block-groups", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=10)

    # Diffusion
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--beta-start", type=float, default=0.0001)
    p.add_argument("--beta-end", type=float, default=0.2)

    # Training
    p.add_argument("--batch-size", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-epochs", type=int, default=5000)
    p.add_argument("--min-epochs", type=int, default=2000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true", default=True)

    # Logging / checkpointing
    p.add_argument("--log-step", type=int, default=50)
    p.add_argument("--sample-epoch", type=int, default=500,
                   help="Generate samples every N epochs")
    p.add_argument("--sample-bs", type=int, default=10)
    p.add_argument("--number-of-samples", type=int, default=100)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--use-wandb", action="store_true", default=False)

    # Distributed
    p.add_argument("--distributed", action="store_true", default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_step(x, y, model, optimizer, device, precision):
    """Execute single training step with optional mixed precision."""
    use_amp = precision == "bf16"
    dtype = torch.bfloat16 if use_amp else torch.float32
    x = x.to(device, dtype=dtype)
    y = y.to(device)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=use_amp):
        loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def val_step(x, y, model, device, precision):
    """Execute single validation step without gradient computation."""
    use_amp = precision == "bf16"
    dtype = torch.bfloat16 if use_amp else torch.float32
    x = x.to(device, dtype=dtype)
    y = y.to(device)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=use_amp):
        loss = model(x, y)
    return loss.item()


# ---------------------------------------------------------------------------
# Main train loop
# ---------------------------------------------------------------------------

def train(args):
    """Main training loop with distributed support and early stopping."""
    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------
    if args.distributed:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        rank = dist.get_rank()
        is_main = rank == 0
        local_batch_size = args.batch_size // dist.get_world_size()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main = True
        local_batch_size = args.batch_size

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset, val_dataset, cell_num_list, numeric_to_tag = get_dataset(
        data_path=args.data_path,
        saved_data_path=args.saved_data_path,
        load_saved_data=args.load_saved_data,
        debug=args.debug,
    )

    train_dl, train_sampler = get_dataloader(
        train_dataset, local_batch_size, args.num_workers, args.distributed, args.pin_memory
    )
    val_dl, _ = get_dataloader(
        val_dataset, local_batch_size, args.num_workers, args.distributed, args.pin_memory
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    unet = UNet(
        dim=args.dim,
        dim_mults=args.dim_mults,
        resnet_block_groups=args.resnet_block_groups,
        num_classes=args.num_classes,
    )
    model = Diffusion(
        model=unet,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if is_main and args.use_wandb:
        wandb_id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=wandb_id, config=vars(args))

    # ------------------------------------------------------------------
    # Checkpoint dir
    # ------------------------------------------------------------------
    if is_main:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path("data/outputs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_files = []
    global_step = 0

    for epoch in tqdm(range(args.num_epochs), disable=not is_main, desc="Epochs"):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # ---- train ----
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            loss = train_step(x, y, model, optimizer, device, args.precision)
            train_loss = loss
            global_step += 1

            if is_main and global_step % args.log_step == 0 and args.use_wandb:
                wandb.log({"train_loss": loss, "epoch": epoch}, step=global_step)

        # ---- validate ----
        model.eval()
        val_losses = []
        for x, y in val_dl:
            val_losses.append(val_step(x, y, model, device, args.precision))
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")

        if is_main:
            tqdm.write(f"Epoch {epoch:4d} | train_loss={train_loss:.4f} | val_loss={avg_val_loss:.4f}")
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": avg_val_loss, "epoch": epoch}, step=global_step)

        # ---- early stopping / checkpointing ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            if is_main:
                # Unwrap DDP if needed
                raw_model = model.module if args.distributed else model
                ckpt = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                    "numeric_to_tag": numeric_to_tag,
                }
                ckpt_path = f"{args.checkpoint_dir}/model_epoch{epoch}_step{global_step}_val{best_val_loss:.4f}.pt"
                torch.save(ckpt, ckpt_path)
                checkpoint_files.append(ckpt_path)
                # Keep only top-2 checkpoints
                if len(checkpoint_files) > 2:
                    os.remove(checkpoint_files.pop(0))
        else:
            patience_counter += 1

        # ---- early stopping check ----
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            if is_main:
                print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

        # ---- periodic sampling ----
        if is_main and (epoch + 1) % args.sample_epoch == 0:
            raw_model = model.module if args.distributed else model
            for cell_id in cell_num_list:
                generate_samples(
                    model=raw_model,
                    cell_type=cell_id,
                    numeric_to_tag=numeric_to_tag,
                    num_samples=args.number_of_samples,
                    sample_bs=args.sample_bs,
                    cond_weight=1.0,
                    output_dir="data/outputs",
                )

    if is_main:
        print(f"\nTraining complete. Best checkpoint: {checkpoint_files[-1] if checkpoint_files else 'none'}")

    if args.distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
