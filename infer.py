"""Inference entry point: generate cell-type-specific DNA sequences from trained checkpoint.

Usage examples:
    # Generate all cell types (using mapping recorded in checkpoint)
    python infer.py --checkpoint checkpoints/model_epoch999_step12000_val0.0123.pt

    # Generate only specified cell types
    python infer.py --checkpoint checkpoints/model.pt --cell-types K562 HepG2

    # Adjust guidance strength and sample count
    python infer.py --checkpoint checkpoints/model.pt --cond-weight 7.0 --num-samples 1000
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constants import NUCLEOTIDES, SEQ_LEN
from diffusion import Diffusion
from unet import UNet
from utils import convert_to_seq


# ---------------------------------------------------------------------------
# Core generation function (also called by train.py during training)
# ---------------------------------------------------------------------------

def generate_samples(
    model: Diffusion,
    cell_type: int,
    numeric_to_tag: dict[int, str],
    num_samples: int = 1000,
    sample_bs: int = 10,
    cond_weight: float = 1.0,
    output_dir: str = "../data/outputs",
    save_attention: bool = False,
) -> list[str]:
    """
    Generate `num_samples` DNA sequences for a given cell type.

    Parameters
    ----------
    model           : trained Diffusion model (already on correct device)
    cell_type       : integer label for the target cell type
    numeric_to_tag  : mapping from int label → cell type name string
    num_samples     : total sequences to generate
    sample_bs       : batch size per sampling call
    cond_weight     : classifier-free guidance weight (1.0 = no extra guidance)
    output_dir      : directory to write output .txt file
    save_attention  : whether to save cross-attention maps as .npy

    Returns
    -------
    list of generated nucleotide strings
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = model.device

    cell_name = numeric_to_tag[cell_type]
    sequences = []
    num_batches = max(1, num_samples // sample_bs)

    for _ in tqdm(range(num_batches), desc=f"Sampling {cell_name}", leave=False):
        labels = torch.tensor([cell_type] * sample_bs, dtype=torch.float32, device=device)
        shape = (sample_bs, 1, 4, SEQ_LEN)

        if save_attention:
            imgs, cross_maps = model.sample_cross(labels, shape, cond_weight)
            att_path = Path(output_dir) / f"cross_att_{cell_name}.npy"
            np.save(str(att_path), np.array([c for c in cross_maps if c is not None]))
        else:
            imgs, _ = model.sample_cross(labels, shape, cond_weight)

        # imgs is a list of T arrays; take the last (fully denoised) step
        final_step = imgs[-1]  # shape: (sample_bs, 1, 4, 200)
        for sample in final_step:
            seq = convert_to_seq(sample, NUCLEOTIDES, SEQ_LEN)
            sequences.append(seq)

    # Write FASTA-style output
    out_path = Path(output_dir) / f"{cell_name}.txt"
    with open(out_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{cell_name}_{i}\n{seq}\n")

    print(f"  Wrote {len(sequences)} sequences → {out_path}")
    return sequences


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, device: str) -> tuple[Diffusion, dict, dict]:
    """
    Load a Diffusion model from a .pt checkpoint saved by train.py.

    Returns
    -------
    model          : Diffusion model in eval mode on `device`
    numeric_to_tag : int → cell type name mapping
    args_dict      : training args stored in checkpoint (may be empty dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    args_dict      = ckpt.get("args", {})
    numeric_to_tag = ckpt.get("numeric_to_tag", {})

    # Reconstruct model with stored or default hyperparameters
    unet = UNet(
        dim=args_dict.get("dim", 200),
        dim_mults=args_dict.get("dim_mults", [1, 2, 4]),
        resnet_block_groups=args_dict.get("resnet_block_groups", 4),
        num_classes=args_dict.get("num_classes", 10),
    )
    model = Diffusion(
        model=unet,
        timesteps=args_dict.get("timesteps", 50),
        beta_start=args_dict.get("beta_start", 0.0001),
        beta_end=args_dict.get("beta_end", 0.2),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  epoch={ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model, numeric_to_tag, args_dict


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for inference configuration."""
    p = argparse.ArgumentParser(description="Generate DNA sequences with a trained Diffusion model")

    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--cell-types", nargs="*", default=None,
                   help="Cell type names to generate (default: all in checkpoint)")
    p.add_argument("--num-samples", type=int, default=1000,
                   help="Number of sequences per cell type")
    p.add_argument("--sample-bs", type=int, default=10, help="Batch size per sampling call")
    p.add_argument("--cond-weight", type=float, default=1.0,
                   help="Classifier-free guidance scale (1.0 = standard conditioning)")
    p.add_argument("--output-dir", default="data/outputs",
                   help="Directory to write generated sequences")
    p.add_argument("--save-attention", action="store_true", default=False,
                   help="Save cross-attention maps as .npy files")
    p.add_argument("--device", default=None,
                   help="Device override (e.g. 'cpu', 'cuda:0'). Auto-detected if omitted.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for inference script."""
    args = parse_args()

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model, numeric_to_tag, _ = load_model_from_checkpoint(args.checkpoint, device)

    if not numeric_to_tag:
        raise RuntimeError(
            "Checkpoint does not contain numeric_to_tag mapping. "
            "Please supply --cell-types manually and update load_model_from_checkpoint."
        )

    # Resolve which cell types to generate
    if args.cell_types is not None:
        tag_to_numeric = {v: k for k, v in numeric_to_tag.items()}
        selected = {}
        for ct in args.cell_types:
            if ct in tag_to_numeric:
                selected[tag_to_numeric[ct]] = ct
            else:
                # Fuzzy match
                matches = [t for t in tag_to_numeric if ct.lower() in t.lower()]
                if len(matches) == 1:
                    print(f"  Matched '{ct}' → '{matches[0]}'")
                    selected[tag_to_numeric[matches[0]]] = matches[0]
                elif len(matches) > 1:
                    print(f"  Warning: '{ct}' is ambiguous: {matches}. Skipping.")
                else:
                    print(f"  Warning: cell type '{ct}' not found. Available: {list(tag_to_numeric)}")
        cell_ids = list(selected.keys())
    else:
        cell_ids = list(numeric_to_tag.keys())

    if not cell_ids:
        raise ValueError("No valid cell types to generate.")

    print(f"\nGenerating {args.num_samples} sequences for: {[numeric_to_tag[c] for c in cell_ids]}")

    # Generate
    for cell_id in cell_ids:
        generate_samples(
            model=model,
            cell_type=cell_id,
            numeric_to_tag=numeric_to_tag,
            num_samples=args.num_samples,
            sample_bs=args.sample_bs,
            cond_weight=args.cond_weight,
            output_dir=args.output_dir,
            save_attention=args.save_attention,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
