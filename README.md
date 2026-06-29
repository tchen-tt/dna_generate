# DNA-Diffusion

A diffusion model for generating cell-type-specific regulatory DNA sequences. This project implements a UNet-based denoising diffusion probabilistic model (DDPM) with classifier-free guidance to generate synthetic 200bp DNA sequences conditioned on cell type.

## Features

- **Conditional Generation**: Generate DNA sequences for specific cell types (K562, hESCT0, HepG2, GM12878)
- **Classifier-Free Guidance**: Control generation quality with guidance scale parameter
- **Attention Mechanism**: Cross-attention between sequence and cell-type embeddings
- **Quality Evaluation**: Comprehensive metrics including GC content, k-mer frequencies, and diversity analysis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the diffusion model on DNA sequence data:

```bash
# Basic training
python train.py

# Debug mode (single sequence for fast iteration)
python train.py --debug

# Custom configuration
python train.py \
  --data-path data/sequences.txt \
  --batch-size 64 \
  --num-epochs 3000 \
  --lr 1e-4 \
  --timesteps 50
```

**Key arguments:**
- `--data-path`: Path to training data (TSV format with columns: chr, sequence, TAG)
- `--batch-size`: Batch size for training (default: 120)
- `--num-epochs`: Maximum training epochs (default: 5000)
- `--min-epochs`: Minimum epochs before early stopping (default: 2000)
- `--patience`: Early stopping patience (default: 10)
- `--precision`: Training precision, "fp32" or "bf16" (default: bf16)
- `--use-wandb`: Enable Weights & Biases logging

### Inference

Generate DNA sequences from a trained checkpoint:

```bash
# Generate for all cell types in checkpoint
python infer.py --checkpoint checkpoints/model_best.pt

# Generate for specific cell types
python infer.py \
  --checkpoint checkpoints/model_best.pt \
  --cell-types K562 HepG2 \
  --num-samples 1000

# Adjust guidance strength
python infer.py \
  --checkpoint checkpoints/model_best.pt \
  --cond-weight 1.0 \
  --num-samples 500
```

**Key arguments:**
- `--checkpoint`: Path to trained model checkpoint (.pt file)
- `--cell-types`: Cell types to generate (default: all types in checkpoint)
- `--num-samples`: Number of sequences per cell type (default: 1000)
- `--cond-weight`: Classifier-free guidance scale (default: 1.0, higher = stronger conditioning)
- `--output-dir`: Output directory for generated sequences (default: ../data/outputs)
- `--save-attention`: Save cross-attention maps

**Output:** Generated sequences are saved as FASTA files in `output-dir/{cell_type}.txt`

### Evaluation

Evaluate quality of generated sequences:

```bash
# Evaluate all cell types in output directory
python evaluate.py \
  --data-path data/sequences.txt \
  --output-dir data/outputs \
  --plot

# Evaluate specific cell types
python evaluate.py \
  --data-path data/sequences.txt \
  --output-dir data/outputs \
  --cell-types K562 HepG2 \
  --plot
```

**Key arguments:**
- `--data-path`: Path to original training data (for comparison)
- `--output-dir`: Directory containing generated sequences
- `--cell-types`: Cell types to evaluate (default: all .txt files in output-dir)
- `--plot`: Generate visualization plots (default: True)

**Evaluation metrics:**
1. **GC Content**: Distribution comparison between real and generated sequences
2. **k-mer Frequency**: Pearson correlation for 1-mer, 2-mer, and 4-mer frequencies
3. **Duplication Rate**: Internal duplicates and overlap with training set
4. **Diversity**: Hamming distance distribution among generated sequences

**Output:** Evaluation results printed to console, plots saved to `output-dir/plots/`

## Data Format

Training data should be a tab-separated file with columns:
- `chr`: Chromosome identifier (chr1=test, chr2=validation, others=training)
- `sequence`: DNA sequence (200bp, ACGT only, no N's)
- `TAG`: Cell type label

Example:
```
chr	sequence	TAG
chr3	ATCGATCG...	K562_ENCLB843GMH
chr4	GCTAGCTA...	HepG2_ENCLB029COU
```

## Model Architecture

- **Backbone**: UNet with ResNet blocks and attention layers
- **Time Embedding**: Learnable sinusoidal positional encoding
- **Conditioning**: Cell type labels embedded and added to time embedding
- **Diffusion Steps**: 50 timesteps with linear beta schedule (default)
- **Training Loss**: Smooth L1 loss on predicted noise

## Project Structure

```
.
├── constants.py       # Shared constants
├── dataloader.py      # Data loading and preprocessing
├── diffusion.py       # DDPM diffusion model
├── unet.py           # UNet architecture
├── layers.py         # Neural network building blocks
├── utils.py          # Utility functions
├── train.py          # Training script
├── infer.py          # Inference script
├── evaluate.py       # Evaluation script
└── requirements.txt  # Python dependencies
```

## Citation

If you use this code, please cite the original DNA-Diffusion paper:

```
[Add citation here when available]
```

## License

[Add license information]
