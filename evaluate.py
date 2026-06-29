"""Generated sequence quality evaluation script.

Evaluation metrics:
1. GC content distribution comparison (generated vs real)
2. k-mer frequency correlation (1-mer, 2-mer, 4-mer)
3. Training set duplication rate
4. Sequence diversity (pairwise edit distance sampling)

Usage:
    python evaluate.py \
        --data-path data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt \
        --output-dir data/outputs \
        --cell-type K562_ENCLB843GMH \
        --plot
"""

import argparse
import os
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def gc_content(seq: str) -> float:
    """Calculate GC content ratio of a DNA sequence."""
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def kmer_freq(seq: str, k: int) -> Counter:
    """Count k-mer frequencies in a sequence."""
    return Counter(seq[i:i+k] for i in range(len(seq) - k + 1))


def all_kmers(k: int) -> list[str]:
    """Generate all possible k-mers from ACGT alphabet."""
    return ["".join(p) for p in product("ACGT", repeat=k)]


def kmer_freq_vector(seqs: list[str], k: int) -> np.ndarray:
    """Calculate average k-mer frequency vector for a set of sequences, normalized to frequencies."""
    keys = all_kmers(k)
    total = Counter()
    for seq in seqs:
        total += kmer_freq(seq.upper(), k)
    n = sum(total.values())
    return np.array([total[key] / n for key in keys])


def read_fasta(path: str) -> list[str]:
    """Read FASTA format file and return list of sequences."""
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seqs.append(line.upper())
    return seqs


def read_training_seqs(data_path: str, cell_type: str) -> list[str]:
    """Read real sequences of specified cell type from training data TSV (training set only)."""
    df = pd.read_csv(data_path, sep="\t")
    mask = (df["TAG"] == cell_type) & (df["data_label"] == "training")
    seqs = df[mask]["sequence"].dropna().tolist()
    seqs = [s.upper() for s in seqs if "N" not in s.upper()]
    return seqs


# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------

def eval_gc_content(real_seqs: list[str], gen_seqs: list[str]) -> dict:
    """Evaluate GC content distribution between real and generated sequences."""
    real_gc = [gc_content(s) for s in real_seqs]
    gen_gc  = [gc_content(s) for s in gen_seqs]
    return {
        "real_mean": np.mean(real_gc),
        "real_std":  np.std(real_gc),
        "gen_mean":  np.mean(gen_gc),
        "gen_std":   np.std(gen_gc),
        "real_gc":   real_gc,
        "gen_gc":    gen_gc,
    }


def eval_kmer(real_seqs: list[str], gen_seqs: list[str], k: int) -> dict:
    """Evaluate k-mer frequency correlation between real and generated sequences."""
    real_vec = kmer_freq_vector(real_seqs, k)
    gen_vec  = kmer_freq_vector(gen_seqs,  k)
    corr, pval = pearsonr(real_vec, gen_vec)
    return {
        "k":        k,
        "pearsonr": corr,
        "pval":     pval,
        "real_vec": real_vec,
        "gen_vec":  gen_vec,
        "keys":     all_kmers(k),
    }


def eval_duplication(real_seqs: list[str], gen_seqs: list[str]) -> dict:
    """Evaluate duplication rate with training set and internal duplicates."""
    real_set = set(real_seqs)
    gen_set  = set(gen_seqs)
    dup_with_train = gen_set & real_set
    internal_dup   = len(gen_seqs) - len(gen_set)
    return {
        "total_generated":    len(gen_seqs),
        "unique_generated":   len(gen_set),
        "internal_dup_rate":  internal_dup / len(gen_seqs),
        "trainset_dup_count": len(dup_with_train),
        "trainset_dup_rate":  len(dup_with_train) / len(gen_seqs),
    }


def eval_diversity(gen_seqs: list[str], n_sample: int = 200) -> dict:
    """Evaluate sequence diversity using Hamming distance sampling.

    Randomly sample sequence pairs and calculate Hamming distance (edit distance proxy for equal-length sequences).
    Full edit distance is computationally expensive; Hamming distance is used as a fast proxy metric.
    """
    seqs = gen_seqs[:n_sample] if len(gen_seqs) > n_sample else gen_seqs
    n = len(seqs)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = sum(a != b for a, b in zip(seqs[i], seqs[j]))
            dists.append(d)
    return {
        "mean_hamming":   np.mean(dists),
        "median_hamming": np.median(dists),
        "max_seq_len":    len(seqs[0]) if seqs else 0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_gc_distribution(gc_result: dict, cell_type: str, save_path: str):
    """Plot GC content distribution histogram for real vs generated sequences."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(gc_result["real_gc"], bins=bins, alpha=0.6, label="Real", color="#2196F3", density=True)
    ax.hist(gc_result["gen_gc"],  bins=bins, alpha=0.6, label="Generated", color="#FF5722", density=True)
    ax.axvline(gc_result["real_mean"], color="#2196F3", linestyle="--", linewidth=1.5,
               label=f"Real mean {gc_result['real_mean']:.3f}")
    ax.axvline(gc_result["gen_mean"],  color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"Gen mean {gc_result['gen_mean']:.3f}")
    ax.set_xlabel("GC Content")
    ax.set_ylabel("Density")
    ax.set_title(f"GC Content Distribution — {cell_type}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_kmer_scatter(kmer_result: dict, cell_type: str, save_path: str):
    """Plot k-mer frequency scatter plot comparing real vs generated."""
    real = kmer_result["real_vec"]
    gen  = kmer_result["gen_vec"]
    k    = kmer_result["k"]
    corr = kmer_result["pearsonr"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real, gen, s=8, alpha=0.5, color="#4CAF50")
    lim = max(real.max(), gen.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y=x")
    ax.set_xlabel("Real k-mer Frequency")
    ax.set_ylabel("Generated k-mer Frequency")
    ax.set_title(f"{k}-mer Frequency Scatter (Pearson r={corr:.4f}) — {cell_type}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_kmer_bar(kmer_result: dict, cell_type: str, save_path: str, top_n: int = 20):
    """Plot bar chart comparing top N most frequent k-mers between real and generated sequences."""
    keys = kmer_result["keys"]
    real = kmer_result["real_vec"]
    gen  = kmer_result["gen_vec"]
    k    = kmer_result["k"]

    # Sort by real frequency and take top_n
    idx = np.argsort(real)[::-1][:top_n]
    top_keys = [keys[i] for i in idx]
    top_real = real[idx]
    top_gen  = gen[idx]

    x = np.arange(top_n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.5), 4))
    ax.bar(x - width/2, top_real, width, label="Real", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, top_gen,  width, label="Generated", color="#FF5722", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top-{top_n} {k}-mer Frequency Comparison — {cell_type}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(data_path: str, output_dir: str, cell_type: str, plot: bool):
    """Run complete evaluation pipeline for a given cell type."""
    gen_path = os.path.join(output_dir, f"{cell_type}.txt")
    if not os.path.exists(gen_path):
        print(f"Generated file not found: {gen_path}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating cell type: {cell_type}")
    print(f"{'='*60}")

    # Load sequences
    gen_seqs  = read_fasta(gen_path)
    real_seqs = read_training_seqs(data_path, cell_type)
    print(f"Generated sequences: {len(gen_seqs)}")
    print(f"Real sequences: {len(real_seqs)}")

    if not gen_seqs:
        print("No generated sequences, skipping.")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. GC content
    # ------------------------------------------------------------------
    print("\n[1] GC Content")
    gc = eval_gc_content(real_seqs, gen_seqs)
    print(f"    Real: {gc['real_mean']:.4f} ± {gc['real_std']:.4f}")
    print(f"    Generated: {gc['gen_mean']:.4f} ± {gc['gen_std']:.4f}")
    diff = abs(gc['real_mean'] - gc['gen_mean'])
    print(f"    Mean difference: {diff:.4f}  {'✓ Close' if diff < 0.05 else '✗ Large deviation'}")
    if plot:
        plot_gc_distribution(gc, cell_type,
                             os.path.join(plot_dir, f"{cell_type}_gc.png"))

    # ------------------------------------------------------------------
    # 2. k-mer correlation
    # ------------------------------------------------------------------
    print("\n[2] k-mer Frequency Correlation (Pearson r)")
    for k in [1, 2, 4]:
        km = eval_kmer(real_seqs, gen_seqs, k)
        status = "✓" if km["pearsonr"] > 0.9 else ("△" if km["pearsonr"] > 0.7 else "✗")
        print(f"    {k}-mer: r = {km['pearsonr']:.4f}  {status}")
        if plot:
            plot_kmer_scatter(km, cell_type,
                              os.path.join(plot_dir, f"{cell_type}_{k}mer_scatter.png"))
            if k == 4:
                plot_kmer_bar(km, cell_type,
                              os.path.join(plot_dir, f"{cell_type}_{k}mer_bar.png"))

    # ------------------------------------------------------------------
    # 3. Duplication rate
    # ------------------------------------------------------------------
    print("\n[3] Duplication Rate")
    dup = eval_duplication(real_seqs, gen_seqs)
    print(f"    Total generated:       {dup['total_generated']}")
    print(f"    Unique generated:      {dup['unique_generated']}")
    print(f"    Internal dup rate:     {dup['internal_dup_rate']:.2%}")
    print(f"    Training set dup count: {dup['trainset_dup_count']}")
    print(f"    Training set dup rate:  {dup['trainset_dup_rate']:.2%}  "
          f"{'✓ No memorization' if dup['trainset_dup_rate'] < 0.01 else '✗ Memorization detected'}")

    # ------------------------------------------------------------------
    # 4. Diversity
    # ------------------------------------------------------------------
    print("\n[4] Sequence Diversity (Hamming distance, 200 pairs sampled)")
    div = eval_diversity(gen_seqs)
    seq_len = div["max_seq_len"]
    norm = div["mean_hamming"] / seq_len if seq_len > 0 else 0
    print(f"    Mean Hamming distance: {div['mean_hamming']:.1f} / {seq_len} bp ({norm:.2%})")
    print(f"    Median Hamming distance: {div['median_hamming']:.1f}")

    print(f"\n{'='*60}\n")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",  default="data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt")
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--cell-types", nargs="*", default=None,
                        help="Specify cell types, default: evaluate all .txt files in output_dir")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate visualization plots")
    args = parser.parse_args()

    # Auto-discover cell types from output_dir
    if args.cell_types:
        cell_types = args.cell_types
    else:
        cell_types = [
            f[:-4] for f in os.listdir(args.output_dir)
            if f.endswith(".txt") and not f.startswith(".")
        ]
        if not cell_types:
            print(f"No .txt files found in {args.output_dir}.")
            return

    for ct in cell_types:
        evaluate(args.data_path, args.output_dir, ct, args.plot)


if __name__ == "__main__":
    main()
