#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "matplotlib>=3.5.0",
#   "numpy>=1.21.0",
#   "scipy>=1.7.0",
#   "tqdm>=4.62.0",
# ]
# ///
"""
Generate statistical analysis of Gumbel-Softmax top-k selection.

Real data from actual Gumbel-Softmax computations showing:
- Mask value distributions
- Temperature and scale effects
- Comparison with sigmoid-based method
- Exploration properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import json

def gumbel_noise(rng):
    """Generate Gumbel noise: G = -log(-log(U))"""
    u = rng.uniform(1e-10, 1.0 - 1e-10)
    return -np.log(-np.log(u))

def softmax(logits):
    """Numerically stable softmax"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def gumbel_softmax(logits, temperature, scale, rng):
    """Gumbel-Softmax sampling"""
    n = len(logits)
    gumbel_logits = np.zeros(n)
    for i in range(n):
        g = gumbel_noise(rng)
        gumbel_logits[i] = (g + scale * logits[i]) / temperature
    return softmax(gumbel_logits)

def relaxed_topk_gumbel(scores, k, temperature, scale, rng):
    """Relaxed top-k using Gumbel-Softmax"""
    n = len(scores)
    max_mask = np.zeros(n)
    for _ in range(k):
        mask = gumbel_softmax(scores, temperature, scale, rng)
        max_mask = np.maximum(max_mask, mask)
    return max_mask

def sigmoid_topk(scores, k, regularization):
    """Sigmoid-based top-k (simplified)"""
    # Sort and create soft mask
    sorted_indices = np.argsort(scores)[::-1]
    mask = np.zeros(len(scores))
    for i, idx in enumerate(sorted_indices):
        if i < k:
            # Soft indicator for top-k
            mask[idx] = 1.0 / (1.0 + np.exp(-regularization * (k - i)))
        else:
            mask[idx] = 1.0 / (1.0 + np.exp(regularization * (i - k + 1)))
    return mask

def main():
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Gumbel-Softmax analysis with real data...")
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    n_runs = 1000
    n_docs = 20
    k = 5
    
    # Realistic reranker scores
    base_scores = np.linspace(0.9, 0.1, n_docs)
    
    # Collect data
    gumbel_masks = []
    sigmoid_masks = []
    gumbel_variances = []
    
    print(f"Computing {n_runs} Gumbel-Softmax samples...")
    for run in tqdm(range(n_runs)):
        rng = np.random.RandomState(42 + run)
        
        # Gumbel-Softmax
        gumbel_mask = relaxed_topk_gumbel(base_scores, k, 0.5, 1.0, rng)
        gumbel_masks.append(gumbel_mask)
        
        # Sigmoid-based (deterministic, same each time)
        if run == 0:
            sigmoid_mask = sigmoid_topk(base_scores, k, 1.0)
            sigmoid_masks.append(sigmoid_mask)
    
    gumbel_masks = np.array(gumbel_masks)
    sigmoid_masks = np.array(sigmoid_masks)
    
    # Calculate statistics
    gumbel_mean = np.mean(gumbel_masks, axis=0)
    gumbel_std = np.std(gumbel_masks, axis=0)
    gumbel_variance = np.var(gumbel_masks, axis=0)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gumbel-Softmax Top-k Analysis (Real Data)', fontsize=16, fontweight='bold')
    
    # 1. Mask value distribution by document rank
    ax = axes[0, 0]
    doc_indices = np.arange(n_docs)
    ax.errorbar(doc_indices, gumbel_mean, yerr=gumbel_std, 
                fmt='o-', label='Gumbel (mean ± std)', capsize=3, alpha=0.7)
    ax.plot(doc_indices, sigmoid_masks[0], 's-', label='Sigmoid (deterministic)', alpha=0.7)
    ax.axvline(k-0.5, color='red', linestyle='--', alpha=0.5, label=f'Top-{k} cutoff')
    ax.set_xlabel('Document Rank (by score)')
    ax.set_ylabel('Mask Value')
    ax.set_title('Mask Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Variance analysis (exploration)
    ax = axes[0, 1]
    ax.bar(doc_indices, gumbel_variance, alpha=0.7, color='steelblue')
    ax.axvline(k-0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Document Rank')
    ax.set_ylabel('Variance')
    ax.set_title('Exploration: Variance Across Runs')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Temperature effect
    ax = axes[1, 0]
    temperatures = [0.1, 0.3, 0.5, 1.0, 2.0]
    temp_means = []
    temp_stds = []
    
    for temp in temperatures:
        temp_masks = []
        for run in range(100):
            rng = np.random.RandomState(1000 + run)
            mask = relaxed_topk_gumbel(base_scores, k, temp, 1.0, rng)
            temp_masks.append(np.mean(mask[:k]))  # Average mask for top-k
        temp_means.append(np.mean(temp_masks))
        temp_stds.append(np.std(temp_masks))
    
    ax.errorbar(temperatures, temp_means, yerr=temp_stds, 
                fmt='o-', capsize=3, alpha=0.7)
    ax.set_xlabel('Temperature (τ)')
    ax.set_ylabel('Mean Mask Value (top-k)')
    ax.set_title('Temperature Effect on Selection Sharpness')
    ax.grid(True, alpha=0.3)
    
    # 4. Selection frequency (how often each doc is selected)
    ax = axes[1, 1]
    selection_freq = np.mean(gumbel_masks > 0.5, axis=0)
    ax.bar(doc_indices, selection_freq, alpha=0.7, color='coral')
    ax.axvline(k-0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Document Rank')
    ax.set_ylabel('Selection Frequency')
    ax.set_title(f'Selection Frequency (mask > 0.5, {n_runs} runs)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "gumbel_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    # Generate comparison plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Gumbel vs Sigmoid: Key Differences', fontsize=16, fontweight='bold')
    
    # Left: Determinism
    ax = axes2[0]
    # Show 5 Gumbel runs vs 1 sigmoid run
    for run in range(5):
        rng = np.random.RandomState(200 + run)
        gumbel_mask = relaxed_topk_gumbel(base_scores, k, 0.5, 1.0, rng)
        ax.plot(doc_indices, gumbel_mask, 'o-', alpha=0.5, label=f'Gumbel run {run+1}' if run < 1 else '')
    ax.plot(doc_indices, sigmoid_masks[0], 's-', linewidth=2, label='Sigmoid (deterministic)')
    ax.set_xlabel('Document Rank')
    ax.set_ylabel('Mask Value')
    ax.set_title('Stochastic vs Deterministic')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Exploration benefit
    ax = axes2[1]
    # Show how Gumbel explores different combinations
    top_k_combinations = []
    for run in range(100):
        rng = np.random.RandomState(300 + run)
        mask = relaxed_topk_gumbel(base_scores, k, 0.5, 1.0, rng)
        selected = np.where(mask > 0.5)[0]
        if len(selected) >= k:
            top_k_combinations.append(tuple(sorted(selected[:k])))
    
    from collections import Counter
    combo_counts = Counter(top_k_combinations)
    top_combos = combo_counts.most_common(5)
    
    combo_labels = [f"Combo {i+1}" for i in range(len(top_combos))]
    combo_freqs = [count for _, count in top_combos]
    
    ax.barh(combo_labels, combo_freqs, alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_title('Top-k Combinations Explored (Gumbel)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path2 = output_dir / "gumbel_vs_sigmoid.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path2}")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Documents: {n_docs}, Top-k: {k}, Runs: {n_runs}")
    print(f"  Gumbel mean mask (top-{k}): {np.mean(gumbel_mean[:k]):.3f}")
    print(f"  Gumbel mean mask (bottom): {np.mean(gumbel_mean[k:]):.3f}")
    print(f"  Average variance (top-{k}): {np.mean(gumbel_variance[:k]):.4f}")
    print(f"  Average variance (bottom): {np.mean(gumbel_variance[k:]):.4f}")
    print(f"  Selection frequency (top-{k}): {np.mean(selection_freq[:k]):.2%}")
    print(f"  Selection frequency (bottom): {np.mean(selection_freq[k:]):.2%}")
    
    print(f"\n✅ Analysis complete! See {output_dir} for visualizations.")

if __name__ == "__main__":
    main()

