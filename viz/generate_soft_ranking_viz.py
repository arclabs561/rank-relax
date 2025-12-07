#!/usr/bin/env python3
"""
Generate soft ranking visualization charts using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def soft_rank(values, alpha):
    """Compute soft ranks for given values and regularization strength."""
    n = len(values)
    ranks = np.zeros(n)
    for i in range(n):
        if not np.isfinite(values[i]):
            ranks[i] = np.nan
            continue
        rank_sum = 0.0
        for j in range(n):
            if i != j and np.isfinite(values[j]):
                diff = values[i] - values[j]
                rank_sum += sigmoid(alpha * diff)
        ranks[i] = rank_sum / (n - 1)
    return ranks

# Example values
values = np.array([5.0, 1.0, 2.0, 4.0, 3.0])
true_ranks = np.array([4.0, 0.0, 1.0, 3.0, 2.0])  # Discrete ranks

# 1. Effect of Regularization Strength
fig, ax = plt.subplots(figsize=(12, 7))

alphas = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

for i, (val, true_rank, color) in enumerate(zip(values, true_ranks, colors)):
    soft_ranks = [soft_rank(values, alpha)[i] for alpha in alphas]
    ax.plot(alphas, soft_ranks, marker='o', linewidth=2.5, markersize=7,
            label=f'Value {val} (true rank: {true_rank})', color=color)

ax.set_xlabel('Regularization Strength (α)', fontweight='bold', fontsize=12)
ax.set_ylabel('Soft Rank', fontweight='bold', fontsize=12)
ax.set_title('Soft Ranks Converge to Discrete Ranks as α Increases\nHigher α = sharper, more discrete-like behavior', 
             fontweight='bold', pad=15)
ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(output_dir / 'soft_ranking_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: soft_ranking_convergence.png")

# 2. Convergence Error
fig, ax = plt.subplots(figsize=(10, 6))

alpha_range = np.logspace(-1, 2, 50)  # 0.1 to 100
errors = []
for alpha in alpha_range:
    soft_ranks = soft_rank(values, alpha)
    error = np.mean(np.abs(soft_ranks - true_ranks))
    errors.append(error)

ax.plot(alpha_range, errors, linewidth=2.5, color='#ff6b9d')
ax.fill_between(alpha_range, errors, alpha=0.3, color='#ff6b9d')

ax.set_xlabel('Regularization Strength (α)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Absolute Error from Discrete Ranks', fontweight='bold', fontsize=12)
ax.set_title('Soft Ranks Approach Discrete Ranks as α → ∞\nError decreases as regularization increases', 
             fontweight='bold', pad=15)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Add annotation
min_error_idx = np.argmin(errors)
min_alpha = alpha_range[min_error_idx]
min_error = errors[min_error_idx]
ax.annotate(f'Min error: {min_error:.4f}\nat α = {min_alpha:.1f}',
            xy=(min_alpha, min_error), xytext=(min_alpha*2, min_error*2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'soft_ranking_error.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: soft_ranking_error.png")

# 3. Comparison: Discrete vs Soft (low/high alpha)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Discrete ranks
axes[0].bar(range(len(values)), true_ranks, color='#00d9ff', alpha=0.8, 
            edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Element Index', fontweight='bold')
axes[0].set_ylabel('Rank', fontweight='bold')
axes[0].set_title('Discrete Ranking\n(True ranks)', fontweight='bold')
axes[0].set_xticks(range(len(values)))
axes[0].set_xticklabels([f'v={v}' for v in values])
axes[0].grid(True, alpha=0.3, axis='y')
for i, (val, rank) in enumerate(zip(values, true_ranks)):
    axes[0].text(i, rank, f'{rank:.0f}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=11)

# Soft ranks (low alpha)
soft_low = soft_rank(values, 0.5)
axes[1].bar(range(len(values)), soft_low, color='#ff6b9d', alpha=0.8, 
            edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Element Index', fontweight='bold')
axes[1].set_ylabel('Soft Rank', fontweight='bold')
axes[1].set_title('Soft Ranking (α = 0.5)\nSmooth, differentiable', fontweight='bold')
axes[1].set_xticks(range(len(values)))
axes[1].set_xticklabels([f'v={v}' for v in values])
axes[1].grid(True, alpha=0.3, axis='y')
for i, (val, rank) in enumerate(zip(values, soft_low)):
    axes[1].text(i, rank, f'{rank:.2f}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=10)

# Soft ranks (high alpha)
soft_high = soft_rank(values, 50.0)
axes[2].bar(range(len(values)), soft_high, color='#00ff88', alpha=0.8, 
            edgecolor='black', linewidth=1.5)
axes[2].set_xlabel('Element Index', fontweight='bold')
axes[2].set_ylabel('Soft Rank', fontweight='bold')
axes[2].set_title('Soft Ranking (α = 50.0)\nClose to discrete', fontweight='bold')
axes[2].set_xticks(range(len(values)))
axes[2].set_xticklabels([f'v={v}' for v in values])
axes[2].grid(True, alpha=0.3, axis='y')
for i, (val, rank) in enumerate(zip(values, soft_high)):
    axes[2].text(i, rank, f'{rank:.2f}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'soft_ranking_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: soft_ranking_comparison.png")

print("\n✅ All soft ranking visualizations generated!")

