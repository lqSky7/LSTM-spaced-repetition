"""
Visualize the different forgetting curve formulas
Shows how each variant transforms the decay parameter
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Range of decay parameters (LSTM outputs)
o_x = np.linspace(-5, 2, 200)

# Fixed parameters
R = 0.9  # Target recall
alpha = 1.2  # Power law exponent
a, b = 0.1, 0.05  # Linear parameters
beta = 0.5  # Sigmoid modulation

# Calculate intervals for each variant
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# V1: Original
intervals_v1 = -np.log(R) / np.exp(o_x)

# V2: Power Law
intervals_v2 = -np.log(R) / (np.exp(o_x) ** alpha)

# V3: Linear Decay
intervals_v3 = -np.log(R) / (np.abs(a) * o_x + np.abs(b) + 1e-6)
intervals_v3 = np.clip(intervals_v3, 1, 90)

# V4: Sigmoid Modulated
intervals_v4 = -np.log(R) / (np.exp(o_x) * sigmoid(beta * o_x))

# V5: Adaptive Target (using fixed R for visualization)
intervals_v5 = -np.log(R) / np.exp(o_x)  # Same as V1 with fixed R

# Clip all to reasonable range
intervals_v1 = np.clip(intervals_v1, 1, 90)
intervals_v2 = np.clip(intervals_v2, 1, 90)
intervals_v4 = np.clip(intervals_v4, 1, 90)
intervals_v5 = np.clip(intervals_v5, 1, 90)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Forgetting Curve Formula Transformations', fontsize=16, fontweight='bold')

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

# Plot each variant
variants = [
    ('V1: Original\ninterval = -log(R) / exp(o(x))', intervals_v1, colors[0]),
    ('V2: Power Law\ninterval = -log(R) / (exp(o(x)))^α', intervals_v2, colors[1]),
    ('V3: Linear Decay\ninterval = -log(R) / (a*o(x) + b)', intervals_v3, colors[2]),
    ('V4: Sigmoid Modulated\ninterval = -log(R) / (exp(o(x)) * sigmoid(β*o(x)))', intervals_v4, colors[3]),
    ('V5: Adaptive Target\ninterval = -log(R(x)) / exp(o(x))', intervals_v5, colors[4])
]

for idx, (title, intervals, color) in enumerate(variants):
    ax = axes[idx // 3, idx % 3]
    ax.plot(o_x, intervals, linewidth=3, color=color)
    ax.set_xlabel('LSTM Output o(x)', fontsize=11)
    ax.set_ylabel('Review Interval (days)', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 90])
    
    # Add reference lines
    ax.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label='1 week')
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='1 month')
    ax.legend(fontsize=9)

# Comparison plot
ax = axes[1, 2]
ax.plot(o_x, intervals_v1, linewidth=2, label='V1: Original', color=colors[0])
ax.plot(o_x, intervals_v2, linewidth=2, label='V2: Power Law', color=colors[1], linestyle='--')
ax.plot(o_x, intervals_v3, linewidth=2, label='V3: Linear', color=colors[2], linestyle='-.')
ax.plot(o_x, intervals_v4, linewidth=2, label='V4: Sigmoid', color=colors[3], linestyle=':')
ax.set_xlabel('LSTM Output o(x)', fontsize=11)
ax.set_ylabel('Review Interval (days)', fontsize=11)
ax.set_title('All Variants Comparison', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 90])

plt.tight_layout()
plt.savefig('variant_results/formula_transformations.png', dpi=300, bbox_inches='tight')
print("✓ Saved formula transformation visualization")
plt.close()

# Create decay rate comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Decay Rate Transformations', fontsize=14, fontweight='bold')

# Decay rates
decay_v1 = np.exp(o_x)
decay_v2 = np.exp(o_x) ** alpha
decay_v3 = np.abs(a) * o_x + np.abs(b)
decay_v4 = np.exp(o_x) * sigmoid(beta * o_x)

ax = axes[0]
ax.plot(o_x, decay_v1, linewidth=2, label='V1: exp(o(x))', color=colors[0])
ax.plot(o_x, decay_v2, linewidth=2, label='V2: (exp(o(x)))^α', color=colors[1])
ax.plot(o_x, decay_v3, linewidth=2, label='V3: a*o(x) + b', color=colors[2])
ax.plot(o_x, decay_v4, linewidth=2, label='V4: exp(o(x)) * sigmoid(β*o(x))', color=colors[3])
ax.set_xlabel('LSTM Output o(x)')
ax.set_ylabel('Decay Rate')
ax.set_title('Decay Rate Functions')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Log scale
ax = axes[1]
ax.semilogy(o_x, decay_v1, linewidth=2, label='V1: exp(o(x))', color=colors[0])
ax.semilogy(o_x, decay_v2, linewidth=2, label='V2: (exp(o(x)))^α', color=colors[1])
ax.semilogy(o_x[o_x > -5], decay_v3[o_x > -5], linewidth=2, label='V3: a*o(x) + b', color=colors[2])
ax.semilogy(o_x, decay_v4, linewidth=2, label='V4: exp(o(x)) * sigmoid(β*o(x))', color=colors[3])
ax.set_xlabel('LSTM Output o(x)')
ax.set_ylabel('Decay Rate (log scale)')
ax.set_title('Decay Rate Functions (Log Scale)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variant_results/decay_rate_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved decay rate comparison")
plt.close()

print("\n✓ All formula visualizations saved to variant_results/")
