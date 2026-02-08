"""
Create visual diagrams showing the architectural differences between variants
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('LSTM Forgetting Curve Variants - Architecture Comparison', 
             fontsize=18, fontweight='bold', y=0.995)

def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=11):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Variant 1: Original
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('V1: Original Exponential Decay', fontsize=14, fontweight='bold', pad=20)

draw_box(ax, 1, 7, 2, 1.5, 'Input\nSequence', 'lightgreen')
draw_box(ax, 4, 7, 2, 1.5, 'LSTM', 'lightblue')
draw_box(ax, 7, 7, 2, 1.5, 'FC\nLayers', 'lightcoral')
draw_arrow(ax, 3, 7.75, 4, 7.75)
draw_arrow(ax, 6, 7.75, 7, 7.75)

draw_box(ax, 4, 4.5, 2, 1.2, 'o(x)', 'lightyellow')
draw_arrow(ax, 8, 7, 5, 5.7)

draw_box(ax, 1, 4.5, 2, 1.2, 'R = 0.9\n(fixed)', 'lightgray')

draw_box(ax, 3.5, 2, 3, 1.5, 'interval =\n-log(R) / exp(o(x))', 'wheat')
draw_arrow(ax, 5, 4.5, 5, 3.5)
draw_arrow(ax, 3, 5.1, 4, 3.2)

draw_box(ax, 3.5, 0.2, 3, 1, 'Review\nInterval', 'lightgreen')
draw_arrow(ax, 5, 2, 5, 1.2)

ax.text(5, 9.3, '✓ Simple & interpretable\n✓ Fixed target recall', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Variant 2: Power Law
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('V2: Power Law Decay', fontsize=14, fontweight='bold', pad=20)

draw_box(ax, 1, 7, 2, 1.5, 'Input\nSequence', 'lightgreen')
draw_box(ax, 4, 7, 2, 1.5, 'LSTM', 'lightblue')
draw_box(ax, 7, 7, 2, 1.5, 'FC\nLayers', 'lightcoral')
draw_arrow(ax, 3, 7.75, 4, 7.75)
draw_arrow(ax, 6, 7.75, 7, 7.75)

draw_box(ax, 4, 4.5, 2, 1.2, 'o(x)', 'lightyellow')
draw_arrow(ax, 8, 7, 5, 5.7)

draw_box(ax, 0.5, 4.5, 2, 1.2, 'R = 0.9', 'lightgray')
draw_box(ax, 7, 4.5, 2, 1.2, 'α = 1.2\n(learnable)', 'orange')

draw_box(ax, 3, 2, 4, 1.5, 'interval =\n-log(R) / (exp(o(x)))^α', 'wheat')
draw_arrow(ax, 5, 4.5, 5, 3.5)
draw_arrow(ax, 2.5, 5.1, 3.5, 3.2)
draw_arrow(ax, 8, 4.5, 6.5, 3.2)

draw_box(ax, 3.5, 0.2, 3, 1, 'Review\nInterval', 'lightgreen')
draw_arrow(ax, 5, 2, 5, 1.2)

ax.text(5, 9.3, '✓ Learnable exponent α\n✓ Adapts curve steepness', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Variant 3: Linear Decay
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('V3: Linear Decay (Failed)', fontsize=14, fontweight='bold', pad=20)

draw_box(ax, 1, 7, 2, 1.5, 'Input\nSequence', 'lightgreen')
draw_box(ax, 4, 7, 2, 1.5, 'LSTM', 'lightblue')
draw_box(ax, 7, 7, 2, 1.5, 'FC\nLayers', 'lightcoral')
draw_arrow(ax, 3, 7.75, 4, 7.75)
draw_arrow(ax, 6, 7.75, 7, 7.75)

draw_box(ax, 4, 4.5, 2, 1.2, 'o(x)', 'lightyellow')
draw_arrow(ax, 8, 7, 5, 5.7)

draw_box(ax, 0.3, 4.5, 1.5, 1.2, 'R = 0.9', 'lightgray')
draw_box(ax, 6.5, 4.5, 3, 1.2, 'a, b\n(learnable)', 'orange')

draw_box(ax, 2.5, 2, 5, 1.5, 'interval =\n-log(R) / (a*o(x) + b)', 'wheat')
draw_arrow(ax, 5, 4.5, 5, 3.5)
draw_arrow(ax, 1.8, 5.1, 3, 3.2)
draw_arrow(ax, 8, 4.5, 7, 3.2)

draw_box(ax, 3.5, 0.2, 3, 1, 'Review\nInterval', 'lightgreen')
draw_arrow(ax, 5, 2, 5, 1.2)

ax.text(5, 9.3, '✗ Removes exponential\n✗ MAE = 3.69 (worst!)', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='#ffcccc'))

# Variant 4: Sigmoid Modulated
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('V4: Sigmoid Modulated Decay', fontsize=14, fontweight='bold', pad=20)

draw_box(ax, 1, 7, 2, 1.5, 'Input\nSequence', 'lightgreen')
draw_box(ax, 4, 7, 2, 1.5, 'LSTM', 'lightblue')
draw_box(ax, 7, 7, 2, 1.5, 'FC\nLayers', 'lightcoral')
draw_arrow(ax, 3, 7.75, 4, 7.75)
draw_arrow(ax, 6, 7.75, 7, 7.75)

draw_box(ax, 4, 4.5, 2, 1.2, 'o(x)', 'lightyellow')
draw_arrow(ax, 8, 7, 5, 5.7)

draw_box(ax, 0.3, 4.5, 1.5, 1.2, 'R = 0.9', 'lightgray')
draw_box(ax, 6.5, 4.5, 3, 1.2, 'β = 0.5\n(learnable)', 'orange')

draw_box(ax, 2, 2, 6, 1.5, 'interval = -log(R) /\n(exp(o(x)) * sigmoid(β*o(x)))', 'wheat', fontsize=10)
draw_arrow(ax, 5, 4.5, 5, 3.5)
draw_arrow(ax, 1.8, 5.1, 2.5, 3.2)
draw_arrow(ax, 8, 4.5, 7.5, 3.2)

draw_box(ax, 3.5, 0.2, 3, 1, 'Review\nInterval', 'lightgreen')
draw_arrow(ax, 5, 2, 5, 1.2)

ax.text(5, 9.3, '✓ Smooth modulation\n✓ Best validation loss', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Variant 5: Adaptive Target
ax = axes[2, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('V5: Adaptive Target Recall (BEST)', fontsize=14, fontweight='bold', pad=20)

draw_box(ax, 1, 7, 2, 1.5, 'Input\nSequence', 'lightgreen')
draw_box(ax, 4, 7, 2, 1.5, 'LSTM', 'lightblue')
draw_arrow(ax, 3, 7.75, 4, 7.75)

# Two branches
draw_box(ax, 7, 8, 2, 1, 'FC\nDecay', 'lightcoral')
draw_box(ax, 7, 6, 2, 1, 'FC\nTarget', 'plum')
draw_arrow(ax, 6, 8.2, 7, 8.5)
draw_arrow(ax, 6, 7.3, 7, 6.5)

draw_box(ax, 7, 4.5, 2, 1.2, 'o(x)', 'lightyellow')
draw_box(ax, 1, 4.5, 2, 1.2, 'R(x)\n0.7-0.95', 'orange')
draw_arrow(ax, 8, 8, 8, 5.7)
draw_arrow(ax, 8, 6, 2, 5.7)

draw_box(ax, 3, 2, 4, 1.5, 'interval =\n-log(R(x)) / exp(o(x))', 'wheat')
draw_arrow(ax, 8, 4.5, 6.5, 3.2)
draw_arrow(ax, 3, 4.5, 4, 3.2)

draw_box(ax, 3.5, 0.2, 3, 1, 'Review\nInterval', 'lightgreen')
draw_arrow(ax, 5, 2, 5, 1.2)

ax.text(5, 9.5, '✓✓ Personalized target recall\n✓✓ MAE = 1.57 (BEST!)', 
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='#ccffcc'))

# Summary comparison
ax = axes[2, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

summary_text = """
RESULTS (MAE in days):

V1: Original           1.66 ⭐⭐⭐
V2: Power Law          1.65 ⭐⭐⭐⭐
V3: Linear Decay       3.69 ❌
V4: Sigmoid Mod        1.61 ⭐⭐⭐⭐
V5: Adaptive Target    1.57 🏆

KEY INSIGHTS:

• Exponential decay is essential
  (V3 linear failed badly)

• Learnable parameters help
  (V2, V4, V5 all improved)

• Personalization wins
  (V5 best by 6%)

• Smooth modulation aids
  generalization (V4)
"""

ax.text(5, 5, summary_text, ha='center', va='center', fontsize=11,
       family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                     edgecolor='black', linewidth=2))

plt.tight_layout()
plt.savefig('variant_results/architecture_diagrams.png', dpi=300, bbox_inches='tight')
print("✓ Saved architecture diagrams")
plt.close()

# Create a simple formula comparison chart
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

title = "Forgetting Curve Formula Comparison"
ax.text(5, 9.5, title, ha='center', fontsize=18, fontweight='bold')

formulas = [
    ("V1: Original", "interval = -log(R) / exp(o(x))", "1.66", "Baseline"),
    ("V2: Power Law", "interval = -log(R) / (exp(o(x)))^α", "1.65", "Learnable exponent"),
    ("V3: Linear", "interval = -log(R) / (a·o(x) + b)", "3.69", "Failed - no exponential"),
    ("V4: Sigmoid", "interval = -log(R) / (exp(o(x)) · sigmoid(β·o(x)))", "1.61", "Smooth modulation"),
    ("V5: Adaptive", "interval = -log(R(x)) / exp(o(x))", "1.57", "Personalized target"),
]

y_start = 8.5
y_step = 1.6

for i, (name, formula, mae, desc) in enumerate(formulas):
    y = y_start - i * y_step
    
    # Color based on performance
    if mae == "1.57":
        color = '#ccffcc'  # Best - green
    elif mae == "3.69":
        color = '#ffcccc'  # Worst - red
    else:
        color = '#ffffcc'  # Good - yellow
    
    # Draw box
    box = FancyBboxPatch((0.5, y-0.6), 9, 1.4, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(1, y+0.3, name, fontsize=12, fontweight='bold', va='center')
    ax.text(5, y+0.3, formula, fontsize=11, va='center', family='monospace')
    ax.text(8.5, y+0.3, f"MAE: {mae}", fontsize=11, va='center', fontweight='bold')
    ax.text(5, y-0.2, desc, fontsize=9, va='center', style='italic', color='#555555')

plt.tight_layout()
plt.savefig('variant_results/formula_comparison_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved formula comparison chart")
plt.close()

print("\n✓ All architecture diagrams saved to variant_results/")
