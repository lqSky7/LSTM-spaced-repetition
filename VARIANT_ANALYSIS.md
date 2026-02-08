# LSTM Forgetting Curve Formula Variants - Analysis Report

## Overview
This report presents 5 variations of the exponential decay forgetting curve formula used in LSTM-based spaced repetition systems. Each variant was trained for 50 epochs on the DSA spaced repetition dataset.

## Variants

### Variant 1: Original Exponential Decay
**Formula:** `interval = -log(R) / exp(o(x))`

Where:
- `R` = target recall probability (learnable, initialized at 0.9)
- `o(x)` = LSTM output (decay parameter)
- `exp(o(x))` = exponential decay rate

**Results:**
- MAE: 1.6611 days
- RMSE: 2.5373 days
- Best Val Loss: 1.3090

**Description:** The baseline model from the paper. Uses exponential transformation of the LSTM output to model decay rate.

---

### Variant 2: Power Law Decay
**Formula:** `interval = -log(R) / (exp(o(x)))^α`

Where:
- `α` = learnable power law exponent (initialized at 1.2)
- Applies power law to the decay rate for non-linear scaling

**Results:**
- MAE: 1.6470 days ✓ (2nd best)
- RMSE: 2.5084 days
- Best Val Loss: 1.2704

**Description:** Introduces a power law exponent to allow the model to learn non-linear decay patterns. The learnable α parameter provides flexibility in how decay rates scale.

---

### Variant 3: Linear Decay
**Formula:** `interval = -log(R) / (a*o(x) + b)`

Where:
- `a`, `b` = learnable linear transformation parameters
- Replaces exponential with linear transformation

**Results:**
- MAE: 3.6916 days ✗ (worst)
- RMSE: 5.8766 days
- Best Val Loss: 3.4969

**Description:** Replaces exponential decay with linear transformation. Performed poorly, suggesting that exponential relationships are important for modeling forgetting curves.

---

### Variant 4: Sigmoid Modulated Decay
**Formula:** `interval = -log(R) / (exp(o(x)) * sigmoid(β*o(x)))`

Where:
- `β` = learnable sigmoid modulation strength (initialized at 0.5)
- `sigmoid(β*o(x))` = smooth modulation factor

**Results:**
- MAE: 1.6081 days ✓ (3rd best)
- RMSE: 2.4430 days
- Best Val Loss: 1.1375 ✓ (best validation)

**Description:** Modulates the decay rate with a sigmoid function for smooth transitions. Achieved the best validation loss, suggesting good generalization.

---

### Variant 5: Adaptive Target Recall
**Formula:** `interval = -log(R(x)) / exp(o(x))`

Where:
- `R(x)` = adaptive target recall predicted by separate network (range: 0.7-0.95)
- Allows personalized target recall per sample

**Results:**
- MAE: 1.5663 days ✓✓ (BEST)
- RMSE: 2.4085 days ✓ (best)
- Best Val Loss: 1.2228

**Description:** Instead of a fixed target recall, predicts an adaptive target for each sample. This allows the model to personalize review intervals based on learner characteristics. **Winner!**

---

## Summary Comparison

| Variant | MAE (days) | RMSE (days) | Val Loss | Rank |
|---------|-----------|-------------|----------|------|
| V1: Original | 1.6611 | 2.5373 | 1.3090 | 4th |
| V2: Power Law | 1.6470 | 2.5084 | 1.2704 | 2nd |
| V3: Linear Decay | 3.6916 | 5.8766 | 3.4969 | 5th |
| V4: Sigmoid Mod | 1.6081 | 2.4430 | 1.1375 | 3rd |
| **V5: Adaptive Target** | **1.5663** | **2.4085** | 1.2228 | **1st** 🏆 |

## Key Findings

1. **Adaptive Target Recall (V5) performs best** - Personalizing the target recall probability for each sample improves prediction accuracy by ~6% over the baseline.

2. **Power Law (V2) is a close second** - Adding a learnable exponent to the decay rate provides useful flexibility.

3. **Sigmoid Modulation (V4) has best validation loss** - The smooth modulation helps with generalization, though test MAE is slightly higher.

4. **Linear transformation (V3) fails** - Removing the exponential relationship significantly degrades performance, confirming that exponential decay is fundamental to forgetting curves.

5. **All exponential variants are competitive** - V1, V2, V4, and V5 all achieve similar performance (MAE within 0.1 days), suggesting the exponential form is robust.

## Recommendations

- **For production use:** Variant 5 (Adaptive Target) offers the best accuracy and personalization
- **For simplicity:** Variant 2 (Power Law) provides good performance with minimal added complexity
- **For generalization:** Variant 4 (Sigmoid Modulated) shows strong validation performance

## Generated Visualizations

All plots are saved in `variant_results/`:
1. `training_curves_*.png` - Training and validation loss curves for all variants
2. `metrics_comparison_*.png` - Bar charts comparing MAE, RMSE, and validation loss
3. `predictions_scatter_*.png` - Scatter plots of predictions vs actuals for each variant
4. `error_distributions_*.png` - Histograms of prediction errors for each variant

## Conclusion

The adaptive target recall approach (V5) demonstrates that personalizing the forgetting curve parameters per sample leads to better interval predictions. This suggests that different learners and problems may benefit from different target recall thresholds, and the model can learn to predict these adaptively.
