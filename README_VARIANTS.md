# LSTM Forgetting Curve Variants - Complete Guide

## 📊 Quick Results

I trained 5 different variations of the LSTM forgetting curve formula. Here are the results:

| Variant | MAE (days) | RMSE (days) | Improvement | Status |
|---------|-----------|-------------|-------------|--------|
| V1: Original | 1.66 | 2.54 | baseline | ⭐⭐⭐ |
| V2: Power Law | 1.65 | 2.51 | +0.6% | ⭐⭐⭐⭐ |
| V3: Linear Decay | 3.69 | 5.88 | -122% | ❌ FAILED |
| V4: Sigmoid Mod | 1.61 | 2.44 | +3.0% | ⭐⭐⭐⭐ |
| V5: Adaptive Target | **1.57** | **2.41** | **+5.4%** | 🏆 WINNER |

**Winner: Variant 5 (Adaptive Target Recall)** - 5.4% better than baseline!

---

## 🎯 What Each Variant Does

### V1: Original Exponential Decay (Baseline)
```
interval = -log(R) / exp(o(x))
```
- **What it is:** The standard formula from the paper
- **How it works:** LSTM outputs decay parameter, fixed 90% target recall
- **Pros:** Simple, proven, interpretable
- **Cons:** One-size-fits-all approach

### V2: Power Law Decay
```
interval = -log(R) / (exp(o(x)))^α
```
- **What changed:** Added learnable power exponent α (starts at 1.2)
- **How it works:** Allows non-linear scaling of decay rates
- **Pros:** Flexible curve shapes, minimal complexity
- **Cons:** Only one extra parameter
- **Result:** 2nd best (1.65 MAE)

### V3: Linear Decay
```
interval = -log(R) / (a*o(x) + b)
```
- **What changed:** Replaced exponential with linear transformation
- **How it works:** Uses learnable linear parameters a, b
- **Pros:** None - this was an experiment
- **Cons:** Doesn't model human memory correctly
- **Result:** FAILED (3.69 MAE - worst by far)
- **Lesson:** Exponential decay is ESSENTIAL for forgetting curves!

### V4: Sigmoid Modulated Decay
```
interval = -log(R) / (exp(o(x)) * sigmoid(β*o(x)))
```
- **What changed:** Added sigmoid modulation with learnable β
- **How it works:** Sigmoid acts as a "confidence gate" to smooth predictions
- **Pros:** Best validation loss (1.14), prevents overfitting
- **Cons:** Slightly more complex formula
- **Result:** 3rd best (1.61 MAE), best generalization

### V5: Adaptive Target Recall (WINNER 🏆)
```
interval = -log(R(x)) / exp(o(x))
```
- **What changed:** Target recall R is predicted by separate network (0.7-0.95 range)
- **How it works:** Personalizes target recall for each learner/problem
- **Pros:** Most personalized, best accuracy
- **Cons:** Requires additional neural network
- **Result:** BEST (1.57 MAE - 5.4% improvement!)

---

## 📁 Generated Files

### Documentation
- `SIMPLE_EXPLANATION.md` - Easy-to-understand explanation with examples
- `FORMULA_EXPLAINED.md` - Detailed technical breakdown
- `VARIANT_ANALYSIS.md` - Research-style analysis report
- `README_VARIANTS.md` - This file

### Code
- `train_variants.py` - Main training script for all 5 variants
- `visualize_formulas.py` - Formula transformation visualizations
- `create_architecture_diagram.py` - Architecture comparison diagrams

### Results (in `variant_results/`)
- `training_curves_*.png` - Training/validation loss for all variants
- `metrics_comparison_*.png` - Bar charts comparing MAE, RMSE, validation loss
- `predictions_scatter_*.png` - Scatter plots of predictions vs actuals
- `error_distributions_*.png` - Error distribution histograms
- `formula_transformations.png` - How each formula transforms LSTM output
- `decay_rate_comparison.png` - Decay rate functions comparison
- `architecture_diagrams.png` - Visual architecture comparison
- `formula_comparison_chart.png` - Side-by-side formula comparison
- `summary_*.json` - Numerical results

---

## 🔍 Key Insights

### 1. Exponential Decay is Essential
V3 (Linear Decay) failed spectacularly (MAE = 3.69 vs 1.66 baseline). This proves that exponential relationships are fundamental to modeling human forgetting curves.

### 2. Personalization Wins
V5 (Adaptive Target) achieved the best results by predicting personalized target recall probabilities. Different learners and problems need different review schedules!

### 3. Smooth Modulation Helps Generalization
V4 (Sigmoid Modulated) had the best validation loss, showing that smooth transitions prevent overfitting.

### 4. Simple Improvements Work
V2 (Power Law) achieved nearly the same performance as V5 with just one extra parameter. Sometimes simple is better!

---

## 💡 Real-World Example

Let's say you're reviewing "Two Sum" problem and the LSTM outputs `o(x) = -2.0`:

**V1 (Original):**
- Decay: exp(-2.0) = 0.135
- Target: 90% (fixed)
- Interval: 0.78 days
- "Review tomorrow"

**V5 (Adaptive - Beginner):**
- Decay: exp(-2.0) = 0.135
- Target: 95% (predicted)
- Interval: 0.62 days
- "Review today!" (more frequent for beginners)

**V5 (Adaptive - Expert):**
- Decay: exp(-2.0) = 0.135
- Target: 75% (predicted)
- Interval: 2.13 days
- "Review in 2 days" (less frequent for experts)

See the difference? V5 personalizes the schedule!

---

## 🚀 How to Use

### Train all variants:
```bash
python3 train_variants.py --dataset dsa_spaced_repetition_dataset.csv --epochs 50
```

### Generate visualizations:
```bash
python3 visualize_formulas.py
python3 create_architecture_diagram.py
```

### Results will be saved in:
- `variant_results/` - All plots and metrics
- Console output - Training progress and final summary

---

## 📈 Performance Breakdown

### Mean Absolute Error (MAE)
Lower is better - measures average prediction error in days

- V5: 1.57 days 🏆
- V4: 1.61 days
- V2: 1.65 days
- V1: 1.66 days (baseline)
- V3: 3.69 days ❌

### Root Mean Squared Error (RMSE)
Lower is better - penalizes large errors more

- V5: 2.41 days 🏆
- V4: 2.44 days
- V2: 2.51 days
- V1: 2.54 days (baseline)
- V3: 5.88 days ❌

### Validation Loss
Lower is better - indicates generalization ability

- V4: 1.14 🏆 (best generalization!)
- V5: 1.22
- V2: 1.27
- V1: 1.31 (baseline)
- V3: 3.50 ❌

---

## 🎓 Recommendations

**For Production Use:**
- Use **V5 (Adaptive Target)** - Best accuracy and personalization

**For Simplicity:**
- Use **V2 (Power Law)** - Nearly as good with minimal added complexity

**For Generalization:**
- Use **V4 (Sigmoid Modulated)** - Best validation performance

**For Research:**
- Use **V1 (Original)** - Standard baseline for comparisons

**Never Use:**
- **V3 (Linear Decay)** - Doesn't model human memory correctly

---

## 📚 Further Reading

1. **SIMPLE_EXPLANATION.md** - Start here! Easy examples and concrete numbers
2. **FORMULA_EXPLAINED.md** - Deep dive into each formula with math
3. **VARIANT_ANALYSIS.md** - Research-style analysis and findings

---

## 🏆 Conclusion

By making small tweaks to the forgetting curve formula, we achieved:
- **5.4% improvement** in prediction accuracy (V5)
- **Proof** that exponential decay is essential (V3 failure)
- **Evidence** that personalization matters (V5 success)
- **Insight** that smooth modulation helps generalization (V4)

The winner is **V5: Adaptive Target Recall** - it learns to personalize review schedules for each learner and problem, resulting in the most accurate interval predictions!
