# Forgetting Curve Formula Variants - Detailed Explanation

## The Original LSTM + Spaced Repetition Formula

**Base Formula:**
```
interval = -log(R) / exp(o(x))
```

Where:
- `interval` = days until next review
- `R` = target recall probability (fixed at 0.9, meaning we want 90% chance of remembering)
- `o(x)` = raw output from LSTM network (a single number, can be negative or positive)
- `exp(o(x))` = exponential function that converts LSTM output to decay rate

**How it works:**
1. LSTM looks at your learning history and outputs a number `o(x)` (e.g., -3.5)
2. We convert this to a decay rate: `exp(-3.5) = 0.03` (small number = slow forgetting)
3. We calculate: `interval = -log(0.9) / 0.03 = 3.5 days`
4. The model learns to output smaller `o(x)` for hard problems (shorter intervals) and larger `o(x)` for easy problems (longer intervals)

---

## Variant 1: Original (Baseline)

**Formula:** `interval = -log(R) / exp(o(x))`

**What's different:** Nothing! This is the baseline from the paper.

**Key characteristics:**
- Fixed target recall R = 0.9 (but learnable during training)
- Exponential decay rate
- Simple and interpretable

**Results:** MAE = 1.66 days

---

## Variant 2: Power Law Decay

**Formula:** `interval = -log(R) / (exp(o(x)))^α`

**What changed:**
```
Original:  decay_rate = exp(o(x))
Power Law: decay_rate = (exp(o(x)))^α
```

Where `α` (alpha) is a **learnable parameter** (starts at 1.2).

**Why this matters:**
- When α = 1.0 → same as original
- When α > 1.0 → decay rate grows faster (more aggressive spacing)
- When α < 1.0 → decay rate grows slower (more conservative spacing)

**Example:**
```
If LSTM outputs o(x) = -2:
- Original:  decay = exp(-2) = 0.135
- Power Law: decay = (exp(-2))^1.2 = 0.135^1.2 = 0.095
             → Slower decay = longer interval
```

**The insight:** Not all problems follow the same exponential curve. Some might need steeper or gentler curves. The model learns the best α.

**Results:** MAE = 1.65 days ✓ (2nd best - very close to best!)

---

## Variant 3: Linear Decay

**Formula:** `interval = -log(R) / (a*o(x) + b)`

**What changed:**
```
Original: decay_rate = exp(o(x))          [exponential]
Linear:   decay_rate = a*o(x) + b         [linear]
```

Where `a` and `b` are **learnable parameters**.

**Why this matters:**
- Removes the exponential relationship
- Tests if linear transformation is sufficient
- Much simpler mathematically

**Example:**
```
If LSTM outputs o(x) = -2:
- Original: decay = exp(-2) = 0.135
- Linear:   decay = 0.1*(-2) + 0.05 = -0.15 (needs abs value)
            → Completely different scaling!
```

**The insight:** This variant FAILED badly (MAE = 3.69 days). This proves that the exponential relationship is crucial for modeling forgetting curves. Human memory doesn't decay linearly!

**Results:** MAE = 3.69 days ✗ (worst by far)

---

## Variant 4: Sigmoid Modulated Decay

**Formula:** `interval = -log(R) / (exp(o(x)) * sigmoid(β*o(x)))`

**What changed:**
```
Original: decay_rate = exp(o(x))
Sigmoid:  decay_rate = exp(o(x)) * sigmoid(β*o(x))
```

Where:
- `sigmoid(x) = 1 / (1 + exp(-x))` (S-shaped curve, outputs 0 to 1)
- `β` (beta) is a **learnable parameter** controlling modulation strength

**Why this matters:**
- Adds a smooth "gating" mechanism
- When o(x) is very negative → sigmoid ≈ 0 → much slower decay
- When o(x) is very positive → sigmoid ≈ 1 → normal decay
- Creates smooth transitions instead of sharp exponential jumps

**Example:**
```
If LSTM outputs o(x) = -2:
- sigmoid(-2 * 0.5) = sigmoid(-1) = 0.27
- Original: decay = exp(-2) = 0.135
- Sigmoid:  decay = 0.135 * 0.27 = 0.036
            → Much slower decay = longer interval

If LSTM outputs o(x) = 2:
- sigmoid(2 * 0.5) = sigmoid(1) = 0.73
- Original: decay = exp(2) = 7.39
- Sigmoid:  decay = 7.39 * 0.73 = 5.39
            → Slightly slower decay
```

**The insight:** The sigmoid acts like a "confidence gate". When the LSTM is uncertain (extreme values), it moderates the decay rate. This helps with generalization!

**Results:** MAE = 1.61 days, Best Val Loss = 1.14 ✓ (best validation = best generalization)

---

## Variant 5: Adaptive Target Recall

**Formula:** `interval = -log(R(x)) / exp(o(x))`

**What changed:**
```
Original: R = 0.9 (fixed for everyone)
Adaptive: R(x) = predicted by separate neural network (0.7 to 0.95)
```

**Architecture:**
- Main LSTM → predicts decay parameter o(x)
- **Second network** → predicts personalized target recall R(x)

**Why this matters:**
- Different students have different goals!
- Beginner might need R = 0.95 (review more often, 95% recall)
- Expert might be fine with R = 0.75 (review less, 75% recall is enough)
- Hard problems might need higher R, easy problems lower R
- The model learns what target recall is appropriate for each situation

**Example:**
```
Student A (beginner) on hard problem:
- LSTM predicts: o(x) = -2, R(x) = 0.95
- interval = -log(0.95) / exp(-2) = 0.38 days (review soon!)

Student B (expert) on easy problem:
- LSTM predicts: o(x) = -1, R(x) = 0.75
- interval = -log(0.75) / exp(-1) = 0.78 days (can wait longer)
```

**The insight:** One-size-fits-all target recall is suboptimal. Different learners and problems need different recall thresholds. This is the most personalized approach!

**Results:** MAE = 1.57 days 🏆 (BEST! 6% better than baseline)

---

## Visual Comparison

### How decay rates differ:

```
LSTM Output (o(x)):  -4    -2     0     2
─────────────────────────────────────────
V1 Original:        0.02  0.14  1.00  7.39
V2 Power Law:       0.01  0.09  1.00  10.6  (steeper curve)
V3 Linear:          0.05  0.05  0.05  0.25  (flat, bad!)
V4 Sigmoid:         0.00  0.04  0.50  5.39  (moderated)
V5 Adaptive:        0.02  0.14  1.00  7.39  (same decay, different R)
```

### Key differences in behavior:

| Variant | Main Innovation | Best For |
|---------|----------------|----------|
| V1 | Baseline exponential | Standard use |
| V2 | Learnable power law exponent | Non-standard forgetting curves |
| V3 | Linear (failed) | ❌ Don't use |
| V4 | Smooth modulation | Better generalization |
| V5 | Personalized targets | Individual learners |

---

## Why These Changes Matter

### 1. **Exponential is essential** (V3 proves this)
Human forgetting follows exponential/power law curves, not linear. Removing exponential = disaster.

### 2. **Flexibility helps** (V2, V4, V5 all improve)
Adding learnable parameters (α, β, R(x)) lets the model adapt to different patterns.

### 3. **Personalization wins** (V5 is best)
Different learners need different review schedules. Predicting personalized target recall is the most effective approach.

### 4. **Smooth transitions matter** (V4 has best validation)
The sigmoid modulation prevents overfitting by smoothing extreme predictions.

---

## Bottom Line

**Original formula:** Works well but assumes everyone wants 90% recall.

**Best improvement (V5):** Let the model predict what recall target each person needs for each problem. Some need 95% (review often), others are fine with 75% (review less). This personalization reduces prediction error by 6%.

**Runner-up (V2):** Add a power law exponent to handle non-standard forgetting patterns. Almost as good as V5 but simpler.

**Surprising finding:** Linear decay (V3) completely fails, proving that exponential relationships are fundamental to how humans forget information.
