# Simple Explanation: What Each Variant Does Differently

## The Core Idea

All variants try to predict: **"How many days until you should review this problem again?"**

The formula structure is always:
```
interval = -log(target_recall) / decay_rate
```

The **decay_rate** determines how fast you forget. The variants differ in HOW they calculate this decay rate.

---

## Variant 1: Original (Baseline)

**What it does:**
```python
decay_rate = exp(o(x))
target_recall = 0.9  # fixed
interval = -log(0.9) / exp(o(x))
```

**In plain English:**
- LSTM outputs a number `o(x)` (like -3.2)
- Convert it to decay rate: `exp(-3.2) = 0.04`
- Calculate interval: `-log(0.9) / 0.04 = 2.6 days`
- Everyone gets the same target (90% recall)

**Concrete example:**
```
You solved "Two Sum" 3 times, got it right twice
LSTM says: o(x) = -2.5
Decay rate: exp(-2.5) = 0.082
Interval: -log(0.9) / 0.082 = 1.3 days
→ Review in 1.3 days
```

---

## Variant 2: Power Law

**What it does:**
```python
decay_rate = (exp(o(x))) ^ alpha
alpha = 1.2  # learnable, can change during training
interval = -log(0.9) / decay_rate
```

**What changed:** Raises the decay rate to a power (alpha)

**Why it matters:**
- If alpha = 1.0 → same as original
- If alpha = 1.2 → decay rate grows 20% faster
- If alpha = 0.8 → decay rate grows 20% slower

**Concrete example:**
```
Same "Two Sum" problem
LSTM says: o(x) = -2.5
Original decay: exp(-2.5) = 0.082
Power law decay: (0.082)^1.2 = 0.058  (slower!)
Interval: -log(0.9) / 0.058 = 1.8 days
→ Review in 1.8 days (longer than original)
```

**The insight:** Some problems need steeper curves, others need gentler curves. The model learns the best power.

---

## Variant 3: Linear Decay (FAILED)

**What it does:**
```python
decay_rate = a * o(x) + b
a = 0.1, b = 0.05  # learnable
interval = -log(0.9) / decay_rate
```

**What changed:** Uses linear math instead of exponential

**Concrete example:**
```
Same "Two Sum" problem
LSTM says: o(x) = -2.5
Linear decay: 0.1 * (-2.5) + 0.05 = -0.20 (negative!)
Need to use absolute value: 0.20
Interval: -log(0.9) / 0.20 = 0.5 days
→ Review in 0.5 days (way too short!)
```

**Why it failed:** Human memory doesn't decay linearly! It follows exponential curves. This variant proved that exponential is essential.

**Result:** MAE = 3.69 days (worst by far)

---

## Variant 4: Sigmoid Modulated

**What it does:**
```python
decay_rate = exp(o(x)) * sigmoid(beta * o(x))
beta = 0.5  # learnable
sigmoid(x) = 1 / (1 + exp(-x))  # S-curve, outputs 0 to 1
interval = -log(0.9) / decay_rate
```

**What changed:** Multiplies decay rate by a sigmoid "gate"

**Why it matters:**
- Sigmoid acts like a confidence score
- When LSTM is uncertain → sigmoid reduces decay rate
- Creates smooth transitions instead of jumps

**Concrete example:**
```
Same "Two Sum" problem
LSTM says: o(x) = -2.5
Exponential part: exp(-2.5) = 0.082
Sigmoid part: sigmoid(0.5 * -2.5) = sigmoid(-1.25) = 0.22
Combined decay: 0.082 * 0.22 = 0.018  (much slower!)
Interval: -log(0.9) / 0.018 = 5.8 days
→ Review in 5.8 days (much longer!)
```

**The insight:** When the model is uncertain (extreme LSTM outputs), it moderates the prediction. This prevents overfitting!

**Result:** Best validation loss (1.14) = best generalization

---

## Variant 5: Adaptive Target (WINNER)

**What it does:**
```python
decay_rate = exp(o(x))  # same as original
target_recall = predict_target(x)  # NEW! Predicted by separate network
interval = -log(target_recall) / decay_rate
```

**What changed:** Instead of fixed 90% target, predicts a personalized target (70% to 95%)

**Why it matters:**
- Beginners might need 95% recall (review more often)
- Experts might be fine with 75% recall (review less often)
- Hard problems need higher targets, easy problems lower targets

**Concrete example:**

**Beginner on hard problem:**
```
LSTM predicts: o(x) = -2.5, target = 0.95
Decay rate: exp(-2.5) = 0.082
Interval: -log(0.95) / 0.082 = 0.6 days
→ Review in 0.6 days (soon!)
```

**Expert on easy problem:**
```
LSTM predicts: o(x) = -2.5, target = 0.75
Decay rate: exp(-2.5) = 0.082
Interval: -log(0.75) / 0.082 = 3.5 days
→ Review in 3.5 days (can wait longer)
```

**The insight:** One-size-fits-all doesn't work! Different people need different review schedules. This is the most personalized approach.

**Result:** MAE = 1.57 days (6% better than baseline) 🏆

---

## Side-by-Side Comparison

Let's say LSTM outputs `o(x) = -2.0` for a problem:

| Variant | Decay Rate Calculation | Decay Rate | Target | Interval |
|---------|----------------------|------------|--------|----------|
| V1 Original | exp(-2.0) | 0.135 | 0.90 | 0.78 days |
| V2 Power Law | (exp(-2.0))^1.2 | 0.095 | 0.90 | 1.11 days |
| V3 Linear | 0.1*(-2.0)+0.05 | 0.15 | 0.90 | 0.70 days |
| V4 Sigmoid | exp(-2.0)*sigmoid(-1.0) | 0.037 | 0.90 | 2.85 days |
| V5 Adaptive | exp(-2.0) | 0.135 | 0.85 | 1.21 days |

See how different they are? Same LSTM output, but very different intervals!

---

## What Makes Each Special?

### V1: Original
- **Strength:** Simple, proven, interpretable
- **Weakness:** Fixed target for everyone
- **Use when:** You want the baseline

### V2: Power Law
- **Strength:** Flexible curve shape
- **Weakness:** Only one extra parameter
- **Use when:** You want simple improvement

### V3: Linear
- **Strength:** None
- **Weakness:** Doesn't model human memory
- **Use when:** Never! This proved exponential is essential

### V4: Sigmoid Modulated
- **Strength:** Best generalization, smooth predictions
- **Weakness:** Slightly more complex
- **Use when:** You care about avoiding overfitting

### V5: Adaptive Target
- **Strength:** Most personalized, best accuracy
- **Weakness:** Needs separate network (more parameters)
- **Use when:** You want the best performance

---

## The Big Picture

**Question:** Why not just use the original formula?

**Answer:** Because people are different!

- Some students need to review at 95% recall (perfectionist)
- Others are fine with 75% recall (efficient learner)
- Hard problems need different schedules than easy ones
- Beginners need different schedules than experts

**V5 (Adaptive Target) learns all of this automatically!**

Instead of forcing everyone to aim for 90% recall, it predicts:
- "This beginner on this hard problem needs 95% recall"
- "This expert on this easy problem is fine with 75% recall"

That's why it wins by 6%!

---

## Final Ranking

1. **V5: Adaptive Target** - 1.57 MAE 🏆 (personalized targets)
2. **V2: Power Law** - 1.65 MAE (flexible curves)
3. **V4: Sigmoid Mod** - 1.61 MAE (smooth predictions)
4. **V1: Original** - 1.66 MAE (baseline)
5. **V3: Linear** - 3.69 MAE ❌ (don't use)

**Bottom line:** If you want the best spaced repetition system, use V5. It personalizes review schedules for each learner and problem!
