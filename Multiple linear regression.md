# Multiple Linear Regression — Deep Mechanical Notes

> **Core Mental Model:** Multiple Linear Regression is the **exact same single neuron** (`nn.Linear(k, 1)` in PyTorch) with **no hidden layers** and **no activation switch**. Instead of 1 input column (`x`), it takes **`k` input columns** (`x1, x2, ..., xk`), gives each column its own shared **weight knob** (`w1, w2, ..., wk`), and still keeps **only 1 shared bias (`b`)**.

---

## 1. Architecture (What Is Inside the Machine)

Compare Simple vs. Multiple side-by-side:

| Model | Formula | Knobs Learned | PyTorch Equivalent |
| :--- | :--- | :--- | :--- |
| **Simple Linear Regression** | `(w * x) + b` | `w`, `b` (2 knobs) | `nn.Linear(1, 1)` |
| **Multiple Linear Regression** | `(w1 * x1) + (w2 * x2) + ... + (wk * xk) + b` | `w1 ... wk`, `b` (`k + 1` knobs) | `nn.Linear(k, 1)` |

```text
  [ Input: x1 ] ──( multiply by shared w1 )──┐
                                             ├──> (+) ──> [ Output: guess ]
  [ Input: x2 ] ──( multiply by shared w2 )──┘     ^
                                                   |
                                          [ Shared Bias: b ]
```

### Every Part in a 2-Feature Model

| Part | Role | Changes Per Row? |
| :--- | :--- | :--- |
| **Inputs (`x1`, `x2`)** | Raw feature values for the current row (e.g., rooms, bathrooms) | **Yes** (new values every row) |
| **Weights (`w1`, `w2`)** | One multiplier per column deciding how strongly that column pushes the output | **No** (shared across all rows) |
| **Bias (`b`)** | Single base offset when **all** inputs are `0` | **No** (still only **1** bias total) |
| **Output (`guess`)** | `(w1 * x1) + (w2 * x2) + b` | **Yes** (changes because `x1`, `x2` change) |

---

## 2. What `w1`, `w2`, and `b` Physically Mean

* **Weight `w1`:** How much the guess changes when `x1` goes up by `+1`, **while `x2` stays frozen**.
  * Example: If `w1 = 2` and `w2 = 3`, adding `+1` room (`x1`) adds `+2` to the price *if bathrooms (`x2`) don't change*.
* **Weight `w2`:** How much the guess changes when `x2` goes up by `+1`, **while `x1` stays frozen**.
* **Bias `b`:** The starting baseline when `x1 = 0` AND `x2 = 0`.
* **Why is there still only 1 bias `b`?** Adding `b1 + b2` is pointless because two plain numbers added together are just one combined number (`3 + 4` is just `7`). So the model only ever needs **one** bias.

---

## 3. The Training Engine (Gradient Descent Loop)

Nothing in the loop changes—you just add **one extra blame bucket** for every new input column.

Start blind: `w1 = 0`, `w2 = 0`, `b = 0`, `lr = 0.05`. Every round runs through **all `N` rows** in 6 exact steps:

1. **Forward Pass (Guess every row):**
   `guess = (w1 * x1) + (w2 * x2) + b`
2. **Individual Miss (Check every row):**
   `miss = real - guess`
   * Notice there is still only **one `miss`** per row—all weights share responsibility for the same prediction error.
3. **Overall Error (MSE — Error Meter):**
   `mse = sum(miss * miss) / N`
4. **Collect Blame Across All Rows (One Bucket Per Knob):**
   * **Bucket for `w1`:** `sum(x1 * miss)` — column `x1` only gets blamed in proportion to `x1`'s size.
   * **Bucket for `w2`:** `sum(x2 * miss)` — column `x2` only gets blamed in proportion to `x2`'s size.
   * **Bucket for `b`:** `sum(miss)` — every row votes equally on `b`.
5. **Compute Gradients (Same `-(2 / N)` Multiplier for All):**
   * `w1_grad = -(2 / N) * sum(x1 * miss)`
   * `w2_grad = -(2 / N) * sum(x2 * miss)`
   * `b_grad  = -(2 / N) * sum(miss)`
6. **Nudge All Knobs Simultaneously:**
   * `w1 = w1 - (lr * w1_grad)`
   * `w2 = w2 - (lr * w2_grad)`
   * `b  = b  - (lr * b_grad)`

> **Crucial Rule:** All weights (`w1`, `w2`, `b`) stay **locked** during the inner row loop. Update all of them **once** at the end of the round after every row has voted.

---

## 4. Complete Numerical Trace (4 Rows, 2 Inputs)

**Dataset (`N = 4`):** Suppose `x1` = rooms, `x2` = bathrooms, `real` = price.
Start blind: **`w1 = 0`**, **`w2 = 0`**, **`b = 0`**, **`lr = 0.05`**.

### Round 1 (`w1 = 0`, `w2 = 0`, `b = 0`)

| `x1` | `x2` | `real` | `guess` | `miss` (`real - guess`) | `miss^2` | `x1 * miss` (`w1` blame) | `x2 * miss` (`w2` blame) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1` | `2` | `9` | `0` | `+9` | `81` | `9` | `18` |
| `2` | `1` | `8` | `0` | `+8` | `64` | `16` | `8` |
| `3` | `2` | `13` | `0` | `+13` | `169` | `39` | `26` |
| `2` | `3` | `14` | `0` | `+14` | `196` | `28` | `42` |
| **Total** | | | | **`44`** | **`510`** | **`92`** | **`94`** |

* **MSE:** `510 / 4 = 127.5`
* **Gradients (`-(2 / 4) = -0.5`):**
  * `w1_grad = -0.5 * 92 = -46.0`
  * `w2_grad = -0.5 * 94 = -47.0`
  * `b_grad  = -0.5 * 44 = -22.0`
* **Updated Knobs (`lr = 0.05`):**
  * `w1 = 0 - (0.05 * -46.0) = 2.30`
  * `w2 = 0 - (0.05 * -47.0) = 2.35`
  * `b  = 0 - (0.05 * -22.0) = 1.10`

*(Repeating this loop ~500 times freezes the knobs at **`w1 = 2.0`**, **`w2 = 3.0`**, **`b = 1.0`**.)*

### Final Frozen Model Check (`w1 = 2`, `w2 = 3`, `b = 1`)

| `x1` | `x2` | `real` | `guess` (`2*x1 + 3*x2 + 1`) | `miss` |
| :--- | :--- | :--- | :--- | :--- |
| `1` | `2` | `9` | `2 + 6 + 1 = 9` | `0` |
| `2` | `1` | `8` | `4 + 3 + 1 = 8` | `0` |
| `3` | `2` | `13` | `6 + 6 + 1 = 13` | `0` |
| `2` | `3` | `14` | `4 + 9 + 1 = 14` | `0` |

* **Prediction on brand-new row (`x1 = 4, x2 = 1`):** `(2 * 4) + (3 * 1) + 1 = 12`.

---

## 5. Pure Python Implementation (From Scratch)

```python
# Dataset (4 rows, 2 input columns)
X1 = [1, 2, 3, 2]
X2 = [2, 1, 2, 3]
Y  = [9, 8, 13, 14]
N  = len(Y)

# 1. Start blind
w1 = 0.0
w2 = 0.0
b  = 0.0
lr = 0.05

# 2. Training loop (Gradient Descent)
for epoch in range(1000):
    # Reset buckets every round
    sum_miss = 0.0
    sum_sq_miss = 0.0
    sum_x1_miss = 0.0
    sum_x2_miss = 0.0

    # Walk through all rows with LOCKED w1, w2, b
    for i in range(N):
        x1 = X1[i]
        x2 = X2[i]
        real = Y[i]

        guess = (w1 * x1) + (w2 * x2) + b
        miss = real - guess

        sum_miss += miss
        sum_sq_miss += miss * miss
        sum_x1_miss += x1 * miss
        sum_x2_miss += x2 * miss

    # Overall MSE (error meter)
    mse = sum_sq_miss / N

    # Overall gradients across all rows
    w1_grad = -(2 / N) * sum_x1_miss
    w2_grad = -(2 / N) * sum_x2_miss
    b_grad  = -(2 / N) * sum_miss

    # Nudge all knobs ONCE per round
    w1 = w1 - (lr * w1_grad)
    w2 = w2 - (lr * w2_grad)
    b  = b  - (lr * b_grad)

print(f"Frozen w1:  {w1:.4f}")   # 2.0000
print(f"Frozen w2:  {w2:.4f}")   # 3.0000
print(f"Frozen b:   {b:.4f}")    # 1.0000
print(f"Final MSE:  {mse:.4f}")  # 0.0000

# 3. Predict on brand-new row (x1 = 4, x2 = 1)
print("Guess:", round((w1 * 4) + (w2 * 1) + b, 2))  # 12.0
```

---

## 6. Production Version (`scikit-learn` — Uses OLS Internally)

In production, `scikit-learn` uses **Matrix OLS** (solves for all weights simultaneously in one shot—no `lr`, no epochs).

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 2D table: 4 rows, 2 columns [x1, x2]
X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [2, 3]
])
y = np.array([9, 8, 13, 14])

model = LinearRegression()
model.fit(X, y)  # Solves for w1, w2, b in one shot

print("Frozen weights [w1, w2]:", np.round(model.coef_, 4))  # [2. 3.]
print("Frozen bias b:          ", round(model.intercept_, 4)) # 1.0

# Predict for 1 new row: [x1=4, x2=1] (must stay 2D: double brackets)
print("Guess:", model.predict([[4, 1]])[0])  # 12.0
```

* **`model.coef_`** returns a 1D array with one weight per column (`[w1, w2]`).
* **`model.intercept_`** returns the single bias `b`.
* **`model.predict([[4, 1]])[0]`**: Double brackets = 1 row with 2 columns; `[0]` pulls the plain number out of the returned list.

---

## 7. The 2 New Traps That Only Exist With Multiple Columns

### Trap 1: Scale Mismatch (Feature Scaling)
Suppose `x1` = **House Size (`2000` sq ft)** and `x2` = **Bedrooms (`3`)**.
* **What breaks:** `x1 * miss` is roughly **600x larger** than `x2 * miss`.
* In Gradient Descent, if `lr` is small enough to stop `w1` from exploding, `w2` barely moves. If `lr` is big enough to move `w2`, `w1` explodes to `NaN`.
* **Fix:** Scale all input columns to the same range (mean `0`, spread `1`) using `StandardScaler` before training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale training inputs
# For new test data, ONLY call transform (never fit_transform):
# X_new_scaled = scaler.transform([[4, 1]])
```

> **Note:** Plain `LinearRegression()` (OLS) handles unscaled numbers mathematically without exploding, **but** scaling is mandatory for Gradient Descent (`SGDRegressor`), Neural Networks, Ridge, Lasso, and ElasticNet.

---

### Trap 2: Copycat Columns (Multicollinearity)
Suppose `x1` = **Size in Sq Ft** and `x2` = **Size in Sq Meters** (two columns carrying the exact same information).
* **What breaks:** The model has infinite ways to split the weight between `w1` and `w2` and still get the exact same guess:
  * `w1 = 2, w2 = 0` gives the same output as `w1 = 1000, w2 = -998`.
* **Result:** Predictions may still look okay on training data, **but the individual weights become wildly unstable garbage**—tiny changes in data flip `w1` and `w2` completely.
* **Fix:** Drop duplicate/highly correlated columns, or use **Ridge / Lasso Regularization** to keep weights under control.

---

## 8. Common Traps & Debugging Checklist

- [ ] **Forgetting to reset ALL buckets:** Every new feature needs its own `sum_xk_miss = 0.0` reset inside `for epoch in range(...)`.
- [ ] **Updating weights inside the row loop:** Always keep `w1_grad`, `w2_grad`, `b_grad` and weight updates at **4 spaces** (outside the inner row loop).
- [ ] **Passing a 1D list to `predict()`:** `model.predict([4, 1])` crashes. Always pass a 2D table: `model.predict([[4, 1]])`.
- [ ] **Using `fit_transform()` on test data:** Always `fit_transform()` on training data, and only `transform()` on new/test data so you don't leak test stats into the scaler.
- [ ] **Trusting raw weights when features aren't scaled:** A bigger weight does **not** automatically mean a more important feature if the columns are in different units (e.g., rupees vs. kilometers).
