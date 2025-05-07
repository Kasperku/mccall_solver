# McCall Job Search Model Solver
A Python script for solving the classic McCall job search model. Compute the optimal reservation wage and expected unemployment value for a worker facing stochastic wage offers.

---

##  Installation

```bash
pip install numpy scipy
```

*Requires Python 3.7+.*

---

##  How It Works

1. **Value of Unemployment (`V_U`)**:

   ```math
   V_U = -c + \beta \int_{0}^{\infty} \max(w, R) f(w) \, dw
   ```
2. **Reservation Wage (`R`)** solves:

   ```math
   R = (1 - \beta) V_U
   ```

The solver iterates on `R` until the fixed‐point condition above is satisfied.

---

## Configuration

You can customize:

* **`wage_dist`**: Any object with a PDF method and sampling (see `UniformWage`, `TruncatedNormalWage`).
* **`search_cost`**: Float ≥ 0.
* **`beta`**: Discount factor in (0, 1).
* **Integration settings**: Tolerances for numerical integration (optional parameters).

---

## Interpreting Results

* **Reservation Wage (`R`)**:  Reject any offer below `R`.
* **Unemployment Value (`V_U`)**: Expected discounted payoff when unemployed.

Relationship:  `R = (1 - β) · V_U`.

