# CMA-ES Optimization Configuration Guide

Photometric shader optimization for CORTO Mars/Phobos pipeline.

---

## 1. Key Concepts

| Term | Meaning |
|---|---|
| **N** | Number of optimized parameters (currently **24**) |
| **λ (popsize)** | Candidate solutions sampled per generation |
| **μ** | Parent (elite) count = `⌊λ/2⌋` |
| **σ₀ (sigma0)** | Initial step-size (search radius) |
| **maxiter** | Maximum generations |
| **N_FRAMES** | PDS images rendered per evaluation (currently **4**) |
| **t_eval** | Wall-clock time for 1 evaluation (all frames) |

> [!IMPORTANT]
> **λ (popsize) and N_FRAMES are independent.** Each evaluation renders ALL N_FRAMES images regardless of popsize. Popsize controls how many *different parameter sets* are tested per generation.

---

## 2. Measure Your t_eval

Before planning, measure 1 evaluation time on your hardware:

```
t_eval = t_render × N_FRAMES + t_score × N_FRAMES
```

### Reference Benchmarks (RTX 4090, 4 frames)

| RENDER_SAMPLES | t_render/frame | t_score/frame (w/ LPIPS) | t_eval (4 frames) |
|---|---|---|---|
| 16 | ~5s | ~4s | ~36s |
| 64 | ~12s | ~4s | ~64s |
| 128 | ~22s | ~4s | ~104s |
| 256 | ~42s | ~4s | ~184s |

> [!TIP]
> Run `debug_optimize.py` with popsize=2, maxiter=1 and divide total time by 2 to get your t_eval.

---

## 3. Core Formulas

### 3.1 Minimum Population Size

From Hansen & Ostermeier (2001), the CMA-ES default population:

```
λ_default = 4 + ⌊3 · ln(N)⌋
```

For N=24: `λ_default = 4 + ⌊3 × 3.178⌋ = 4 + 9 = 13`

**Rules:**
- `λ_min = λ_default` — below this, covariance estimation becomes unreliable
- Use **even numbers** so μ = λ/2 is clean (→ **14** instead of 13)
- Larger λ = better exploration but slower convergence per wall-clock hour

### 3.2 Total Evaluations from Time Budget

```
E_total = ⌊T_budget / t_eval⌋
```

Where `T_budget` is in seconds.

### 3.3 Maximum Generations

```
maxiter = ⌊E_total / λ⌋
```

### 3.4 Convergence Quality Check

CMA-ES typically needs **10N to 30N** function evaluations for well-conditioned problems, 
and up to **100N** for ill-conditioned ones (Hansen 2016, "The CMA Evolution Strategy: A Tutorial").

```
E_minimum  = 10 × N = 240    (bare minimum, may not converge)
E_moderate = 30 × N = 720    (reasonable convergence)
E_good     = 100 × N = 2400  (thorough exploration)
```

> [!WARNING]
> If `E_total < E_minimum`, your time budget is too tight. Reduce N_FRAMES, RENDER_SAMPLES, or fix some parameters.

### 3.5 Sigma0 Selection

| Scenario | sigma0 | Rationale |
|---|---|---|
| Cold start (no prior) | **0.20** | ~20% of normalized range; explores broadly |
| Warm start (from CSV) | **0.05** | Fine-tune around known good point |
| Re-optimization (small tweak) | **0.02** | Very local refinement |

Reference: Hansen (2016) recommends σ₀ ≈ 1/4 of the expected distance to the optimum in each coordinate. For [0,1] normalized bounds this gives ~0.25. We use slightly less since our x0 is usually informed.

---

## 4. Step-by-Step Planning

### Step 1: Define your time budget

```
T_budget_hours = ?
T_budget = T_budget_hours × 3600  (seconds)
```

### Step 2: Measure or estimate t_eval

```
t_eval ≈ (RENDER_SAMPLES/16 × 5 + 4) × N_FRAMES  (rough estimate, seconds)
```

### Step 3: Calculate total evaluations

```
E_total = ⌊T_budget / t_eval⌋
```

### Step 4: Choose popsize

```
λ = max(14, next_even(4 + ⌊3 × ln(N)⌋))
```

If `E_total / λ < 50`, consider reducing λ to λ_default (but never below 10).

### Step 5: Calculate maxiter

```
maxiter = ⌊E_total / λ⌋
```

### Step 6: Validate convergence quality

```
quality = E_total / N
if quality < 10:  → WARNING: likely won't converge
if quality 10-30: → OK for warm-start
if quality > 30:  → Good
if quality > 100: → Excellent
```

---

## 5. Ready-Made Configurations

All examples assume: **N=24, N_FRAMES=4, RENDER_SAMPLES=64, t_eval≈64s**

### 🔬 Quick Test (1 hour)

```python
# E_total = 3600/64 ≈ 56 evals, quality = 56/24 = 2.3x → DEBUG ONLY
ALGO_SETTINGS = {
    "cmaes": {"sigma0": 0.05, "popsize": 8, "maxiter": 7},
}
RENDER_SAMPLES = 16  # reduce for speed
```

### ⚡ Overnight (8 hours)

```python
# E_total = 28800/64 ≈ 450 evals, quality = 450/24 = 18.8x → OK with warm-start
ALGO_SETTINGS = {
    "cmaes": {"sigma0": 0.10, "popsize": 14, "maxiter": 32},
}
```

### 🎯 Standard (24 hours)

```python
# E_total = 86400/64 ≈ 1350 evals, quality = 1350/24 = 56x → Good
ALGO_SETTINGS = {
    "cmaes": {"sigma0": 0.15, "popsize": 14, "maxiter": 96},
}
```

### 🏆 Full (72 hours)

```python
# E_total = 259200/64 ≈ 4050 evals, quality = 4050/24 = 169x → Excellent
ALGO_SETTINGS = {
    "cmaes": {"sigma0": 0.20, "popsize": 14, "maxiter": 289},
}
```

### 🚀 Marathon (168 hours / 1 week)

```python
# E_total = 604800/64 ≈ 9450 evals, quality = 9450/24 = 394x → Research-grade
ALGO_SETTINGS = {
    "cmaes": {"sigma0": 0.25, "popsize": 18, "maxiter": 525},
}
```

---

## 6. Quick Calculator

Copy-paste this into Python to compute your config:

```python
import math

# ── USER INPUTS ──
T_hours       = 72        # Time budget (hours)
N             = 24        # Optimized parameters
N_FRAMES      = 4         # PDS images
RENDER_SAMPLES = 64       # Blender samples
warm_start    = False     # True if starting from a known-good CSV

# ── DERIVED ──
t_render_frame = RENDER_SAMPLES / 16 * 5   # seconds (GPU-dependent)
t_score_frame  = 4                          # seconds (with LPIPS)
t_eval         = (t_render_frame + t_score_frame) * N_FRAMES

T_budget  = T_hours * 3600
E_total   = int(T_budget / t_eval)
lam_default = 4 + int(3 * math.log(N))
lam       = max(14, lam_default + (lam_default % 2))  # round up to even
maxiter   = E_total // lam
quality   = E_total / N

if warm_start:
    sigma0 = 0.05
elif quality > 100:
    sigma0 = 0.20
elif quality > 30:
    sigma0 = 0.15
else:
    sigma0 = 0.10

print(f"Time budget:   {T_hours}h ({T_budget}s)")
print(f"t_eval:        {t_eval:.0f}s")
print(f"E_total:       {E_total} evaluations")
print(f"popsize (λ):   {lam}")
print(f"maxiter:       {maxiter}")
print(f"sigma0:        {sigma0}")
print(f"Quality:       {quality:.1f}x  (>30 = good, >100 = excellent)")
print(f"\n# config.py")
print(f'ALGO_SETTINGS = {{')
print(f'    "cmaes": {{"sigma0": {sigma0}, "popsize": {lam}, "maxiter": {maxiter}}},')
print(f'}}')
```

---

## 7. Advanced: Reducing N to Fit Tighter Budgets

If your quality ratio is below 10x, consider **fixing** some parameters to reduce N:

```python
# Example: Fix well-known parameters
ParamDef("atm_scale_height", "atm", ..., optimize=False, ...),  # Fix at 116.06
ParamDef("mars_ior",         "mars", ..., optimize=False, ...),  # Fix at 2.38
```

Impact on popsize and convergence:

| N (optimized) | λ_default | E_minimum (10N) | E_good (30N) |
|---|---|---|---|
| 24 | 13 | 240 | 720 |
| 18 | 12 | 180 | 540 |
| 12 | 11 | 120 | 360 |
| 8 | 10 | 80 | 240 |

---

## 8. RENDER_SAMPLES vs Accuracy Tradeoff

Lower samples = faster but noisier renders = noisier objective function.

| RENDER_SAMPLES | Render noise | Suitable for |
|---|---|---|
| 16 | High | Debug / quick test only |
| 64 | Medium | Standard optimization |
| 128 | Low | Final refinement pass |
| 256 | Very low | Publication-quality validation |

> [!TIP]
> A good strategy: run the first 70% of your budget at RENDER_SAMPLES=64, then re-run the best parameters at RENDER_SAMPLES=256 for final validation.

---

## 9. References

1. **Hansen, N. (2016).** "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772. 
   - Section 4: Default parameter settings, population size formula.
2. **Hansen, N. & Ostermeier, A. (2001).** "Completely Derandomized Self-Adaptation in Evolution Strategies." Evolutionary Computation, 9(2), 159-195.
   - Original λ_default = 4 + ⌊3·ln(N)⌋ derivation.
3. **Hansen, N. (2009).** "Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed." GECCO Workshop.
   - BIPOP restart strategy; typical budget: 10⁴·N to 10⁵·N for hard functions.

---

## 10. Checklist Before Starting

- [ ] Measured `t_eval` on your hardware with target RENDER_SAMPLES
- [ ] `quality = E_total / N > 10` (minimum), ideally > 30
- [ ] `popsize ≥ 14` for N=24
- [ ] `sigma0` matches your start point quality (warm vs cold)
- [ ] CSV logging enabled (`optimization_output/` or `debug_opt_output/`)
- [ ] `CHECKPOINT_EVERY` set (saves coefficients.json periodically)
- [ ] `K_SWEEP_ENABLED` = True (shadow mask noise filtering)
- [ ] `SAVE_DEBUG_IMAGES` = False for production (saves disk I/O)
