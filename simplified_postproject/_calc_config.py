import math

scenarios = [1, 8, 20, 24, 48, 72, 80, 96]
N = 24
N_FRAMES = 5
RENDER_SAMPLES = 64

t_render_frame = RENDER_SAMPLES / 16 * 5
t_score_frame = 4
t_eval = (t_render_frame + t_score_frame) * N_FRAMES

print(f"t_eval = {t_eval:.0f}s ({t_eval/60:.1f} min)")
lam_d = 4 + int(3 * math.log(N))
print(f"lambda_default = 4 + floor(3*ln({N})) = {lam_d}")
print()
header = f"{'Hours':>6} {'E_total':>8} {'popsize':>8} {'maxiter':>8} {'sigma0':>8} {'Quality':>10}"
print(header)
print("-" * len(header))
for h in scenarios:
    T = h * 3600
    E = int(T / t_eval)
    lam = max(14, lam_d + (lam_d % 2))
    mi = E // lam
    q = E / N
    if q > 100: s = 0.20
    elif q > 30: s = 0.15
    else: s = 0.10
    tag = "DEBUG" if q < 10 else ("OK" if q < 30 else ("GOOD" if q < 100 else "EXCELLENT"))
    print(f"{h:>5}h {E:>8} {lam:>8} {mi:>8} {s:>8.2f} {q:>8.1f}x {tag}")
#