import numpy as np

d = np.genfromtxt("height_tracking.csv", delimiter=",", names=True)
t, tgt, act = d["t_s"], d["target_m"], d["actual_m"]
err = act - tgt
print(f"RMSE over run:        {np.sqrt(np.mean(err**2))*100:.2f} cm")
print(f"Mean abs error:       {np.mean(np.abs(err))*100:.2f} cm")

# Settled error: last 2 s of every stretch where the target is constant for >= 3 s
change = np.flatnonzero(np.diff(tgt) != 0) + 1
bounds = np.concatenate([[0], change, [len(t)]])
for lo_t, hi_t in [(0.22, 0.2201), (0.339, 0.341)]:
    vals = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if lo_t <= tgt[a] <= hi_t and t[b-1] - t[a] >= 3.0:
            m = t >= t[b-1] - 2.0
            m[:a] = False; m[b:] = False
            vals.append(np.mean(err[m]))
    if vals:
        print(f"Settled error at target {lo_t:.2f}: "
              + ", ".join(f"{v*100:+.1f} cm" for v in vals))