import json

import numpy as np
from scipy.optimize import curve_fit

CACHE_PATH = "cache_m33.json"


def power_law(k, a, b):
    return a * np.power(k, b)


def main():
    with open(CACHE_PATH) as f:
        cached = json.load(f)
    data = cached["data"]
    ks = np.array(data["ks"], dtype=float)
    times = np.array(data["times"], dtype=float)

    _b0, _log_a0 = np.polyfit(np.log(ks), np.log(times), 1)
    p0 = (np.exp(_log_a0), _b0)

    (a_fit, b_fit), _pcov = curve_fit(power_law, ks, times, p0=p0, maxfev=10000)

    preds = power_law(ks, a_fit, b_fit)
    ss_res = np.sum((times - preds) ** 2)
    ss_tot = np.sum((times - times.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    b_loglinear, log_a_loglinear = np.polyfit(np.log(ks), np.log(times), 1)
    a_loglinear = np.exp(log_a_loglinear)

    print(f"Points: {list(zip(data['ks'], [round(t, 2) for t in data['times']]))}\n")
    print("scipy.optimize.curve_fit (nonlinear, original space):")
    print(f"    a = {a_fit:.4f}")
    print(f"    b = {b_fit:.4f}")
    print(f"    R^2 = {r_squared:.4f}")
    print()
    print("notebook's own np.polyfit (linear, log-log space) -- for comparison:")
    print(f"    a = {a_loglinear:.4f}")
    print(f"    b = {b_loglinear:.4f}")

    print("\n_ext_j_result = {")
    print(f'    "a_fit": {a_fit:.6f},')
    print(f'    "b_fit": {b_fit:.6f},')
    print(f'    "r_squared": {r_squared:.6f},')
    print(f'    "a_loglinear": {a_loglinear:.6f},')
    print(f'    "b_loglinear": {b_loglinear:.6f},')
    print("}")


if __name__ == "__main__":
    main()