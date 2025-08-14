import time
import numpy as np
from jax import random

from zavala_funcs import (
    # stochastic
    zavala,
    generate_instance,
    price_distortion,
    probability_feasible,
    expected_cumulative_regret,
    # deterministic bits (already implemented in zavala_funcs.py)
    zavala_deterministic_da,
    zavala_rt_energy_only,
    expected_caps_from_scenarios,
)

# =========================
# Run Zavala baseline (stochastic) + deterministic
# =========================
num_instances = 10
key = random.key(2)
keys = random.split(key, num_instances)
instances = []
for key in keys:
    instances.append(generate_instance(key, num_scenarios=10, num_g=10, num_d=10))

# --- stochastic accumulators (your existing ones) ---
zavala_times = []
zavala_distortions = []
zavala_regrets = []
probs_feasible = []

# --- deterministic accumulators (new) ---
det_distortions = []
det_regrets = []

for i in range(len(instances)):
    probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar = instances[i]

    # ===== Stochastic Zavala (your existing code) =====
    start_time = time.time()
    z_g_i, z_d_j, _, _, z_pi, z_Pi = zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    end_time = time.time()

    probs_feasible.append(probability_feasible(probs, z_g_i, z_d_j, g_i_bar, d_j_bar))
    zavala_times.append(end_time - start_time)
    zavala_distortions.append(price_distortion(probs, z_pi, z_Pi))
    zavala_regrets.append(expected_cumulative_regret(probs, z_g_i, z_d_j, z_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

    # ===== Deterministic Zavala (energy-only, no network) =====
    # Use expected capacities for DA (as in §3.1 text)
    gbar_det, dbar_det = expected_caps_from_scenarios(probs, g_i_bar, d_j_bar)

    # Day-ahead deterministic solve
    g_det, d_det, pi_det = zavala_deterministic_da(mc_g_i, mv_d_j, gbar_det, dbar_det)

    # Real-time per scenario (to compute Π(ω) and distortion)
    Pi_det = []
    for p in range(len(probs)):
        _, _, Pi_p = zavala_rt_energy_only(mc_g_i, mv_d_j, g_i_bar[p], d_j_bar[p])
        Pi_det.append(Pi_p)
    Pi_det = np.array(Pi_det)

    det_distortions.append(price_distortion(probs, pi_det, Pi_det))
    det_regrets.append(expected_cumulative_regret(probs, g_det, d_det, pi_det, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

# ============== Prints =================
print(f'Stochastic Zavala mean distortion: {np.mean(zavala_distortions)}')
print(f'Stochastic Zavala mean regret: {np.mean(zavala_regrets)}')

print(f'Deterministic Zavala mean distortion: {np.mean(det_distortions)}')
print(f'Deterministic Zavala mean regret: {np.mean(det_regrets)}')