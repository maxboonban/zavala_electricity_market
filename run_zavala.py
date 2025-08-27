import time
import numpy as np
from jax import random

from zavala_funcs import (
    # stochastic
    zavala,
    zavala_cvar,
    generate_instance,
    price_distortion,
    probability_feasible,
    expected_cumulative_regret,
    # deterministic bits
    zavala_deterministic_da,
    zavala_rt_energy_only,
    expected_caps_from_scenarios,
    # >>> ADDED: printing helpers
    _print_da_rt_summary,
    _stack_rt,
)

# =========================
# Run Zavala baseline (stochastic) + deterministic + CVaR
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

# --- CVaR accumulators (new) ---
cvar_distortions = []
cvar_regrets = []

PRINT_FIRST_INSTANCE_ONLY = True  # keeps output readable

for i in range(len(instances)):
    probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar = instances[i]

    # ===== Stochastic Zavala =====
    start_time = time.time()
    # >>> CHANGED: capture Z_G, Z_D instead of discarding
    z_g_i, z_d_j, Z_G, Z_D, z_pi, z_Pi = zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    end_time = time.time()

    probs_feasible.append(probability_feasible(probs, z_g_i, z_d_j, g_i_bar, d_j_bar))
    zavala_times.append(end_time - start_time)
    zavala_distortions.append(price_distortion(probs, z_pi, z_Pi))
    zavala_regrets.append(expected_cumulative_regret(probs, z_g_i, z_d_j, z_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

    # >>> ADDED: print DA/RT summary (stochastic) for first instance
    if (not PRINT_FIRST_INSTANCE_ONLY) or (i == 0):
        _print_da_rt_summary(
            label="STOCHASTIC",
            pi=float(z_pi),
            g_da=np.array(z_g_i),
            d_da=np.array(z_d_j),
            Pi=np.array(z_Pi),      # length S
            G_rt=np.array(Z_G),     # S x G
            D_rt=np.array(Z_D),     # S x D
            probs=np.array(probs),
        )

    # ===== CVaR Zavala =====
    # >>> CHANGED: capture C_G, C_D so we can print
    cvar_g_i, cvar_d_j, C_G, C_D, cvar_pi, cvar_Pi = zavala_cvar(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    cvar_distortions.append(price_distortion(probs, cvar_pi, cvar_Pi))
    cvar_regrets.append(expected_cumulative_regret(probs, cvar_g_i, cvar_d_j, cvar_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

    # >>> ADDED: print DA/RT summary (CVaR) for first instance
    if (not PRINT_FIRST_INSTANCE_ONLY) or (i == 0):
        _print_da_rt_summary(
            label="CVAR",
            pi=float(cvar_pi),
            g_da=np.array(cvar_g_i),
            d_da=np.array(cvar_d_j),
            Pi=np.array(cvar_Pi),
            G_rt=np.array(C_G),
            D_rt=np.array(C_D),
            probs=np.array(probs),
        )

    # ===== Deterministic Zavala (energy-only, no network) =====
    # Use expected capacities for DA
    gbar_det, dbar_det = expected_caps_from_scenarios(probs, g_i_bar, d_j_bar)

    # Day-ahead deterministic solve
    g_det, d_det, pi_det = zavala_deterministic_da(mc_g_i, mv_d_j, gbar_det, dbar_det)

    # Real-time per scenario — also collect RT dispatch to print deltas
    G_det_list, D_det_list, Pi_det_list = [], [], []
    for p in range(len(probs)):
        Gp, Dp, Pi_p = zavala_rt_energy_only(mc_g_i, mv_d_j, g_i_bar[p], d_j_bar[p])
        G_det_list.append(Gp)
        D_det_list.append(Dp)
        Pi_det_list.append(Pi_p)
    G_det_rt, D_det_rt = _stack_rt(G_det_list, D_det_list)
    Pi_det = np.array(Pi_det_list)

    det_distortions.append(price_distortion(probs, pi_det, Pi_det))
    det_regrets.append(expected_cumulative_regret(probs, g_det, d_det, pi_det, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

    # >>> ADDED: print DA/RT summary (deterministic) for first instance
    if (not PRINT_FIRST_INSTANCE_ONLY) or (i == 0):
        _print_da_rt_summary(
            label="DETERMINISTIC",
            pi=float(pi_det),
            g_da=np.array(g_det),
            d_da=np.array(d_det),
            Pi=Pi_det,
            G_rt=G_det_rt,
            D_rt=D_det_rt,
            probs=np.array(probs),
        )

# ============== Aggregates =================
print(f'Stochastic Zavala mean distortion: {np.mean(zavala_distortions)}')
print(f'Stochastic Zavala mean regret: {np.mean(zavala_regrets)}')

print(f'Stochastic Zavala with CVaR mean distortion: {np.mean(cvar_distortions)}')
print(f'Stochastic Zavala with CVaR mean regret: {np.mean(cvar_regrets)}')

print(f'Deterministic mean distortion: {np.mean(det_distortions)}')
print(f'Deterministic mean regret: {np.mean(det_regrets)}')