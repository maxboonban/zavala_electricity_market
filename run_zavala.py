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
    compute_social_surplus,
    compare_tail_welfare_stoch_vs_cvar
)

# =========================
# Run Zavala baseline (stochastic) + deterministic + CVaR
# =========================
num_instances = 100
key = random.key(200)
keys = random.split(key, num_instances)
instances = []
for key in keys:
    instances.append(generate_instance(key, num_scenarios=20, num_g=10, num_d=10))

# --- stochastic accumulators ---
zavala_times = []
zavala_distortions = []
zavala_regrets = []
probs_feasible = []

# --- deterministic accumulators ---
det_distortions = []
det_regrets = []

# --- CVaR accumulators ---
cvar_distortions = []
cvar_regrets = []

# --- social-surplus accumulators ---
stoch_ss_neg_total, stoch_ss_neg_supplier, stoch_ss_neg_consumer, stoch_ss = [], [], [], []
det_ss_neg_total,   det_ss_neg_supplier,   det_ss_neg_consumer,   det_ss   = [], [], [], []
cvar_ss_neg_total,  cvar_ss_neg_supplier,  cvar_ss_neg_consumer,  cvar_ss  = [], [], [], []

for i in range(len(instances)):
    probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar = instances[i]

    # ===== Stochastic Zavala =====
    # >>> CHANGED: capture Z_G, Z_D instead of discarding
    z_g_i, z_d_j, Z_G, Z_D, z_pi, z_Pi = zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)

    probs_feasible.append(probability_feasible(probs, z_g_i, z_d_j, g_i_bar, d_j_bar))
    zavala_distortions.append(price_distortion(probs, z_pi, z_Pi))
    zavala_regrets.append(expected_cumulative_regret(probs, z_g_i, z_d_j, z_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar))
    ss_stoch = compute_social_surplus(
        probs, mc_g_i, mv_d_j,
        g_da=z_g_i, d_da=z_d_j,
        G_rt=Z_G, D_rt=Z_D,
        # optional: pass custom deltas to match your experiments
        # mc_g_i_delta=mc_g_i/10.0, mv_d_j_delta=mv_d_j/10.0
    )
    stoch_ss_neg_total.append(ss_stoch["E_neg_total"])
    stoch_ss_neg_supplier.append(ss_stoch["E_neg_supplier"])
    stoch_ss_neg_consumer.append(ss_stoch["E_neg_consumer"])
    stoch_ss.append(ss_stoch["E_social_surplus"])

    # ===== CVaR Zavala =====
    # >>> CHANGED: capture C_G, C_D so we can print
    cvar_g_i, cvar_d_j, C_G, C_D, cvar_pi, cvar_Pi, cvar_link_dual = zavala_cvar(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    cvar_distortions.append(price_distortion(probs, cvar_pi, cvar_Pi))
    cvar_regrets.append(expected_cumulative_regret(probs, cvar_g_i, cvar_d_j, cvar_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar))
    ss_cvar = compute_social_surplus(
        probs, mc_g_i, mv_d_j,
        g_da=cvar_g_i, d_da=cvar_d_j,
        G_rt=C_G, D_rt=C_D,
    )
    cvar_ss_neg_total.append(ss_cvar["E_neg_total"])
    cvar_ss_neg_supplier.append(ss_cvar["E_neg_supplier"])
    cvar_ss_neg_consumer.append(ss_cvar["E_neg_consumer"])
    cvar_ss.append(ss_cvar["E_social_surplus"])          

    # ===== Deterministic Zavala (energy-only, no network) =====
    # Use expected capacities for DA
    gbar_det, dbar_det = expected_caps_from_scenarios(probs, g_i_bar, d_j_bar)

    # Day-ahead deterministic solve
    g_det, d_det, pi_det = zavala_deterministic_da(mc_g_i, mv_d_j, gbar_det, dbar_det)

    # Real-time per scenario — also collect RT dispatch to print deltas
    G_det_list, D_det_list, Pi_det_list = [], [], []
    for p in range(len(probs)):
        Gp, Dp, Pi_p = zavala_rt_energy_only(mc_g_i, mv_d_j, g_det, d_det, g_i_bar[p], d_j_bar[p])
        G_det_list.append(Gp)
        D_det_list.append(Dp)
        Pi_det_list.append(Pi_p)
    G_det_rt, D_det_rt = _stack_rt(G_det_list, D_det_list)
    Pi_det = np.array(Pi_det_list)

    det_distortions.append(price_distortion(probs, pi_det, Pi_det))
    det_regrets.append(expected_cumulative_regret(probs, g_det, d_det, pi_det, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

    # --- NEW: §4.1 social surplus for deterministic run ---
    ss_det = compute_social_surplus(
        probs, mc_g_i, mv_d_j,
        g_da=g_det, d_da=d_det,
        G_rt=G_det_rt, D_rt=D_det_rt,
        # mc_g_i_delta=mc_g_i/10.0, mv_d_j_delta=mv_d_j/10.0,  # (optional) explicitly pass deltas
    )
    det_ss_neg_total.append(ss_det["E_neg_total"])
    det_ss_neg_supplier.append(ss_det["E_neg_supplier"])
    det_ss_neg_consumer.append(ss_det["E_neg_consumer"])
    det_ss.append(ss_det["E_social_surplus"])

    # ss_stoch["ss_per_scenario"] and ss_cvar["ss_per_scenario"] should both be (S,)
    cmp = compare_tail_welfare_stoch_vs_cvar(
        ss_stoch["ss_per_scenario"],   # baseline per-scenario NEG-SS
        ss_cvar["ss_per_scenario"],    # CVaR per-scenario NEG-SS
        probs,
        tail=0.05,                     # 5% tail
        worst="low"                    # treat more negative as worse; use "high" if opposite
    )

    # print(f'############# iter {i+1} ##################### \n')
    # # print("Tail indices (baseline-defined):", cmp["idx"])
    # # print("Tail probs:", cmp["probs"], "sum =", cmp["stoch_tail_prob_sum"])
    # print("Baseline NEG-SS on tail:", cmp["stoch_neg_ss"])
    # print("CVaR     NEG-SS on tail:", cmp["cvar_neg_ss"])
    # print("Baseline tail weighted-mean NEG-SS:", cmp["stoch_tail_weighted_mean_neg_ss"])
    # print("CVaR on same tail weighted-mean NEG-SS:", cmp["cvar_on_stoch_tail_weighted_mean_neg_ss"])
    # print(f"CVaR link dual = {cvar_link_dual}")
# print(f"Price distortion stochastic = {zavala_distortions}")
# print(f"Price distortion CVaR = {cvar_distortions}")


# # ============== Overall Expectation Results =================
print(f'Stochastic Zavala mean distortion: {np.mean(zavala_distortions)}')
print(f'Stochastic Zavala with CVaR mean distortion: {np.mean(cvar_distortions)}')
print(f'Deterministic mean distortion: {np.mean(det_distortions)}')

# print(f"Stochastic mean E[-SS]: {np.mean(stoch_ss_neg_total)} "
#       f"(suppliers {np.mean(stoch_ss_neg_supplier)}, consumers {np.mean(stoch_ss_neg_consumer)})")
print(f"Stochastic mean E[SS]:  {np.mean(stoch_ss)}")

# print(f'CVaR mean distortion: {np.mean(cvar_distortions)}')

# print(f"CVaR       mean E[-SS]: {np.mean(cvar_ss_neg_total)} "
#       f"(suppliers {np.mean(cvar_ss_neg_supplier)}, consumers {np.mean(cvar_ss_neg_consumer)})")
print(f"CVaR       mean E[SS]:  {np.mean(cvar_ss)}")

# print(f"Deterministic mean E[-SS]: {np.mean(det_ss_neg_total)} "
#       f"(suppliers {np.mean(det_ss_neg_supplier)}, consumers {np.mean(det_ss_neg_consumer)})")
print(f"Deterministic mean E[SS]:  {np.mean(det_ss)}")

# print("Stochastic Case \n")
# print(f"Total Welfare = {stoch_ss_neg_total}")
# print(f"Day-ahead price = {z_pi}")
# print(f"Real-time price = {z_Pi}")
# print(f"Probability = {probs} \n")

# print("CVaR Stochastic Case \n")
# print(f"Total Welfare = {cvar_ss_neg_total}")
# print(f"Day-ahead price = {cvar_pi}")
# print(f"Real-time price = {cvar_Pi}")
# print(f"Probability = {probs} \n")