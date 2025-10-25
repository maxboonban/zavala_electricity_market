import time
import numpy as np
import matplotlib.pyplot as plt
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
    tail_worst_indices_by_value
)

from plot_visualization import (
    plot_ss_bar_with_errorlabels,
    plot_tail_welfare_means_with_errorbars,
    plot_rt_price_histograms,
    plot_ss_and_tail_overlay
)

# =========================
# Run Zavala baseline (stochastic) + deterministic + CVaR
# =========================
num_instances = 10
key = random.key(200)
keys = random.split(key, num_instances)
instances = []
for key in keys:
    r = int(random.randint(key, shape=(), minval=0, maxval=1_000_000))
    instances.append(generate_instance(key, num_scenarios=500, num_g=10, num_d=10, r=r))
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

# Price accumulators
z_Pi_all = []
cvar_Pi_all = []

stoch_tail_distortions, cvar_tail_distortions, det_tail_distortions = [], [], []
stoch_tail_welfare, cvar_tail_welfare, det_tail_welfare = [], [], []

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
    z_Pi_all.append(np.asarray(z_Pi).ravel())

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
    cvar_Pi_all.append(np.asarray(cvar_Pi).ravel())      

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
    

    # Zavala Stochastic Tail Metrics
    stoch_tail_welfare_indices = tail_worst_indices_by_value(ss_stoch["ss_per_scenario"], probs, tail=0.05, worst="high")
    stoch_tail_welfare.append(-np.mean(ss_stoch['ss_per_scenario'][stoch_tail_welfare_indices]))     # Mean (report positive social surplus)
    z_Pi_tail = z_Pi[stoch_tail_welfare_indices]
    stoch_tail_distortions.append(np.mean(np.abs(z_pi - z_Pi_tail)))

    # CVaR Stochastic Tail Metrics
    cvar_tail_welfare_indices = tail_worst_indices_by_value(ss_cvar["ss_per_scenario"], probs, tail=0.05, worst="high")
    cvar_tail_welfare.append(-np.mean(ss_cvar['ss_per_scenario'][cvar_tail_welfare_indices])) 
    cvar_Pi_tail = cvar_Pi[cvar_tail_welfare_indices]
    cvar_tail_distortions.append(np.mean(np.abs(cvar_pi - cvar_Pi_tail)))

    # Deterministic Tail Metrics
    det_tail_welfare_indices = tail_worst_indices_by_value(ss_det["ss_per_scenario"], probs, tail=0.05, worst="high")
    det_tail_welfare.append(-np.mean(ss_det["ss_per_scenario"][det_tail_welfare_indices]))
    det_Pi_tail = Pi_det[det_tail_welfare_indices]
    det_tail_distortions.append(np.mean(np.abs(pi_det - det_Pi_tail)))

print("============== Overall Expectation Results =================")
print(f'Stochastic Zavala mean distortion: {np.mean(zavala_distortions)}')
print(f'Stochastic Zavala with CVaR mean distortion: {np.mean(cvar_distortions)}')
print(f'Deterministic mean distortion: {np.mean(det_distortions)}')

print(f"Stochastic mean E[-SS]: {np.mean(stoch_ss_neg_total)} "
      f"(suppliers {np.mean(stoch_ss_neg_supplier)}, consumers {np.mean(stoch_ss_neg_consumer)})")
print(f"Stochastic mean E[SS]:  {np.mean(stoch_ss)}")

print(f"CVaR       mean E[-SS]: {np.mean(cvar_ss_neg_total)} "
      f"(suppliers {np.mean(cvar_ss_neg_supplier)}, consumers {np.mean(cvar_ss_neg_consumer)})")
print(f"CVaR mean E[SS]:  {np.mean(cvar_ss)}")

print(f"Deterministic mean E[-SS]: {np.mean(det_ss_neg_total)} "
      f"(suppliers {np.mean(det_ss_neg_supplier)}, consumers {np.mean(det_ss_neg_consumer)})")
print(f"Deterministic mean E[SS]:  {np.mean(det_ss)}")


print("######################## Tail Metrics #############################################")
print(f"Stochastic Tail distortions = {np.mean(stoch_tail_distortions)}")
print(f"Stochastic Tail welfare = {np.mean(stoch_tail_welfare)}")

print(f"CVaR Tail distortions = {np.mean(cvar_tail_distortions)}")
print(f"CVaR Tail welfare = {np.mean(cvar_tail_welfare)}")

print(f"Deterministic Tail distortions = {np.mean(det_tail_distortions)}")
print(f"Deterministic Tail welfare = {np.mean(det_tail_welfare)}")

print("######################## Prices #############################################")
print(f"Mean Real-time prices (Stoch) = {np.mean(z_Pi)}")
print(f"Mean Real-time prices (CVaR) = {np.mean(cvar_Pi)}")
print(f"Mean Real-time prices (deterministic) = {np.mean(Pi_det)}")

# Compare committed vs real-time prices
print(f"Day-ahead prices (Stoch) = {z_pi}")
print(f"Tail real-time prices (Stoch) = {z_Pi_tail}")

print(f"Day-ahead prices (CVaR) = {cvar_pi}")
print(f"Tail real-time prices (CVaR) = {cvar_Pi_tail}")

print(f"Day-ahead price (deterministic) = {pi_det}")
print(f"Tail real-time price (deterministic) = {det_Pi_tail}")

print(f"Real-time prices (Stoch) = {z_Pi}")
print(f"Real-time prices (CVaR) = {cvar_Pi}")
print(f"Real-time prices (deterministic) = {Pi_det}")

print("############################ Unit Commitments #############################")
# Unit commitments for Generators
print(f"Day-ahead g committed quantities (Stochastic) = {z_g_i}")
print(f"Tail Real-time g committed quantities (Stochastic) = {Z_G[stoch_tail_welfare_indices]}")

print(f"Day-ahead g committed quantities (CVaR) = {cvar_g_i}")
print(f"Tail Real-time g committed quantities (CVaR) = {C_G[cvar_tail_welfare_indices]}")

print(f"Day-ahead g committed quantities (deterministic) = {g_det}")
print(f"Tail Real-time g committed quantities (deterministic) = {G_det_rt[det_tail_welfare_indices]}")

# Unit commitments for Consumers
print(f"Day-ahead d committed quantities (Stochastic) = {z_d_j}")
print(f"Tail Real-time d committed quantities (Stochastic) = {Z_D[stoch_tail_welfare_indices]}")

print(f"Day-ahead d committed quantities (CVaR) = {cvar_d_j}")
print(f"Tail Real-time d committed quantities (CVaR) = {C_D[cvar_tail_welfare_indices]}")

print(f"Day-ahead d committed quantities (deterministic) = {d_det}")
print(f"Tail Real-time d committed quantities (deterministic) = {D_det_rt[det_tail_welfare_indices]}")


print(f"Stochastic Welfare Total = {stoch_ss}")

############ OTHER visualization plots #############

# plot_ss_bar_with_errorlabels(
#     stoch_ss, cvar_ss, det_ss,
#     err="std",
#     title="E[SS] — mean with SD error bars",
#     savepath="visual_outputs/mean_ss_bar_sd.png",
#     show=False
# )
# # === Plot: mean tail welfare with vertical error bars ===
# plot_tail_welfare_means_with_errorbars(
#     stoch_tail_welfare,
#     cvar_tail_welfare,
#     det_tail_welfare,
#     err="std",         # use "sem" if you prefer standard error
#     show=True,         # set False if running headless
#     save=True,
#     outdir="visual_outputs",
#     filename="tail_welfare_means.png",
# )

# # === Aggregate real-time prices across instances and plot histograms ===
# if len(z_Pi_all) > 0 and len(cvar_Pi_all) > 0:
#     z_prices_all = np.concatenate(z_Pi_all)
#     cvar_prices_all = np.concatenate(cvar_Pi_all)

#     plot_rt_price_histograms(
#         z_prices_all,
#         cvar_prices_all,
#         bins=30,
#         show=False,  # headless-safe
#         savepath="visual_outputs/z_Pi_vs_cvar_Pi_histograms.png",
#     )
#     print("\nSaved side-by-side histograms to visual_outputs/z_Pi_vs_cvar_Pi_histograms.png")

plot_ss_and_tail_overlay(
    stoch_ss, cvar_ss, det_ss,
    stoch_tail_welfare, cvar_tail_welfare, det_tail_welfare,
    savepath="visual_outputs/ss_vs_tail_overlay.png",
    show=False  # or True if you want to display
)