# run_zavala_system_1.py
import numpy as np
from zavala_funcs import (
    build_system_1_data,
    zavala_system_1_stochastic,
    zavala_system_1_deterministic_da,
    zavala_system_1_rt_network,
    price_distortion_system_1,
)

def expected_cumulative_regret_system_1(
    probs,            # (S,)
    g_da, d_da,       # (3,), (3,)  DA schedules per node
    pi_da,            # (3,)        DA nodal prices
    alpha_gen,        # (3,)        generator bids per node
    alpha_load,       # (3,)        load bids per node (VOLL at node 2)
    gen_cap_scn,      # (S,3)       per-scenario gen caps
    load_cap_scn      # (S,3)       per-scenario load caps
):
    """
    Expected participant-utility regret (sum over nodes), evaluated at DA nodal prices.
    For each scenario s, each generator best-responds with min(cap, argmax at margin pi - alpha_gen),
    and each load with min(cap, argmax at margin alpha_load - pi). Network constraints are *not*
    part of individual best-responses (price-taking assumption), matching your energy-only regret.
    """
    probs = np.asarray(probs, float)
    g_da = np.asarray(g_da, float)
    d_da = np.asarray(d_da, float)
    pi_da = np.asarray(pi_da, float)
    alpha_gen = np.asarray(alpha_gen, float)
    alpha_load = np.asarray(alpha_load, float)
    gen_cap_scn = np.asarray(gen_cap_scn, float)
    load_cap_scn = np.asarray(load_cap_scn, float)

    margins_g = np.maximum(pi_da - alpha_gen, 0.0)       # (3,)
    margins_d = np.maximum(alpha_load - pi_da, 0.0)      # (3,)

    # scenario-optimal utilities if each participant best-responds at price pi_da
    g_ind_u = (gen_cap_scn * margins_g).sum(axis=1)      # (S,)
    d_ind_u = (load_cap_scn * margins_d).sum(axis=1)     # (S,)

    # utilities under current DA allocations at price pi_da
    g_cur_u = float((g_da * (pi_da - alpha_gen)).sum())
    d_cur_u = float((d_da * (alpha_load - pi_da)).sum())

    # regret per scenario (always >= 0 if DA schedules are feasible w.r.t caps)
    scen_regret = (g_ind_u - g_cur_u) + (d_ind_u - d_cur_u)  # (S,)

    return float((probs * scen_regret).sum())

def main():
    data = build_system_1_data()
    probs = data["scenario_probs"]
    alpha_gen, alpha_load = data["alpha_gen"], data["alpha_load"]
    gen_cap_scn = data["gen_cap_scenarios"]
    load_cap_scn = data["load_cap_scenarios"]
    S = int(data["num_scenarios"])

    # ===== STOCHASTIC System 1 =====
    (g_da_s, d_da_s, f_da_s, theta_da_s,
     G_rt_s, D_rt_s, f_rt_s, theta_rt_s,
     pi_da_s, Pi_rt_s) = zavala_system_1_stochastic()

    M_avg_s, M_max_s = price_distortion_system_1(pi_da_s, Pi_rt_s, probs)
    regret_s = expected_cumulative_regret_system_1(
        probs, g_da_s, d_da_s, pi_da_s, alpha_gen, alpha_load, gen_cap_scn, load_cap_scn
    )

    # ===== DETERMINISTIC System 1 =====
    g_da_d, d_da_d, f_da_d, theta_da_d, pi_da_d = zavala_system_1_deterministic_da()

    # RT prices per scenario (for distortion metric)
    rt_prices_det = []
    for s in range(S):
        _, _, _, _, Pi_s = zavala_system_1_rt_network(gen_cap_scn[s], load_cap_scn[s])
        rt_prices_det.append(np.asarray(Pi_s, float))
    rt_prices_det = np.vstack(rt_prices_det)  # (S,3)

    M_avg_d, M_max_d = price_distortion_system_1(pi_da_d, rt_prices_det, probs)
    regret_d = expected_cumulative_regret_system_1(
        probs, g_da_d, d_da_d, pi_da_d, alpha_gen, alpha_load, gen_cap_scn, load_cap_scn
    )

    # ===== Prints =====
    print("=== System 1: Stochastic vs Deterministic ===")
    print(f"Stochastic   →  M_avg_distortion={M_avg_s:.3f}, M_max_distortion={M_max_s:.3f}, Expected_Regret={regret_s:.3f}")
    print(f"Deterministic→  M_avg_distortion={M_avg_d:.3f}, M_max_distortion={M_max_d:.3f}, Expected_Regret={regret_d:.3f}")

if __name__ == "__main__":
    main()