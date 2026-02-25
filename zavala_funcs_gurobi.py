# zavala_funcs_gurobi.py
import numpy as np
import jax.numpy as jnp
import gurobipy as gp

def zavala_gurobi(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar):
    # Coerce to numpy (supports callers passing jax arrays)
    probs   = np.asarray(probs,   dtype=float).reshape(-1)
    mc_g_i  = np.asarray(mc_g_i,  dtype=float).reshape(-1)
    mv_d_j  = np.asarray(mv_d_j,  dtype=float).reshape(-1)
    g_i_bar = np.asarray(g_i_bar, dtype=float)
    d_j_bar = np.asarray(d_j_bar, dtype=float)

    S = probs.shape[0]
    G = mc_g_i.shape[0]
    D = mv_d_j.shape[0]
    if g_i_bar.shape != (S, G) or d_j_bar.shape != (S, D):
        raise ValueError("Shape mismatch: g_i_bar must be (S,G) and d_j_bar must be (S,D).")

    # Incremental bids (same heuristic as your CVXPY)
    delta_g = mc_g_i / 10.0
    delta_d = mv_d_j / 10.0

    m = gp.Model("zavala_gurobi")

    # Day-ahead variables (free, to match your unconstrained cp.Variable())
    g = m.addMVar(G, lb=-gp.GRB.INFINITY, name="g")
    d = m.addMVar(D, lb=-gp.GRB.INFINITY, name="d")

    # Real-time variables (nonnegative with per-scenario caps)
    G_rt = m.addMVar((S, G), lb=0.0, name="G_rt")
    D_rt = m.addMVar((S, D), lb=0.0, name="D_rt")

    # Deviation auxiliaries for (·)_+ linearization
    Ugp = m.addMVar((S, G), lb=0.0, name="Ugp")  # (G_rt - g)_+
    Ugn = m.addMVar((S, G), lb=0.0, name="Ugn")  # (g - G_rt)_+
    Udp = m.addMVar((S, D), lb=0.0, name="Udp")  # (D_rt - d)_+
    UdN = m.addMVar((S, D), lb=0.0, name="UdN")  # (d - D_rt)_+

    # Capacity constraints
    m.addConstr(G_rt <= g_i_bar, name="gen_cap")
    m.addConstr(D_rt <= d_j_bar, name="dem_cap")

    # Day-ahead balance (keep handle for dual/price)
    day_balance = m.addConstr(d.sum() - g.sum() == 0.0, name="day_balance")

    # Real-time balance per scenario (keep handles for duals/prices)
    rt_balance = []
    for s in range(S):
        rt_balance.append(
            m.addConstr(D_rt[s].sum() - d.sum() == G_rt[s].sum() - g.sum(),
                        name=f"rt_balance[{s}]")
        )
        # Deviation definitions → (·)_+ via inequalities; nonneg enforced by lb
        m.addConstr(Ugp[s] >= G_rt[s] - g, name=f"ugp_ge[{s}]")
        m.addConstr(Ugn[s] >= g - G_rt[s], name=f"ugn_ge[{s}]")
        m.addConstr(Udp[s] >= D_rt[s] - d, name=f"udp_ge[{s}]")
        m.addConstr(UdN[s] >= d - D_rt[s], name=f"udn_ge[{s}]")

    # Objective: sum_p probs[p]*( α^g·G_rt[p] − α^d·D_rt[p]
    #                              + Δg·(Ugp[p]+Ugn[p]) + Δd·(Udp[p]+UdN[p]) )
    obj = gp.LinExpr(0.0)
    for s in range(S):
        obj += probs[s] * (
            mc_g_i @ G_rt[s]
            - mv_d_j @ D_rt[s]
            + delta_g @ (Ugp[s] + Ugn[s])
            + delta_d @ (Udp[s] + UdN[s])
        )
    m.setObjective(obj, gp.GRB.MINIMIZE)

    # Solver settings (match your cp.GUROBI options)
    m.Params.Method = 2          # barrier
    m.Params.Crossover = 0       # no crossover → central duals
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol  = 1e-9
    m.Params.BarConvTol     = 1e-12
    # m.Params.OutputFlag = 0     # uncomment to silence logs

    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi solve failed: status={m.Status}")

    # Extract solutions (match return types/shapes from your CVXPY version)
    g_val  = jnp.array(g.X)
    d_val  = jnp.array(d.X)
    G_val  = jnp.array(G_rt.X)
    D_val  = jnp.array(D_rt.X)

    # Prices: divide RT duals by probs[p], same as your CVXPY code
    pi = -float(day_balance.Pi)
    Pi = jnp.array([-float(rt_balance[s].Pi) / probs[s] for s in range(S)])

    return g_val, d_val, G_val, D_val, pi, Pi