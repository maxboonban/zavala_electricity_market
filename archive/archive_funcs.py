
# gurobi version
def zavala_system_1_stochastic():
    data = build_system_1_data()
    N, L, S = data["num_nodes"], data["num_lines"], data["num_scenarios"]
    alpha_gen,  delta_gen  = data["alpha_gen"],  data["delta_gen"]
    alpha_load, delta_load = data["alpha_load"], data["delta_load"]
    B, Fbar = data["susceptance"], data["flow_limit"]
    delta_flow, delta_theta = data["delta_flow"], data["delta_theta"]
    probs = data["scenario_probs"]
    Gbar_s, Dbar_s = data["gen_cap_scenarios"], data["load_cap_scenarios"]
    Gbar_max, Dbar_max = data["Gbar_max"], data["Dbar_max"]
    gbar_det = data["gen_cap_det"]

    # DA vars
    g_da = cp.Variable(N, nonneg=True)
    d_da = cp.Variable(N, nonneg=True)
    theta_da = cp.Variable(N)
    f_da = cp.Variable(L)

    # RT vars
    G_rt = cp.Variable((S, N), nonneg=True)
    D_rt = cp.Variable((S, N), nonneg=True)
    theta_rt = cp.Variable((S, N))
    f_rt = cp.Variable((S, L))

    # Up/Down deviations (split abs)
    Ugp = cp.Variable((S, N), nonneg=True); Ugn = cp.Variable((S, N), nonneg=True)  # G_rt - g_da = Ugp - Ugn
    Udp = cp.Variable((S, N), nonneg=True); UdN = cp.Variable((S, N), nonneg=True)  # D_rt - d_da = Udp - UdN
    Wfp = cp.Variable((S, L), nonneg=True); WfN = cp.Variable((S, L), nonneg=True)  # f_rt - f_da = Wfp - WfN
    Ztp = cp.Variable((S, N), nonneg=True); ZtN = cp.Variable((S, N), nonneg=True)  # theta_rt - theta_da = Ztp - ZtN

    cons = []

    # DA network
    cons += [theta_da[1] == 0]
    cons += [f_da[0] == B[0]*(theta_da[0]-theta_da[1]),
             f_da[1] == B[1]*(theta_da[1]-theta_da[2]) ]
    cons += [ -Fbar <= f_da, f_da <= Fbar ]
    # DA nodal balances (keep for DA prices)
    da_bal_1 = g_da[0] - d_da[0] - f_da[0] == 0
    da_bal_2 = g_da[1] - d_da[1] + f_da[0] - f_da[1] == 0
    da_bal_3 = g_da[2] - d_da[2] + f_da[1] == 0
    cons += [da_bal_1, da_bal_2, da_bal_3]

    # DA caps to avoid unbounded base term and anchor economics
    # cons += [g_da <= Gbar_max, d_da <= Dbar_max]

    # DA load is deterministic: [0, 100, 0]
    cons += [d_da[0] == 0.0, d_da[1] == 100.0, d_da[2] == 0.0]
    cons += [d_da <= np.array([0.0, 100.0, 0.0])]

    # CAp DA generation
    cons += [g_da <= np.array([50.0, 50.0, 50.0])]

    # Scenario constraints + deviation linkers
    rt_bal_constraints = []
    for s in range(S):
        cons += [theta_rt[s,1] == 0]
        cons += [f_rt[s,0] == B[0]*(theta_rt[s,0]-theta_rt[s,1]),
                 f_rt[s,1] == B[1]*(theta_rt[s,1]-theta_rt[s,2])]
        cons += [ -Fbar <= f_rt[s], f_rt[s] <= Fbar ]
        # RT nodal balances (for RT prices later)
        rt1 = G_rt[s,0] - D_rt[s,0] - f_rt[s,0] == 0
        rt2 = G_rt[s,1] - D_rt[s,1] + f_rt[s,0] - f_rt[s,1] == 0
        rt3 = G_rt[s,2] - D_rt[s,2] + f_rt[s,1] == 0
        cons += [rt1, rt2, rt3]
        rt_bal_constraints.append([rt1, rt2, rt3])

        # capacity
        cons += [G_rt[s] <= Gbar_s[s], D_rt[s] <= Dbar_s[s]]

        # deviation linkers
        cons += [G_rt[s] - g_da == Ugp[s] - Ugn[s]]
        cons += [D_rt[s] - d_da == Udp[s] - UdN[s]]
        cons += [f_rt[s] - f_da == Wfp[s] - WfN[s]]
        cons += [theta_rt[s] - theta_da == Ztp[s] - ZtN[s]]

    # # Objective: DA base + expected deviation costs (no RT linear welfare term)
    # dev_cost = 0
    # for s in range(S):
    #     dev_cost += probs[s]*( cp.sum(cp.multiply(delta_gen,  Ugp[s]+Ugn[s])) +
    #                            cp.sum(cp.multiply(delta_load, Udp[s]+UdN[s])) +
    #                            delta_flow  * cp.sum(Wfp[s]+WfN[s]) +
    #                            delta_theta * cp.sum(Ztp[s]+ZtN[s]) )
    # obj = cp.Minimize(alpha_gen @ g_da - alpha_load @ d_da + dev_cost)

    # prob = cp.Problem(obj, cons)

    # --- objective pieces ---
    base_DA = alpha_gen @ g_da - alpha_load @ d_da
    obj_terms = []
    for s in range(S):
        rt_lin   = alpha_gen @ G_rt[s] - alpha_load @ D_rt[s]  # <-- put this back
        dev_gen  = cp.sum(cp.multiply(delta_gen,  Ugp[s] + Ugn[s]))
        dev_load = cp.sum(cp.multiply(delta_load, Udp[s] + UdN[s]))
        dev_flow = delta_flow  * cp.sum(Wfp[s] + WfN[s])
        dev_th   = delta_theta * cp.sum(Ztp[s] + ZtN[s])
        obj_terms.append(probs[s] * (rt_lin + dev_gen + dev_load + dev_flow + dev_th))

    prob = cp.Problem(cp.Minimize(base_DA + cp.sum(obj_terms)), cons)
    prob.solve(solver=cp.GUROBI, Method=2, Crossover=0, FeasibilityTol=1e-9, OptimalityTol=1e-9)

    # Which DA bounds are active?
    print("DA gen at bounds?:", (np.isclose(g_da.value, 0) | np.isclose(g_da.value, Gbar_max)).astype(int))
    print("DA load fixed  :", d_da.value.round(4))

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"System 1 stochastic solve failed: status={prob.status}")

    # values
    g_da_v = jnp.array(g_da.value); d_da_v = jnp.array(d_da.value)
    f_da_v = jnp.array(f_da.value); theta_da_v = jnp.array(theta_da.value)
    G_rt_v = jnp.array(G_rt.value); D_rt_v = jnp.array(D_rt.value)
    f_rt_v = jnp.array(f_rt.value); theta_rt_v = jnp.array(theta_rt.value)

    # prices
    pi_da = -jnp.array([float(da_bal_1.dual_value),
                        float(da_bal_2.dual_value),
                        float(da_bal_3.dual_value)])
    Pi_rt = []
    for s in range(S):
        Pi_rt.append([float(c.dual_value)/float(probs[s]) for c in rt_bal_constraints[s]])
    Pi_rt = -jnp.array(Pi_rt)

    # Print DA prices
    print("=== Day-Ahead (DA) Nodal Prices ===")
    print(f"Node 1: ${pi_da[0]:.3f}/MWh")
    print(f"Node 2: ${pi_da[1]:.3f}/MWh") 
    print(f"Node 3: ${pi_da[2]:.3f}/MWh")

    # Print RT prices
    print("\n=== Real-Time (RT) Nodal Prices by Scenario ===")
    for s in range(S):
        print(f"Scenario {s+1} (prob={probs[s]:.3f}):")
        print(f"  Node 1: ${Pi_rt[s,0]:.3f}/MWh")
        print(f"  Node 2: ${Pi_rt[s,1]:.3f}/MWh")
        print(f"  Node 3: ${Pi_rt[s,2]:.3f}/MWh")

    # Print expected RT prices
    expected_rt_prices = jnp.sum(Pi_rt * probs[:, jnp.newaxis], axis=0)
    print(f"\n=== Expected RT Prices ===")
    print(f"Node 1: ${expected_rt_prices[0]:.3f}/MWh")
    print(f"Node 2: ${expected_rt_prices[1]:.3f}/MWh")
    print(f"Node 3: ${expected_rt_prices[2]:.3f}/MWh")

    return g_da_v, d_da_v, f_da_v, theta_da_v, G_rt_v, D_rt_v, f_rt_v, theta_rt_v, pi_da, Pi_rt



# cvxpy version
def zavala_system_1_stochastic():
    """
    Two-stage STOCHASTIC clearing for System 1 with DC network and incremental bids
    on generators, load, line flows, and phase angles.
    Returns:
      g_da (3,), d_da (3,), f_da (2,), theta_da (3,),
      G_rt (S,3), D_rt (S,3), f_rt (S,2), theta_rt (S,3),
      pi_da (3,), Pi_rt (S,3)
    """
    import numpy as np
    data = build_system_1_data()
    N, L, S = data["num_nodes"], data["num_lines"], data["num_scenarios"]
    alpha_gen,  delta_gen  = data["alpha_gen"],  data["delta_gen"]
    alpha_load, delta_load = data["alpha_load"], data["delta_load"]
    B, Fbar = data["susceptance"], data["flow_limit"]
    delta_flow, delta_theta = data["delta_flow"], data["delta_theta"]
    probs = data["scenario_probs"]
    Gbar_s, Dbar_s = data["gen_cap_scenarios"], data["load_cap_scenarios"]

    # ----- Variables -----
    g_da = cp.Variable(N, nonneg=True)            # day-ahead generation per node
    d_da = cp.Variable(N, nonneg=True)            # day-ahead demand per node
    theta_da = cp.Variable(N)                     # day-ahead phase angles
    f_da = cp.Variable(L)                         # day-ahead line flows [1-2, 2-3]

    G_rt = cp.Variable((S, N), nonneg=True)       # real-time generation
    D_rt = cp.Variable((S, N), nonneg=True)       # real-time demand
    theta_rt = cp.Variable((S, N))                # real-time angles
    f_rt = cp.Variable((S, L))                    # real-time flows

    cons = []

    # ----- DA network constraints -----
    cons += [theta_da[1] == 0]  # reference bus at node 2
    cons += [f_da[0] == B[0]*(theta_da[0] - theta_da[1])]
    cons += [f_da[1] == B[1]*(theta_da[1] - theta_da[2])]
    cons += [ -Fbar <= f_da, f_da <= Fbar ]

    # DA nodal balances (store to read duals as DA prices)
    da_bal_1 = g_da[0] - d_da[0] - f_da[0] == 0
    da_bal_2 = g_da[1] - d_da[1] + f_da[0] - f_da[1] == 0
    da_bal_3 = g_da[2] - d_da[2] + f_da[1] == 0
    cons += [da_bal_1, da_bal_2, da_bal_3]

    # # # Optional DA caps to keep things tight
    # cons += [g_da <= np.array([50.0, 75.0, 50.0])]
    # cons += [d_da <= np.array([0.0, 100.0, 0.0])]

    # ----- RT constraints per scenario -----
    rt_bal_constraints = []  # to read duals for RT prices
    for s in range(S):
        cons += [theta_rt[s,1] == 0]
        cons += [f_rt[s,0] == B[0]*(theta_rt[s,0] - theta_rt[s,1])]
        cons += [f_rt[s,1] == B[1]*(theta_rt[s,1] - theta_rt[s,2])]
        cons += [ -Fbar <= f_rt[s], f_rt[s] <= Fbar ]

        rt_bal_1 = G_rt[s,0] - D_rt[s,0] - f_rt[s,0] == 0
        rt_bal_2 = G_rt[s,1] - D_rt[s,1] + f_rt[s,0] - f_rt[s,1] == 0
        rt_bal_3 = G_rt[s,2] - D_rt[s,2] + f_rt[s,1] == 0
        cons += [rt_bal_1, rt_bal_2, rt_bal_3]
        rt_bal_constraints.append([rt_bal_1, rt_bal_2, rt_bal_3])

        cons += [G_rt[s] <= Gbar_s[s], D_rt[s] <= Dbar_s[s]]

    # ----- Objective: expected RT welfare + deviation penalties -----
    obj_terms = []
    for s in range(S):
        # RT linear welfare
        rt_lin = alpha_gen @ G_rt[s] - alpha_load @ D_rt[s]
        # deviations around DA plan
        dev_gen   = cp.sum(cp.multiply(delta_gen,  cp.abs(G_rt[s] - g_da)))
        dev_load  = cp.sum(cp.multiply(delta_load, cp.abs(D_rt[s] - d_da)))
        dev_flow  = delta_flow  * cp.norm1(f_rt[s]    - f_da)
        dev_theta = delta_theta * cp.norm1(theta_rt[s] - theta_da)
        obj_terms.append(probs[s]*(rt_lin + dev_gen + dev_load + dev_flow + dev_theta))

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), cons)
    # prob.solve()
    prob.solve(
        solver=cp.GUROBI,
        Method=2,        # barrier
        Crossover=0,     # no crossover → central duals
        FeasibilityTol=1e-9,
        OptimalityTol=1e-9,
        BarConvTol=1e-12,
        # OutputFlag=0,  # uncomment to silence GUROBI logs
    )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"System 1 stochastic solve failed: status={prob.status}")

    # ----- Outputs -----
    g_da_v = jnp.array(g_da.value); d_da_v = jnp.array(d_da.value)
    f_da_v = jnp.array(f_da.value); theta_da_v = jnp.array(theta_da.value)
    G_rt_v = jnp.array(G_rt.value); D_rt_v = jnp.array(D_rt.value)
    f_rt_v = jnp.array(f_rt.value); theta_rt_v = jnp.array(theta_rt.value)

    # DA nodal prices from duals
    pi_da = -jnp.array([float(da_bal_1.dual_value),
                       float(da_bal_2.dual_value),
                       float(da_bal_3.dual_value)])
#    # Print DA prices
    print("=== Day-Ahead (DA) Nodal Prices ===")
    print(f"Node 1: ${pi_da[0]:.3f}/MWh")
    print(f"Node 2: ${pi_da[1]:.3f}/MWh") 
    print(f"Node 3: ${pi_da[2]:.3f}/MWh")

    # RT nodal prices per scenario (divide duals by scenario prob)
    Pi_rt = []
    for s in range(S):
        Pi_rt.append([float(c.dual_value)/float(probs[s]) for c in rt_bal_constraints[s]])
    Pi_rt = -jnp.array(Pi_rt)  # shape (S, 3)

        # Print RT prices
    print("\n=== Real-Time (RT) Nodal Prices by Scenario ===")
    for s in range(S):
        print(f"Scenario {s+1} (prob={probs[s]:.3f}):")
        print(f"  Node 1: ${Pi_rt[s,0]:.3f}/MWh")
        print(f"  Node 2: ${Pi_rt[s,1]:.3f}/MWh")
        print(f"  Node 3: ${Pi_rt[s,2]:.3f}/MWh")

        # Print expected RT prices
    expected_rt_prices = jnp.sum(Pi_rt * probs[:, jnp.newaxis], axis=0)
    print(f"\n=== Expected RT Prices ===")
    print(f"Node 1: ${expected_rt_prices[0]:.3f}/MWh")
    print(f"Node 2: ${expected_rt_prices[1]:.3f}/MWh")
    print(f"Node 3: ${expected_rt_prices[2]:.3f}/MWh")

    # Print price differences
    price_diff = pi_da - expected_rt_prices
    # print(f"\n=== DA vs Expected RT Price Differences ===")
    # print(f"Node 1: ${price_diff[0]:.3f}/MWh")
    # print(f"Node 2: ${price_diff[1]:.3f}/MWh")
    # print(f"Node 3: ${price_diff[2]:.3f}/MWh")

    return g_da_v, d_da_v, f_da_v, theta_da_v, G_rt_v, D_rt_v, f_rt_v, theta_rt_v, pi_da, Pi_rt

# 2025.09.09 cvxpy version
import cvxpy as cp
import jax.numpy as jnp
import numpy as np

def zavala_system_1_stochastic():
    # --- pull data exactly like GUROBI version ---
    data = build_system_1_data()
    N, L, S = data["num_nodes"], data["num_lines"], data["num_scenarios"]
    alpha_gen,  delta_gen  = data["alpha_gen"],  data["delta_gen"]
    alpha_load, delta_load = data["alpha_load"], data["delta_load"]
    B, Fbar = data["susceptance"], data["flow_limit"]
    delta_flow, delta_theta = data["delta_flow"], data["delta_theta"]  # not used in obj (keeping your objective)
    probs = data["scenario_probs"]
    Gbar_s, Dbar_s = data["gen_cap_scenarios"], data["load_cap_scenarios"]
    Gbar_max, Dbar_max = data["Gbar_max"], data["Dbar_max"]            # used for DA caps (like GUROBI)
    gbar_det = data["gen_cap_det"]                                     # not used here

    assert N == 3 and L == 2, "This version assumes the 3-bus System 1 topology."

    # --- variables (DA & RT) ---
    g_i = [cp.Variable(nonneg=True) for _ in range(N)]
    d_j = [cp.Variable(nonneg=True) for _ in range(N)]
    G_i = [[cp.Variable(nonneg=True) for _ in range(N)] for _ in range(S)]
    D_j = [[cp.Variable(nonneg=True) for _ in range(N)] for _ in range(S)]

    # network vars
    theta_da = cp.Variable(N)
    f_da = cp.Variable(L)
    theta_rt = cp.Variable((S, N))
    f_rt = cp.Variable((S, L))

    # --- objective (KEEP your original form; use alpha_/delta_) ---
    objective = cp.Minimize(
        sum(sum((alpha_gen[i] * G_i[p][i]
                 + delta_gen[i] * cp.max(cp.vstack([G_i[p][i] - g_i[i], 0]))
                 + delta_gen[i] * cp.max(cp.vstack([g_i[i] - G_i[p][i], 0])))
                * probs[p] for p in range(S)) for i in range(N))
        + sum(sum((-alpha_load[j] * D_j[p][j]
                 + delta_load[j] * cp.max(cp.vstack([d_j[j] - D_j[p][j], 0]))
                 + delta_load[j] * cp.max(cp.vstack([D_j[p][j] - d_j[j], 0])))
                * probs[p] for p in range(S)) for j in range(N))
    )

    cons = []

    # ===== Day-ahead network (explicit, like GUROBI) =====
    cons += [theta_da[1] == 0]  # ref bus node 2
    cons += [f_da[0] == B[0] * (theta_da[0] - theta_da[1]),
             f_da[1] == B[1] * (theta_da[1] - theta_da[2])]
    cons += [-Fbar[0] <= f_da[0], f_da[0] <= Fbar[0],
             -Fbar[1] <= f_da[1], f_da[1] <= Fbar[1]]

    # DA nodal balances (exactly as in GUROBI)
    da_bal_1 = g_i[0] - d_j[0] - f_da[0] == 0
    da_bal_2 = g_i[1] - d_j[1] + f_da[0] - f_da[1] == 0
    da_bal_3 = g_i[2] - d_j[2] + f_da[1] == 0
    cons += [da_bal_1, da_bal_2, da_bal_3]

    # DA deterministic pieces (match GUROBI ground truth)
    cons += [d_j[0] == 0.0, d_j[1] == 100.0, d_j[2] == 0.0]
    cons += [g_i[i] <= Gbar_max[i] for i in range(N)]  # == [50,50,50]

    # ===== Real-time network per scenario (explicit) =====
    rt_bal_constraints = []
    for p in range(S):
        cons += [theta_rt[p, 1] == 0]
        cons += [f_rt[p, 0] == B[0] * (theta_rt[p, 0] - theta_rt[p, 1]),
                 f_rt[p, 1] == B[1] * (theta_rt[p, 1] - theta_rt[p, 2])]
        cons += [-Fbar[0] <= f_rt[p, 0], f_rt[p, 0] <= Fbar[0],
                 -Fbar[1] <= f_rt[p, 1], f_rt[p, 1] <= Fbar[1]]

        # RT nodal balances (as in GUROBI)
        rt1 = G_i[p][0] - D_j[p][0] - f_rt[p, 0] == 0
        rt2 = G_i[p][1] - D_j[p][1] + f_rt[p, 0] - f_rt[p, 1] == 0
        rt3 = G_i[p][2] - D_j[p][2] + f_rt[p, 1] == 0
        cons += [rt1, rt2, rt3]
        rt_bal_constraints.append([rt1, rt2, rt3])

        # scenario caps
        cons += [G_i[p][i] <= Gbar_s[p, i] for i in range(N)]
        cons += [D_j[p][j] <= Dbar_s[p, j] for j in range(N)]

    # --- solve ---
    prob = cp.Problem(objective, cons)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"System 1 stochastic solve failed: status={prob.status}")

    # --- values (for printing & return) ---
    g_i_arr = np.array([g_i[i].value for i in range(N)])
    d_j_arr = np.array([d_j[j].value for j in range(N)])
    G_arr   = np.array([[G_i[p][i].value for i in range(N)] for p in range(S)])
    D_arr   = np.array([[D_j[p][j].value for j in range(N)] for p in range(S)])
    f_da_v = np.array(f_da.value);  theta_da_v = np.array(theta_da.value)
    f_rt_v = np.array(f_rt.value);  theta_rt_v = np.array(theta_rt.value)

    # --- prices (same convention as GUROBI) ---
    pi_vec = -np.array([float(da_bal_1.dual_value),
                        float(da_bal_2.dual_value),
                        float(da_bal_3.dual_value)])
    Pi_mat = -np.array([[float(c.dual_value) / float(probs[p]) for c in rt_bal_constraints[p]]
                        for p in range(S)])

    # ===== PRINTS to match your GUROBI output =====
    # Which DA bounds are active?
    print("DA gen at bounds?:", (np.isclose(g_i_arr, 0) | np.isclose(g_i_arr, Gbar_max)).astype(int))
    print("DA load fixed  :", d_j_arr.round(4))

    # Print DA prices
    print("=== Day-Ahead (DA) Nodal Prices ===")
    print(f"Node 1: ${pi_vec[0]:.3f}/MWh")
    print(f"Node 2: ${pi_vec[1]:.3f}/MWh")
    print(f"Node 3: ${pi_vec[2]:.3f}/MWh")

    # Print RT prices
    print("\n=== Real-Time (RT) Nodal Prices by Scenario ===")
    for p in range(S):
        print(f"Scenario {p+1} (prob={probs[p]:.3f}):")
        print(f"  Node 1: ${Pi_mat[p,0]:.3f}/MWh")
        print(f"  Node 2: ${Pi_mat[p,1]:.3f}/MWh")
        print(f"  Node 3: ${Pi_mat[p,2]:.3f}/MWh")

    # Print expected RT prices
    expected_rt_prices = (Pi_mat * probs[:, None]).sum(axis=0)
    print(f"\n=== Expected RT Prices ===")
    print(f"Node 1: ${expected_rt_prices[0]:.3f}/MWh")
    print(f"Node 2: ${expected_rt_prices[1]:.3f}/MWh")
    print(f"Node 3: ${expected_rt_prices[2]:.3f}/MWh")

    # --- return (same structure you used before) ---
    return (
        jnp.array(g_i_arr), jnp.array(d_j_arr),
        jnp.array(f_da_v), jnp.array(theta_da_v),
        jnp.array(G_arr), jnp.array(D_arr),
        jnp.array(f_rt_v), jnp.array(theta_rt_v),
        jnp.array(pi_vec), jnp.array(Pi_mat),
    )