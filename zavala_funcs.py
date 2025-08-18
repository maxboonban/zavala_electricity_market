import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

def zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar):
    num_g = len(mc_g_i)
    num_d = len(mv_d_j)

    # Define the variables
    g_i = [cp.Variable() for i in range(num_g)]
    G_i = [[cp.Variable() for i in range(num_g)] for p in range(len(probs))]
    d_j = [cp.Variable() for j in range(num_d)]
    D_j = [[cp.Variable() for j in range(num_d)] for p in range(len(probs))]

    # Compute incremental bids (just a heuristic)
    mc_g_i_delta = mc_g_i / 10.0
    mv_d_j_delta = mv_d_j / 10.0

    # Define the objective
    objective = cp.Minimize(sum(sum((mc_g_i[i] * G_i[p][i] + mc_g_i_delta[i] * cp.max(cp.vstack([G_i[p][i] - g_i[i], 0])) + mc_g_i_delta[i] * cp.max(cp.vstack([g_i[i] - G_i[p][i], 0]))) * probs[p] for p in range(len(probs))) for i in range(num_g)) \
                            + sum(sum((-mv_d_j[j] * D_j[p][j] + mv_d_j_delta[j] * cp.max(cp.vstack([d_j[j] - D_j[p][j], 0])) + mv_d_j_delta[j] * cp.max(cp.vstack([D_j[p][j] - d_j[j], 0]))) * probs[p] for p in range(len(probs))) for j in range(num_d)))

    # Define the constraints
    day_ahead_balance = [
                    sum(d_j[j] for j in range(num_d)) == sum(g_i[i] for i in range(num_g))
                ]
    real_time_balance = [
                    sum(D_j[p][j] - d_j[j] for j in range(num_d)) == sum(G_i[p][i] - g_i[i] for i in range(num_g)) for p in range(len(probs))
                ]

    constraints = day_ahead_balance \
                + real_time_balance + [
                    G_i[p][i] >= 0 for p in range(len(probs)) for i in range(num_g)
                ] + [
                    G_i[p][i] <= g_i_bar[p][i] for p in range(len(probs)) for i in range(num_g)
                ] + [
                    D_j[p][j] >= 0 for p in range(len(probs)) for j in range(num_d)
                ] + [
                    D_j[p][j] <= d_j_bar[p][j] for p in range(len(probs)) for j in range(num_d)
                ]

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Print solver statistics for debugging
    print(f"Solver: {prob.solver_stats.solver_name}, Status: {prob.status}, Iterations: {prob.solver_stats.num_iters}")

    g_i_jax = jnp.array([g_i[i].value for i in range(num_g)])
    d_j_jax = jnp.array([d_j[j].value for j in range(num_d)])
    G_i_jax = jnp.array([[G_i[p][i].value for i in range(num_g)] for p in range(len(probs))])
    D_j_jax = jnp.array([[D_j[p][j].value for j in range(num_d)] for p in range(len(probs))])
    pi = day_ahead_balance[0].dual_value
    Pi = jnp.array([real_time_balance[p].dual_value / probs[p] for p in range(len(probs))])

    return g_i_jax, d_j_jax, G_i_jax, D_j_jax, pi, Pi

def generate_instance(key, num_scenarios = 10, num_g = 10, num_d = 10, minval = 1, maxval = 100):
    probs_key, mc_key, mv_key, g_key, d_key = random.split(key, 5)
    probs = random.dirichlet(probs_key, jnp.ones(num_scenarios))
    mc_g_i = random.uniform(mc_key, (num_g,), minval=minval, maxval=maxval)
    mv_d_j = random.uniform(mv_key, (num_d,), minval=minval, maxval=maxval)
    g_i_bar = random.uniform(g_key, (num_scenarios, num_g), minval=minval, maxval=maxval)
    d_j_bar = random.uniform(d_key, (num_scenarios, num_d), minval=minval, maxval=maxval)
    return probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar

def price_distortion(probs, pi, Pi):
    return abs(pi - sum(probs * Pi))

# allows some error!
def feasible(g_i, d_j, scenario_g_i_bar, scenario_d_j_bar):
    return jnp.all(g_i <= scenario_g_i_bar + 0.1) and jnp.all(d_j <= scenario_d_j_bar + 0.1)

def probability_feasible(probs, g_i, d_j, g_i_bar, d_j_bar):
    probability_sum = 0
    for p in range(len(probs)):
        if feasible(g_i, d_j, g_i_bar[p], d_j_bar[p]):
            probability_sum += probs[p]
    return probability_sum

def cumulative_regret(g_i, d_j, pi, mc_g_i, mv_d_j, scenario_g_i_bar, scenario_d_j_bar):
    if feasible(g_i, d_j, scenario_g_i_bar, scenario_d_j_bar):
        g_ind_u = jnp.sum(scenario_g_i_bar * jnp.maximum(pi - mc_g_i, 0))
        g_cur_u = jnp.sum(g_i * (pi - mc_g_i))
        d_ind_u = jnp.sum(scenario_d_j_bar * jnp.maximum(mv_d_j - pi, 0))
        d_cur_u = jnp.sum(d_j * (mv_d_j - pi))
        return (g_ind_u - g_cur_u) + (d_ind_u - d_cur_u)
    else:
        return None

def expected_cumulative_regret(probs, g_i, d_j, pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar):
    expected_regret = 0
    for p in range(len(probs)):
        regret = cumulative_regret(g_i, d_j, pi, mc_g_i, mv_d_j, g_i_bar[p], d_j_bar[p])
        if regret is not None:
            expected_regret += probs[p] * regret
    return expected_regret

# === NEW: deterministic DA (energy-only, no network) =========================
def zavala_deterministic_da(mc_g_i, mv_d_j, g_i_bar_det, d_j_bar_det):
    """
    Solve Zavala §3.1 deterministic day-ahead formulation (energy-only, no network).
    min   sum_i α^g_i g_i  - sum_j α^d_j d_j
    s.t.  sum_j d_j = sum_i g_i
          0 ≤ g_i ≤ ḡ_i,  0 ≤ d_j ≤ d̄_j

    Returns:
        g_i (jnp.ndarray), d_j (jnp.ndarray), pi (float scalar dual of balance)
    """

    num_g = mc_g_i.size
    num_d = mv_d_j.size

    # Vector variables (nonnegative)
    g = cp.Variable(num_g, nonneg=True)
    d = cp.Variable(num_d, nonneg=True)

    # Objective: negative social surplus (linear)
    obj = cp.Minimize(mc_g_i @ g - mv_d_j @ d)

    # Single-bus energy balance
    balance_con = cp.sum(d) - cp.sum(g) == 0

    # Bounds
    cons = [
        balance_con,
        g <= g_i_bar_det,
        d <= d_j_bar_det,
    ]

    prob = cp.Problem(obj, cons)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Deterministic DA solve failed: status={prob.status}")

    # Dual of the balance is the (single) day-ahead price π
    pi = float(balance_con.dual_value) if balance_con.dual_value is not None else np.nan
    
    # # Print Energy-Only Deterministic DA price
    # print("\n=== Energy-Only Deterministic Day-Ahead Price ===")
    # print(f"Single Bus Price: ${pi:.3f}/MWh")

    return jnp.array(g.value), jnp.array(d.value), pi


# === NEW: real-time energy-only per scenario (to get Π(ω)) ===================
def zavala_rt_energy_only(mc_g_i, mv_d_j, g_i_bar_rt, d_j_bar_rt):
    """
    Real-time energy-only clearing for a single scenario ω (no network):
    min   sum_i α^g_i G_i(ω) - sum_j α^d_j D_j(ω)
    s.t.  sum_j D_j(ω) = sum_i G_i(ω)
          0 ≤ G_i(ω) ≤ Ḡ_i(ω),  0 ≤ D_j(ω) ≤ D̄_j(ω)

    Returns:
        G (jnp.ndarray), D (jnp.ndarray), Pi (float scalar dual of balance)
    """
    num_g = mc_g_i.size
    num_d = mv_d_j.size

    G = cp.Variable(num_g, nonneg=True)
    D = cp.Variable(num_d, nonneg=True)

    obj = cp.Minimize(mc_g_i @ G - mv_d_j @ D)
    balance_con = cp.sum(D) - cp.sum(G) == 0
    cons = [balance_con, G <= g_i_bar_rt, D <= d_j_bar_rt]

    prob = cp.Problem(obj, cons)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"RT solve failed: status={prob.status}")

    Pi = float(balance_con.dual_value) if balance_con.dual_value is not None else np.nan
    
    # # Print Energy-Only Real-Time price
    # print("\n=== Energy-Only Real-Time Price (Single Scenario) ===")
    # print(f"Single Bus Price: ${Pi:.3f}/MWh")
    # print(f"(No network constraints - energy balance only)")
    # print(f"Generation Capacity: {g_i_bar_rt}")
    # print(f"Load Capacity: {d_j_bar_rt}")
    
    return jnp.array(G.value), jnp.array(D.value), Pi


# === NEW: helper to convert scenarios → deterministic capacities = E[·] =======
def expected_caps_from_scenarios(probs, g_i_bar_scn, d_j_bar_scn):
    """
    probs: shape (S,)
    g_i_bar_scn: shape (S, G)
    d_j_bar_scn: shape (S, D)
    Returns:
        gbar_det (G,), dbar_det (D,)  as expectation under probs
    """
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    gbar = np.tensordot(probs, np.asarray(g_i_bar_scn, dtype=np.float64), axes=(0, 0))
    dbar = np.tensordot(probs, np.asarray(d_j_bar_scn, dtype=np.float64), axes=(0, 0))
    return gbar, dbar

# =========================
# System I (3-bus DC network) — ASCII variable names
# =========================

def build_system_1_data():
    """
    Nodes: 1--2--3  (index 0,1,2)
    Lines: (1->2), (2->3)
    Two deterministic generators at nodes 1 & 3, stochastic at node 2.
    """
    import numpy as np
    num_nodes = 3
    num_lines = 2
    num_scenarios = 3

    # bids / incremental bids
    alpha_gen  = np.array([10.0, 1.0, 20.0])      # generator bids at nodes [1,2,3]
    delta_gen  = np.array([1.0, 0.1, 2.0])        # incremental generator bids
    alpha_load = np.array([0.0, 1000.0, 0.0])     # value of lost load at node 2
    delta_load = np.array([0.0, 0.001, 0.0])      # incremental load bids (node 2)

    susceptance = np.array([50.0, 50.0])          # for lines [1-2, 2-3]
    flow_limit  = np.array([25.0, 50.0])          # thermal limits [1-2, 2-3]
    delta_flow  = 0.001                           # deviation price for flows
    delta_theta = 0.001                           # deviation price for angles

    # generator caps per scenario: g1=50, g2 in {25,50,75}, g3=50
    gen_cap_scenarios = np.zeros((num_scenarios, 3))
    gen_cap_scenarios[0] = np.array([50.0, 25.0, 50.0])
    gen_cap_scenarios[1] = np.array([50.0, 50.0, 50.0])
    gen_cap_scenarios[2] = np.array([50.0, 75.0, 50.0])

    # demand caps per scenario (only node 2 has 100; deterministic across scenarios)
    load_cap_scenarios = np.zeros((num_scenarios, 3))
    load_cap_scenarios[:, 1] = 100.0

    scenario_probs = np.array([1/3, 1/3, 1/3], dtype=float)

    # deterministic expected caps (for deterministic DA run)
    gen_cap_det  = np.array([50.0, 50.0, 50.0])
    load_cap_det = np.array([0.0, 100.0, 0.0])

    return dict(
        num_nodes=num_nodes, num_lines=num_lines, num_scenarios=num_scenarios,
        alpha_gen=alpha_gen, delta_gen=delta_gen,
        alpha_load=alpha_load, delta_load=delta_load,
        susceptance=susceptance, flow_limit=flow_limit,
        delta_flow=delta_flow, delta_theta=delta_theta,
        scenario_probs=scenario_probs,
        gen_cap_scenarios=gen_cap_scenarios,
        load_cap_scenarios=load_cap_scenarios,
        gen_cap_det=gen_cap_det, load_cap_det=load_cap_det
    )


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

    # # Optional DA caps to keep things tight
    cons += [g_da <= np.array([50.0, 75.0, 50.0])]
    cons += [d_da <= np.array([0.0, 100.0, 0.0])]

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
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"System 1 stochastic solve failed: status={prob.status}")

    # ----- Outputs -----
    g_da_v = jnp.array(g_da.value); d_da_v = jnp.array(d_da.value)
    f_da_v = jnp.array(f_da.value); theta_da_v = jnp.array(theta_da.value)
    G_rt_v = jnp.array(G_rt.value); D_rt_v = jnp.array(D_rt.value)
    f_rt_v = jnp.array(f_rt.value); theta_rt_v = jnp.array(theta_rt.value)

    # DA nodal prices from duals
    pi_da = jnp.array([float(da_bal_1.dual_value),
                       float(da_bal_2.dual_value),
                       float(da_bal_3.dual_value)])
#    # Print DA prices
#     print("=== Day-Ahead (DA) Nodal Prices ===")
#     print(f"Node 1: ${pi_da[0]:.3f}/MWh")
#     print(f"Node 2: ${pi_da[1]:.3f}/MWh") 
#     print(f"Node 3: ${pi_da[2]:.3f}/MWh")

    # RT nodal prices per scenario (divide duals by scenario prob)
    Pi_rt = []
    for s in range(S):
        Pi_rt.append([float(c.dual_value)/float(probs[s]) for c in rt_bal_constraints[s]])
    Pi_rt = jnp.array(Pi_rt)  # shape (S, 3)

    #     # Print RT prices
    # print("\n=== Real-Time (RT) Nodal Prices by Scenario ===")
    # for s in range(S):
    #     print(f"Scenario {s+1} (prob={probs[s]:.3f}):")
    #     print(f"  Node 1: ${Pi_rt[s,0]:.3f}/MWh")
    #     print(f"  Node 2: ${Pi_rt[s,1]:.3f}/MWh")
    #     print(f"  Node 3: ${Pi_rt[s,2]:.3f}/MWh")
    
        # Print expected RT prices
    expected_rt_prices = jnp.sum(Pi_rt * probs[:, jnp.newaxis], axis=0)
    # print(f"\n=== Expected RT Prices ===")
    # print(f"Node 1: ${expected_rt_prices[0]:.3f}/MWh")
    # print(f"Node 2: ${expected_rt_prices[1]:.3f}/MWh")
    # print(f"Node 3: ${expected_rt_prices[2]:.3f}/MWh")
    
    # Print price differences
    price_diff = pi_da - expected_rt_prices
    # print(f"\n=== DA vs Expected RT Price Differences ===")
    # print(f"Node 1: ${price_diff[0]:.3f}/MWh")
    # print(f"Node 2: ${price_diff[1]:.3f}/MWh")
    # print(f"Node 3: ${price_diff[2]:.3f}/MWh")

    return g_da_v, d_da_v, f_da_v, theta_da_v, G_rt_v, D_rt_v, f_rt_v, theta_rt_v, pi_da, Pi_rt


def zavala_system_1_deterministic_da():
    """
    Deterministic (here-and-now) DA clearing for System 1 with DC network.
    Uses expected wind cap (50 MWh). No deviation terms.
    Returns:
      g_da (3,), d_da (3,), f_da (2,), theta_da (3,), pi_da (3,)
    """
    import numpy as np
    data = build_system_1_data()
    alpha_gen, alpha_load = data["alpha_gen"], data["alpha_load"]
    B, Fbar = data["susceptance"], data["flow_limit"]
    gbar_det, dbar_det = data["gen_cap_det"], data["load_cap_det"]

    g_da = cp.Variable(3, nonneg=True)
    d_da = cp.Variable(3, nonneg=True)
    theta_da = cp.Variable(3)
    f_da = cp.Variable(2)

    cons = [theta_da[1] == 0]
    cons += [f_da[0] == B[0]*(theta_da[0]-theta_da[1])]
    cons += [f_da[1] == B[1]*(theta_da[1]-theta_da[2])]
    cons += [ -Fbar <= f_da, f_da <= Fbar ]

    bal1 = g_da[0] - d_da[0] - f_da[0] == 0
    bal2 = g_da[1] - d_da[1] + f_da[0] - f_da[1] == 0
    bal3 = g_da[2] - d_da[2] + f_da[1] == 0
    cons += [bal1, bal2, bal3]

    cons += [g_da <= gbar_det, d_da <= dbar_det]

    obj = cp.Minimize(alpha_gen @ g_da - alpha_load @ d_da)
    prob = cp.Problem(obj, cons)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"System 1 deterministic DA failed: status={prob.status}")

    pi_da = -jnp.array([float(bal1.dual_value), float(bal2.dual_value), float(bal3.dual_value)])
    
    # Print Deterministic DA prices
    print("\n=== Deterministic Day-Ahead (DA) Nodal Prices ===")
    print(f"Node 1: ${pi_da[0]:.3f}/MWh")
    print(f"Node 2: ${pi_da[1]:.3f}/MWh")
    print(f"Node 3: ${pi_da[2]:.3f}/MWh")
    
    return jnp.array(g_da.value), jnp.array(d_da.value), jnp.array(f_da.value), jnp.array(theta_da.value), pi_da


def zavala_system_1_rt_network(gen_cap_rt, load_cap_rt):
    """
    Single-scenario RT NETWORK clearing (used for deterministic evaluation).
    Inputs:
      gen_cap_rt (3,), load_cap_rt (3,)
    Returns:
      G (3,), D (3,), f (2,), theta (3,), Pi (3,)
    """
    import numpy as np
    data = build_system_1_data()
    alpha_gen, alpha_load = data["alpha_gen"], data["alpha_load"]
    B, Fbar = data["susceptance"], data["flow_limit"]

    gen_cap_rt = np.asarray(gen_cap_rt, float)
    load_cap_rt = np.asarray(load_cap_rt, float)

    G = cp.Variable(3, nonneg=True)
    D = cp.Variable(3, nonneg=True)
    theta = cp.Variable(3)
    f = cp.Variable(2)

    cons = [theta[1] == 0]
    cons += [f[0] == B[0]*(theta[0]-theta[1])]
    cons += [f[1] == B[1]*(theta[1]-theta[2])]
    cons += [ -Fbar <= f, f <= Fbar ]

    bal1 = G[0] - D[0] - f[0] == 0
    bal2 = G[1] - D[1] + f[0] - f[1] == 0
    bal3 = G[2] - D[2] + f[1] == 0
    cons += [bal1, bal2, bal3]

    cons += [G <= gen_cap_rt, D <= load_cap_rt]

    obj = cp.Minimize(alpha_gen @ G - alpha_load @ D)
    prob = cp.Problem(obj, cons)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"System 1 RT solve failed: status={prob.status}")

    Pi = -jnp.array([float(bal1.dual_value), float(bal2.dual_value), float(bal3.dual_value)])
    
    # Print RT Network prices
    print("\n=== Deterministic Real-Time Network Nodal Prices ===")
    print(f"Node 1: ${Pi[0]:.3f}/MWh")
    print(f"Node 2: ${Pi[1]:.3f}/MWh")
    print(f"Node 3: ${Pi[2]:.3f}/MWh")
    print(f"Generation Capacity: {gen_cap_rt}")
    print(f"Load Capacity: {load_cap_rt}")
    
    return jnp.array(G.value), jnp.array(D.value), jnp.array(f.value), jnp.array(theta.value), Pi


def price_distortion_system_1(pi_da, rt_prices, probs):
    """
    Node-wise price distortion metrics for System 1:
      M_avg = mean_n |pi_da[n] - E[rt_prices[:,n]]|
      M_max = max_n  |pi_da[n] - E[rt_prices[:,n]]|
    """
    import numpy as np
    pi_da = np.asarray(pi_da, float)
    rt_prices = np.asarray(rt_prices, float)
    probs = np.asarray(probs, float)

    exp_rt = probs @ rt_prices
    diff = np.abs(pi_da - exp_rt)
    return float(diff.mean()), float(diff.max())
