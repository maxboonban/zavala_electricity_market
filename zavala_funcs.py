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