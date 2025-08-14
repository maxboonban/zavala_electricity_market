import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

def zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar):
    # Cast inputs to float64 for numerical stability
    probs = np.asarray(probs, dtype=np.float64)
    mc_g_i = np.asarray(mc_g_i, dtype=np.float64)
    mv_d_j = np.asarray(mv_d_j, dtype=np.float64)
    g_i_bar = np.asarray(g_i_bar, dtype=np.float64)
    d_j_bar = np.asarray(d_j_bar, dtype=np.float64)
    
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
    
    # Try to solve with tight tolerances, using whatever solver is available
    try:
        prob.solve(abstol=1e-8, reltol=1e-8, feastol=1e-8)
    except:
        # Fallback to default solver if tolerance setting fails
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