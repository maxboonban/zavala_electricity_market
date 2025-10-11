import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import gurobipy as gp

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

def beta_scaled(rng, shape, a, b, lo, hi):
    """Generate beta-distributed random numbers scaled to [lo, hi] range."""
    z = random.beta(rng, a, b, shape=shape)
    return lo + z * (hi - lo)

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

    # # Print solver statistics for debugging
    # print(f"Solver: {prob.solver_stats.solver_name}, Status: {prob.status}, Iterations: {prob.solver_stats.num_iters}")

    g_i_jax = jnp.array([g_i[i].value for i in range(num_g)])
    d_j_jax = jnp.array([d_j[j].value for j in range(num_d)])
    G_i_jax = jnp.array([[G_i[p][i].value for i in range(num_g)] for p in range(len(probs))])
    D_j_jax = jnp.array([[D_j[p][j].value for j in range(num_d)] for p in range(len(probs))])
    pi = day_ahead_balance[0].dual_value
    Pi = jnp.array([real_time_balance[p].dual_value / probs[p] for p in range(len(probs))])


    return g_i_jax, d_j_jax, G_i_jax, D_j_jax, pi, Pi

def truncated_lognormal(rng, shape, mean, sigma, lo, hi):
    x = np.random.lognormal(mean, sigma, size = shape)
    x = np.clip(x, lo, hi)
    return x

def _beta_mixture_left_heavy(rng, n, low=0.0, high=100.0,
                             w=0.65,  # weight on the "bad/low" component
                             a1=0.8, b1=3.5,  # very left-skewed beta
                             a2=4.5, b2=2.0   # right-skewed but bounded
                             ):
    import numpy as np
    u = rng.random(n)
    x1 = rng.beta(a1, b1, size=n)
    x2 = rng.beta(a2, b2, size=n)
    x  = np.where(u < w, x1, x2)
    return low + (high - low) * x

def _dirichlet_near_uniform(rng, n, kappa=500.0):
    # probabilities close to uniform but not identical (helps smoothness)
    import numpy as np
    alpha = np.full(n, kappa / n)
    p = rng.dirichlet(alpha)
    return p / p.sum()

  
def generate_instance(key, num_scenarios = 10, num_g = 10, num_d = 10, minval = 1, maxval = 100):
    input_scenario = "s_htoy_mix"  # "s_1", "s_2", "s_3", "s_7", "s_htoy", "s_htoy_mix"

    if input_scenario == "s_1":
        # Sid's original synthetic case with uniform distribution
        probs_key, mc_key, mv_key, g_key, d_key = random.split(key, 5)
        probs = random.dirichlet(probs_key, jnp.ones(num_scenarios))
        mc_g_i = random.uniform(mc_key, (num_g,), minval=minval, maxval=maxval)
        mv_d_j = random.uniform(mv_key, (num_d,), minval=minval, maxval=maxval)
        g_i_bar = random.uniform(g_key, (num_scenarios, num_g), minval=minval, maxval=maxval)
        d_j_bar = random.uniform(d_key, (num_scenarios, num_d), minval=minval, maxval=maxval)

    elif input_scenario == "s_2":
        # Supply-shortage tails: gen caps left-heavy (more mass near 0)
        # Beta(a<1,b>1). Demand caps remain uniform.
        probs_key, mc_key, mv_key, g_key, d_key = random.split(key, 5)
        probs = random.dirichlet(probs_key, jnp.ones(num_scenarios))
        mc_g_i = random.uniform(mc_key, (num_g,), minval=minval, maxval=maxval)
        mv_d_j = random.uniform(mv_key, (num_d,), minval=minval, maxval=maxval)
        g_i_bar = beta_scaled(g_key, (num_scenarios, num_g), a=0.4, b=4.0, lo=minval, hi=maxval)
        d_j_bar = random.uniform(d_key, (num_scenarios, num_d), minval=minval, maxval=maxval)

    elif input_scenario == "s_3":
        # Demand-surge tails: demand caps right-heavy (more mass near max)
        # Beta(a>1,b<1). Gen caps remain uniform.
        probs_key, mc_key, mv_key, g_key, d_key = random.split(key, 5)
        probs = random.dirichlet(probs_key, jnp.ones(num_scenarios))
        mc_g_i = random.uniform(mc_key, (num_g,), minval=minval, maxval=maxval)
        mv_d_j = random.uniform(mv_key, (num_d,), minval=minval, maxval=maxval)
        g_i_bar = random.uniform(g_key, (num_scenarios, num_g), minval=minval, maxval=maxval)
        d_j_bar = beta_scaled(d_key, (num_scenarios, num_d), a=4.0, b=0.4, lo=minval, hi=maxval)
    
    elif input_scenario == "s_7":
        #print("Using s_7: Original Sid uniform but with truncated lognormal gen caps")
        probs_key, mc_key, mv_key, g_key, d_key = random.split(key, 5)
        probs = random.dirichlet(probs_key, jnp.ones(num_scenarios))
        mc_g_i = random.uniform(mc_key, (num_g,), minval=minval, maxval=maxval)
        mv_d_j = random.uniform(mv_key, (num_d,), minval=minval, maxval=maxval)

        g_i_bar = maxval - truncated_lognormal(g_key, shape=(num_scenarios, num_g), mean=3, sigma=1.0, lo=1, hi=100) 
        d_j_bar = random.uniform(d_key, (num_scenarios, num_d), minval=40, maxval=43)
    
    elif input_scenario == "s_htoy":
        # Single-bus, 2 generators (cheap-unreliable & expensive-reliable), 1 load.
        probs   = jnp.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=float)

        # Marginal bids: cheap unreliable vs. expensive reliable
        mc_g_i  = jnp.array([10.0, 40.0])   # α^g
        mv_d_j  = jnp.array([1000.0])       # VOLL-like α^d to punish curtailment

       
        g1      = jnp.array([20.0, 30.0, 80.0, 90.0, 100.0])
        # Reliable expensive unit
        g2      = jnp.array([55.0, 55.0, 55.0, 55.0, 55.0])

        # Demand cap (deterministic): 90 MW — creates shortfalls when g1 is 20 or 30
        d1      = jnp.array([90.0, 90.0, 90.0, 90.0, 90.0])

        g_i_bar = jnp.stack([g1, g2], axis=1)  
        d_j_bar = d1.reshape(-1, 1)            

        return probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar
    
    elif input_scenario == "s_htoy_mix":
        rng = np.random.default_rng(12)  

        S = max(num_scenarios, 200)  # ensure plenty of scenarios
        probs = _dirichlet_near_uniform(rng, S, kappa=800.0)

        # bids: cheap unreliable vs. reliable but pricier
        mc_g_i = np.array([10.0, 45.0], dtype=float)
        mv_d_j = np.array([1000.0], dtype=float)  # inelastic-ish

        
        g1 = _beta_mixture_left_heavy(rng, S, low=0.0, high=100.0,
                                      w=0.6, a1=0.7, b1=4.0, a2=4.5, b2=2.0)

        # reliable pricey cap
        g2_cap = 55.0
        g2 = np.full(S, g2_cap, dtype=float)

        # demand with mild noise, set to strain system in bad g1 cases
        d_mean = 92.0
        d_sd   = 3.0
        d1 = np.clip(rng.normal(d_mean, d_sd, size=S), 86.0, 98.0)

        g_i_bar = np.stack([g1, g2], axis=1)  
        d_j_bar = d1.reshape(S, 1)            

        
        return probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar
    
    elif input_scenario == "s_real10_mix":
        rng = np.random.default_rng(2025)

        S = num_scenarios
        #S = max(num_scenarios, 400)  
        probs = _dirichlet_near_uniform(rng, S, kappa=1500.0)

        G = 10  # total generators

       
        mc_unrel = rng.uniform(8.0, 14.0, size=6)    # α^g for unreliable
        # Reliable/expensive: higher bids
        mc_rel   = rng.uniform(35.0, 55.0, size=4)   # α^g for reliable
        mc_g_i   = np.concatenate([mc_unrel, mc_rel])

        # Load willingness / VOLL (single inelastic-ish load)
        mv_d_j   = np.array([1000.0], dtype=float)

 
        unrel_ranges = [(25, 90), (30, 95), (20, 85), (35, 100), (28, 92), (22, 88)]
        rel_ranges   = [(20, 30), (22, 32), (18, 28), (24, 34)]  # tighter & higher floor

        # states: 0=no shock, 1=mild shock, 2=severe shock
        # probabilities tuned to put enough mass in the tail without being spiky
        shock_state = rng.choice([0, 1, 2], size=S, p=[0.70, 0.20, 0.10])
        # multiplicative factors applied to ALL unreliable gens in that scenario
        shock_factor = np.ones(S)
        # mild: cut to 60-85% output, severe: cut to 15-40% → drives tail
        shock_factor[shock_state == 1] *= rng.uniform(0.60, 0.85, size=(shock_state == 1).sum())
        shock_factor[shock_state == 2] *= rng.uniform(0.15, 0.40, size=(shock_state == 2).sum())


        def _jitter(n, lo=0.92, hi=1.08):
            return rng.uniform(lo, hi, size=n)

        # Build unreliable caps across scenarios
        unrel_caps = []
        for (lo, hi) in unrel_ranges:
            base = _beta_mixture_left_heavy(rng, S, low=lo, high=hi,
                                            w=0.62, a1=0.7, b1=4.3, a2=5.0, b2=2.1)
            # apply common shock + idiosyncratic jitter
            cap = base * shock_factor * _jitter(S)
            unrel_caps.append(np.clip(cap, 0.0, None))
        unrel_caps = np.column_stack(unrel_caps)  # (S,6)

        rel_caps = []
        for (lo, hi) in rel_ranges:
            # concentrate near upper end (reliable), small variance
            x = rng.beta(9.0, 2.0, size=S)  # skewed high
            base = lo + (hi - lo) * x
            cap = base * _jitter(S, 0.98, 1.03)
            rel_caps.append(np.clip(cap, 0.0, None))
        rel_caps = np.column_stack(rel_caps)     

        g_i_bar = np.column_stack([unrel_caps, rel_caps])  

        # base demand ~ N(μ,σ) and higher during shocks → exacerbates tail
        d_mean, d_sd = 220.0, 10.0
        D = rng.normal(d_mean, d_sd, size=S)
        D += np.where(shock_state == 1, rng.uniform(8, 15, size=S), 0.0)  
        D += np.where(shock_state == 2, rng.uniform(18, 30, size=S), 0.0) 
        D = np.clip(D, 190.0, 265.0)
        d_j_bar = D.reshape(S, 1)

        return probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar

    else:
        raise ValueError(f"Unknown input_scenario: {input_scenario}")

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
        raise RuntimeError(f"Deterministic DA solve failed: status={prob.status}")

    # Dual of the balance is the (single) day-ahead price π
    pi = float(balance_con.dual_value) if balance_con.dual_value is not None else np.nan
    
    # # Print Energy-Only Deterministic DA price
    # print("\n=== Energy-Only Deterministic Day-Ahead Price ===")
    # print(f"Single Bus Price: ${pi:.3f}/MWh")

    return jnp.array(g.value), jnp.array(d.value), pi


# === NEW: real-time energy-only per scenario (to get Π(ω)) ===================
def zavala_rt_energy_only(mc_g_i, mv_d_j, g_da, d_da, g_i_bar_rt, d_j_bar_rt):
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

    # Compute incremental bids (just a heuristic)
    mc_g_i_delta = mc_g_i / 10.0
    mv_d_j_delta = mv_d_j / 10.0
    delta_alpha_g = np.asarray(mc_g_i_delta, float).reshape(num_g)  # generator incremental bid prices
    delta_alpha_d = np.asarray(mv_d_j_delta, float).reshape(num_d)  # load incremental bid prices

    G = cp.Variable(num_g, nonneg=True)
    D = cp.Variable(num_d, nonneg=True)

    obj = cp.Minimize(mc_g_i @ G - mv_d_j @ D)
    # obj = cp.Minimize(delta_alpha_g @ cp.abs(G - g_da) + delta_alpha_d @ cp.abs(D - d_da))


    balance_con = cp.sum(D) - cp.sum(G) == 0
    cons = [balance_con, G <= g_i_bar_rt, D <= d_j_bar_rt]

    prob = cp.Problem(obj, cons)
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

    # bids prices
    alpha_gen  = np.array([10.0, 1.0, 20.0])      # generator bids at nodes [1,2,3]
    alpha_load = np.array([0.0, 1000.0, 0.0])     # value of lost load at node 2

    # incremental bid prices
    delta_gen = np.array([1.0, 0.10, 2.0])
    # incremental load bids
    delta_load = np.array([0.0, 0.001, 0.0])

    susceptance = np.array([50.0, 50.0])          # for lines [1-2, 2-3]
    flow_limit  = np.array([25.0, 50.0])          # thermal limits [1-2, 2-3]
    delta_flow  = 1e-3                          # deviation price for flows
    delta_theta = 1e-3                          # deviation price for angles


    # generator caps per scenario: g1=50, g2 in {25,50,75}, g3=50
    gen_cap_scenarios = np.zeros((num_scenarios, 3))
    gen_cap_scenarios[0] = np.array([50.0, 25.0, 50.0])
    gen_cap_scenarios[1] = np.array([50.0, 50.0, 50.0])
    gen_cap_scenarios[2] = np.array([50.0, 75.0, 50.0])

    # demand caps per scenario (only node 2 has 100; deterministic across scenarios)
    load_cap_scenarios = np.zeros((num_scenarios, 3))
    load_cap_scenarios[:, 1] = 100.0

    Gbar_max = gen_cap_scenarios.max(axis=0)  # -> [50., 75., 50.]
    Dbar_max = load_cap_scenarios.max(axis=0) # -> [ 0.,100.,  0.]

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
        gen_cap_det=gen_cap_det, load_cap_det=load_cap_det,
        Gbar_max=Gbar_max, Dbar_max=Dbar_max
    )

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

    # Up/Down deviations/auxiliary variables
    Ugp = cp.Variable((S, N), nonneg=True); Ugn = cp.Variable((S, N), nonneg=True)  # Generator Up/down deviations:G_rt - g_da = Ugp - Ugn
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

    # CAp DA generation
    cons += [g_da <= np.array([50.0, 50.0, 50.0])]

    # Network constraints
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
        cons += [G_rt[s] - g_da == Ugp[s] - Ugn[s]] # Generator real-time - DA deviation = Generator Up/down deviations
        cons += [D_rt[s] - d_da == Udp[s] - UdN[s]] # Load real-time - DA deviation = Load Up/down deviations
        cons += [f_rt[s] - f_da == Wfp[s] - WfN[s]] # Flow real-time - DA deviation = Flow Up/down deviations
        cons += [theta_rt[s] - theta_da == Ztp[s] - ZtN[s]] # Angle real-time - DA deviation = Angle Up/down deviations

    # objective function: split into a two-step approach
    base_DA = alpha_gen @ g_da - alpha_load @ d_da
    obj_terms = []
    for s in range(S):
        rt_lin   = alpha_gen @ G_rt[s] - alpha_load @ D_rt[s]
        dev_gen  = cp.sum(cp.multiply(delta_gen,  Ugp[s] + Ugn[s]))
        dev_load = cp.sum(cp.multiply(delta_load, Udp[s] + UdN[s]))
        dev_flow = delta_flow  * cp.sum(Wfp[s] + WfN[s])
        dev_th   = delta_theta * cp.sum(Ztp[s] + ZtN[s])
        obj_terms.append(probs[s] * (rt_lin + dev_gen + dev_load + dev_flow + dev_th))

    objective = base_DA + cp.sum(obj_terms)
    prob = cp.Problem(cp.Minimize(objective), cons)
    prob.solve(solver=cp.GUROBI, Method=2, Crossover=0, FeasibilityTol=1e-9, OptimalityTol=1e-9)
    print(f"Objective value: {objective.value:.10f}")

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

def zavala_cvar(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar):
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

    # --- CVaR knobs ---
    beta = 0.95
    lambda_cvar = 0.1  # set >0 to turn on CVaR regularization

    # --- Build per-scenario loss L_p ---
    Lp = []
    for p in range(len(probs)):
        loss_gen = sum(
            mc_g_i[i] * G_i[p][i]
            + mc_g_i_delta[i] * cp.max(cp.vstack([G_i[p][i] - g_i[i], 0]))
            + mc_g_i_delta[i] * cp.max(cp.vstack([g_i[i] - G_i[p][i], 0]))
            for i in range(num_g)
        )
        loss_dem = sum(
            -mv_d_j[j] * D_j[p][j]
            + mv_d_j_delta[j] * cp.max(cp.vstack([d_j[j] - D_j[p][j], 0]))
            + mv_d_j_delta[j] * cp.max(cp.vstack([D_j[p][j] - d_j[j], 0]))
            for j in range(num_d)
        )
        Lp.append(loss_gen + loss_dem)

    # --- Rockafellar–Uryasev CVaR(β): alpha and scenario slacks ---
    alpha = cp.Variable()
    s = [cp.Variable(nonneg=True) for _ in range(len(probs))]
    cvar_link = [s[p] >= Lp[p] - alpha for p in range(len(probs))]
    cvar_term = alpha + (1.0 / (1.0 - beta)) * sum(probs[p] * s[p] for p in range(len(probs)))

    # Your original expected-loss expression, unchanged
    expected_loss = (
        sum(sum((mc_g_i[i] * G_i[p][i]
                + mc_g_i_delta[i] * cp.max(cp.vstack([G_i[p][i] - g_i[i], 0]))
                + mc_g_i_delta[i] * cp.max(cp.vstack([g_i[i] - G_i[p][i], 0])))
                * probs[p] for p in range(len(probs))) for i in range(num_g))
        + sum(sum((-mv_d_j[j] * D_j[p][j]
                + mv_d_j_delta[j] * cp.max(cp.vstack([d_j[j] - D_j[p][j], 0]))
                + mv_d_j_delta[j] * cp.max(cp.vstack([D_j[p][j] - d_j[j], 0])))
                * probs[p] for p in range(len(probs))) for j in range(num_d))
    )

    objective = cp.Minimize(expected_loss + lambda_cvar * cvar_term)

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
                ] + cvar_link
    

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
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

    # # Print solver statistics for debugging
    # print(f"Solver: {prob.solver_stats.solver_name}, Status: {prob.status}, Iterations: {prob.solver_stats.num_iters}")

    g_i_jax = jnp.array([g_i[i].value for i in range(num_g)])
    d_j_jax = jnp.array([d_j[j].value for j in range(num_d)])
    G_i_jax = jnp.array([[G_i[p][i].value for i in range(num_g)] for p in range(len(probs))])
    D_j_jax = jnp.array([[D_j[p][j].value for j in range(num_d)] for p in range(len(probs))])
    pi = day_ahead_balance[0].dual_value
    Pi = jnp.array([real_time_balance[p].dual_value / probs[p] for p in range(len(probs))])
    cvar_link_dual = jnp.array([float(np.asarray(c.dual_value).ravel()[0]) if c.dual_value is not None else np.nan for c in cvar_link])

    return g_i_jax, d_j_jax, G_i_jax, D_j_jax, pi, Pi, cvar_link_dual

def _print_da_rt_summary(label, pi, g_da, d_da, Pi, G_rt, D_rt, probs=None):
    """
    Prints a compact DA/RT summary.
    - label: str, "STOCH" or "DET"
    - pi: scalar DA price
    - g_da, d_da: 1D arrays of DA generator and demand commitments
    - Pi: 1D array of RT prices per scenario (length S)
    - G_rt, D_rt: 2D arrays (S x G) and (S x D) of RT dispatches
    - probs: optional scenario probabilities (for weighted summaries)
    """
    import numpy as np
    S = Pi.shape[0]
    G = g_da.shape[0]
    D = d_da.shape[0]

    dG = G_rt - g_da[None, :]   # S x G
    dD = D_rt - d_da[None, :]   # S x D

    print(f"\n==== {label} — Day-Ahead ====")
    print(f"DA price π: {pi:.4f}")
    print(f"DA gen commits g_i (len={G}): {np.array(g_da).round(4)}")
    print(f"DA load commits d_j (len={D}): {np.array(d_da).round(4)}")

    print(f"\n==== {label} — Real-Time (per scenario) ====")
    print(f"RT prices Π(ω) (len={S}): {np.array(Pi).round(4)}")
    print("ΔG = G(ω) - g (negative ⇒ scaled back):")
    print(np.array(dG).round(4))
    print("ΔD = D(ω) - d (positive ⇒ demand up):")
    print(np.array(dD).round(4))

    # Small summary by generator (how often scaled back / up)
    tol = 1e-9
    frac_back = (dG < -tol).mean(axis=0)
    frac_up   = (dG >  tol).mean(axis=0)
    print("\nPer-generator fraction of scenarios:")
    print(f"  scaled back (ΔG<0): {np.array(frac_back).round(2)}")
    print(f"  ramped up  (ΔG>0): {np.array(frac_up).round(2)}")

    if probs is not None:
        # probability-weighted average adjustments
        w_avg_dG = (probs[:, None] * dG).sum(axis=0)
        w_avg_dD = (probs[:, None] * dD).sum(axis=0)
        print("\nProbability-weighted average ΔG per generator:")
        print(np.array(w_avg_dG).round(4))
        print("Probability-weighted average ΔD per load:")
        print(np.array(w_avg_dD).round(4))

def _stack_rt(G_list, D_list):
    """Convert lists of per-scenario RT vectors into arrays shaped (S, G/D)."""
    import numpy as np
    G_rt = np.vstack([np.asarray(x) for x in G_list])
    D_rt = np.vstack([np.asarray(x) for x in D_list])
    return G_rt, D_rt

def compute_social_surplus(
    probs,
    mc_g_i, mv_d_j,          # α^g, α^d  (vectors)
    g_da, d_da,              # day-ahead commitments (vectors)
    G_rt, D_rt,              # arrays shaped (S, G) and (S, D)
    mc_g_i_delta=None,       # optional Δα for generators (scalar or vector)
    mv_d_j_delta=None,       # optional Δα for loads (scalar or vector)
):
    """
    Expected NEGATIVE social surplus (cost), evaluated in the Eq. 10 'incremental-delta' form.

      Generators:   α^g · G(ω) + Δα^{g,+}(G-g)_+ + Δα^{g,−}(g-G)_+
      Loads:       −α^d · D(ω) + Δα^{d,+}(D-d)_+ + Δα^{d,−}(d-D)_+

    If only mc_g_i_delta / mv_d_j_delta are provided, they are treated as symmetric:
        Δα^{•,+} = Δα^{•,−} = given delta
    """
    import numpy as np

    p  = np.asarray(probs,  dtype=float)
    ag = np.asarray(mc_g_i, dtype=float).reshape(-1)   # α^g
    ad = np.asarray(mv_d_j, dtype=float).reshape(-1)   # α^d

    g  = np.asarray(g_da,   dtype=float).reshape(-1)
    d  = np.asarray(d_da,   dtype=float).reshape(-1)
    G  = np.asarray(G_rt,   dtype=float)              # (S,G)
    D  = np.asarray(D_rt,   dtype=float)              # (S,D)

    S, Gdim = G.shape
    _, Ddim = D.shape

    # symmetric increments by default (matches your solver’s Δ = α/10 heuristic)
    if mc_g_i_delta is None:
        mc_g_i_delta = ag / 10.0
    if mv_d_j_delta is None:
        mv_d_j_delta = ad / 10.0

    mc_g_i_delta = np.broadcast_to(np.asarray(mc_g_i_delta, dtype=float).reshape(-1), ag.shape)
    mv_d_j_delta = np.broadcast_to(np.asarray(mv_d_j_delta, dtype=float).reshape(-1), ad.shape)

    # Allow asymmetric deltas later if you choose:
    # Δ+ = Δ− = given delta (for now)
    dg_plus  = mc_g_i_delta
    dg_minus = mc_g_i_delta
    dd_plus  = mv_d_j_delta
    dd_minus = mv_d_j_delta

    # helpers
    def pos(x): return np.maximum(x, 0.0)

    # Per-scenario NEGATIVE social surplus (cost)
    #   gen:  α^g · G  +  Δg+·(G−g)_+ + Δg−·(g−G)_+
    #   load: −α^d · D  +  Δd+·(D−d)_+ + Δd−·(d−D)_+
    ss_per_scn = np.empty(S, dtype=float)
    for s in range(S):
        dG = G[s] - g
        dD = D[s] - d

        gen_cost  = ag @ G[s]  +  dg_plus @ pos(dG)  +  dg_minus @ pos(-dG)
        load_cost = - ad @ D[s] +  dd_plus @ pos(dD) +  dd_minus @ pos(-dD)

        ss_per_scn[s] = gen_cost + load_cost   # NEGATIVE social surplus for scenario s

    # Expectations
    E_neg_total     = float(p @ ss_per_scn)
    E_social_surplus = -E_neg_total

    # Optional split (supplier vs consumer) if you want:
    # E_neg_supplier = float(p @ (ag @ G.T + (dg_plus @ pos(G - g).T) + (dg_minus @ pos(g - G).T)))
    # E_neg_consumer = float(p @ (-ad @ D.T + (dd_plus @ pos(D - d).T) + (dd_minus @ pos(d - D).T)))

    # If you want to preserve keys present in your code:
    E_neg_supplier = np.nan
    E_neg_consumer = np.nan

    return {
        "E_neg_supplier": E_neg_supplier,
        "E_neg_consumer": E_neg_consumer,
        "E_neg_total":    E_neg_total,
        "E_social_surplus": E_social_surplus,
        "ss_per_scenario": ss_per_scn,   # NEG-SS per scenario (useful for tail metrics)
    }

# def compute_social_surplus(
#     probs,
#     mc_g_i, mv_d_j,          # α^g, α^d  (vectors)
#     g_da, d_da,              # day-ahead commitments
#     G_rt, D_rt,              # arrays shaped (S, G) and (S, D)
#     mc_g_i_delta=None,       # optional Δα for generators
#     mv_d_j_delta=None,       # optional Δα for loads
# ):
#     """
#     Implements §4.1 Social Surplus:

#       C_i^g(ω) = +α_i^g g_i
#                   + α_i^{g,+} (G_i(ω) - g_i)_+
#                   - α_i^{g,−} (G_i(ω) - g_i)_−

#       C_j^d(ω) = −α_j^d d_j
#                   + α_j^{d,+} (D_j(ω) - d_j)_+
#                   - α_j^{d,−} (D_j(ω) - d_j)_−

#     Negative social surplus per scenario is  Σ_i C_i^g(ω) + Σ_j C_j^d(ω).
#     We return its expectation, its components, and (+)social surplus (the negative of it).

#     Returns dict with:
#       E_neg_supplier, E_neg_consumer, E_neg_total, E_social_surplus
#     """
#     import numpy as np

#     probs   = np.asarray(probs, dtype=float)
#     mc_g_i  = np.asarray(mc_g_i, dtype=float)
#     mv_d_j  = np.asarray(mv_d_j, dtype=float)
#     g_da    = np.asarray(g_da, dtype=float)
#     d_da    = np.asarray(d_da, dtype=float)
#     G_rt    = np.asarray(G_rt, dtype=float)  # (S, G)
#     D_rt    = np.asarray(D_rt, dtype=float)  # (S, D)

#     # Default incremental prices: your current heuristic
#     if mc_g_i_delta is None:
#         mc_g_i_delta = mc_g_i / 10.0
#     if mv_d_j_delta is None:
#         mv_d_j_delta = mv_d_j / 10.0

#     # Construct α^{·,+} and α^{·,−} from Δα, per paper (Δα>0):
#     #   Δα^{g,+} = α^{g,+} − α^g  ⇒ α^{g,+} = α^g + Δα^{g,+}
#     #   Δα^{g,−} = α^g − α^{g,−} ⇒ α^{g,−} = α^g − Δα^{g,−}
#     alpha_g_plus  = mc_g_i + mc_g_i_delta
#     alpha_g_minus = mc_g_i - mc_g_i_delta
#     alpha_d_plus  = mv_d_j + mv_d_j_delta
#     alpha_d_minus = mv_d_j - mv_d_j_delta

#     # Deviations
#     dG = G_rt - g_da[None, :]   # (S, G)
#     dD = D_rt - d_da[None, :]   # (S, D)
#     pos = lambda x: np.maximum(x, 0.0)
#     neg = lambda x: np.maximum(-x, 0.0)

#     # Supplier negative cost per scenario (vector shape (S,))
#     const_sup = float(np.sum(mc_g_i * g_da))
#     Cg = const_sup + pos(dG) @ alpha_g_plus - neg(dG) @ alpha_g_minus

#     # Consumer negative cost per scenario
#     const_dem = -float(np.sum(mv_d_j * d_da))
#     Cd = const_dem + pos(dD) @ alpha_d_plus - neg(dD) @ alpha_d_minus

#     neg_total = Cg + Cd

#     E_neg_supplier = float(probs @ Cg)
#     E_neg_consumer = float(probs @ Cd)
#     E_neg_total    = float(probs @ neg_total)
#     E_social_surplus = -E_neg_total

#     ss_per_scenario = neg_total

#     return {
#         "E_neg_supplier": E_neg_supplier,
#         "E_neg_consumer": E_neg_consumer,
#         "E_neg_total":    E_neg_total,
#         "E_social_surplus": E_social_surplus,
#         "ss_per_scenario": ss_per_scenario
#     }

# === Tail scenario helpers for tail welfare analysis ===
def tail_worst_indices_by_value(values, probs, tail=0.05, worst="high"):
    """
    Return indices of the worst `tail` probability mass using the supplied per-scenario
    `values` and `probs`. We pick WHOLE scenarios (no fractional split) until the
    cumulative probability reaches / exceeds `tail`.

    Parameters
    ----------
    values : array-like, shape (S,)
        Per-scenario metric used to rank 'worst' (e.g., NEGATIVE social surplus from
        compute_social_surplus(...)[ "ss_per_scenario" ]).
    probs : array-like, shape (S,)
        Scenario probabilities (sum to 1).
    tail : float, default 0.05
        Target tail probability mass, e.g., 0.05 for 5%.
    worst : {"low","high"}, default "low"
        How to define 'worse':
          - "low": smaller values are worse (use this if `values` are NEG-SS and
                   you consider more negative to be worse).
          - "high": larger values are worse.

    Returns
    -------
    idx : np.ndarray of int, shape (K,)
        Scenario indices (0..S-1) that make up the selected worst tail, ordered
        from worst to less-worse according to `worst`.
    """
    import numpy as np

    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(probs,  dtype=float).ravel()
    if v.shape != p.shape:
        raise ValueError("values and probs must be 1D arrays of equal length.")

    if worst not in ("low", "high"):
        raise ValueError("worst must be 'low' or 'high'")

    order = np.argsort(v) if worst == "low" else np.argsort(v)[::-1]

    cum = 0.0
    chosen = []
    for k in order:
        chosen.append(int(k))
        cum += float(p[k])
        if cum >= float(tail) - 1e-15:
            break

    return np.asarray(chosen, dtype=int)
