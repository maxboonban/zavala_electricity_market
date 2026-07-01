# %%
import os
import argparse
import numpy as np
import pandas as pd

# Paths: notebook is in zavala_electricity_market, data in dataset/IM-3-GO-WEST
# Run from project root (prj_market) or from zavala_electricity_market
_cwd = os.getcwd()
if os.path.basename(_cwd) == "zavala_electricity_market":
    BASE_DIR = os.path.dirname(_cwd)
else:
    BASE_DIR = _cwd
DATA_DIR = os.path.join(BASE_DIR, "dataset", "IM-3-GO-WEST")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(_cwd, "dataset", "IM-3-GO-WEST")

# Ensure zavala_electricity_market is on path (when running from project root)
import sys
ZAVALA_DIR = os.path.join(BASE_DIR, "zavala_electricity_market")
if os.path.isdir(ZAVALA_DIR) and ZAVALA_DIR not in sys.path:
    sys.path.insert(0, ZAVALA_DIR)

# --- Minimal customization for current repo layout ---

# 1) If the original DATA_DIR doesn't exist, try archive/dataset/IM-3-GO-WEST
if not os.path.isdir(DATA_DIR):
    alt_data_dir = os.path.join(BASE_DIR, "archive", "dataset", "IM-3-GO-WEST")
    if os.path.isdir(alt_data_dir):
        DATA_DIR = alt_data_dir

# 2) If ZAVALA_DIR isn't a valid dir (e.g. code is at repo root),
#    fall back to BASE_DIR itself when it has zavala_funcs.py
if not os.path.isdir(ZAVALA_DIR):
    repo_root_candidate = BASE_DIR
    if os.path.isfile(os.path.join(repo_root_candidate, "zavala_funcs.py")):
        if repo_root_candidate not in sys.path:
            sys.path.insert(0, repo_root_candidate)
        ZAVALA_DIR = repo_root_candidate

print("Using DATA_DIR:", DATA_DIR)
print("Using ZAVALA_DIR:", ZAVALA_DIR)
print("Files:", os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else "not found")

from zavala_funcs import (
    zavala,
    zavala_cvar,
    zavala_deterministic_da,
    zavala_rt_energy_only,
    expected_caps_from_scenarios,
    price_distortion,
    probability_feasible,
    expected_cumulative_regret,
    compute_social_surplus,
    tail_worst_indices_by_value,
    _stack_rt,
)

print("Data dir:", DATA_DIR)
print("Files:", os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else "not found")

# %%
parser = argparse.ArgumentParser(description="Zavala electricity market experiment")
parser.add_argument("--num-wind",        type=int, default=10,          help="Number of wind generators")
parser.add_argument("--num-solar",       type=int, default=10,          help="Number of solar generators")
parser.add_argument("--num-thermal",     type=int, default=4,           help="Number of thermal generators")
parser.add_argument("--num-instances",   type=int, default=10,          help="Number of instances to run")
parser.add_argument("--num-scenarios",   type=int, default=500,         help="Scenarios per instance")
parser.add_argument("--lambda-cvar",     type=float, default=0.1,       help="CVaR regularization weight (0 recovers plain stochastic clearing)")
parser.add_argument("--beta",            type=float, default=0.95,      help="CVaR tail confidence level (e.g. 0.90, 0.95, 0.98)")
parser.add_argument("--experiment-name", type=str, default="experiment", help="Name for output files")
parser.add_argument("--data-dir",        type=str, default=None,        help="Override data directory")
args = parser.parse_args()

num_solar   = args.num_solar
num_wind    = args.num_wind
num_thermal = args.num_thermal

if args.data_dir:
    DATA_DIR = args.data_dir

print(f"Experiment : {args.experiment_name}")
print(f"Generators — solar: {num_solar}, wind: {num_wind}, thermal: {num_thermal}")
print(f"Instances  : {args.num_instances}, Scenarios per instance: {args.num_scenarios}")
print(f"CVaR params: lambda_cvar={args.lambda_cvar}, beta={args.beta}")

# %%
# Load nodal time series (rows = time, columns = bus_XXXXX)
solar = pd.read_csv(os.path.join(DATA_DIR, "nodal_solar.csv"))
wind = pd.read_csv(os.path.join(DATA_DIR, "nodal_wind.csv"))
load_df = pd.read_csv(os.path.join(DATA_DIR, "nodal_load.csv"))
thermal_df = pd.read_csv(os.path.join(DATA_DIR, "thermal_gens.csv"))

T = len(solar)
assert len(wind) == T and len(load_df) == T, "Solar, wind, load must have same length"
print(f"Time steps: {T}")
print(f"Solar columns: {solar.shape[1]}, Wind: {wind.shape[1]}, Load: {load_df.shape[1]}")
print(f"Thermal generators: {len(thermal_df)}")

# %%
# Choose buses with non-trivial solar: columns with max > threshold
solar_cols = [c for c in solar.columns if solar[c].max() > 50]
wind_cols = [c for c in wind.columns if wind[c].max() > 50]
# Pick x solar and y wind (unreliable)
solar_buses = solar_cols[:num_solar] if len(solar_cols) >= num_solar else list(solar.columns[:num_solar])
wind_buses = wind_cols[:num_wind] if len(wind_cols) >= num_wind else list(wind.columns[:num_wind])

# Reliable: aggregate thermal by bus, pick num_thermal buses with largest capacity
thermal_by_bus = thermal_df.groupby("Bus")["Max_Cap"].sum().sort_values(ascending=False)
thermal_buses_numeric = list(thermal_by_bus.head(num_thermal).index)
thermal_bus_cols = [f"bus_{b}" for b in thermal_buses_numeric]  # for load alignment if needed

# Load: use total system load (sum over all buses)
load_total = load_df.sum(axis=1).values  # (T,)

print("Solar buses (unreliable):", solar_buses)
print("Wind buses (unreliable):", wind_buses)
print("Thermal buses (reliable):", thermal_buses_numeric)
print("Load: system total (sum over all buses)")

# %%
# Filter thermal_df for the selected buses
thermal_selected = thermal_df[thermal_df['Bus'].isin(thermal_buses_numeric)]
print(thermal_selected)

# %% [markdown]
# Debug Logs

# %%
# For diagnostics: breakdown of DA and RT allocations
def _log(msg, logfile=None):
    if logfile is None:
        print(msg)
    else:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

def _tech_breakdown_da(g_da):
    g_da = np.asarray(g_da, dtype=float)
    return {
        "solar": g_da[:num_solar].sum(),
        "wind": g_da[num_solar:num_solar+num_wind].sum(),
        "thermal": g_da[num_solar+num_wind:].sum(),
        "total": g_da.sum(),
    }

def _tech_breakdown_rt(G_rt, probs=None):
    G_rt = np.asarray(G_rt, dtype=float)
    solar = G_rt[:, :num_solar].sum(axis=1)
    wind = G_rt[:, num_solar:num_solar+num_wind].sum(axis=1)
    thermal = G_rt[:, num_solar+num_wind:].sum(axis=1)
    total = G_rt.sum(axis=1)

    if probs is None:
        return {
            "solar": solar.mean(),
            "wind": wind.mean(),
            "thermal": thermal.mean(),
            "total": total.mean(),
        }

    probs = np.asarray(probs, dtype=float)
    return {
        "solar": np.dot(probs, solar),
        "wind": np.dot(probs, wind),
        "thermal": np.dot(probs, thermal),
        "total": np.dot(probs, total),
    }

def _print_case_diag(name, probs, g_da, d_da, G_rt, D_rt, pi, Pi, logfile=None):
    g_da = np.asarray(g_da, dtype=float)
    d_da = np.asarray(d_da, dtype=float)
    G_rt = np.asarray(G_rt, dtype=float)
    D_rt = np.asarray(D_rt, dtype=float)

    da = _tech_breakdown_da(g_da)
    rt = _tech_breakdown_rt(G_rt, probs=probs)

    da_load = float(d_da.sum())
    exp_rt_load = float(np.dot(np.asarray(probs, dtype=float), D_rt.sum(axis=1)))

    da_supply = float(g_da.sum())
    exp_rt_supply = float(np.dot(np.asarray(probs, dtype=float), G_rt.sum(axis=1)))

    _log(f"\n===== {name} =====", logfile)
    _log(f"DA price pi: {float(pi):.6f}", logfile)
    _log(f"E[RT price]: {float(np.dot(np.asarray(probs, dtype=float), np.asarray(Pi, dtype=float))):.6f}", logfile)
    _log("DA allocation by tech: " + str({k: round(v, 4) for k, v in da.items()}), logfile)
    _log("Expected RT allocation by tech: " + str({k: round(v, 4) for k, v in rt.items()}), logfile)

    gen_names = [f"solar_{i+1}" for i in range(num_solar)] + [f"wind_{i+1}" for i in range(num_wind)] + [f"thermal_{i+1}" for i in range(num_thermal)]
    g_da = np.asarray(g_da, dtype=float)
    exp_rt = np.dot(np.asarray(probs, dtype=float), np.asarray(G_rt, dtype=float))

    df = pd.DataFrame({
        "gen": gen_names,
        "DA": g_da,
        "E_RT": exp_rt,
        "RT_minus_DA": exp_rt - g_da,
    })
    _log(df.to_string(index=False), logfile)
    _log(f"DA load: {da_load:.6f}", logfile)
    _log(f"E[RT load]: {exp_rt_load:.6f}", logfile)
    _log(f"DA supply - DA load: {da_supply - da_load:.6f}", logfile)
    _log(f"E[RT supply] - E[RT load]: {exp_rt_supply - exp_rt_load:.6f}", logfile)

def _print_stoch_vs_cvar_diff(z_g_i, cvar_g_i, logfile=None):
    gen_names = [f"solar_{i+1}" for i in range(num_solar)] + [f"wind_{i+1}" for i in range(num_wind)] + [f"thermal_{i+1}" for i in range(num_thermal)]
    z = np.asarray(z_g_i, dtype=float)
    c = np.asarray(cvar_g_i, dtype=float)
    df = pd.DataFrame({
        "gen": gen_names,
        "stoch_DA": z,
        "cvar_DA": c,
        "cvar_minus_stoch": c - z,
    })
    _log("\n===== CVaR - Stochastic DA difference =====", logfile)
    _log(df.to_string(index=False), logfile)
    _log("Tech-level difference: " + str({
        "solar": round((c[:num_solar] - z[:num_solar]).sum(), 4),
        "wind": round((c[num_solar:num_solar+num_wind] - z[num_solar:num_solar+num_wind]).sum(), 4),
        "thermal": round((c[num_solar+num_wind:] - z[num_solar+num_wind:]).sum(), 4),
        "total": round((c - z).sum(), 4),
    }), logfile)

# %%
# Initialize log files
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

debug_log = os.path.join(log_dir, f"{args.experiment_name}_debug.txt")
with open(debug_log, "w", encoding="utf-8") as f:
    f.write(f"Zavala diagnostics log — {args.experiment_name}\n")
    f.write(f"CVaR params: lambda_cvar={args.lambda_cvar}, beta={args.beta}\n")

# %% [markdown]
# Build Real Data

# %%
def build_real_data_instance(solar_df, wind_df, load_total_vec, thermal_by_bus, thermal_buses_numeric,
                              solar_buses, wind_buses, start_idx, num_scenarios, rng=None):
    """
    Build (probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar) from real data for one time window.
    - start_idx: first time index
    - num_scenarios: number of consecutive time steps (scenarios)
    """
    if rng is None:
        rng = np.random.default_rng()
    end_idx = start_idx + num_scenarios
    S = num_scenarios

    # Scenario probabilities: near-uniform (same idea as s_real10_mix)
    kappa = 1500.0
    alpha = np.full(S, kappa / S)
    probs = rng.dirichlet(alpha)
    probs = probs / probs.sum()

    # Marginal costs: cheap for unreliable (solar/wind), higher for reliable (thermal)
    mc_unrel = rng.uniform(8.0, 14.0, size=len(solar_buses) + len(wind_buses))
    mc_rel = rng.uniform(35.0, 55.0, size=len(thermal_buses_numeric))
    mc_g_i = np.concatenate([mc_unrel, mc_rel]).astype(float)

    # Single inelastic load (VOLL)
    mv_d_j = np.array([1000.0], dtype=float)

    # Generator capacities per scenario (S x num_solar+num_wind+num_thermal)
    # Columns 0..num_solar-1: solar, num_solar..num_solar+num_wind-1: wind, rest: thermal
    solar_vals = solar_df.loc[start_idx:end_idx - 1, solar_buses].values  # (S, 3)
    wind_vals = wind_df.loc[start_idx:end_idx - 1, wind_buses].values    # (S, 3)
    unrel_caps = np.clip(np.hstack([solar_vals, wind_vals]), 0.0, None)  # (S, 6)

    rel_caps = np.array([thermal_by_bus[b] for b in thermal_buses_numeric], dtype=float)
    rel_caps = np.broadcast_to(rel_caps, (S, 4))  # (S, 4) constant across scenarios

    g_i_bar = np.hstack([unrel_caps, rel_caps])  # (S, 10)

    # Demand: system total load for each scenario (S x 1)
    d_j_bar = load_total_vec[start_idx:end_idx].reshape(-1, 1).astype(float)
    d_j_bar = np.clip(d_j_bar, 1e-6, None)  # avoid zeros

    return probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar

# %%
# Quick sanity check: one small window
NUM_SCENARIOS = 500
rng = np.random.default_rng(42)
probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar = build_real_data_instance(
    solar, wind, load_total, thermal_by_bus, thermal_buses_numeric,
    solar_buses, wind_buses, start_idx=0, num_scenarios=min(NUM_SCENARIOS, T), rng=rng
)
print("probs.shape:", probs.shape, "sum:", probs.sum())
print("g_i_bar.shape:", g_i_bar.shape)
print("d_j_bar.shape:", d_j_bar.shape)
print("Unreliable (solar+wind) sample mean:", g_i_bar[:, :6].mean(axis=0))
print("Reliable (thermal) constant:", g_i_bar[0, 6:])
print("Load sample:", d_j_bar[:5].ravel())

# %%
def run_zavala_one_instance(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar, logfile=None,
                             lambda_cvar=0.1, beta=0.95):
    """Run stochastic, CVaR, and deterministic Zavala for one instance. Returns dict of metrics.
    Same logic as run_zavala.py, no changes to external files.
    """
    # ----- Stochastic Zavala -----
    z_g_i, z_d_j, Z_G, Z_D, z_pi, z_Pi = zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    prob_f = probability_feasible(probs, z_g_i, z_d_j, g_i_bar, d_j_bar)
    z_dist = price_distortion(probs, z_pi, z_Pi)
    z_reg = expected_cumulative_regret(probs, z_g_i, z_d_j, z_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    ss_stoch = compute_social_surplus(probs, mc_g_i, mv_d_j, g_da=z_g_i, d_da=z_d_j, G_rt=Z_G, D_rt=Z_D)

    # # ----- CVaR Zavala -----
    cvar_g_i, cvar_d_j, C_G, C_D, cvar_pi, cvar_Pi, _ = zavala_cvar(
        probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar, beta=beta, lambda_cvar=lambda_cvar
    )
    cvar_dist = price_distortion(probs, cvar_pi, cvar_Pi)
    cvar_reg = expected_cumulative_regret(probs, cvar_g_i, cvar_d_j, cvar_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    ss_cvar = compute_social_surplus(probs, mc_g_i, mv_d_j, g_da=cvar_g_i, d_da=cvar_d_j, G_rt=C_G, D_rt=C_D)

    # ----- Deterministic (expected capacities) -----
    gbar_det, dbar_det = expected_caps_from_scenarios(probs, g_i_bar, d_j_bar)
    g_det, d_det, pi_det = zavala_deterministic_da(mc_g_i, mv_d_j, gbar_det, dbar_det)
    G_det_list, D_det_list, Pi_det_list = [], [], []
    for p in range(len(probs)):
        Gp, Dp, Pi_p = zavala_rt_energy_only(mc_g_i, mv_d_j, g_det, d_det, g_i_bar[p], d_j_bar[p])
        G_det_list.append(Gp)
        D_det_list.append(Dp)
        Pi_det_list.append(Pi_p)
    G_det_rt, D_det_rt = _stack_rt(G_det_list, D_det_list)
    Pi_det = np.array(Pi_det_list)
    det_dist = price_distortion(probs, pi_det, Pi_det)
    det_reg = expected_cumulative_regret(probs, g_det, d_det, pi_det, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    ss_det = compute_social_surplus(probs, mc_g_i, mv_d_j, g_da=g_det, d_da=d_det, G_rt=G_det_rt, D_rt=D_det_rt)

    # ----- Tail metrics (5% worst by high neg-surplus) -----
    tail = 0.05
    stoch_tail_idx = tail_worst_indices_by_value(ss_stoch["ss_per_scenario"], probs, tail=tail, worst="high")
    cvar_tail_idx = tail_worst_indices_by_value(ss_cvar["ss_per_scenario"], probs, tail=tail, worst="high")
    det_tail_idx = tail_worst_indices_by_value(ss_det["ss_per_scenario"], probs, tail=tail, worst="high")

    stoch_tail_welfare = -np.mean(ss_stoch["ss_per_scenario"][stoch_tail_idx])
    cvar_tail_welfare = -np.mean(ss_cvar["ss_per_scenario"][cvar_tail_idx])
    det_tail_welfare = -np.mean(ss_det["ss_per_scenario"][det_tail_idx])

    stoch_tail_dist = np.mean(np.abs(z_pi - np.array(z_Pi)[stoch_tail_idx]))
    cvar_tail_dist = np.mean(np.abs(cvar_pi - np.array(cvar_Pi)[cvar_tail_idx]))
    det_tail_dist = np.mean(np.abs(pi_det - Pi_det[det_tail_idx]))

    # Log diagnostics/output results analysis
    _print_case_diag("Stochastic", probs, z_g_i, z_d_j, Z_G, Z_D, z_pi, z_Pi, logfile=logfile)
    _print_case_diag("CVaR", probs, cvar_g_i, cvar_d_j, C_G, C_D, cvar_pi, cvar_Pi, logfile=logfile)
    _print_case_diag("Deterministic", probs, g_det, d_det, G_det_rt, D_det_rt, pi_det, Pi_det, logfile=logfile)
    _print_stoch_vs_cvar_diff(z_g_i, cvar_g_i, logfile=logfile)

    return {
        "prob_feasible": prob_f,
        "stoch_distortion": z_dist, "stoch_regret": z_reg, "stoch_ss": ss_stoch["E_social_surplus"],
        "stoch_tail_welfare": stoch_tail_welfare, "stoch_tail_distortion": stoch_tail_dist,
        "cvar_distortion": cvar_dist, "cvar_regret": cvar_reg, "cvar_ss": ss_cvar["E_social_surplus"],
        "cvar_tail_welfare": cvar_tail_welfare, "cvar_tail_distortion": cvar_tail_dist,
        "det_distortion": det_dist, "det_regret": det_reg, "det_ss": ss_det["E_social_surplus"],
        "det_tail_welfare": det_tail_welfare, "det_tail_distortion": det_tail_dist,
    }

# %%
NUM_INSTANCES = args.num_instances
NUM_SCENARIOS = args.num_scenarios
rng = np.random.default_rng(2025)

max_start = T - NUM_SCENARIOS
if max_start <= 0:
    raise ValueError(f"Need at least {NUM_SCENARIOS} time steps; have {T}")

# Random start indices for each instance (non-overlapping or random)
start_indices = rng.integers(0, max_start + 1, size=NUM_INSTANCES)

results_list = []
for i in range(NUM_INSTANCES):
    start = int(start_indices[i])
    probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar = build_real_data_instance(
        solar, wind, load_total, thermal_by_bus, thermal_buses_numeric,
        solar_buses, wind_buses, start_idx=start, num_scenarios=NUM_SCENARIOS, rng=rng
    )
    print(f"the shapes of probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar are {probs.shape}, {mc_g_i.shape}, {mv_d_j.shape}, {g_i_bar.shape}, {d_j_bar.shape}")
    res = run_zavala_one_instance(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar, logfile=debug_log,
                                   lambda_cvar=args.lambda_cvar, beta=args.beta)
    results_list.append(res)
    print(f"Instance {i+1}/{NUM_INSTANCES} (start={start}) done.")

# %%
# Aggregate and average
keys = list(results_list[0].keys())
means = {k: np.mean([r[k] for r in results_list]) for k in keys}
stds = {k: np.std([r[k] for r in results_list]) for k in keys}

print("============== Real-data results (averaged over {} instances) ================".format(NUM_INSTANCES))
print("Distortion (DA vs E[RT price]):")
print("  Stochastic:", means["stoch_distortion"], "±", stds["stoch_distortion"])
print("  CVaR:", means["cvar_distortion"], "±", stds["cvar_distortion"])
print("  Deterministic:", means["det_distortion"], "±", stds["det_distortion"])
print("E[Social Surplus]:")
print("  Stochastic:", means["stoch_ss"], "±", stds["stoch_ss"])
print("  CVaR:", means["cvar_ss"], "±", stds["cvar_ss"])
print("  Deterministic:", means["det_ss"], "±", stds["det_ss"])
print("Tail (5%) welfare (mean positive SS in worst tail):")
print("  Stochastic:", means["stoch_tail_welfare"], "±", stds["stoch_tail_welfare"])
print("  CVaR:", means["cvar_tail_welfare"], "±", stds["cvar_tail_welfare"])
print("  Deterministic:", means["det_tail_welfare"], "±", stds["det_tail_welfare"])
print("Tail (5%) price distortion:")
print("  Stochastic:", means["stoch_tail_distortion"], "±", stds["stoch_tail_distortion"])
print("  CVaR:", means["cvar_tail_distortion"], "±", stds["cvar_tail_distortion"])
print("  Deterministic:", means["det_tail_distortion"], "±", stds["det_tail_distortion"])
print("Probability feasible:", means["prob_feasible"], "±", stds["prob_feasible"])

# %%
summary = pd.DataFrame({
    "Method": ["Stochastic", "CVaR", "Deterministic"] * 3,
    "Metric": ["Distortion", "Distortion", "Distortion", "E[SS]", "E[SS]", "E[SS]", "Tail welfare", "Tail welfare", "Tail welfare"],
    "Mean": [
        means["stoch_distortion"], means["cvar_distortion"], means["det_distortion"],
        means["stoch_ss"], means["cvar_ss"], means["det_ss"],
        means["stoch_tail_welfare"], means["cvar_tail_welfare"], means["det_tail_welfare"],
    ],
    "Std": [
        stds["stoch_distortion"], stds["cvar_distortion"], stds["det_distortion"],
        stds["stoch_ss"], stds["cvar_ss"], stds["det_ss"],
        stds["stoch_tail_welfare"], stds["cvar_tail_welfare"], stds["det_tail_welfare"],
    ],
})
summary["Std/Mean"] = summary["Std"] / summary["Mean"]

# %%
from pathlib import Path

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Summary dataframe ===")
print(summary.to_string(index=False))

with open(LOG_DIR / f"{args.experiment_name}_summary.log", "w") as f:
    f.write(summary.to_string(index=False))
    f.write("\n")


