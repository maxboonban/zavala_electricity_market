"""Driver for the IM3 GO-WEST economic experiments E1-E5.

  python run_im3_experiments.py --experiments E1 E2 E3 E4 E5 \
      --scenarios 300 --eps 0.3 --beta 0.95 --solver GUROBI --tag im3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import im3_data as D
import im3_experiments as X


class Reporter:
    def __init__(self, logfile=None):
        self.logfile = logfile
        self.lines = []
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            open(logfile, "w").close()

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(s)
        if self.logfile:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(s + "\n")


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+",
                    default=["E1", "E2", "E3", "E4", "E5", "E6"])
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--solver", default="GUROBI")
    ap.add_argument("--hour-of-day", type=int, default=18,
                    help="delivery hour the scenarios condition on")
    ap.add_argument("--season", default="summer",
                    choices=["winter", "spring", "summer", "autumn", "all"])
    ap.add_argument("--sampling", default="conditional",
                    choices=["conditional", "consecutive"])
    ap.add_argument("--num-solar", type=int, default=6)
    ap.add_argument("--num-wind", type=int, default=6)
    ap.add_argument("--num-thermal", type=int, default=6)
    ap.add_argument("--renewable-scale", type=float, default=1.0)
    ap.add_argument("--shares", nargs="+", type=float,
                    default=[0.15, 0.30, 0.45, 0.60, 0.75],
                    help="E3: target renewable shares at fixed total capacity")
    ap.add_argument("--reps", type=int, default=20, help="replications for E4/E5")
    ap.add_argument("--train", type=int, default=200, help="E4 training scenarios")
    ap.add_argument("--test", type=int, default=200, help="E4 held-out scenarios")
    ap.add_argument("--eps-list", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--deltas", nargs="+", type=float,
                    default=[0.02, 0.05, 0.10, 0.20, 0.40],
                    help="E6: deviation bid price as a fraction of mc / VOLL")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--tag", default="im3")
    args = ap.parse_args()

    season = None if args.season == "all" else args.season
    base_kw = dict(num_solar=args.num_solar, num_wind=args.num_wind,
                   num_thermal=args.num_thermal, hour_of_day=args.hour_of_day,
                   season=season, sampling=args.sampling,
                   renewable_scale=args.renewable_scale, seed=args.seed)

    logfile = Path("logs") / f"im3_{args.tag}.log"
    rep = Reporter(str(logfile))
    t0 = time.time()
    rep("=" * 78)
    rep("IM3 GO-WEST economic experiments")
    rep("=" * 78)
    rep(f"  solver={args.solver}  S={args.scenarios}  eps={args.eps}  beta={args.beta}")
    rep(f"  delivery hour={args.hour_of_day}  season={args.season}  "
        f"sampling={args.sampling}  seed={args.seed}")

    inst = D.build_instance(n_scenarios=args.scenarios, **base_kw)
    probs, mc, mv, gbar, dbar, meta = inst
    rep(f"\n  fleet: {args.num_solar} solar + {args.num_wind} wind + "
        f"{args.num_thermal} thermal buses, 1 load (VOLL ${meta['voll']:.0f}/MWh)")
    rep(f"  thermal capacity {meta['thermal_capacity_mw']:,.0f} MW,  "
        f"mean renewable {meta['renewable_capacity_mw']:,.0f} MW  "
        f"(share {meta['renewable_share']:.1%})")
    rep(f"  mean load {meta['mean_load_mw']:,.0f} MW  "
        f"(real WECC shape, level scaled by {meta['load_scale']:.4f} to a "
        f"{meta['reserve_margin']:.0%} reserve margin)")
    rep(f"  marginal costs $/MWh: " + ", ".join(
        f"{t}={c:.2f}" for t, c in zip(meta["tech"], mc)))
    rep(f"  NOTE {meta['n_heat_rate_floored']} of 280 thermal heat rates were "
        f"floored to physical minima (the dataset's gas heat rates are not physical)")

    results = {"config": vars(args), "meta": _json_safe(
        {k: v for k, v in meta.items() if k != "hours"})}

    if "E1" in args.experiments:
        results["E1"] = X.exp_E1(inst, args.eps, args.beta, args.solver, rep)
    if "E2" in args.experiments:
        results["E2"] = X.exp_E2(inst, args.eps, args.beta, args.solver, rep)
    if "E3" in args.experiments:
        kw = dict(base_kw); kw["n_scenarios"] = args.scenarios
        results["E3"] = X.exp_E3(kw, args.shares, args.eps, args.beta, args.solver, rep)
    if "E4" in args.experiments:
        results["E4"] = X.exp_E4(base_kw, args.train, args.test, args.eps,
                                 args.beta, args.solver, args.reps, rep)
    if "E5" in args.experiments:
        results["E5"] = X.exp_E5(base_kw, args.scenarios, args.eps_list,
                                 args.beta, args.solver, args.reps, rep)

    if "E6" in args.experiments:
        results["E6"] = X.exp_E6(inst, args.eps, args.beta, args.solver,
                                 args.deltas, rep)

    rep(f"\n{'=' * 78}")
    rep(f"done in {time.time() - t0:.1f}s")
    rep("=" * 78)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / f"im3_{args.tag}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2)
    rep(f"\nResults written to {jpath}")
    rep(f"Log written to {logfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
