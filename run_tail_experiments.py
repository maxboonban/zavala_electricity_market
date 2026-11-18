#!/usr/bin/env python
"""Driver for the tail-welfare experiments W0-W6.

    python run_tail_experiments.py --experiments W0 W1 W2 W3 W4 W5 W6 \
        --scenarios 500 --seeds 30 --eps 0.3 --beta 0.95 \
        --solver GUROBI --tag tail

Writes one JSON per run to --out and a human-readable log to logs/.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import cvxpy as cp

import tail_experiments as X


class Reporter:
    def __init__(self, logfile=None):
        self.lines, self.logfile = [], logfile
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            open(logfile, "w").close()

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))
        if self.logfile:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _provenance():
    """Record enough to reproduce the run exactly."""
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:                                       # noqa: BLE001
            return None
    return dict(
        commit=_git("rev-parse", "HEAD"),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(_git("status", "--porcelain")),
        python=platform.python_version(),
        cvxpy=cp.__version__,
        numpy=np.__version__,
        node=platform.node(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+",
                    default=["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7"],
                    choices=["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7"])
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=30,
                    help="number of paired seeds")
    ap.add_argument("--seed0", type=int, default=2025, help="first seed")
    ap.add_argument("--sweep-seeds", type=int, default=15,
                    help="seeds per level in W2/W3/W4/W6 (sweeps are wider)")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--solver", default="GUROBI")

    # scenario-family parameters (the base configuration all sweeps perturb)
    ap.add_argument("--ramp-premium", type=float, default=1.0)
    ap.add_argument("--backdown-discount", type=float, default=0.40)
    ap.add_argument("--adequacy", type=float, default=1.12)
    ap.add_argument("--voll", type=float, default=1000.0)
    ap.add_argument("--renewable-penetration", type=float, default=0.45)
    ap.add_argument("--p-stress", type=float, default=0.12)
    ap.add_argument("--ramp-limit-frac", type=float, default=None,
                    help="start-up constraint; EXTENSION beyond the paper's "
                         "Eq. (63)-(65). Omit to stay within verified theory.")

    # sweep levels
    ap.add_argument("--asymmetry-levels", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--adequacy-levels", nargs="+", type=float,
                    default=[0.95, 1.00, 1.05, 1.12, 1.20, 1.30])
    ap.add_argument("--voll-levels", nargs="+", type=float,
                    default=[200.0, 500.0, 1000.0, 2000.0, 5000.0])
    ap.add_argument("--eps-levels", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    # W7: four independent physical drivers of a heavy supply tail.
    ap.add_argument("--stress-prob-levels", nargs="+", type=float,
                    default=[0.02, 0.06, 0.12, 0.22, 0.35])
    ap.add_argument("--stress-depth-levels", nargs="+", type=float,
                    default=[-0.5, -1.2, -1.9, -2.8, -3.6])
    ap.add_argument("--correlation-levels", nargs="+", type=float,
                    default=[0.1, 0.4, 0.7, 0.9, 0.98])
    ap.add_argument("--outage-levels", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5, 0.7])

    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--tag", default="tail")
    ap.add_argument("--check-reference", action="store_true",
                    help="assert the vectorised model matches zavala_cvar "
                         "(needs jax + Gurobi; cluster only)")
    args = ap.parse_args()

    solver = getattr(cp, args.solver)
    seeds = [args.seed0 + i for i in range(args.seeds)]
    sweep_seeds = seeds[:args.sweep_seeds]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Reporter(str(Path("logs") / f"tail_{args.tag}.log"))

    base_kw = dict(
        ramp_premium=args.ramp_premium,
        backdown_discount=args.backdown_discount,
        adequacy=args.adequacy,
        voll=args.voll,
        renewable_penetration=args.renewable_penetration,
        p_stress=args.p_stress,
        ramp_limit_frac=args.ramp_limit_frac,
    )
    use_ramp = args.ramp_limit_frac is not None

    rep("=" * 78)
    rep("Tail-welfare experiments for risk-aware market clearing")
    rep("=" * 78)
    prov = _provenance()
    rep(f"commit {prov['commit']} ({prov['branch']}"
        f"{', DIRTY' if prov['dirty'] else ''})  on {prov['node']}")
    rep(f"solver={args.solver}  S={args.scenarios}  seeds={len(seeds)}  "
        f"eps={args.eps}  beta={args.beta}")
    rep(f"base: {base_kw}")
    if use_ramp:
        rep("")
        rep("  *** ramp_limit_frac is SET. The start-up constraint")
        rep("  *** G_i(w) - g_i <= ramp_cap_i is an EXTENSION beyond the")
        rep("  *** paper's Eq. (63)-(65). Report these results separately.")

    if args.check_reference:
        rep("\nPreflight: vectorised model vs zavala_cvar")
        from tail_scenarios import generate_tail_instance
        from tail_model import assert_matches_reference
        inst = generate_tail_instance(args.seed0, S=min(args.scenarios, 60),
                                      **base_kw)
        # Two kinds of failure, and they must not be conflated. If the REFERENCE
        # cannot be loaded at all -- zavala_funcs imports jax, which is fragile
        # across machines -- that says nothing about the model being reported,
        # and must not abort a run that is otherwise fine. If the reference
        # loads and DISAGREES, every number downstream is suspect and the run
        # must stop.
        try:
            d = assert_matches_reference(inst, eps=args.eps, beta=args.beta,
                                         solver=solver, rep=rep)
            rep(f"  [PASS] objective d_rho_RA = {d['d_rho_RA']:.3e}")
            rep(f"         diagnostics: " + "  ".join(
                f"{k}={v:.2e}" for k, v in d.items() if k != "d_rho_RA"))
        except (ImportError, OSError, RuntimeError) as exc:
            rep(f"  [SKIP] reference unavailable: {type(exc).__name__}: {exc}")
            rep("  The vectorised model could NOT be cross-checked against")
            rep("  zavala_cvar on this machine. Results below stand on their")
            rep("  own; say so if they are reported without this check.")
        except Exception as exc:                                # noqa: BLE001
            rep(f"  [FAIL] model disagrees with the reference: "
                f"{type(exc).__name__}: {exc}")
            raise

    t0 = time.time()
    results = {}

    if "W0" in args.experiments:
        results["W0"] = X.W0_structural_ceiling(
            seeds, args.scenarios, args.eps, args.beta, solver, base_kw, rep)

    if "W1" in args.experiments:
        results["W1"] = X.W1_headline(
            seeds, args.scenarios, args.eps, args.beta, solver, base_kw, rep)

    if "W2" in args.experiments:
        results["W2"] = X.W2_asymmetry(
            sweep_seeds, args.scenarios, args.eps, args.beta, solver,
            base_kw, args.asymmetry_levels, rep)

    if "W3" in args.experiments:
        results["W3"] = X.W3_adequacy(
            sweep_seeds, args.scenarios, args.eps, args.beta, solver,
            base_kw, args.adequacy_levels, rep)

    if "W4" in args.experiments:
        results["W4"] = X.W4_voll(
            sweep_seeds, args.scenarios, args.eps, args.beta, solver,
            base_kw, args.voll_levels, rep)

    if "W5" in args.experiments:
        results["W5"] = X.W5_decomposition(
            seeds, args.scenarios, args.eps, args.beta, solver, base_kw, rep)

    if "W6" in args.experiments:
        results["W6"] = X.W6_frontier(
            sweep_seeds, args.scenarios, args.beta, solver, base_kw,
            args.eps_levels, rep)

    if "W7" in args.experiments:
        results["W7"] = X.W7_tail_heaviness(
            sweep_seeds, args.scenarios, args.eps, args.beta, solver, base_kw,
            dict(p_stress=args.stress_prob_levels,
                 stress_shift=args.stress_depth_levels,
                 rho_wind=args.correlation_levels,
                 common_mode_prob=args.outage_levels),
            rep)

    rep(f"\nelapsed {time.time() - t0:.1f}s")

    payload = dict(args=vars(args), provenance=prov, base_kw=base_kw,
                   seeds=seeds, results=results)
    path = out_dir / f"tail_{args.tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    rep(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
