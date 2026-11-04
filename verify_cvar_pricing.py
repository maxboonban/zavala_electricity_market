#!/usr/bin/env python
"""Verify the CVaR price identification in `zavala_cvar` against the paper.

Checks, on small instances (cheap: a few scenarios, solved locally):

  Eq. (69)  sum_w gamma_w        = eps
  Eq. (67)  rho_tilde_w          = (1 - eps) * rho_w + gamma_w,  sum_w rho_tilde_w = 1
  Eq. (70)  0 <= gamma_w         <= eps * rho_w / (1 - beta)
  Eq. (107) p_w                  = nu_w / rho_tilde_w

and confirms the eps = 0 limit reduces to the risk-neutral prices of `zavala`
(paper Remark 10: when eps = 0, gamma_w = 0, rho_tilde_w = rho_w).

Run:  python verify_cvar_pricing.py [--solver CLARABEL]
"""
import argparse
import numpy as np
import cvxpy as cp

from zavala_funcs import zavala, zavala_cvar, generate_instance
from jax import random


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return ok


def run_case(probs, mc, mv, gbar, dbar, eps, beta, solver):
    print(f"\n--- eps={eps}, beta={beta}, S={len(probs)}, G={len(mc)}, D={len(mv)} ---")
    g, d, G, D, pi, Pi, gamma = zavala_cvar(
        probs, mc, mv, gbar, dbar, beta=beta, lambda_cvar=eps,
        check_invariants=False, solver=solver, verbose=False,
    )
    probs = np.asarray(probs, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float)
    rho_tilde = (1.0 - eps) * probs + gamma

    tol = 1e-6
    results = []
    results.append(check("Eq.(69)  sum gamma = eps",
                         abs(gamma.sum() - eps) < max(tol, tol * max(eps, 1.0)),
                         f"sum={gamma.sum():.10g} eps={eps:.10g} diff={gamma.sum()-eps:+.3e}"))
    results.append(check("Eq.(67)  sum rho_tilde = 1",
                         abs(rho_tilde.sum() - 1.0) < tol,
                         f"sum={rho_tilde.sum():.12g} diff={rho_tilde.sum()-1.0:+.3e}"))
    ub = eps * probs / (1.0 - beta)
    results.append(check("Eq.(70)  0 <= gamma <= eps*rho/(1-beta)",
                         bool((gamma >= -tol).all() and (gamma <= ub + tol).all()),
                         f"min={gamma.min():+.3e} max={gamma.max():.3e} ub_max={ub.max():.3e}"))
    results.append(check("rho_tilde > 0 (RT prices well defined)",
                         bool((rho_tilde > 0).all()),
                         f"min={rho_tilde.min():.3e}"))

    n_tail = int((gamma > tol).sum())
    print(f"  info: {n_tail}/{len(probs)} scenarios carry tail mass (gamma_w > 0); "
          f"pi={float(pi):.6f}  E[Pi]={float(probs @ np.asarray(Pi)):.6f}")

    # Size of the correction that was wrong before: old denominator was rho + gamma
    old_Pi = np.asarray([None])
    with np.errstate(divide="ignore", invalid="ignore"):
        old_den = probs + gamma
        old_Pi = np.where(old_den > 0, np.asarray(Pi) * rho_tilde / np.where(old_den > 0, old_den, 1.0), np.nan)
    rel = np.abs(np.asarray(Pi) - old_Pi) / np.maximum(np.abs(np.asarray(Pi)), 1e-12)
    print(f"  info: old (buggy) denominator would change E[Pi] by "
          f"{float(probs @ old_Pi) - float(probs @ np.asarray(Pi)):+.6f} "
          f"($/MWh), max per-scenario relative change {100*float(rel.max()):.3f}%")
    return all(results), gamma, Pi, pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="CLARABEL",
                    help="cvxpy solver name (use GUROBI on the cluster)")
    ap.add_argument("--scenarios", type=int, default=40)
    args = ap.parse_args()
    solver = getattr(cp, args.solver)

    print(f"Solver: {args.solver}")
    key = random.key(0)
    probs, mc, mv, gbar, dbar = generate_instance(
        key, num_scenarios=args.scenarios, num_g=10, num_d=10, r=2025,
        input_scenario="s_real10_mix",
    )

    all_ok = True
    for eps in (0.1, 0.3, 0.5, 1.0):
        ok, _, _, _ = run_case(probs, mc, mv, gbar, dbar, eps, 0.95, solver)
        all_ok &= ok

    # Remark 10: eps = 0 must reproduce the risk-neutral prices exactly.
    print("\n--- Remark 10: eps = 0 reduces to risk-neutral `zavala` ---")
    _, _, _, _, pi_rn, Pi_rn = zavala(probs, mc, mv, gbar, dbar, solver=solver, verbose=False)
    g0, d0, G0, D0, pi0, Pi0, gamma0 = zavala_cvar(
        probs, mc, mv, gbar, dbar, beta=0.95, lambda_cvar=0.0,
        check_invariants=False, solver=solver, verbose=False,
    )
    dpi = abs(float(pi0) - float(pi_rn))
    dPi = float(np.max(np.abs(np.asarray(Pi0) - np.asarray(Pi_rn))))
    all_ok &= check("pi(eps=0) == pi_risk_neutral", dpi < 1e-4, f"|diff|={dpi:.3e}")
    all_ok &= check("Pi(eps=0) == Pi_risk_neutral", dPi < 1e-3, f"max|diff|={dPi:.3e}")
    all_ok &= check("gamma(eps=0) == 0", float(np.abs(np.asarray(gamma0)).max()) < 1e-8,
                    f"max|gamma|={float(np.abs(np.asarray(gamma0)).max()):.3e}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
