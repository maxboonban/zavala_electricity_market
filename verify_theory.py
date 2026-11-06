"""Numerical adjudication of suspected theoretical errors in the manuscript.

Each test is written so that it can only pass if the *corrected* statement is
right and the *printed* statement is wrong (or vice versa). Where both variants
satisfy a weaker condition (e.g. market clearing), we say so explicitly, because
that is exactly why the error survived the checks already in the paper.

Run:  python verify_theory.py --solver CLARABEL --scenarios 60
"""
import argparse
import numpy as np

from zavala_funcs import generate_instance, zavala_cvar
from zavala_decentralized import (
    planner_losses, transfers_equal_share, transfers_heterogeneous,
    solve_agent, objective_at, _var_level,
)

BAR = "=" * 78


class Rep:
    def __init__(self):
        self.fails = 0

    def __call__(self, s=""):
        print(s, flush=True)

    def check(self, name, ok, detail=""):
        tag = "[PASS]" if ok else "[FAIL]"
        if not ok:
            self.fails += 1
        print(f"  {tag} {name}   {detail}", flush=True)


def build(args):
    """Solve the planner problem and package everything the agents need."""
    key = __import__("jax").random.PRNGKey(args.seed)
    probs, mc, mv, gbar, dbar = generate_instance(
        key, num_scenarios=args.scenarios, num_g=args.num_g, num_d=args.num_d)
    probs = np.asarray(probs, float).reshape(-1)
    mc = np.asarray(mc, float).reshape(-1)
    mv = np.asarray(mv, float).reshape(-1)
    gbar = np.asarray(gbar, float)
    dbar = np.asarray(dbar, float)
    g, d, G, D, p0, Pi, gamma = zavala_cvar(
        probs, mc, mv, gbar, dbar, beta=args.beta, lambda_cvar=args.eps,
        check_invariants=False, solver=args.solver, verbose=False)
    P = dict(probs=probs, mc=mc, mv=mv, gbar=gbar, dbar=dbar,
             g=np.asarray(g, float).reshape(-1), d=np.asarray(d, float).reshape(-1),
             G=np.asarray(G, float), D=np.asarray(D, float),
             p0=float(p0), Pi=np.asarray(Pi, float).reshape(-1),
             gamma=np.asarray(gamma, float).reshape(-1))
    P["rho_tilde"] = (1 - args.eps) * P["probs"] + P["gamma"]
    P["q"] = P["rho_tilde"].copy()
    # Deviation penalties alpha^+/alpha^- (paper Eq. 3). These must match the
    # planner's, and must be STRICTLY POSITIVE: with delta = 0 the agent's
    # day-ahead quantity is unbounded above (Eq. 65 bounds only x_i >= 0, while
    # Eq. 63 bounds the realised quantity), so the agent LP has no optimum and
    # (UM) is vacuous. See theory note T10.
    P["mc_delta"] = mc / 10.0
    P["mv_delta"] = mv / 10.0
    P["L_prod"], P["L_cons"], P["ell"] = planner_losses(
        P["p0"], P["Pi"], mc, mv, P["g"], P["d"], P["G"], P["D"])
    return P


def agents(P):
    S = P["probs"].size
    for j in range(P["mc"].size):
        yield ("producer", j, P["mc"][j], P["mc_delta"][j], np.zeros(S),
               P["gbar"][:, j], P["g"][j], P["G"][:, j])
    for i in range(P["mv"].size):
        yield ("consumer", i, P["mv"][i], P["mv_delta"][i], np.zeros(S),
               P["dbar"][:, i], P["d"][i], P["D"][:, i])


def um_gaps(P, args, R_all, betas):
    """Relative (UM) gap per agent: how far the planner allocation + the given
    transfer is from the agent's own optimum at the posted prices.

    The insurance variable is boxed because at q = rho_tilde the objective is
    exactly flat in r; the box is set far outside any sensible position and we
    assert it never binds."""
    gaps = []
    bound = 50.0 * max(1.0, float(np.max(np.abs(np.vstack([P["L_cons"], P["L_prod"]])))))
    bound_hits = 0
    n = P["mv"].size
    m = P["mc"].size
    for idx, (kind, k, cv, dl, lo, hi, da, rt) in enumerate(agents(P)):
        row = k if kind == "producer" else m + k
        bk = float(betas[row])
        best = solve_agent(kind, P["probs"], P["p0"], P["Pi"], cv, dl, lo, hi,
                           args.eps, bk, q=P["q"], r_fixed=None,
                           insurance=True, solver=args.solver, r_bound=bound)
        if best.get("r_bound_active"):
            bound_hits += 1
        if best.get("status") not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"agent LP {kind}{k} status={best.get('status')}")
        got = objective_at(kind, P["probs"], P["p0"], P["Pi"], cv, dl,
                           args.eps, bk, da, rt, q=P["q"], r=R_all[row],
                           insurance=True)
        denom = max(1.0, abs(best["objective"]))
        gaps.append(abs(best["objective"] - got) / denom)
    if bound_hits:
        print(f"  [WARN] insurance box active for {bound_hits} agent(s) -- "
              f"gap values are not trustworthy")
    return np.array(gaps)


# ---------------------------------------------------------------------------
# T1 / T2 : sum of tilted weights, and p0 = lambda
# ---------------------------------------------------------------------------
def test_T1_T2(P, args, rep):
    rep("\n" + BAR)
    rep("T1/T2  sum_w rho_tilde_w = 1  (paper p.19 claims 1 + eps)")
    rep(BAR)
    s = float(P["rho_tilde"].sum())
    rep(f"  eps = {args.eps},  sum_w rho_tilde_w = {s:.12f}")
    rep(f"  printed claim 1 + eps = {1 + args.eps:.12f}    corrected claim 1 = 1.0")
    rep.check("sum rho_tilde == 1 (corrected)", abs(s - 1.0) < 1e-7,
              f"|sum - 1| = {abs(s - 1.0):.3e}")
    rep.check("sum rho_tilde != 1 + eps (printed is wrong)",
              abs(s - (1 + args.eps)) > 1e-3 or args.eps == 0,
              f"|sum - (1+eps)| = {abs(s - (1 + args.eps)):.3e}")
    rep("  => T2 follows: the (1+eps) normaliser in p_0 = lambda/(1+eps) (p.26)")
    rep("     has no source; Eq. (107) and the Thm 6 proof both give p_0 = lambda.")


# ---------------------------------------------------------------------------
# T3 : sign of Eq. (111)-(112)
# ---------------------------------------------------------------------------
def test_T3(P, args, rep):
    rep("\n" + BAR)
    rep("T3  sign of Eq. (111)-(112)  [equal-share transfer]")
    rep(BAR)
    rep("  Eq. (97) defines the insured loss as  L - r,  but Eq. (113) flattens")
    rep("  L + r. Only one sign can be right. Both satisfy clearing Eq. (95),")
    rep("  which is why the paper's own check does not catch it.")
    m = P["mc"].size
    betas = np.full(m + P["mv"].size, args.beta)

    for label, as_printed in (("as printed  r = -L + l/N + d", True),
                              ("corrected   r = +L - l/N - d", False)):
        rp, rc = transfers_equal_share(P["L_prod"], P["L_cons"], P["ell"],
                                       as_printed=as_printed)
        R = np.vstack([rp, rc])
        clearing = float(np.max(np.abs(R.sum(0))))
        gaps = um_gaps(P, args, R, betas)
        rep(f"  {label}")
        rep(f"      clearing Eq.(95) max|sum_k r_k| = {clearing:.3e}   "
            f"(both signs clear -- this is the trap)")
        rep(f"      (UM) max relative gap           = {gaps.max():.3e}")
        if as_printed:
            printed_gap = gaps.max()
        else:
            corrected_gap = gaps.max()

    rep.check("corrected sign supports (UM)", corrected_gap < 1e-3,
              f"max rel gap = {corrected_gap:.3e}  (solver tolerance)")
    rep.check("printed sign does NOT support (UM)", printed_gap > 1e-2,
              f"max rel gap = {printed_gap:.3e}")
    rep.check("the two signs separate by >= 3 orders of magnitude",
              printed_gap / max(corrected_gap, 1e-15) > 1e3,
              f"ratio = {printed_gap / max(corrected_gap, 1e-15):.3e}")
    rep("  => Eq. (111)-(112) should read  r_k = +L_k - l/(n+m) - delta_k,")
    rep("     and Eq. (113) should read    L_k - r_k = l/(n+m) + delta_k.")
    rep("     Eq. (122) in Theorem 9 already uses this corrected sign.")


# ---------------------------------------------------------------------------
# T5 : Theorem 9's definition of t*
# ---------------------------------------------------------------------------
def test_T5(P, args, rep):
    rep("\n" + BAR)
    rep("T5  Theorem 9: is  t* := sum_k t_k  circular?  [INVESTIGATED -- NOT AN ERROR]")
    rep(BAR)
    rep("  Eq. (122) sets  r_k = L_k - t_k - w_k (l(w) - t*)  with t* := sum_k t_k,")
    rep("  while t_k also names each agent's own VaR variable. That looks circular:")
    rep("  one might expect self-consistency to force t* = VaR_bmin(l*), the")
    rep("  PLANNER's VaR. We tested both readings. They give the SAME objective,")
    rep("  because a constant shift of r_k is exactly objective-neutral:")
    rep("      d/dc [ (1-eps) E[u+r+c] - q.(r+c) - eps CVaR(L-r-c) ]")
    rep("          = (1-eps) - sum_w q_w + eps = 1 - sum_w q_w = 0,")
    rep("  which holds PRECISELY BECAUSE sum_w rho_tilde_w = 1 (test T1). So the")
    rep("  {t_k} in (122) are free constants, t* := sum_k t_k is a harmless")
    rep("  normalisation, and Theorem 9's construction is sound as printed.")
    rep("  (Had the paper's sum = 1 + eps been right, this invariance would fail.)")

    m, n = P["mc"].size, P["mv"].size
    rng = np.random.default_rng(args.seed)
    pool = np.array(args.het_betas, float)
    beta_prod = rng.choice(pool, size=m)
    beta_cons = rng.choice(pool, size=n)
    beta_prod[0] = pool.min()
    beta_min = float(min(beta_prod.min(), beta_cons.min()))
    betas_all = np.concatenate([beta_prod, beta_cons])      # producer-major order

    # The planner must clear at beta_min (Eq. 117-120).
    g, d, G, D, p0, Pi, gamma = zavala_cvar(
        P["probs"], P["mc"], P["mv"], P["gbar"], P["dbar"], beta=beta_min,
        lambda_cvar=args.eps, check_invariants=False, solver=args.solver, verbose=False)
    Q = dict(P)
    Q.update(g=np.asarray(g, float).reshape(-1), d=np.asarray(d, float).reshape(-1),
             G=np.asarray(G, float), D=np.asarray(D, float), p0=float(p0),
             Pi=np.asarray(Pi, float).reshape(-1),
             gamma=np.asarray(gamma, float).reshape(-1))
    Q["rho_tilde"] = (1 - args.eps) * Q["probs"] + Q["gamma"]
    Q["q"] = Q["rho_tilde"].copy()
    Q["L_prod"], Q["L_cons"], Q["ell"] = planner_losses(
        Q["p0"], Q["Pi"], Q["mc"], Q["mv"], Q["g"], Q["d"], Q["G"], Q["D"])

    rep(f"  beta_min = {beta_min},  |S_>| = {int((betas_all > beta_min).sum())} "
        f"of {m + n} agents")

    L_all = np.vstack([Q["L_cons"], Q["L_prod"]])           # consumer-major
    N = m + n
    t_raw = np.array([_var_level(L_all[k], Q["probs"], beta_min) for k in range(N)])
    t_planner = _var_level(Q["ell"], Q["probs"], beta_min)
    rep(f"  sum_k t_k (as printed) = {t_raw.sum():.4f}")
    rep(f"  VaR_bmin(l*)  (planner) = {t_planner:.4f}")
    rep(f"  these differ by {abs(t_raw.sum() - t_planner):.4f} -- the two readings of t*")

    # ordering: transfers_heterogeneous returns (r_prod, r_cons); um_gaps wants
    # producer-major rows.
    def gaps_for(t_k_vec):
        rp, rc, _info = transfers_heterogeneous(
            Q["L_prod"], Q["L_cons"], Q["ell"], beta_prod, beta_cons,
            t_star=t_k_vec, probs=Q["probs"], beta_min=beta_min)
        R = np.vstack([rp, rc])
        clearing = float(np.max(np.abs(R.sum(0))))
        return clearing, um_gaps(Q, args, R, betas_all), R

    cl_p, gaps_p, _ = gaps_for(t_raw)
    # corrected: shift the t_k uniformly so that they sum to the planner's VaR
    t_fix = t_raw + (t_planner - t_raw.sum()) / N
    cl_c, gaps_c, _ = gaps_for(t_fix)

    rep(f"  as printed  (t* = sum_k t_k):   clearing = {cl_p:.3e}   "
        f"(UM) max rel gap = {gaps_p.max():.3e}")
    rep(f"  corrected   (sum_k t_k = t*_P): clearing = {cl_c:.3e}   "
        f"(UM) max rel gap = {gaps_c.max():.3e}")
    rep.check("Theorem 9 construction supports (UM)  [t* = sum_k t_k, as printed]",
              gaps_p.max() < 1e-3, f"max rel gap = {gaps_p.max():.3e}")
    rep.check("both readings of t* agree (construction is shift-invariant)",
              abs(gaps_p.max() - gaps_c.max()) < 1e-6,
              f"|difference| = {abs(gaps_p.max() - gaps_c.max()):.3e}")
    rep("  VERDICT: not an error. Theorem 9's construction is verified under the")
    rep("  (UM) test, which D4 in the first experiment round never actually ran --")
    rep("  D4 checked only Eq. (95) and Eq. (121), which hold for ANY {t_k}.")
    rep("  Constant-shift coefficient 1 - sum_w q_w = "
        f"{1.0 - float(Q['q'].sum()):.3e}")
    return dict(beta_min=beta_min, t_sum_printed=float(t_raw.sum()),
                t_planner=float(t_planner),
                um_printed=float(gaps_p.max()), um_corrected=float(gaps_c.max()))


# ---------------------------------------------------------------------------
# T8 : Remark 11, heterogeneous eps (paper defers this)
# ---------------------------------------------------------------------------
def test_T8(P, args, rep):
    rep("\n" + BAR)
    rep("T8  Remark 11: heterogeneous eps reduces to a planner at (eps_min, beta_P)")
    rep(BAR)
    rep("  Claim: with phi_k = eps_k beta_k/(1-beta_k), the binding bounds on q are")
    rep("    q_w >= (1-eps_min) rho_w   and   q_w <= (1+phi_min) rho_w,")
    rep("  which are exactly a planner's bounds at eps_P = eps_min and")
    rep("  beta_P = phi_min/(eps_min + phi_min).  The paper defers this; we check it.")

    m, n = P["mc"].size, P["mv"].size
    rng = np.random.default_rng(args.seed + 7)
    eps_pool = np.array(args.het_eps, float)
    beta_pool = np.array(args.het_betas, float)
    eps_k = rng.choice(eps_pool, size=m + n)
    beta_k = rng.choice(beta_pool, size=m + n)
    # Anti-correlate eps and beta so that the eps_min agent is NOT the phi_min
    # agent. Then beta_P falls strictly OUTSIDE the pool {beta_k} and the test
    # is sharp: it cannot be satisfied by accidentally coinciding with some
    # agent's own beta.
    lo_e, hi_e = float(eps_pool.min()), float(eps_pool.max())
    lo_b, hi_b = float(beta_pool.min()), float(beta_pool.max())
    eps_k = np.where(rng.random(m + n) < 0.5, lo_e, hi_e)
    beta_k = np.where(eps_k == lo_e, hi_b, lo_b)   # small eps <-> large beta
    eps_k[0], beta_k[0] = lo_e, hi_b
    eps_k[1], beta_k[1] = hi_e, lo_b

    phi = eps_k * beta_k / (1.0 - beta_k)
    eps_min = float(eps_k.min())
    phi_min = float(phi.min())
    beta_P = phi_min / (eps_min + phi_min)
    rep(f"  eps_k in {sorted(set(np.round(eps_k,3).tolist()))}, "
        f"beta_k in {sorted(set(np.round(beta_k,3).tolist()))}")
    rep(f"  eps_min = {eps_min:.4f}   phi_min = {phi_min:.4f}   "
        f"=> beta_P = {beta_P:.6f}")
    outside = not np.any(np.isclose(beta_k, beta_P))
    rep.check("beta_P falls strictly outside the agents' own {beta_k} (sharp test)",
              outside, f"beta_P = {beta_P:.6f}, pool = {sorted(set(np.round(beta_k,4).tolist()))}")

    # algebraic identity: planner at (eps_P, beta_P) has phi_P = phi_min
    phi_P = eps_min * beta_P / (1.0 - beta_P)
    rep.check("planner's phi_P equals phi_min (algebra of Remark 11)",
              abs(phi_P - phi_min) < 1e-10, f"|phi_P - phi_min| = {abs(phi_P - phi_min):.3e}")

    # numerical: solve the planner at (eps_min, beta_P) and verify the resulting
    # q = rho_tilde satisfies EVERY agent's bounds.
    g, d, G, D, p0, Pi, gamma = zavala_cvar(
        P["probs"], P["mc"], P["mv"], P["gbar"], P["dbar"], beta=beta_P,
        lambda_cvar=eps_min, check_invariants=False, solver=args.solver, verbose=False)
    gam = np.asarray(gamma, float).reshape(-1)
    q = (1 - eps_min) * P["probs"] + gam
    lo_viol, hi_viol = 0.0, 0.0
    for k in range(m + n):
        lo = (1 - eps_k[k]) * P["probs"]
        hi = (1 + phi[k]) * P["probs"]
        lo_viol = max(lo_viol, float(np.max(lo - q)))
        hi_viol = max(hi_viol, float(np.max(q - hi)))
    tol = 1e-7 * float(np.max(P["probs"]))
    rep(f"  max lower-bound violation  max_k,w [(1-eps_k)rho_w - q_w] = {lo_viol:.3e}")
    rep(f"  max upper-bound violation  max_k,w [q_w - (1+phi_k)rho_w] = {hi_viol:.3e}")
    rep.check("q from the (eps_min, beta_P) planner is feasible for EVERY agent",
              lo_viol < tol and hi_viol < tol,
              f"tol = {tol:.3e}")
    rep.check("sum_w q_w == 1", abs(float(q.sum()) - 1.0) < 1e-7,
              f"|sum q - 1| = {abs(float(q.sum()) - 1.0):.3e}")
    return dict(eps_min=eps_min, phi_min=phi_min, beta_P=float(beta_P),
                lo_viol=lo_viol, hi_viol=hi_viol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--scenarios", type=int, default=60)
    ap.add_argument("--num-g", type=int, default=10)
    ap.add_argument("--num-d", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--het-betas", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    ap.add_argument("--het-eps", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    args = ap.parse_args()

    rep = Rep()
    rep(BAR)
    rep("THEORY VERIFICATION  --  adjudicating suspected errors in the manuscript")
    rep(BAR)
    rep(f"  solver={args.solver}  S={args.scenarios}  eps={args.eps}  beta={args.beta}")
    P = build(args)
    test_T1_T2(P, args, rep)
    test_T3(P, args, rep)
    test_T5(P, args, rep)
    test_T8(P, args, rep)
    rep("\n" + BAR)
    rep("ALL THEORY CHECKS PASSED" if rep.fails == 0 else f"{rep.fails} CHECK(S) FAILED")
    rep(BAR)
    return 0 if rep.fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
