"""Two-stage clearing with asymmetric deviation bid prices, vectorised.

Relationship to `zavala_funcs.py`
---------------------------------
This is the same model as `zavala` / `zavala_cvar`, with two changes:

1. **Asymmetric `alpha^+` / `alpha^-`.** Paper Eq. (3) defines the upward and
   downward deviation bid prices separately; `zavala_funcs.py` hard-codes them
   equal at `mc/10`. Passing `dgp == dgn` recovers that model exactly.

2. **Vectorised.** `zavala_cvar` builds one scalar `cp.Variable()` per
   (scenario, unit) and a `cp.max(cp.vstack(...))` per deviation term, which
   makes CVXPY emit "too many subexpressions" and spend minutes compiling at
   S=500. The experiments here need ~1600 solves, so compilation time is the
   binding constraint. Matrix variables cut it by well over an order of
   magnitude. `assert_matches_reference` checks the two agree numerically.

The pricing readout is identical to `zavala_cvar`, including the `eps < 1`
support handling recorded as theory note T11.

Evaluation protocol
-------------------
`evaluate_da` solves the recourse problem for a *given* day-ahead position
against the full scenario set. Every policy is scored through it, which matters
because the policies otherwise differ in what they report:

  - the stochastic and risk-aware planners carry their own real-time variables,
    which are already the recourse solution;
  - the deterministic planner does not, because its day-ahead was computed
    against expected capacities.

`zavala_rt_energy_only` in `zavala_funcs.py` scored the deterministic policy
with the objective `(mc + delta) |G - g_da|`, which omits the energy cost and
the load's value entirely -- it minimises schedule *change* rather than cost.
Scoring all three policies through one evaluator removes that asymmetry.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

# A scenario counts as priced when the CVaR-tilted measure gives it at least
# this fraction of total mass; see theory note T11 and `zavala_funcs.py`.
PRICE_SUPPORT_REL_TOL = 1e-9

_SOLVE_KW = dict(
    Method=2,            # barrier
    Crossover=0,         # no crossover -> central duals
    FeasibilityTol=1e-9,
    OptimalityTol=1e-9,
    BarConvTol=1e-12,
)


def _solve(prob, solver, verbose=False):
    name = cp.GUROBI if solver is None else solver
    if name == cp.GUROBI:
        prob.solve(solver=cp.GUROBI, verbose=verbose, **_SOLVE_KW)
    else:
        prob.solve(solver=name, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solve failed: status={prob.status}")
    return prob


def _losses(Gr, Dr, g, d, mc, mv, dgp, dgn, ddp, ddn):
    """Per-scenario loss (negative social surplus), paper Eq. (10) form."""
    return (Gr @ mc
            + cp.pos(Gr - g[None, :]) @ dgp
            + cp.pos(g[None, :] - Gr) @ dgn
            - Dr @ mv
            + cp.pos(Dr - d[None, :]) @ ddp
            + cp.pos(d[None, :] - Dr) @ ddn)


def solve_planner(inst, eps=0.0, beta=0.95, solver=None, verbose=False,
                  da_nonneg=True, use_ramp=False, da_cap=True):
    """Risk-aware two-stage clearing, paper Eq. (58)-(65).

    eps = 0 gives the risk-neutral stochastic planner (`zavala`); eps > 0 adds
    the CVaR term (`zavala_cvar`). Returns allocations, prices and duals.

    da_nonneg enforces `x, y >= 0` as Eq. (12)/(18)/(65) require. The existing
    repo leaves day-ahead variables free -- a deviation the experiment log
    already flags -- so this defaults to the paper's convention.

    use_ramp applies `G_i(w) - g_i <= ramp_cap_i` when the instance carries one.
    That constraint is an EXTENSION beyond Eq. (63)-(65): it makes the day-ahead
    schedule physically binding, as start-up times do in a real market. Results
    using it are reported separately from results that do not.
    """
    probs = np.asarray(inst["probs"], float).reshape(-1)
    mc = np.asarray(inst["mc"], float).reshape(-1)
    mv = np.asarray(inst["mv"], float).reshape(-1)
    gbar = np.asarray(inst["gbar"], float)
    dbar = np.asarray(inst["dbar"], float)
    dgp = np.asarray(inst["dgp"], float).reshape(-1)
    dgn = np.asarray(inst["dgn"], float).reshape(-1)
    ddp = np.asarray(inst["ddp"], float).reshape(-1)
    ddn = np.asarray(inst["ddn"], float).reshape(-1)
    S, G = gbar.shape
    D = dbar.shape[1]

    g = cp.Variable(G, nonneg=da_nonneg)
    d = cp.Variable(D, nonneg=da_nonneg)
    Gr = cp.Variable((S, G), nonneg=True)
    Dr = cp.Variable((S, D), nonneg=True)

    L = _losses(Gr, Dr, g, d, mc, mv, dgp, dgn, ddp, ddn)

    da_bal = cp.sum(d) == cp.sum(g)
    rt_bal = (cp.sum(Dr, axis=1) - cp.sum(d)
              == cp.sum(Gr, axis=1) - cp.sum(g))

    cons = [da_bal, rt_bal, Gr <= gbar, Dr <= dbar]

    # Day-ahead quantities cannot exceed nameplate. Theory note T10 shows the
    # program is bounded only because alpha^- > 0; this makes boundedness
    # structural rather than contingent on the deviation prices.
    #
    # This constraint is NOT in the paper's Eq. (63)-(65), which bound only the
    # realised quantity x_i + x_i^+ - x_i^-, and it is not in `zavala_cvar`.
    # It therefore makes this feasible set a strict subset of the reference's,
    # so it must be switched OFF when the two models are compared, or the
    # comparison measures the constraint rather than the implementation.
    if da_cap:
        cons += [g <= gbar.max(axis=0), d <= dbar.max(axis=0)]

    if use_ramp and inst.get("ramp_cap") is not None:
        cons.append(Gr - g[None, :] <= np.asarray(inst["ramp_cap"], float)[None, :])

    expected = probs @ L
    if eps > 0.0:
        alpha = cp.Variable()
        zeta = cp.Variable(S, nonneg=True)
        link = zeta >= L - alpha
        cons.append(link)
        cvar = alpha + (1.0 / (1.0 - beta)) * (probs @ zeta)
        obj = (1.0 - eps) * expected + eps * cvar
    else:
        link = None
        obj = expected

    _solve(cp.Problem(cp.Minimize(obj), cons), solver, verbose)

    g_v = np.asarray(g.value, float).reshape(-1)
    d_v = np.asarray(d.value, float).reshape(-1)

    gamma = (np.zeros(S) if link is None
             else np.asarray(link.dual_value, float).reshape(-1))
    rho_tilde = (1.0 - eps) * probs + gamma
    nu = np.asarray(rt_bal.dual_value, float).reshape(-1)
    p0 = float(np.asarray(da_bal.dual_value).ravel()[0])

    support = rho_tilde > PRICE_SUPPORT_REL_TOL * rho_tilde.sum()
    Pi = np.full(S, np.nan)
    Pi[support] = nu[support] / rho_tilde[support]

    return dict(
        g=g_v, d=d_v,
        G=np.asarray(Gr.value, float), D=np.asarray(Dr.value, float),
        p0=p0, Pi=Pi, gamma=gamma, rho_tilde=rho_tilde, support=support,
        eps=float(eps), beta=float(beta),
        objective=float(obj.value),
    )


def solve_deterministic_da(inst, solver=None, verbose=False, da_nonneg=True):
    """Deterministic day-ahead against expected capacities (the naive policy).

    Expected capacities, single-stage merit order, no recourse term. This is
    the baseline the paper's deterministic comparison intends.
    """
    probs = np.asarray(inst["probs"], float).reshape(-1)
    mc = np.asarray(inst["mc"], float).reshape(-1)
    mv = np.asarray(inst["mv"], float).reshape(-1)
    gbar_e = probs @ np.asarray(inst["gbar"], float)
    dbar_e = probs @ np.asarray(inst["dbar"], float)

    g = cp.Variable(mc.size, nonneg=da_nonneg)
    d = cp.Variable(mv.size, nonneg=da_nonneg)
    bal = cp.sum(d) == cp.sum(g)
    cons = [bal, g <= gbar_e, d <= dbar_e]
    _solve(cp.Problem(cp.Minimize(mc @ g - mv @ d), cons), solver, verbose)
    return dict(g=np.asarray(g.value, float).reshape(-1),
                d=np.asarray(d.value, float).reshape(-1),
                p0=float(np.asarray(bal.dual_value).ravel()[0]))


def evaluate_da(inst, g_da, d_da, solver=None, verbose=False, use_ramp=False):
    """Score a day-ahead position against the full scenario set.

    Solves the recourse problem with `(g, d)` fixed. The scenarios do not couple
    once the day-ahead position is fixed, so this is one block-diagonal LP
    rather than S separate solves.

    Returns per-scenario losses and a cost decomposition. The decomposition is
    what shows *where* risk aversion pays: upward deviation, downward deviation,
    energy, and unserved load are reported separately.
    """
    probs = np.asarray(inst["probs"], float).reshape(-1)
    mc = np.asarray(inst["mc"], float).reshape(-1)
    mv = np.asarray(inst["mv"], float).reshape(-1)
    gbar = np.asarray(inst["gbar"], float)
    dbar = np.asarray(inst["dbar"], float)
    dgp = np.asarray(inst["dgp"], float).reshape(-1)
    dgn = np.asarray(inst["dgn"], float).reshape(-1)
    ddp = np.asarray(inst["ddp"], float).reshape(-1)
    ddn = np.asarray(inst["ddn"], float).reshape(-1)
    S, G = gbar.shape
    D = dbar.shape[1]

    g_da = np.asarray(g_da, float).reshape(-1)
    d_da = np.asarray(d_da, float).reshape(-1)

    Gr = cp.Variable((S, G), nonneg=True)
    Dr = cp.Variable((S, D), nonneg=True)

    L = (Gr @ mc
         + cp.pos(Gr - g_da[None, :]) @ dgp
         + cp.pos(g_da[None, :] - Gr) @ dgn
         - Dr @ mv
         + cp.pos(Dr - d_da[None, :]) @ ddp
         + cp.pos(d_da[None, :] - Dr) @ ddn)

    cons = [cp.sum(Dr, axis=1) == cp.sum(Gr, axis=1), Gr <= gbar, Dr <= dbar]
    if use_ramp and inst.get("ramp_cap") is not None:
        cons.append(Gr - g_da[None, :] <= np.asarray(inst["ramp_cap"], float)[None, :])

    _solve(cp.Problem(cp.Minimize(probs @ L), cons), solver, verbose)

    Gv = np.asarray(Gr.value, float)
    Dv = np.asarray(Dr.value, float)
    dG = Gv - g_da[None, :]
    dD = Dv - d_da[None, :]

    energy = Gv @ mc
    dev_up = np.maximum(dG, 0.0) @ dgp
    dev_dn = np.maximum(-dG, 0.0) @ dgn
    value = Dv @ mv
    load_dev = np.maximum(dD, 0.0) @ ddp + np.maximum(-dD, 0.0) @ ddn
    loss = energy + dev_up + dev_dn - value + load_dev
    unserved = dbar.sum(axis=1) - Dv.sum(axis=1)

    return dict(
        loss=loss, G=Gv, D=Dv,
        energy=energy, dev_up=dev_up, dev_dn=dev_dn,
        value=value, load_dev=load_dev, unserved=unserved,
        unserved_cost=unserved * float(mv[0]),
    )


# --------------------------------------------------------------------------
# risk metrics
# --------------------------------------------------------------------------

def var_level(loss, probs, beta):
    """beta-VaR of a discrete loss distribution."""
    loss = np.asarray(loss, float).reshape(-1)
    probs = np.asarray(probs, float).reshape(-1)
    order = np.argsort(loss)
    cum = np.cumsum(probs[order])
    idx = int(np.searchsorted(cum, beta, side="left"))
    return float(loss[order][min(idx, loss.size - 1)])


def cvar(loss, probs, beta):
    """beta-CVaR via Rockafellar-Uryasev, probability-weighted.

    `tail_worst_indices_by_value` in `zavala_funcs.py` takes an UNWEIGHTED mean
    over whole scenarios whose probability reaches the tail, which is a slightly
    different estimand. This is the quantity the planner actually optimises.
    """
    loss = np.asarray(loss, float).reshape(-1)
    probs = np.asarray(probs, float).reshape(-1)
    t = var_level(loss, probs, beta)
    return float(t + (1.0 / (1.0 - beta)) * (probs @ np.maximum(loss - t, 0.0)))


def score(probs, ev, beta=0.95):
    """Welfare summary for one evaluated policy. Welfare = -loss."""
    loss = ev["loss"]
    return dict(
        expected_welfare=float(-(probs @ loss)),
        cvar_welfare=float(-cvar(loss, probs, beta)),
        var_welfare=float(-var_level(loss, probs, beta)),
        worst_welfare=float(-loss.max()),
        served_mwh=float(probs @ ev["D"].sum(axis=1)),
        expected_unserved=float(probs @ ev["unserved"]),
        cvar_unserved=float(cvar(ev["unserved"], probs, beta)),
        expected_dev_up=float(probs @ ev["dev_up"]),
        expected_dev_dn=float(probs @ ev["dev_dn"]),
        expected_energy=float(probs @ ev["energy"]),
    )


def tail_decomposition(probs, ev, beta=0.95):
    """Average cost components conditional on the beta-tail of total loss.

    Shows what risk aversion actually buys, rather than only that it buys
    something.
    """
    loss = ev["loss"]
    t = var_level(loss, probs, beta)
    m = loss >= t - 1e-12
    w = probs[m]
    w = w / w.sum() if w.sum() > 0 else w
    return {k: float(w @ ev[k][m]) for k in
            ("energy", "dev_up", "dev_dn", "value", "load_dev", "unserved_cost")}


# --------------------------------------------------------------------------
# equivalence check against the verified reference implementation
# --------------------------------------------------------------------------

def assert_matches_reference(inst, eps=0.3, beta=0.95, solver=None,
                             tol_obj=1e-6, tol_diag=1e-3, rep=print):
    """Check the vectorised model reproduces `zavala_funcs.zavala_cvar`.

    Only valid when deviation prices are symmetric, since the reference cannot
    express asymmetry. Run on the cluster, where Gurobi and jax are available.

    What is and is not pinned down
    ------------------------------
    Both models minimise the SAME objective,

        rho_RA[l] = (1 - eps) * E[l] + eps * CVaR_beta[l],

    and at an optimum that value is unique. It is therefore the equivalence
    test, held to `tol_obj`.

    `E[l]` and `CVaR_beta[l]` SEPARATELY are not unique. This LP is degenerate
    -- many allocations achieve the same rho_RA -- so a solution may slide along
    the optimal face, trading expected loss against tail loss at the exchange
    rate eps/(1-eps) while rho_RA does not move. Two correct implementations can
    land on different points of that face.

    `p0` is a dual variable, and the dual is non-unique whenever the primal is
    degenerate, which is the normal case here: with a flat merit order many
    bases support the same allocation. Requiring two formulations to return the
    same marginal price asserts something the LP does not guarantee.

    So those three are reported as DIAGNOSTICS at the looser `tol_diag`. A
    discrepancy in them with rho_RA agreeing tightly is positive evidence of
    alternate optima, not of a modelling error -- and the function says so when
    it sees that signature. A discrepancy in rho_RA is a real disagreement and
    always raises.
    """
    from zavala_funcs import zavala_cvar

    sym = dict(inst)
    sym["dgp"] = sym["dgn"] = np.asarray(inst["mc"], float) / 10.0
    sym["ddp"] = sym["ddn"] = np.asarray(inst["mv"], float) / 10.0
    sym["ramp_cap"] = None

    # da_cap=False: the reference has no day-ahead nameplate cap, so comparing
    # against the capped model measures that constraint, not the
    # implementation. The cap's own effect is quantified separately below.
    mine = solve_planner(sym, eps=eps, beta=beta, solver=solver,
                         da_nonneg=False, use_ramp=False, da_cap=False)

    g_r, d_r, G_r, D_r, pi_r, Pi_r, gam_r = zavala_cvar(
        sym["probs"], sym["mc"], sym["mv"], sym["gbar"], sym["dbar"],
        beta=beta, lambda_cvar=eps, check_invariants=False,
        solver=solver, verbose=False, delta_scale=0.1)

    probs = np.asarray(sym["probs"], float)
    ref_loss = (np.asarray(G_r) @ sym["mc"]
                + np.abs(np.asarray(G_r) - np.asarray(g_r)[None, :]) @ sym["dgp"]
                - np.asarray(D_r) @ sym["mv"]
                + np.abs(np.asarray(D_r) - np.asarray(d_r)[None, :]) @ sym["ddp"])
    mine_loss = (mine["G"] @ sym["mc"]
                 + np.abs(mine["G"] - mine["g"][None, :]) @ sym["dgp"]
                 - mine["D"] @ sym["mv"]
                 + np.abs(mine["D"] - mine["d"][None, :]) @ sym["ddp"])

    scale = max(1.0, abs(float(probs @ ref_loss)))

    # The unique quantity: the objective both models minimise. Recomputed from
    # each solution's own loss vector, so it does not depend on either solver
    # reporting its objective consistently.
    def _rho(loss):
        return (1.0 - eps) * float(probs @ loss) + eps * cvar(loss, probs, beta)

    rho_mine, rho_ref = _rho(mine_loss), _rho(ref_loss)
    d_rho = abs(rho_mine - rho_ref) / scale

    diag = dict(
        d_expected_loss=abs(float(probs @ mine_loss)
                            - float(probs @ ref_loss)) / scale,
        d_cvar=abs(cvar(mine_loss, probs, beta)
                   - cvar(ref_loss, probs, beta)) / scale,
        d_p0=abs(mine["p0"] - float(pi_r)) / max(1.0, abs(float(pi_r))),
    )
    # sum(gamma) = eps is an identity, not an allocation, so it is exact.
    d_sum_gamma = abs(mine["gamma"].sum() - float(np.asarray(gam_r).sum()))

    out = dict(d_rho_RA=d_rho, d_sum_gamma=d_sum_gamma, **diag)

    if not np.isfinite(d_rho) or d_rho > tol_obj:
        raise AssertionError(
            f"vectorised model disagrees with reference on the OBJECTIVE: "
            f"d_rho_RA = {d_rho:.3e} > {tol_obj:.0e}. This is a real "
            f"disagreement, not degeneracy. Full: {out}")
    if not np.isfinite(d_sum_gamma) or d_sum_gamma > 1e-7:
        raise AssertionError(f"sum(gamma) differs: {d_sum_gamma:.3e}")

    loose = {k: v for k, v in diag.items()
             if not np.isfinite(v) or v > tol_diag}
    if loose:
        raise AssertionError(
            f"objective agrees (d_rho_RA = {d_rho:.3e}) but the allocation "
            f"differs by more than degeneracy explains: {loose}")

    # Report the signature explicitly rather than leaving it to be inferred.
    spread = max(diag.values())
    if spread > tol_obj and rep is not None:
        rep(f"  [note] objective matches to {d_rho:.2e} while E[l], CVaR and "
            f"p0 differ by up to {spread:.2e}.")
        rep("         That is alternate optima: this LP is degenerate, so the")
        rep("         solution slides along the optimal face and the dual is")
        rep("         not unique. The equivalence still holds.")

    # The experiments run WITH the day-ahead nameplate cap, which the reference
    # and the paper's Eq. (63)-(65) do not have. Equivalence to the reference
    # therefore does not by itself license the experiments. Measure what the
    # cap costs so the deviation is a reported number, not an assumption.
    capped = solve_planner(sym, eps=eps, beta=beta, solver=solver,
                           da_nonneg=False, use_ramp=False, da_cap=True)
    capped_loss = (capped["G"] @ sym["mc"]
                   + np.abs(capped["G"] - capped["g"][None, :]) @ sym["dgp"]
                   - capped["D"] @ sym["mv"]
                   + np.abs(capped["D"] - capped["d"][None, :]) @ sym["ddp"])
    rho_capped = _rho(capped_loss)
    out["d_rho_da_cap"] = (rho_capped - rho_mine) / scale   # signed; >= 0
    if rep is not None:
        rep(f"  [note] the T10 day-ahead nameplate cap, which the paper's "
            f"Eq. (63)-(65) do not impose,")
        rep(f"         raises the planner objective by "
            f"{out['d_rho_da_cap']:+.2e} relative. The experiments run WITH")
        rep("         it; report that alongside any absolute welfare number.")
    return out
