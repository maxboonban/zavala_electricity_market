"""Diagnostic: what is the day-ahead decision, and what is actually in the tail?

    python check_tail_composition.py          # needs only numpy

Reconstructs the headline IM3 scenario set (summer, delivery hour 18, seed 2025,
S=500) WITHOUT a solver, and answers two questions the result JSONs cannot:

  1. What changes between the risk-neutral and risk-aware day-ahead commitment?
     With VOLL >> mc and firm capacity above peak load, demand is served in full
     in every scenario, so the recourse dispatch is a plain merit-order fill and
     the day-ahead quantities enter the loss ONLY through the symmetric
     deviation terms alpha|x(w) - x|. Minimising E[alpha|dbar - d|] therefore
     puts d at the MEDIAN of dbar; the CVaR tilt moves it to a lower quantile.
     The whole thing is a 1-D quantile problem numpy can solve exactly, and it
     reproduces the Gurobi numbers to ~3%.

  2. Which scenarios are in the beta-tail? Because the loss is
     cost - VOLL*demand and demand is always served, the loss is very nearly a
     decreasing linear function of demand -- so the "tail" is the LOW-demand
     evenings, not a scarcity event. This instance is never scarce.

Written while auditing OSCAR job 9253564; see docs/EXPERIMENT_LOG.md.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(ROOT, "dataset", "IM-3-GO-WEST")

LOAD_SCALE = 0.2362680966458012
VOLL = 1000.0
DELTA = 0.1
EPS, BETA = 0.3, 0.95
S = 500


def load_csv(name):
    p = os.path.join(DD, name)
    with open(p) as f:
        hdr = f.readline().strip().split(",")
    return hdr, np.loadtxt(p, delimiter=",", skiprows=1)


def cvar(x, p, beta):
    o = np.argsort(x)
    xs, ps = x[o], p[o]
    i = min(int(np.searchsorted(np.cumsum(ps), beta, side="left")), xs.size - 1)
    var = xs[i]
    return float(var + (ps @ np.maximum(xs - var, 0.0)) / (1.0 - beta))


# --- reproduce _scenario_hours exactly -------------------------------------
hdr_l, load = load_csv("nodal_load.csv")
hdr_s, solar = load_csv("nodal_solar.csv")
hdr_w, wind = load_csv("nodal_wind.csv")
T = load.shape[0]
hoy = np.arange(T)
hod = hoy % 24
doy = hoy // 24
month = np.clip(np.searchsorted(
    np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), doy,
    side="right") + 1, 1, 12)

rng = np.random.default_rng(2025)
mask = (hod == 18) & np.isin(month, [6, 7, 8])
pool = np.flatnonzero(mask)
hours = rng.choice(pool, size=S, replace=True)
print(f"pool {pool.size} distinct summer 18:00 hours; drew S={S}, "
      f"{np.unique(hours).size} distinct")

# --- fleet, exactly as meta records it -------------------------------------
solar_buses = ["bus_261331", "bus_204901", "bus_400041",
               "bus_500241", "bus_252891", "bus_284011"]
wind_buses = ["bus_252891", "bus_211211", "bus_100931",
              "bus_605141", "bus_606631", "bus_530021"]
si = [hdr_s.index(b) for b in solar_buses]
wi = [hdr_w.index(b) for b in wind_buses]

s_cap = solar[np.ix_(hours, si)]
w_cap = wind[np.ix_(hours, wi)]
# thermal: (capacity MW, capacity-weighted mc $/MWh) per selected bus
TH = [(6922.33, 23.27), (6837.60, 22.46), (6742.10, 13.70),
      (6521.70, 21.22), (4908.60, 22.10), (4800.50, 15.39)]
t_cap = np.array([c for c, _ in TH])
t_mc = np.array([m for _, m in TH])

dbar = load[hours].sum(axis=1) * LOAD_SCALE
probs = np.full(S, 1.0 / S)

print(f"firm thermal {t_cap.sum():,.1f} MW   mean renewable "
      f"{(s_cap.sum(1) + w_cap.sum(1)).mean():,.1f} MW   mean load {dbar.mean():,.1f} MW")
print(f"load spread: min {dbar.min():,.0f}  p5 {np.percentile(dbar,5):,.0f}  "
      f"median {np.median(dbar):,.0f}  p95 {np.percentile(dbar,95):,.0f}  max {dbar.max():,.0f}")
print(f"renewable spread: min {(s_cap.sum(1)+w_cap.sum(1)).min():,.0f}  "
      f"max {(s_cap.sum(1)+w_cap.sum(1)).max():,.0f} MW")
print(f"is firm capacity ever short of demand? "
      f"{bool((t_cap.sum() < dbar).any())}   (max shortfall "
      f"{max(0.0, float((dbar - t_cap.sum()).max())):,.1f} MW)")

# --- merit-order cost of serving dbar in full ------------------------------
# units: 6 solar @0.50, 6 wind @1.51, then the six thermal buses.
mc_all = np.concatenate([np.full(6, 0.50), np.full(6, 1.51), t_mc])
order = np.argsort(mc_all)
cost = np.zeros(S)
for w in range(S):
    cap = np.concatenate([s_cap[w], w_cap[w], t_cap])
    need = dbar[w]
    for k in order:
        take = min(cap[k], need)
        cost[w] += take * mc_all[k]
        need -= take
        if need <= 1e-9:
            break
    assert need <= 1e-6, f"scenario {w} short by {need}"

ell0 = cost - VOLL * dbar          # deviation-free loss
print(f"\ndeviation-free loss l0: mean {ell0.mean():,.0f}  "
      f"CVaR_.95 {cvar(ell0, probs, BETA):,.0f} $/h")

# --- WHAT IS THE TAIL? -----------------------------------------------------
thr = np.sort(ell0)[int(BETA * S)]
tail = ell0 >= thr
print(f"\n--- the {tail.sum()} worst scenarios by l0 ---")
print(f"  demand    in tail: mean {dbar[tail].mean():,.0f} MW   "
      f"vs {dbar[~tail].mean():,.0f} MW elsewhere")
print(f"  renewable in tail: mean {(s_cap.sum(1)+w_cap.sum(1))[tail].mean():,.0f} MW  "
      f"vs {(s_cap.sum(1)+w_cap.sum(1))[~tail].mean():,.0f} MW elsewhere")
print(f"  corr(l0, demand)    = {np.corrcoef(ell0, dbar)[0,1]:+.4f}")
print(f"  corr(l0, renewable) = {np.corrcoef(ell0, s_cap.sum(1)+w_cap.sum(1))[0,1]:+.4f}")
print(f"  demand percentile of tail scenarios: "
      f"{np.mean([ (dbar < dbar[w]).mean() for w in np.flatnonzero(tail)])*100:.1f}%")

# --- the day-ahead decision, load side only --------------------------------
# alpha_load = 100 $/MWh is 43x the largest generator alpha, so the day-ahead
# load commitment is where almost all of the deviation cost lives.
A = DELTA * VOLL


def risk(d, eps):
    l = ell0 + A * np.abs(dbar - d)
    return (1 - eps) * float(probs @ l) + eps * cvar(l, probs, BETA)


grid = np.linspace(dbar.min() - 500, dbar.max() + 500, 400001)
# vectorised scan
L = ell0[None, :] + A * np.abs(dbar[None, :] - grid[:, None])
E = L.mean(axis=1)
srt = np.sort(L, axis=1)
k = int(BETA * S)
var = srt[:, k]
C = var + np.maximum(srt - var[:, None], 0).sum(axis=1) / S / (1 - BETA)
d_rn = grid[np.argmin((1 - 0.0) * E + 0.0 * C)]
d_ra = grid[np.argmin((1 - EPS) * E + EPS * C)]

print(f"\n--- the day-ahead load commitment d ---")
print(f"  risk-neutral  d = {d_rn:,.1f} MW   "
      f"= the {100*(dbar < d_rn).mean():.1f}th percentile of dbar")
print(f"  risk-aware    d = {d_ra:,.1f} MW   "
      f"= the {100*(dbar < d_ra).mean():.1f}th percentile of dbar")
print(f"  shift            {d_ra - d_rn:+,.1f} MW")

l_rn = ell0 + A * np.abs(dbar - d_rn)
l_ra = ell0 + A * np.abs(dbar - d_ra)
dE = float(probs @ l_ra) - float(probs @ l_rn)
dC = cvar(l_ra, probs, BETA) - cvar(l_rn, probs, BETA)
print(f"\n  predicted dE[l]  = {dE:+,.2f} $/h     actual (E1) = +48,995.92")
print(f"  predicted dCVaR  = {dC:+,.2f} $/h     actual (E1) = -223,609.79")
print(f"  predicted ratio  = {dC/dE:.3f}            actual      = -4.564")

ren = s_cap.sum(1) + w_cap.sum(1)

print("\n" + "=" * 70)
print("VARIANCE DECOMPOSITION of the deviation-free loss l0 = cost - VOLL*d")
print("=" * 70)
print(f"  std( l0 )            = {ell0.std():12,.0f} $/h")
print(f"  std( VOLL * dbar )   = {(VOLL*dbar).std():12,.0f} $/h   "
      f"<- {100*(VOLL*dbar).std()/ell0.std():.1f}% of the total")
print(f"  std( merit cost )    = {cost.std():12,.0f} $/h   "
      f"<- {100*cost.std()/ell0.std():.1f}%")
print(f"  mean merit cost      = {cost.mean():12,.0f} $/h")
print(f"  mean VOLL * dbar     = {(VOLL*dbar).mean():12,.0f} $/h   "
      f"<- {(VOLL*dbar).mean()/cost.mean():.0f}x the cost it is netted against")

print("\n" + "=" * 70)
print("WHAT THE TAIL IS, under two definitions of 'loss'")
print("=" * 70)
for name, x in [("welfare  l0 = cost - VOLL*d   (as implemented)", ell0),
                ("cost     c  = merit-order cost only", cost)]:
    thr = np.sort(x)[int(BETA * S)]
    t = x >= thr
    pct = np.mean([(dbar < dbar[w]).mean() for w in np.flatnonzero(t)]) * 100
    pctr = np.mean([(ren < ren[w]).mean() for w in np.flatnonzero(t)]) * 100
    print(f"\n  {name}")
    print(f"    CVaR_.95            = {cvar(x, probs, BETA):14,.0f} $/h")
    print(f"    tail mean demand    = {dbar[t].mean():10,.0f} MW  "
          f"(vs {dbar[~t].mean():,.0f} elsewhere)  -> {pct:.1f}th pctile")
    print(f"    tail mean renewable = {ren[t].mean():10,.0f} MW  "
          f"(vs {ren[~t].mean():,.0f} elsewhere)  -> {pctr:.1f}th pctile")
    print(f"    corr with demand    = {np.corrcoef(x, dbar)[0,1]:+.4f}")
    print(f"    corr with renewable = {np.corrcoef(x, ren)[0,1]:+.4f}")

print("\n" + "=" * 70)
print("Is this instance ever scarce?")
print("=" * 70)
firm = float(t_cap.sum())
print(f"  firm thermal alone        {firm:,.1f} MW")
print(f"  peak demand               {dbar.max():,.1f} MW")
print(f"  firm / peak               {firm/dbar.max():.3f}  "
      f"(planning reserve margin {100*(firm/dbar.max()-1):.1f}%)")
print(f"  scenarios with total capacity < demand: "
      f"{int(((firm + ren) < dbar).sum())} of {S}")
print(f"  scenarios with FIRM capacity < demand:  "
      f"{int((firm < dbar).sum())} of {S}")
print(f"  minimum reserve over all scenarios:     {float((firm - dbar).min()):,.1f} MW")
