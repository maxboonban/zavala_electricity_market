"""IM3 / GO-WEST instance construction for the risk-aware market experiments.

This module replaces two modelling shortcuts in `zavala_real_data.py`:

1. MARGINAL COSTS WERE RANDOM.  `zavala_real_data.py` draws thermal marginal
   costs from Uniform[35, 55] even though `thermal_gens.csv` carries a per-unit
   `Heat_Rate` and `Fuel`. Here marginal cost is built from the data:

       mc = Heat_Rate [MMBtu/MWh] * fuel_price [$/MMBtu] + VOM [$/MWh]

   Caveat, stated plainly: the dataset's natural-gas heat rates are not
   physical (mean 4.02, min 0.62 MMBtu/MWh; a real combined-cycle unit is
   ~6.4 and a combustion turbine ~10). We therefore FLOOR each unit's heat
   rate at a technology-specific physical minimum and report how many units
   were floored. Set `heat_rate_floor=False` to use the raw values.

2. "SCENARIOS" WERE CONSECUTIVE HOURS.  Taking 500 consecutive hours as 500
   scenarios conflates the diurnal cycle with forecast uncertainty: system
   solar runs 0 -> 13.8 -> 0 GW within a single day, so such an instance asks
   "what time is it?" rather than "what will the wind do at 18:00 tomorrow?".
   Here a scenario set is built by CONDITIONING on a delivery hour: we take the
   same hour-of-day across many days (optionally restricted to one season), so
   the spread across scenarios is genuine cross-day uncertainty for a fixed
   delivery period. Use `sampling="consecutive"` to reproduce the old behaviour.

Fuel prices are EIA-style 2023 annual averages for delivery to the power
sector; they are parameters, not data, and are declared here so they can be
cited and varied.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --- declared parameters (not data) ----------------------------------------
FUEL_PRICE = {            # $/MMBtu delivered to the power sector, 2023-ish
    "NG (Natural Gas)": 3.00,
    "BIT (Bituminous Coal)": 2.30,
    "NUC (Nuclear)": 0.65,
}
VOM = {                   # $/MWh variable O&M
    "NG (Natural Gas)": 2.00,
    "BIT (Bituminous Coal)": 4.50,
    "NUC (Nuclear)": 2.40,
}
HEAT_RATE_FLOOR = {       # MMBtu/MWh, physical minimum for the technology
    "NG (Natural Gas)": 6.40,
    "BIT (Bituminous Coal)": 8.80,
    "NUC (Nuclear)": 10.40,
}
RENEWABLE_MC = {"solar": 0.0, "wind": 0.0}   # zero fuel cost
RENEWABLE_VOM = {"solar": 0.50, "wind": 1.50}

SEASONS = {
    "winter": [12, 1, 2], "spring": [3, 4, 5],
    "summer": [6, 7, 8], "autumn": [9, 10, 11],
}


def _data_dir(explicit=None):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "dataset", "IM-3-GO-WEST"),
                 os.path.join(here, "archive", "dataset", "IM-3-GO-WEST"),
                 os.path.join(os.path.dirname(here), "dataset", "IM-3-GO-WEST")):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("IM-3-GO-WEST dataset not found")


_CACHE = {}


def load_raw(data_dir=None):
    """Load and cache the four CSVs."""
    d = _data_dir(data_dir)
    if d in _CACHE:
        return _CACHE[d]
    solar = pd.read_csv(os.path.join(d, "nodal_solar.csv"))
    wind = pd.read_csv(os.path.join(d, "nodal_wind.csv"))
    load = pd.read_csv(os.path.join(d, "nodal_load.csv"))
    thermal = pd.read_csv(os.path.join(d, "thermal_gens.csv"))
    T = len(solar)
    assert len(wind) == T and len(load) == T
    # hour-of-year -> (month, hour-of-day). The series is a non-leap year.
    hoy = np.arange(T)
    hod = hoy % 24
    doy = hoy // 24
    month = np.searchsorted(
        np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), doy, side="right") + 1
    out = dict(solar=solar, wind=wind, load=load, thermal=thermal,
               T=T, hod=hod, doy=doy, month=np.clip(month, 1, 12))
    _CACHE[d] = out
    return out


def thermal_marginal_costs(thermal, heat_rate_floor=True):
    """Per-unit marginal cost from Heat_Rate x fuel price + VOM."""
    hr = thermal["Heat_Rate"].to_numpy(float).copy()
    n_floored = 0
    if heat_rate_floor:
        for fuel, floor in HEAT_RATE_FLOOR.items():
            sel = (thermal["Fuel"] == fuel).to_numpy()
            n_floored += int(np.sum(sel & (hr < floor)))
            hr[sel] = np.maximum(hr[sel], floor)
    price = thermal["Fuel"].map(FUEL_PRICE).to_numpy(float)
    vom = thermal["Fuel"].map(VOM).to_numpy(float)
    return hr * price + vom, n_floored


def _scenario_hours(raw, n_scenarios, rng, hour_of_day=18, season=None,
                    sampling="conditional"):
    """Choose which hours of the year become scenarios."""
    T = raw["T"]
    if sampling == "consecutive":
        start = int(rng.integers(0, max(1, T - n_scenarios)))
        return np.arange(start, start + n_scenarios) % T
    if sampling != "conditional":
        raise ValueError(sampling)
    mask = raw["hod"] == int(hour_of_day)
    if season is not None:
        mask &= np.isin(raw["month"], SEASONS[season])
    pool = np.flatnonzero(mask)
    if pool.size == 0:
        raise ValueError(f"no hours match hour_of_day={hour_of_day}, season={season}")
    replace = pool.size < n_scenarios
    return rng.choice(pool, size=n_scenarios, replace=replace)


def build_instance(n_scenarios=200, num_solar=6, num_wind=6, num_thermal=6,
                   hour_of_day=18, season=None, sampling="conditional",
                   renewable_scale=1.0, thermal_scale=1.0, voll=1000.0, seed=0,
                   heat_rate_floor=True, equal_probs=True, data_dir=None,
                   reserve_margin=0.15, load_scale=None):
    """Build one market instance from IM3 GO-WEST.

    Returns (probs, mc, mv, gbar, dbar, meta). `meta["tech"]` labels every
    generator as "solar" / "wind" / the thermal fuel string, and `meta["is_renewable"]`
    is the boolean mask used by the incidence experiments.

    `renewable_scale` multiplies all solar and wind capacity, which is how the
    penetration sweep varies the renewable share while holding the fleet and
    the load fixed.
    """
    raw = load_raw(data_dir)
    rng = np.random.default_rng(seed)
    hours = _scenario_hours(raw, n_scenarios, rng, hour_of_day, season, sampling)
    S = len(hours)

    solar, wind, load = raw["solar"], raw["wind"], raw["load"]
    # Rank buses by mean output AT THE DELIVERY HOUR, not over the whole year:
    # the top solar bus at 03:00 is meaningless.
    s_rank = solar.iloc[hours].mean(axis=0).sort_values(ascending=False)
    w_rank = wind.iloc[hours].mean(axis=0).sort_values(ascending=False)
    solar_buses = list(s_rank.index[:num_solar])
    wind_buses = list(w_rank.index[:num_wind])

    thermal = raw["thermal"]
    mc_th_all, n_floored = thermal_marginal_costs(thermal, heat_rate_floor)
    th = thermal.assign(_mc=mc_th_all)
    # Aggregate thermal to buses, keep the largest, and use capacity-weighted
    # marginal cost for the aggregate unit.
    grp = th.groupby("Bus").apply(
        lambda df: pd.Series({
            "cap": df["Max_Cap"].sum(),
            "mc": float(np.average(df["_mc"], weights=df["Max_Cap"])),
            "fuel": df.loc[df["Max_Cap"].idxmax(), "Fuel"],
        }), include_groups=False)
    grp = grp.sort_values("cap", ascending=False).head(num_thermal)

    s_cap = solar.loc[hours, solar_buses].to_numpy(float) * renewable_scale
    w_cap = wind.loc[hours, wind_buses].to_numpy(float) * renewable_scale
    t_cap = np.broadcast_to(grp["cap"].to_numpy(float) * float(thermal_scale),
                            (S, len(grp)))
    gbar = np.clip(np.hstack([s_cap, w_cap, t_cap]), 0.0, None)

    mc = np.concatenate([
        np.full(num_solar, RENEWABLE_MC["solar"] + RENEWABLE_VOM["solar"]),
        np.full(num_wind, RENEWABLE_MC["wind"] + RENEWABLE_VOM["wind"]),
        grp["mc"].to_numpy(float)])
    # A strictly increasing tie-break keeps the LP from being massively
    # degenerate across identical renewable units (which would make the
    # per-agent incidence numbers arbitrary).
    mc = mc + 1e-3 * np.arange(mc.size)

    dbar = load.loc[hours].sum(axis=1).to_numpy(float).reshape(-1, 1)

    # SIZING. The full WECC load (mean 82 GW, peak 141 GW) dwarfs any
    # sub-selected fleet, so an instance built from a handful of thermal buses
    # sits in permanent deep scarcity: the price pins to VOLL in every scenario
    # and the clearing rule stops mattering. We therefore rescale the load LEVEL
    # so that firm capacity carries peak load with a planning reserve margin,
    # which is how operational test systems are normally sized. The load SHAPE
    # across scenarios -- the thing the uncertainty lives in -- is untouched.
    firm_cap = float(t_cap[0].sum())
    if load_scale is None:
        peak = float(dbar.max())
        load_scale = (firm_cap / (1.0 + reserve_margin)) / max(peak, 1e-9)
    dbar = np.clip(dbar * float(load_scale), 1e-6, None)
    mv = np.array([float(voll)])

    probs = (np.full(S, 1.0 / S) if equal_probs
             else rng.dirichlet(np.full(S, 1500.0 / S)))
    probs = probs / probs.sum()

    tech = (["solar"] * num_solar + ["wind"] * num_wind
            + list(grp["fuel"].to_numpy()))
    meta = dict(
        hours=hours, tech=np.array(tech, dtype=object),
        is_renewable=np.array([t in ("solar", "wind") for t in tech]),
        solar_buses=solar_buses, wind_buses=wind_buses,
        thermal_buses=list(grp.index), n_heat_rate_floored=n_floored,
        hour_of_day=hour_of_day, season=season, sampling=sampling,
        renewable_scale=renewable_scale, thermal_scale=thermal_scale,
        voll=voll, seed=seed,
        load_scale=float(load_scale), reserve_margin=reserve_margin,
        renewable_capacity_mw=float(np.mean(s_cap.sum(1) + w_cap.sum(1))),
        thermal_capacity_mw=float(t_cap[0].sum()),
        mean_load_mw=float(dbar.mean()),
    )
    meta["renewable_share"] = (meta["renewable_capacity_mw"]
                               / max(1e-9, meta["renewable_capacity_mw"]
                                     + meta["thermal_capacity_mw"]))
    return probs, mc, mv, gbar, dbar, meta


def holdout_split(n_scenarios, n_holdout, **kw):
    """Two disjoint scenario sets drawn from the same conditional pool.

    Used by the out-of-sample experiment: clear the market on `train`, then
    evaluate the realised loss on `test`, which the clearing never saw.
    """
    seed = kw.pop("seed", 0)
    tr = build_instance(n_scenarios=n_scenarios, seed=seed, **kw)
    te = build_instance(n_scenarios=n_holdout, seed=seed + 10_007, **kw)
    return tr, te
