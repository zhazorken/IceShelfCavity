#!/usr/bin/env python3
"""
Automated melt-flux SIGN check for a short cavity run — the one thing worth verifying first.

Melting should push the water in the first cells under the ice toward the freezing point
(cooler) and freshen it (less salty) RELATIVE TO THE AMBIENT profile at the same depth.
This reads the last snapshot and reports the mean near-ice T,S anomaly vs ambient.

    python3 check_melt_sign.py [simname] [--dir OUTDIR]   # default: cputest, dir=output

PASS = near-ice water is cooler AND fresher than ambient (correct sign)
WARN = warmer/saltier ⇒ flip the sign of the returns in scripts/melt_parameterization.jl
"""
import os
import argparse
import re
import numpy as np
try:
    import pandas as pd
    import xarray as xr
except Exception as e:                       # pragma: no cover
    raise SystemExit(f"need pandas/xarray/netcdf4 for the sign check: {e}")

_ap = argparse.ArgumentParser()
_ap.add_argument("simname", nargs="?", default="cputest")
_ap.add_argument("--dir", default="output", help="directory holding <simname>.nc (default: output)")
_args = _ap.parse_args()
sim, outdir = _args.simname, _args.dir
nc = os.path.join(outdir, f"{sim}.nc")
if not os.path.exists(nc):
    raise SystemExit(f"{nc} not found — run the model first (or pass --dir).")

# --- ambient profile, parsed from the same stratification.jl the model used --------------
strat = "scripts/stratification.jl" if os.path.exists("scripts/stratification.jl") else "stratification.jl"
txt = open(strat).read()
def consts(prefix):
    m = re.search(r"const\s+" + prefix + r"\s*=\s*(.+)", txt)
    if not m:
        raise SystemExit(f"could not parse '{prefix}' from {strat}")
    return [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
aT, cT, z0T, hT = consts(r"aT,\s*cT,\s*z0T,\s*hT")
aS, cS, z0S, hS = consts(r"aS,\s*cS,\s*z0S,\s*hS")
Tamb = lambda z: aT + cT * np.tanh((z - z0T) / hT)
Samb = lambda z: aS + cS * np.tanh((z - z0S) / hS)

# --- geometry, same source of truth as the model ----------------------------------------
csv = "data/pineislandbath.csv" if os.path.exists("data/pineislandbath.csv") else "pineislandbath.csv"
geom = pd.read_csv(csv)
gx = geom["x_km"].values * 1e3
ice_base = geom["topo"].values
sea_floor = geom["bathy"].values

ds = xr.open_dataset(nc, decode_times=False).isel(time=-1)

def coord_names(da):
    xn = [d for d in da.dims if d.startswith("x")][0]
    zn = [d for d in da.dims if d.startswith("z")][0]
    return xn, zn

def near_ice_anom(var, ambf, ncells=2):
    """Mean (field − ambient) over the first `ncells` water cells directly under the ice."""
    da = ds[var]
    xn, zn = coord_names(da)
    x = da[xn].values
    z = da[zn].values
    arr = da.transpose(zn, xn).values.astype(float)      # (nz, nx)
    top = np.interp(x, gx, ice_base)
    bot = np.interp(x, gx, sea_floor)
    ZZ = z[:, None]
    solid = (ZZ > top[None, :]) | (ZZ < bot[None, :])
    adj = np.zeros_like(solid)                            # water cells with ice within ncells above
    for i in range(1, ncells + 1):
        adj[:-i, :] |= solid[i:, :]
    adj &= ~solid
    anom = (arr - ambf(ZZ) * np.ones_like(arr))[adj]
    anom = anom[np.isfinite(anom)]
    return (np.nanmean(anom), int(anom.size))

dT, nT = near_ice_anom("T", Tamb)
dS, nS = near_ice_anom("S", Samb)

print(f"run: {nc}   (last snapshot; {nT} near-ice cells sampled)")
print(f"  near-ice ΔT vs ambient = {dT:+.4f} °C    (want < 0: cooler toward freezing)")
print(f"  near-ice ΔS vs ambient = {dS:+.4f} psu   (want < 0: fresher from meltwater)")

if abs(dT) < 1e-3 and abs(dS) < 1e-4:
    print("  ~ near-zero signal — a 30-min smoke run barely melts; run './run_cpu_test.sh check'")
    print("    (finer + longer) to see a clear trend before trusting the verdict.")
elif dT < 0 and dS < 0:
    print("  PASS — melt-flux sign looks correct (near-ice water cools AND freshens).")
else:
    print("  WARN — near-ice water is warmer and/or saltier than ambient.")
    print("         Flip the sign of the returns in scripts/melt_parameterization.jl")
    print("         (melt_heat_flux / melt_salt_flux) and re-run.")
