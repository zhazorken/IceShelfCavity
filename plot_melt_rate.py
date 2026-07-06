#!/usr/bin/env python3
"""
Diagnose the melt rate ṁ(x) along the ice-shelf base from a run's output, using the SAME
max(shear, convective[·slope]) closure as scripts/melt_parameterization.jl (Wild et al. /
Zhao et al. 2024). Plots ṁ vs distance (with the Pine Island 1–80 m/yr band shaded) and ṁ vs
basal slope, so you can check the magnitudes and the slope dependence.

    python3 plot_melt_rate.py [simname] [--dir OUTDIR]     # e.g. iceshelfcavity3d_midy

(The model doesn't output ṁ as a field; this reconstructs it from the near-ice u,v,w,T,S. It
mirrors the Julia closure, so it's the check for "are the melt rates in the PIG ballpark?".)
"""
import os, argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ---- constants (must match scripts/melt_parameterization.jl) ----
cw, Lf, rhow, rhoi = 3974.0, 3.35e5, 1027.25, 916.0
l1, l2, l3 = -5.73e-2, 8.32e-2, 7.61e-4
GT, GS, cB = 0.0235, 6.7e-4, 2.85e-7
kappa, z_ref = 0.4, 1.0
SLOPE_FMIN, SLOPE_FMAX = 0.3, 3.0

ap = argparse.ArgumentParser()
ap.add_argument("simname", nargs="?", default="iceshelfcavity3d_midy")
ap.add_argument("--dir", default="output")
ap.add_argument("--Cd", type=float, default=0.0022)
ap.add_argument("--slope_ref", type=float, default=0.03)
ap.add_argument("--no_slope", action="store_true", help="disable the slope factor (uniform)")
a = ap.parse_args()

nc = os.path.join(a.dir, f"{a.simname}.nc")
if not os.path.exists(nc):
    raise SystemExit(f"{nc} not found (pass simname + --dir).")
ds = xr.open_dataset(nc, decode_times=False).isel(time=-1).squeeze(drop=True)  # drop singleton y (3-D midy slice)

csv = "data/pineislandbath.csv" if os.path.exists("data/pineislandbath.csv") else "pineislandbath.csv"
geom = pd.read_csv(csv); gx = geom["x_km"].values * 1e3
ice_base, sea_floor = geom["topo"].values, geom["bathy"].values

def coord(da, l):
    return [d for d in da.dims if d.startswith(l)][0]

T = ds["T"]; xn, zn = coord(T, "x"), coord(T, "z")
x = T[xn].values; z = T[zn].values
nz, nx = z.size, x.size
def fld(v):  # (nz, nx) on the T (center) grid; averages any staggered face grid onto centers
    if v not in ds:
        return np.zeros((nz, nx))
    da = ds[v].squeeze()
    xv, zv = coord(da, "x"), coord(da, "z")
    a = da.transpose(zv, xv).values.astype(float)
    if a.shape[0] == nz + 1: a = 0.5 * (a[:-1, :] + a[1:, :])   # z-faces -> centers
    if a.shape[1] == nx + 1: a = 0.5 * (a[:, :-1] + a[:, 1:])   # x-faces -> centers
    return a
Tf, Sf = fld("T"), fld("S")
uf, vf, wf = fld("u"), fld("v"), fld("w")

top = np.interp(x, gx, ice_base)          # ice base z(x)
bot = np.interp(x, gx, sea_floor)         # sea floor z(x)
# ice-base slope from geometry
dtopdx = np.gradient(top, x)
sinth = np.abs(dtopdx) / np.sqrt(1 + dtopdx**2)
sfac = np.ones_like(sinth) if a.no_slope else np.clip(np.cbrt(np.maximum(sinth,1e-4)/a.slope_ref), SLOPE_FMIN, SLOPE_FMAX)

def melt_at(U, Tw, Sw, zc, z1, sf):
    z0 = z_ref*np.exp(-kappa/np.sqrt(a.Cd))
    us_sh = kappa*U/np.log(max(z1, 1.001*z0)/z0)
    dT = Tw - (l1*Sw + l2 + l3*zc)
    us_cv = sf*(rhoi*Lf*cB/(rhow*cw*GT))*np.cbrt(abs(dT))
    us = max(us_sh, us_cv, 1e-5)
    gT, gS = GT*us, GS*us
    c0 = l2+l3*zc; r = cw*gT/Lf
    A = r*l1; B = -(gS + r*(Tw-c0)); C = gS*Sw
    D = np.sqrt(max(B*B-4*A*C, 0.0)); Sb = min(max((-B-D)/(2*A), 0.0), Sw); Tb = l1*Sb+c0
    return rhow*cw*gT*(Tw-Tb)/(rhoi*Lf)*3.15e7   # m/yr

mdot = np.full(x.size, np.nan)
for i in range(x.size):
    # shallowest ocean cell in this column (just under the ice)
    ocean = np.where((z < top[i]) & (z > bot[i]))[0]
    if ocean.size == 0:
        continue
    k = ocean[np.argmax(z[ocean])]                 # highest ocean cell = ice-adjacent
    z1 = max(top[i] - z[k], 0.5)                    # distance from ice base to the cell centre
    U = np.sqrt(uf[k,i]**2 + vf[k,i]**2 + wf[k,i]**2)
    if not np.isfinite(U): continue
    mdot[i] = melt_at(U, Tf[k,i], Sf[k,i], z[k], z1, sfac[i])

good = np.isfinite(mdot)
print(f"melt ṁ (m/yr):  min {np.nanmin(mdot):.1f}   median {np.nanmedian(mdot):.1f}   "
      f"mean {np.nanmean(mdot):.1f}   max {np.nanmax(mdot):.1f}")
print("Pine Island observed: ~1-80 m/yr (higher on deep warm steep grounding line).")

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
ax[0].axhspan(1, 80, color="green", alpha=0.08, label="PIG observed 1–80 m/yr")
ax[0].plot(x[good], mdot[good], lw=1.2, color="firebrick")
ax[0].set_ylabel("melt rate  ṁ  (m/yr)")
ax[0].set_xlabel("distance from grounding line (m)")
ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3)
sl_deg = np.degrees(np.arctan(np.abs(dtopdx)))
ax[1].scatter(sl_deg[good], mdot[good], s=6, color="steelblue", alpha=0.5)
ax[1].set_xlabel("basal slope (degrees)"); ax[1].set_ylabel("melt rate  ṁ  (m/yr)")
ax[1].grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(a.dir, f"{a.simname}_meltrate.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
