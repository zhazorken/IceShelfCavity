"""
Quick-look plot of the ice-shelf-cavity LES output: x–z sections of u, w, T and S at the
last output time, with the ice shelf and sea floor shaded grey.

Adapted from the landfast model's plot_landfast.py. The solid region is reconstructed from
the SAME geometry CSV the model read (data/pineislandbath.csv), so the overlay is always
consistent with the run.

    python3 plot_cavity.py [simname] [--dir OUTDIR]      # default simname=iceshelfcavity, dir=output
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

_ap = argparse.ArgumentParser()
_ap.add_argument("simname", nargs="?", default="iceshelfcavity")
_ap.add_argument("--dir", default="output", help="directory holding <simname>.nc (default: output)")
_args = _ap.parse_args()
simname, outdir = _args.simname, _args.dir

csv_name = "pineislandbath.csv"
csv_path = csv_name if os.path.exists(csv_name) else os.path.join("data", csv_name)

ncfile = os.path.join(outdir, f"{simname}.nc")
if not os.path.exists(ncfile):
    avail = [f[:-3] for f in sorted(os.listdir(outdir)) if f.endswith(".nc")] if os.path.isdir(outdir) else []
    raise SystemExit(f"{ncfile} not found. Pass the run name (and --dir), e.g.\n"
                     f"    python3 plot_cavity.py <simname> --dir {outdir}\n"
                     f"Available in {outdir}/: {avail or '(none — run the model first)'}")
ds = xr.open_dataset(ncfile, decode_times=False).isel(time=-1).squeeze(drop=True)  # squeeze drops the singleton y of a 3-D midy slice

# Geometry (x in CSV is km -> m), same source of truth as the model
geom = pd.read_csv(csv_path)
gx = geom["x_km"].values * 1e3
ice_base  = geom["topo"].values    # z(x) of the ice-shelf base
sea_floor = geom["bathy"].values   # z(x) of the sea floor

def coord_names(da):
    """Return (x_name, z_name) for a variable's own staggered dims."""
    xname = [d for d in da.dims if d.startswith("x")][0]
    zname = [d for d in da.dims if d.startswith("z")][0]
    return xname, zname

def ice_adjacent(mask, k):
    """Water cells with solid (ice) within k cells ABOVE them — the first layer
    under the ice, where the strong melt flux produces cell-scale over/undershoots."""
    adj = np.zeros_like(mask)
    for i in range(1, k + 1):
        adj[:-i, :] |= mask[i:, :]
    return adj & (~mask)

def outlier_despeckle(a, thresh, k=1):
    """Replace cells that deviate from their local (2k+1)² neighbour-median by more than
    `thresh` with that median. Catches isolated melt-flux specks while leaving smooth
    gradients untouched."""
    stk = []
    for dz in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dz == 0 and dx == 0:
                continue
            stk.append(np.roll(np.roll(a, dz, axis=0), dx, axis=1))
    med = np.nanmedian(np.stack(stk), axis=0)
    out = a.copy()
    bad = np.abs(a - med) > thresh
    out[bad] = med[bad]
    return out

def solid_mask(da):
    """True where the cell centre is inside ice or sea floor (to be greyed out)."""
    xname, zname = coord_names(da)
    x = da[xname].values
    z = da[zname].values
    top = np.interp(x, gx, ice_base)
    bot = np.interp(x, gx, sea_floor)
    ZZ = z[:, None]                       # (nz, 1)
    return (ZZ > top[None, :]) | (ZZ < bot[None, :]), xname, zname

# --- near-ice cleanup (cosmetic) -------------------------------------------------
OUTLIER_DESPECKLE = {"T": 0.5, "S": 0.3}   # replace cells > this far from local median; {} disables
OUTLIER_K         = 1

panels = [("u", "RdBu_r", dict(robust=True),         "u  (m/s)  [cross-cavity]"),
          ("w", "RdBu_r", dict(robust=True),         "w  (m/s)  [vertical]"),
          ("T", "inferno", dict(robust=True),        "T  (°C)"),
          ("S", "cividis", dict(robust=True),        "S  (psu)")]

fig, axes = plt.subplots(nrows=len(panels), figsize=(11, 10.5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.12)

for ax, (var, cmap, kw, label) in zip(axes, panels):
    if var not in ds:
        ax.set_visible(False)
        continue
    da = ds[var]
    mask, xname, zname = solid_mask(da)
    x = da[xname].values
    z = da[zname].values
    arr = da.transpose(zname, xname).values.astype(float).copy()
    arr[mask] = np.nan
    if var in OUTLIER_DESPECKLE:
        arr = outlier_despeckle(arr, OUTLIER_DESPECKLE[var], OUTLIER_K)
        arr[mask] = np.nan
    da_m = xr.DataArray(arr, coords={zname: z, xname: x}, dims=(zname, xname))
    da_m.plot(ax=ax, rasterized=True, cmap=cmap, add_labels=False,
              cbar_kwargs={"label": label}, **kw)
    # grey the solid region (ice + sea floor)
    grey = np.ma.masked_where(~mask, np.full(mask.shape, 0.6))
    ax.pcolormesh(da[xname].values, da[zname].values, grey,
                  cmap="gray", vmin=0, vmax=1, shading="auto",
                  rasterized=True, zorder=10)
    ax.set_ylabel("z (m)")

axes[-1].set_xlabel("distance from grounding line (m)   [0 = grounding line, right = ice-shelf front]")
plt.tight_layout()
fig.savefig(os.path.join(outdir, f"{simname}_section.pdf"), bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(outdir, f"{simname}_section.png"), dpi=130, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {os.path.join(outdir, simname)}_section.png / .pdf")
