#!/usr/bin/env python3
"""Animate the ice-shelf-cavity fields time-series into an mp4 (or gif fallback).

Reads <dir>/<simname>.nc (x-z, all output times), blanks the ice shelf + sea floor from the
geometry CSV, fixes the colour scales across frames, and renders a 4-panel (u, w, T, S) movie.
Adapted from the Ovall26/newLES_cg make_movie.py.

Usage:
  python3 make_cavity_movie.py iceshelfcavity                       # -> output/iceshelfcavity.mp4
  python3 make_cavity_movie.py iceshelfcavity --dir /glade/work/$USER/cavity_runs --fps 12
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
try:
    import pandas as pd
    import xarray as xr
except ImportError:
    sys.exit("Please: pip install pandas xarray netCDF4 matplotlib")

def coord_names(da):
    xn = [d for d in da.dims if d.startswith("x")][0]
    zn = [d for d in da.dims if d.startswith("z")][0]
    return xn, zn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("simname")
    ap.add_argument("--dir", default="output", help="directory holding <simname>.nc")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fn = os.path.join(a.dir, f"{a.simname}.nc")
    if not os.path.exists(fn):
        sys.exit(f"not found: {fn}")
    ds = xr.open_dataset(fn, decode_times=False)

    # geometry (same source of truth as the model) for the ice/sea-floor mask
    csv = "data/pineislandbath.csv" if os.path.exists("data/pineislandbath.csv") else "pineislandbath.csv"
    geom = pd.read_csv(csv)
    gx = geom["x_km"].values * 1e3
    ice_base, sea_floor = geom["topo"].values, geom["bathy"].values

    panels = [("u", "RdBu_r", "u (m/s)"), ("w", "RdBu_r", "w (m/s)"),
              ("T", "inferno", "T (°C)"), ("S", "cividis", "S (psu)")]
    panels = [(v, c, lab) for (v, c, lab) in panels if v in ds]
    if not panels:
        sys.exit("no u/w/T/S variables in file")

    series, coords, ranges = {}, {}, {}
    nt = ds.sizes.get("time", 1)
    for v, cmap, lab in panels:
        da = ds[v]
        xn, zn = coord_names(da)
        x, z = da[xn].values, da[zn].values
        arr = da.transpose("time", zn, xn).values.astype(float)          # (t, z, x)
        top = np.interp(x, gx, ice_base); bot = np.interp(x, gx, sea_floor)
        solid = (z[:, None] > top[None, :]) | (z[:, None] < bot[None, :])
        arr[:, solid] = np.nan
        series[v], coords[v] = arr, (x, z)
        if v in ("u", "w"):
            m = np.nanpercentile(np.abs(arr), 99) or 1e-6
            ranges[v] = (-m, m)
        else:
            ranges[v] = tuple(np.nanpercentile(arr, [1, 99]))

    times = ds["time"].values if "time" in ds.variables else np.arange(nt)
    fig, axs = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True)
    if len(panels) == 1:
        axs = [axs]
    meshes = []
    for ax, (v, cmap, lab) in zip(axs, panels):
        x, z = coords[v]; lo, hi = ranges[v]
        pc = ax.pcolormesh(x, z, series[v][0], cmap=cmap, vmin=lo, vmax=hi, shading="auto")
        fig.colorbar(pc, ax=ax, label=lab)
        ax.set_ylabel("z (m)")
        meshes.append(pc)
    axs[-1].set_xlabel("distance from grounding line (m)   [0 = grounding line, right = ice-shelf front]")
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(i):
        for pc, (v, cmap, lab) in zip(meshes, panels):
            pc.set_array(series[v][i].ravel())
        suptitle.set_text(f"{a.simname}   t = {times[i]:.0f} s   (frame {i+1}/{nt})")
        return meshes

    ani = manim.FuncAnimation(fig, update, frames=nt, blit=False)
    out = a.out or os.path.join(a.dir, f"{a.simname}.mp4")
    try:
        ani.save(out, writer=manim.FFMpegWriter(fps=a.fps, bitrate=3000))
    except Exception as e:
        out = os.path.splitext(out)[0] + ".gif"
        print(f"  (ffmpeg unavailable [{e}] — writing gif instead)")
        ani.save(out, writer=manim.PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"  wrote {out}  ({nt} frames)")
    ds.close()

if __name__ == "__main__":
    main()
