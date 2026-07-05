# IceShelfCavity

Cavity-scale LES of an **Antarctic ice-shelf cavity** (Pine Island geometry) — the
"cavity-scale LES with realistic basal topography" from the ice–ocean interactions talk.

A 2-D (x–z, flat-in-y) immersed-boundary nonhydrostatic Oceananigans model. It is built on
the numerical skeleton of the original `icecavity.jl` and modernized with the developments
from the coastal landfast-ice model (`landfast.jl`):

- **Three-equation ice–ocean melt** at the ice-shelf base (Holland & Jenkins 1999;
  Jenkins 2011; ISOMIP+ coefficients) — replaces the old fixed-salinity meltwater BC.
- **Idealized two-layer warm-cavity stratification** (cold Winter Water over warm modified
  CDW), as GPU-safe analytic profiles.
- **Shelf-front sponge** that holds the warm-CDW reservoir at the open boundary and absorbs
  outflow — the warm-water intrusion that drives melt.
- **Quadratic drag on all immersed faces and all three velocity components**, sharing the
  same friction velocity as the melt exchange.
- **CPU/GPU auto-selection** (optional CUDA), **auto-resume from checkpoints**, `--outdir` /
  `--wall_time_limit` for cluster runs, and a GitHub → Casper GPU workflow ported from the
  `Ovall26/newLES_cg` project.

Geometry orientation: `x = 0` is the **grounding line** (closed western wall, deepest,
where the ice base meets the sea floor); `x = Lx ≈ 67 km` is the **ice-shelf front**
(eastern open boundary). Warm mCDW enters at depth from the front, melts the ice near the
grounding line, and the buoyant meltwater rises along the ice base as an outflow plume.

## Run it

Easiest path is the wrapper (it uses coarse resolutions sized for this 67 km domain, then
plots and runs the melt-flux sign check):

```bash
./run_cpu_test.sh setup     # one-time: instantiate + precompile (Julia >= 1.10; ~10-30 min)
./run_cpu_test.sh           # quick smoke test  (~3.9k cells, ~1-2 min) -> output/cputest_section.png
./run_cpu_test.sh check     # finer physics sanity (~21k cells, a few minutes)
```

Or drive it by hand:

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project iceshelfcavity.jl --arch=cpu --dz=40 --aspect=8 \
      --stop_time=0.02 --output_interval=5 --max_dt=5 --simname=cputest
python3 plot_cavity.py cputest        # section plot
python3 check_melt_sign.py cputest    # PASS/WARN on the melt-flux sign

# production (GPU):
qsub submit_pbs.sh
```

## GPU runs on Casper (NCAR)

Full instructions in **SETUP_CASPER.md**. In short:

```bash
# push this folder to a GitHub repo (a .gitignore is already set up), then on Casper:
git clone <your-repo> IceShelfCavity && cd IceShelfCavity
JULIA=$HOME/.juliaup/bin/julia ./setup_casper.sh      # instantiate + precompile (once)
qsub submit_pbs.sh                                    # Pine Island, ~3 M cells on one A100
#   qsub -v SIM=coarse,DZ=4 submit_pbs.sh             # quick GPU shakedown

# outputs + checkpoints land in /glade/work/$USER/cavity_runs (off the repo); the job
# auto-resumes from the latest checkpoint if you re-submit. Then:
./postprocess.sh iceshelfcavity                       # on Casper: section PNG + sign check + movie
./fetch_results.sh                                    # on your laptop: rsync the light files home
```

The GPU submit passes `--animate=false` (no inline CairoMakie on the node) and
`--wall_time_limit=11.5` (checkpoint before the PBS walltime). Movies are made post-hoc from
the NetCDF with `make_cavity_movie.py`.

## Files

| file | role |
|---|---|
| `iceshelfcavity.jl` | main driver (2-D x–z) |
| `iceshelfcavity3d.jl` | 3-D driver: same geometry extruded ~10 km in periodic y |
| `submit_pbs_3d.sh` | GPU batch script for the 3-D run |
| `scripts/read_bathymetry.jl` | reads the geometry CSV (`x_km, bathy, topo`) |
| `scripts/utils.jl` | grid-sizing helpers (unchanged from the cavity project) |
| `scripts/melt_parameterization.jl` | 3-equation ice–ocean melt closure |
| `scripts/stratification.jl` | idealized two-layer warm-CDW `T_profile(z)`, `S_profile(z)` |
| `data/pineislandbath.csv` | Pine Island ice-draft / bathymetry transect |
| `plot_cavity.py` | quick-look u/w/T/S section plot (`--dir` for cluster output) |
| `check_melt_sign.py` | automated melt-flux sign check (PASS/WARN) |
| `make_cavity_movie.py` | u/w/T/S time-series → mp4/gif |
| `run_cpu_test.sh` | one-command CPU smoke/physics test (+ plot + sign check) |
| `submit_pbs.sh` | GPU batch script (Casper), CASE/OUTDIR/auto-resume |
| `setup_casper.sh` | one-time cluster env instantiate + precompile |
| `fetch_results.sh` / `postprocess.sh` | pull light results / make plots+movies on the cluster |
| `SETUP_CASPER.md` | GitHub → Casper walkthrough |
| `Project.toml` | dependency pins (Oceananigans 0.109, Oceanostics 0.16, julia 1.10) |

`Manifest.toml` is **git-ignored** (each machine resolves fresh from the Project.toml pins);
your local one is the tested 1.10.11 stack. See **NOTES.md** for the CPU recipe, the pressure
solver, what changed vs. `icecavity.jl`, and the tuning knobs.
