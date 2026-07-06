# Ice-shelf-cavity LES — build notes

A 2-D (x–z, flat-in-y) immersed-boundary nonhydrostatic Oceananigans model of an Antarctic
ice-shelf cavity (Pine Island). It takes the numerical skeleton of the original
`icecavity.jl` and ports in the developments from the landfast-ice model (`landfast.jl`).
The numerics are deliberately unchanged; only the physics that differs has been swapped or
added.

## What changed vs. the original `icecavity.jl`

| Aspect | original `icecavity.jl` | this model |
|---|---|---|
| Ice tracer BC | fixed meltwater salinity (`S=25` on the ice base) | **3-equation melt parameterization** (heat + salt flux, depth-dependent freezing point) |
| Stratification | linear salinity from a single `N²` | **idealized two-layer** cold Winter Water over warm mCDW (analytic tanh profiles) |
| Front forcing | open eastern inflow only | open east **plus a shelf-front sponge** relaxing T,S→ambient and u→0 (warm reservoir + absorbing layer) |
| Drag (immersed) | `u,w` only, speed `√(u²+w²)` | **all faces, all three components** `u,v,w`, full speed `√(u²+v²+w²)` — consistent with the melt u★ |
| Architecture | GPU if present, else CPU | `--arch=auto/cpu/gpu`; **CUDA is an optional import** so it instantiates CPU-only |
| Resuming | none | **Checkpointer + `--pickup`** (resume from latest checkpoint) |
| `--stop_time` / `--output_interval` | seconds / minutes literals | interpreted in **days / minutes**; added `--max_dt`, `--nu`, `--checkpoint_interval` |
| Time step | internal-wave estimate | small initial Δt grown by the `TimeStepWizard` up to `--max_dt` |
| Plot | 3-panel (u, w, S) | **5-panel** (u, v, w, T, S) animation + `plot_cavity.py` section |
| Coriolis / geometry | `FPlane(latitude=-75)`, Pine Island CSV | unchanged (still Antarctic Pine Island) |

The melt closure, the optional-CUDA/checkpoint/arch machinery, `read_bathymetry.jl`,
`utils.jl` and the plotting approach are ported from `landfast.jl`; the geometry, the
Southern-Hemisphere Coriolis, the ~67 km × ~1 km cavity dimensions, and the buoyancy-driven
(rather than throughflow-driven) circulation are the ice-shelf-cavity physics.

## Testing on CPU first

**This stack needs Julia ≥ 1.10. Julia 1.9 cannot run it** (its packages reference symbols
that don't exist before 1.10). The active **`Manifest.toml` is the tested CG stack resolved
with Julia 1.10.11** (Oceananigans 0.109.2, Oceanostics 0.16.17) — the same one that runs the
`Ovall26/newLES_cg` plume model — so it instantiates directly on Julia 1.10:

```bash
julia --version                                  # must be >= 1.10 (1.10.x recommended)
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Using juliaup (matches the cluster's 1.10.9): `juliaup add 1.10 && juliaup default 1.10`, or
point `JULIA` straight at it: `JULIA=~/.juliaup/bin/julia ./run_cpu_test.sh setup`.

`Manifest.shipped.toml` is the older 1.12.6-resolved manifest, kept only as a backup — it
references the 1.11+ stdlib `JuliaSyntaxHighlighting` and will NOT instantiate on 1.10, so
don't rename it back over `Manifest.toml` unless you're on Julia ≥ 1.11.

On a laptop with no CUDA, delete the `CUDA = …` line from `Project.toml` before instantiating
— the driver warns and runs on CPU. Run **single-threaded** on CPU (the immersed-BC path on
the newer Oceananigans + Julia 1.12 threaded GC can crash; the pinned 0.109.2 avoids it).
CPU is for sanity checks only; production resolution is a GPU job.

Cheap CPU knobs (production values in parentheses):

| flag | production | CPU test | effect |
|---|---|---|---|
| `--arch` | auto/gpu | cpu | force CPU |
| `--dz` | 2.0 | 8.0 | vertical spacing (m); bigger ⇒ far fewer cells |
| `--aspect` | 4 | 4 | Δx = aspect·Δz |
| `--stop_time` | 10 | 0.05 | run length in **days** |
| `--output_interval` | 30 | 10 | output/frame cadence in **minutes** |
| `--max_dt` | 5.0 | 5.0 | max time step in **seconds** (CFL cap; keep ~2–10 s) |
| `--nu` | 1e-4 | 1e-4 | constant ν=κ [m²/s]; lower ⇒ less dissipation |
| `--checkpoint_interval` | 0.5 | 0.5 | checkpoint cadence in **days** |
| `--pickup` | off | off | resume from latest checkpoint |
| `--cg_reltol` | 1e-5 | 1e-5 | CG Poisson solver relative tolerance (looser ⇒ fewer iters) |
| `--cg_maxiter` | 30 | 30 | CG Poisson solver max iterations per pressure solve |

```bash
# compile-and-step smoke test — a few thousand cells, runs in seconds–minutes
julia --project iceshelfcavity.jl --arch=cpu --dz=8 --aspect=4 \
      --stop_time=0.05 --output_interval=10 --simname=cputest
python3 plot_cavity.py cputest
```

### What to check on the first short run
1. It compiles and steps without error; the progress messenger shows sane CFL (< 0.5) and
   max velocities that grow from zero as the melt-driven overturning spins up.
2. **Melt-flux sign** — the one thing worth verifying. The near-ice layer should trend
   toward the freezing point (cooling) and freshen. If it warms/salinifies instead, flip the
   sign of the returns in `scripts/melt_parameterization.jl` (see its header note).
3. The two-layer T,S structure and the cavity geometry mask look right in `plot_cavity.py`.

Expect the overturning to develop over **hours** of model time, not minutes: melt buoyancy
near the deep grounding line sets up a plume that rises along the ice base, Coriolis turns it
into an along-cavity `v`, and the front sponge keeps warm mCDW supplied at depth. A 15-minute
smoke test will still look quiescent — it only confirms the model compiles, steps, and has
the right structure.

## Pressure solver & immersed boundaries (no spurious near-wall flow)

The model uses the **`ConjugateGradientPoissonSolver`**, *not* the default FFT solver. This
is the fix from the `Ovall26/newLES_cg` project: the FFT pressure solve is only approximate
at immersed boundaries and produces **spurious near-wall divergence / velocities** along the
ice base and sea floor. The CG solver enforces continuity on the actual `GridFittedBoundary`
cavity geometry, so the immersed boundary is clean. It uses the FFT solver as a
preconditioner, so a modest `--cg_maxiter` (30) and a looser `--cg_reltol` (1e-5) keep the
iteration count low.

Paired with **`timestepper = :QuasiAdamsBashforth2`**, which does one pressure solve per step
instead of three (the RK3 default) — ~3× fewer CG solves, i.e. ~3× faster, at no cost to the
immersed-boundary treatment. (The original `icecavity.jl`/`landfast.jl` used the CG solver but
kept RK3; this model adds QAB2 to match the tuned `newLES_cg` setup.)

If the progress messenger ever shows the pressure residual failing to converge (CFL fine but
velocities blowing up near walls), raise `--cg_maxiter` or tighten `--cg_reltol`.

## Melt parameterization (`scripts/melt_parameterization.jl`)

**Melt = max(shear, convective)** (Wild et al. v2.2 / Zhao et al. 2024 GRL). At each ice-base
point we take the larger of a shear-driven and a buoyancy-driven melt estimate — "which process
dominates the turbulent transfer."

- **Shear** — three-equation thermodynamics with γ = Γ·u★. To make it **resolution-independent**
  the friction velocity is the log-law wall value `u★_shear = κ|U₁|/ln(z₁/z₀)` (not √(Cd)·|U₁|,
  which drifts with Δz because |U₁| is sampled at the first cell). `z₀` is set so it recovers
  √(Cd)·U at a fixed reference height (Cd = `--melt_Cd`, default 0.0022).
- **Convective** — the velocity-independent Kerr & McConnochie scaling `ṁ_conv = c_B·|ΔT|^{4/3}`
  (c_B = 2.85×10⁻⁷ m/s), recast as an equivalent `u★_conv ∝ |ΔT|^{1/3}` so the same solver
  produces it. This is the key piece: it keeps melt realistic when the plume is slow (spin-up),
  because it does not vanish as U→0.

Constants are the Wild/Davis (Thwaites East) set: Γᵀ=0.0235, Γˢ=6.7×10⁻⁴, and λ₁,λ₂,λ₃ for the
depth-dependent freezing point. These give **Pine-Island-ballpark melt** (≈ 3–50 m/yr across the
cavity, higher on the deep warm steep grounding line) — check: grounding line ≈ 36, mid-cavity
≈ 15, near the cold front ≈ 3 m/yr, with the convective branch dominant during spin-up. Momentum
drag (the τ BCs) is **separate** and still uses the roughness drag.

Applied via field-dependent flux BCs on the `top` immersed faces (downward-facing ice base).
**Next refinements** (earmarked): make the convective branch explicitly basal-slope-dependent
(McConnochie & Kerr), and treat melt on steeply sloping ice faces (needs face tagging).

## Tuning knobs
- **Stratification** (`scripts/stratification.jl`): warm/cold end members `aT,cT / aS,cS`;
  thermocline depth `z0T,z0S` (≈ −600 m); interface sharpness `hT,hS` (smaller ⇒ sharper).
- **Front sponge** (`let` block in `iceshelfcavity.jl`): `τ_nudge` (relaxation timescale),
  `sponge_width` (Gaussian half-width of the front nudging region). Lower `τ_nudge` ⇒ stiffer
  reservoir; widen `sponge_width` to push the warm supply further into the cavity.
- **Geometry**: swap `--bathymetry` for another `x_km, bathy, topo` transect (the immersed
  mask is evaluated on the native grid, so keels/topography are felt at grid resolution).
- **Closure**: constant `ScalarDiffusivity(ν=κ=--nu)` to match the cavity project; for a true
  3-D LES switch to an SGS closure (AMD / Smagorinsky) — the 3-D driver does this by default.
- **Melt shear drag** (`--melt_Cd`, default 0.0022): the shear drag in the max(shear,convective)
  closure (see below). The convective branch is what sets melt at low shear, so this mostly
  matters once the plume is fast. Momentum drag is separate.
- **Tide**: `--tide` (amplitude m/s, 0 = off) and `--tide_period` (hours, default 12.42 = M2)
  add an idealized barotropic tide by oscillating the shelf-front sponge's u-target at the
  tidal frequency. The tidal exchange enters from the front and interacts with the basal
  topography — the tidal–topography interaction from the talk. Start it after a melt-driven
  spin-up (two-phase submit) so you separate the tidal signal from the spin-up transient.

## 3-D extruded run (`iceshelfcavity3d.jl`)

Same x–z Pine Island geometry, **extruded in a periodic y channel** (`--Ly`, default 10 km),
so 3-D structure the 2-D run can't have — convective rolls, along-slope instabilities, fully
3-D tidal–topography interaction — can develop. The geometry is independent of y, so the
immersed mask ignores y; drag, melt, sponge and profiles all use 3-D `(x,y,z,t)` signatures,
and the closure defaults to **AnisotropicMinimumDissipation** (SGS LES) unless `--nu>0`.

A 3-D LES of 67 km × 10 km × ~0.74 km is huge, so the boundary layer is **necessarily coarser
than the 2-D run**. Defaults: `--dz=8` (Δx=Δy=32 m) → ~60 M cells, comfortable on an 80 GB
A100. The driver prints the cell count + a rough memory estimate at startup — check it before
committing GPU hours. `--dy` defaults to Δx (isotropic horizontal); set it larger for fewer
cells. To refine near the ice you'd shrink `--Ly` or move to a stretched/nested grid (future).

Outputs (to `--outdir`): the full 3-D field `<sim>.nc` written sparsely (`--fields3d_interval`,
hours — these are the big files, kept on GLADE), plus two **light x–z files** written every
`--output_interval`: `<sim>_midy.nc` (mid-channel slice, directly comparable to the 2-D run)
and `<sim>_yavg.nc` (the y-average = mean overturning). Plot/movie those with the same tools:
`python3 plot_cavity.py <sim>_midy --dir OUTDIR`. Submit with `submit_pbs_3d.sh`; `postprocess.sh`
and `fetch_results.sh` already special-case the 3-D slice files and never pull the big 3-D field.

## Note on scope / what is idealized
The stratification here is a **representative, tunable two-layer profile**, not a specific
hydrographic cast — set the end members and thermocline to your target ice shelf, or replace
`stratification.jl` with an analytic fit to a real CTD (the same pattern the landfast model
uses in `ctd_fit_params.jl`). The sponge imposes a steady warm reservoir; time-dependent
(e.g. tidal or seasonal) forcing at the front is a natural next step and maps directly onto
the tidal–topography and buoyancy–topography interactions discussed in the talk.
