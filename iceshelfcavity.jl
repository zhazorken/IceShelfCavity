if ("PBS_JOBID" in keys(ENV))  @info "Job ID" ENV["PBS_JOBID"] end # Print job ID if this is a PBS simulation
# Set up the environment ONCE from the shell (do not uncomment here):
#     julia --project -e 'using Pkg; Pkg.instantiate()'
# Leave this commented. Running Pkg.instantiate() on every launch is slow, and doing it
# against a Manifest.toml from a different Julia version errors. This stack needs
# Julia >= 1.10; see NOTES.md "Testing on CPU first".
#versioninfo()
#
# =====================================================================================
#  Cavity-scale LES of an ANTARCTIC ICE-SHELF CAVITY (Pine Island geometry).
#  2-D (x–z, flat-in-y) immersed-boundary nonhydrostatic Oceananigans model.
#
#  This is the "cavity-scale LES with realistic basal topography" of the talk, rebuilt on
#  the numerical skeleton of the Antarctic cavity model `icecavity.jl` and modernized with
#  the developments from the landfast-ice model (`landfast.jl`):
#    • a three-equation ice–ocean melt parameterization at the ice-shelf base
#      (Holland & Jenkins 1999; Jenkins 2011; ISOMIP+ coefficients),
#    • an idealized two-layer warm-cavity stratification (cold Winter Water over warm mCDW),
#    • an offshore (shelf-front) sponge that maintains the warm-water reservoir and absorbs
#      outflow — i.e. the warm-CDW intrusion that drives melt,
#    • quadratic drag on ALL immersed faces and ALL three velocity components,
#    • CPU/GPU auto-selection (optional CUDA), checkpoint + --pickup resume,
#    • a 5-panel (u, v, w, T, S) animation and robust file/path handling.
#
#  Geometry orientation (Pine Island transect, `data/pineislandbath.csv`):
#    x = 0    GROUNDING LINE  (closed western wall; deepest; ice base meets the sea floor)
#    x = Lx   ICE-SHELF FRONT (eastern OPEN boundary; warm mCDW reservoir at depth)
#  Warm modified CDW enters at depth from the front, melts the ice near the deep grounding
#  line, and the buoyant meltwater rises along the ice base as an outflow plume — the
#  "upside-down estuarine circulation" of the talk.
# =====================================================================================
using ArgParse
using Oceananigans
using Oceananigans: on_architecture
using Oceananigans.Units
using Oceananigans.Fields: @compute
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans: prettytime

# CUDA is optional — only needed for GPU runs. Wrapping the import lets you instantiate
# and run on a CPU-only machine (e.g. a laptop) without the CUDA stack installed.
const CUDA_LOADED = try
    @eval import CUDA
    true
catch err
    @warn "CUDA not loaded — CPU only." exception=(err, catch_backtrace())
    false
end
gpu_functional() = CUDA_LOADED && CUDA.functional()

using Interpolations: LinearInterpolation, Flat as FlatBC   # alias: avoid clashing with Oceananigans.Flat (topology)
using Statistics: mean
using Printf

#+++ Parse initial arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--simname"
            help = "Simulation name for output"
            default = "iceshelfcavity"
            arg_type = String

        "--arch"
            help = "Architecture: auto (GPU if available, else CPU), cpu, or gpu"
            default = "auto"
            arg_type = String

        "--aspect"
            help = "Horizontal-to-vertical grid aspect ratio (Δx = aspect·Δz)"
            default = 4.0
            arg_type = Float64

        "--dz"
            help = "Vertical grid spacing in meters"
            default = 2.0            # matches the icecavity_dz2 production run
            arg_type = Float64

        "--stop_time"
            help = "Simulation stop time in DAYS"
            default = 2.0            # quick test; a real spin-up is many days
            arg_type = Float64

        "--output_interval"
            help = "Output/animation interval in MINUTES"
            default = 30.0
            arg_type = Float64

        "--max_dt"
            help = "Maximum time step in SECONDS (CFL-limited; keep modest, ~2-10 s)"
            default = 5.0
            arg_type = Float64

        "--nu"
            help = "Constant viscosity = diffusivity [m^2/s]. Lower (e.g. 1e-5 or 0) ⇒ less"*
                   " dissipation, closer to eddying (WENO supplies grid-scale dissipation)."
            default = 1e-4
            arg_type = Float64

        "--checkpoint_interval"
            help = "Checkpoint cadence in DAYS (full model state, for resuming)"
            default = 0.5
            arg_type = Float64

        "--pickup"
            help = "Resume from the latest checkpoint in checkpoints/ instead of starting at t=0"
            action = :store_true

        "--bathymetry"
            help = "Geometry CSV in data/ (columns x_km, bathy, topo)"
            default = "pineislandbath.csv"
            arg_type = String

        "--cg_reltol"
            help = "Conjugate-gradient Poisson solver relative tolerance (looser ⇒ fewer iters)"
            default = 1e-5
            arg_type = Float64

        "--cg_maxiter"
            help = "Conjugate-gradient Poisson solver max iterations per pressure solve"
            default = 30
            arg_type = Int

        "--outdir"
            help = "Directory for NetCDF output + checkpoints (default <rundir>/output). Keep it OFF"*
                   " the git repo (e.g. /glade/work/USER/cavity_runs) for cluster runs."
            default = ""
            arg_type = String

        "--wall_time_limit"
            help = "Stop gracefully after this many HOURS of wall time (writes a checkpoint so a"*
                   " re-submit resumes). Default: no limit."
            default = Inf
            arg_type = Float64

        "--animate"
            help = "Write the inline CairoMakie mp4 (leave on for CPU tests; set false for big GPU"*
                   " runs and make movies post-hoc with make_cavity_movie.py)"
            default = true
            arg_type = Bool

        "--tide"
            help = "Barotropic tide amplitude [m/s] injected at the shelf front (0 = off). Drives a"*
                   " tidal exchange flow that interacts with the basal topography."
            default = 0.0
            arg_type = Float64

        "--tide_period"
            help = "Tidal period in HOURS (default 12.42 = M2 semidiurnal)"
            default = 12.42
            arg_type = Float64

        "--melt_Cd"
            help = "Shear drag coefficient for the melt closure at the reference height (Wild et al.;"*
                   " default 0.0022). Melt = max(shear[this Cd], convective[Kerr]); resolution-independent."
            default = 0.0022
            arg_type = Float64

        "--melt_slope"
            help = "Basal-slope-dependent convective melt (McConnochie & Kerr): 1 = on, 0 = off (uniform)."
            default = 1
            arg_type = Int

        "--slope_ref"
            help = "Reference sinθ at which the slope factor = 1 (default 0.03 ≈ 1.7°); steeper ⇒ more melt."
            default = 0.03
            arg_type = Float64

    end
    return parse_args(settings, as_symbols=true)
end

params = (; parse_command_line_arguments()...)
rundir = @__DIR__
#---

#+++ Figure out architecture
arch = if params.arch == "cpu"
    CPU()
elseif params.arch == "gpu"
    GPU()
elseif gpu_functional()
    GPU()
else
    CPU()
end
@info "Starting simulation $(params.simname) with dz=$(params.dz) on $arch (--arch=$(params.arch))\n"
#---

#+++ Get bathymetry file and secondary simulation parameters
# Resolve helper files whether they live in scripts/ (repo layout) or alongside this file.
locate(name) = (p = joinpath(@__DIR__, "scripts", name); isfile(p) ? p : joinpath(@__DIR__, name))
include(locate("read_bathymetry.jl"))          # geometry CSV reader
include(locate("utils.jl"))                    # grid-sizing helpers
include(locate("melt_parameterization.jl"))    # 3-equation ice–ocean melt
include(locate("stratification.jl"))           # idealized two-layer warm-CDW T,S profiles

bathymetry = get_bathymetry_from(params.bathymetry)

top_interp = LinearInterpolation(bathymetry.x, bathymetry.top, extrapolation_bc=FlatBC())       # ice base z(x)
bottom_interp = LinearInterpolation(bathymetry.x, bathymetry.bottom, extrapolation_bc=FlatBC()) # sea floor z(x)

let
    #+++ Geometry
    Lx = maximum(bathymetry.x)
    zᵇ, zᵗ = (minimum(bathymetry.bottom), maximum(bathymetry.top))
    Lz = zᵗ - zᵇ
    #---

    #+++ Simulation size
    Nx = max(ceil(Int, Lx / (params.aspect * params.dz)), 5)
    Nz = max(ceil(Int, Lz / params.dz), 2)

    Nx = closest_factor_number((2, 3, 5), Nx)
    Nz = closest_factor_number((2, 3, 5), Nz)
    N_total = Nx * Nz
    #---

    #+++ Dynamically-relevant secondary parameters
    f = FPlane(latitude=-75).f        # Pine Island, Antarctica (Southern Hemisphere ⇒ f < 0)
    N² = 1e-5/second^2                # representative interfacial N² (reference/logging only;
                                      # the actual profile is the two-layer state set below)
    #---

    #+++ Scales
    T_inertial = 2π / abs(f)
    z₀ = 0.1meters                    # under-ice / sea-bed roughness length (Δz≈2 m cavity grid)
    #---

    global params = merge(params, Base.@locals)
end
#---

#+++ Base grid
grid_base = RectilinearGrid(arch; topology = (Bounded, Flat, Bounded),
                            size = (params.Nx, params.Nz),
                            x = (0, params.Lx), z = (params.zᵇ, params.zᵗ))
@info grid_base
#---

#+++ Immersed boundary
# 0 = ocean (between sea floor and ice base); 1 = solid (ice above, sea floor below)
cavity_mask(x, z) = ifelse((z > bottom_interp(x)) & (z < top_interp(x)), 0, 1)
GFB = GridFittedBoundary(cavity_mask)

grid = ImmersedBoundaryGrid(grid_base, GFB)
@info grid
#---

#+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
z₁ = minimum_zspacing(grid_base, Center(), Center(), Center())/2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz = (κᵛᵏ / log(z₁/params.z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ (x, y, z) =" params.c_dz

# Quadratic drag uses the FULL speed |U| = √(u²+v²+w²) and is applied to all three
# components — including the along-cavity v that the Coriolis-turned overturning generates.
# The same u★ = √(Cᴰ)·|U| feeds the melt exchange velocities, so momentum/heat/salt are
# mutually consistent.
@inline _spd(u, v, w) = √(u^2 + v^2 + w^2)
@inline τᵘ_drag(x, z, t, u, v, w, p) = -p.Cᴰ * u * _spd(u, v, w)
@inline τᵛ_drag(x, z, t, u, v, w, p) = -p.Cᴰ * v * _spd(u, v, w)
@inline τʷ_drag(x, z, t, u, v, w, p) = -p.Cᴰ * w * _spd(u, v, w)

τᵘ = FluxBoundaryCondition(τᵘ_drag, field_dependencies=(:u, :v, :w), parameters=(; Cᴰ = params.c_dz))
τᵛ = FluxBoundaryCondition(τᵛ_drag, field_dependencies=(:u, :v, :w), parameters=(; Cᴰ = params.c_dz))
τʷ = FluxBoundaryCondition(τʷ_drag, field_dependencies=(:u, :v, :w), parameters=(; Cᴰ = params.c_dz))
#---

#+++ Ambient water-mass structure (idealized two-layer warm cavity; scripts/stratification.jl)
#    Cold, fresh Winter Water over warm, salty modified CDW. `T_profile(z)`, `S_profile(z)`
#    are GPU-safe analytic tanh fits (no interpolation object inside kernels).
eos = LinearEquationOfState()   # salinity dominates the density structure here
buoyancy_formulation = SeawaterBuoyancy(equation_of_state=eos)

let
    τ_nudge      = 30minutes  # relaxation timescale toward the ambient profile in the front sponge
    sponge_width = 3000.0     # m  Gaussian half-width of the shelf-front (open-ocean) nudging region
    global params = merge(params, Base.@locals)
end

T₀(z) = T_profile(z);  T₀(x, z) = T_profile(z)   # signatures for set!
S₀(z) = S_profile(z);  S₀(x, z) = S_profile(z)
Tᵉ(z, t) = T_profile(z)                          # eastern (shelf-front) inflow profile
Sᵉ(z, t) = S_profile(z)
#---

#+++ Boundary conditions
# Momentum: quadratic drag on ALL immersed faces (ice-shelf base + sea floor). The eastern
# (shelf-front) boundary is open/radiating with zero mean normal velocity — the exchange
# flow is set up by melt buoyancy and maintained by the front sponge, not by a prescribed
# inflow current.
u_bcs = FieldBoundaryConditions(immersed = τᵘ,
                                east = OpenBoundaryCondition(0,
                                          scheme = PerturbationAdvection(inflow_timescale=10minutes)))
v_bcs = FieldBoundaryConditions(immersed = τᵛ)
w_bcs = FieldBoundaryConditions(immersed = τʷ)

# Tracers: three-equation melt flux on the ice-shelf base (`top` immersed faces) and the
# ambient two-layer profiles prescribed at the eastern open boundary.
#
# Melt closure = max(shear, convective) (Wild et al.; scripts/melt_parameterization.jl). The
# convective (Kerr) branch is velocity-independent, so melt stays PIG-realistic even while the
# plume is still spinning up. --melt_Cd is the shear drag; z₁ (first-cell height) makes the
# shear friction velocity resolution-independent (log law). Momentum drag BCs are separate.
# Precompute the per-x basal-slope factor from the ice-base geometry (CPU, at setup) and move it
# to the architecture; the melt kernel just indexes it by x. --melt_slope=0 disables it (all 1).
Δx_grid = params.Lx / params.Nx
sfac_cpu = map(1:params.Nx) do i
    xc = (i - 0.5) * Δx_grid; δ = 0.5Δx_grid
    dtopdx = (top_interp(xc + δ) - top_interp(xc - δ)) / (2δ)
    sinθ = abs(dtopdx) / sqrt(1 + dtopdx^2)
    params.melt_slope == 1 ? slope_factor(sinθ, params.slope_ref) : 1.0
end
sfac_dev = on_architecture(arch, collect(Float64, sfac_cpu))
melt_params = (; Cd = params.melt_Cd, z1 = z₁, sfac = sfac_dev, dx = Δx_grid, Nx = params.Nx)
@info "Melt closure = max(shear[Cd], convective[Kerr·slope])" melt_Cd=params.melt_Cd z1=z₁ slope=(params.melt_slope==1) slope_factor_range=(minimum(sfac_cpu), maximum(sfac_cpu))
T_melt = FluxBoundaryCondition(melt_heat_flux, field_dependencies=(:u, :v, :w, :T, :S), parameters=melt_params)
S_melt = FluxBoundaryCondition(melt_salt_flux, field_dependencies=(:u, :v, :w, :T, :S), parameters=melt_params)

T_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = T_melt),
                                east = ValueBoundaryCondition(Tᵉ))
S_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = S_melt),
                                east = ValueBoundaryCondition(Sᵉ))
#---

#+++ Shelf-front "warm reservoir" nudging (sponge)
# Relax T,S back toward the ambient two-layer profile — and u toward 0 — inside a Gaussian
# region centred on the shelf front (east boundary). This holds the warm mCDW reservoir at
# the open-ocean side so it can keep intruding shoreward and driving melt, and acts as an
# absorbing layer that damps noise radiating out of the cavity toward the open boundary.
#
# NOTE: the grid is Flat in y, so the mask is called with (x, z) and the target with
# (x, z, t). Constants are captured in a `let` so the closures are isbits (CPU & GPU safe).
front_sponge_mask = let xc = params.Lx, w = params.sponge_width
    (x, z) -> exp(-(x - xc)^2 / (2 * w^2))
end
T∞(x, z, t) = T_profile(z)
S∞(x, z, t) = S_profile(z)

T_sponge = Relaxation(rate = 1/params.τ_nudge, mask = front_sponge_mask, target = T∞)
S_sponge = Relaxation(rate = 1/params.τ_nudge, mask = front_sponge_mask, target = S∞)

# Optional barotropic tide: oscillate the front-sponge u-target at the tidal frequency, so a
# tidal exchange flow enters from the shelf front and interacts with the basal topography.
# --tide=0 (default) ⇒ target is 0, i.e. the plain absorbing sponge. Captured in a `let` so
# the closure is isbits (GPU-safe).
utide = let A = params.tide, ω = 2π / (params.tide_period * hours)
    (x, z, t) -> A * sin(ω * t)
end
u_sponge = Relaxation(rate = 1/params.τ_nudge, mask = front_sponge_mask, target = utide)
#---

#+++ Model setup
# Immersed boundaries need the ConjugateGradientPoissonSolver, NOT the default FFT solver:
# the FFT pressure solve is only approximate at immersed boundaries and produces spurious
# near-wall divergence/velocities. The CG solver enforces continuity on the actual cavity
# geometry and uses the FFT solver as a preconditioner, so a modest maxiter + looser reltol
# keep the iteration count low (tune via --cg_reltol / --cg_maxiter). QuasiAdamsBashforth2
# does ONE pressure solve per step vs THREE for the RK3 default — ~3× fewer CG solves.
coriolis = FPlane(; params.f)
model = NonhydrostaticModel(grid;
                            timestepper = :QuasiAdamsBashforth2,
                            buoyancy = buoyancy_formulation,
                            coriolis,
                            advection = WENO(order=5),
                            tracers = (:T, :S),
                            closure = params.nu > 0 ? ScalarDiffusivity(ν=params.nu, κ=params.nu) : nothing,
                            forcing = (u = u_sponge, T = T_sponge, S = S_sponge),
                            pressure_solver = ConjugateGradientPoissonSolver(grid;
                                reltol = params.cg_reltol, maxiter = params.cg_maxiter),
                            boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs))

# Initial condition: the two-layer ambient T,S from the profiles; no mean flow (the melt
# buoyancy drives the overturning); a tiny w kick breaks symmetry.
set!(model, T=T₀, S=S₀, w=(x, z) -> 1e-4 * randn())
#---

#+++ Simulation setup
stop_time = params.stop_time * days
# Initial time step [s]. The TimeStepWizard adapts it from here using the advective CFL.
# The vertical CFL w·Δt/Δz is the real limit; keep --max_dt modest so a transient plume
# can't overshoot CFL between the wizard's samples.
Δt = 0.5
# --wall_time_limit stops the run gracefully (writing a checkpoint) before the PBS walltime
# kills it, so a re-submit resumes cleanly. Default Inf = no wall-time stop.
wtl = params.wall_time_limit
simulation = Simulation(model; Δt, stop_time,
                        wall_time_limit = isfinite(wtl) ? wtl * 3600 : Inf)

wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=params.max_dt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

#+++ Progress messenger
using Oceanostics.ProgressMessengers
walltime_per_timestep = StepDuration(with_prefix=false) # This needs to instantiated here, and not in the function below
walltime = Walltime()
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + walltime + MaxVelocities()
                              + TimeStep() + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                              + "step dur = " * walltime_per_timestep)(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(40))
#---
#---

#+++ Plotting (inline CairoMakie animation; skipped when --animate=false, e.g. big GPU runs)
if params.animate
animations_dir = joinpath(rundir, "anims")
mkpath(animations_dir)

u, v, w = model.velocities
T, S = model.tracers

using CairoMakie
fig = Figure(size = (2400, 700));

xz_axis_kwargs = (
    xlabel = "distance from grounding line (m)",
    ylabel = "depth (m)",
    xgridvisible = false,
    ygridvisible = false,
)

ax_0 = Axis(fig[1, 1:7])
ax_1 = Axis(fig[1, 1]; title = "u (cross-cavity)", xz_axis_kwargs...)
ax_v = Axis(fig[1, 3]; title = "v (along-cavity)", xz_axis_kwargs...)
ax_2 = Axis(fig[1, 5]; title = "w (vertical)", xz_axis_kwargs...)
ax_3 = Axis(fig[2, 1]; title = "Temperature", xz_axis_kwargs...)
ax_4 = Axis(fig[2, 3]; title = "Salinity", xz_axis_kwargs...)
linkaxes!(ax_1, ax_v, ax_2, ax_3, ax_4)
hideydecorations!(ax_v)
hideydecorations!(ax_2)
hideydecorations!(ax_4)

u_scale = 0.15
w_scale = 0.02
v_scale = 0.1
T_cold, T_warm = T_profile(params.zᵗ), T_profile(params.zᵇ)   # cold/fresh top, warm/salty bottom
S_fresh, S_salty = S_profile(params.zᵗ), S_profile(params.zᵇ)

@compute u_slc = Field(Field(u + 0))
@compute v_slc = Field(Field(v + 0))
@compute w_slc = Field(Field(w + 0))
@compute T_slc = Field(Field(T + 0))
@compute S_slc = Field(Field(S + 0))

function update_plot!(sim)
    hm_u = heatmap!(fig[1, 1], u_slc, colorrange=(-u_scale, u_scale), colormap=:balance, nan_color=:gray)
    hm_v = heatmap!(fig[1, 3], v_slc, colorrange=(-v_scale, v_scale), colormap=:balance, nan_color=:gray)
    hm_w = heatmap!(fig[1, 5], w_slc, colorrange=(-w_scale, w_scale), colormap=:balance, nan_color=:gray)
    hm_T = heatmap!(fig[2, 1], T_slc, colorrange=(T_cold, T_warm), colormap=:thermal, nan_color=:gray)
    hm_S = heatmap!(fig[2, 3], S_slc, colorrange=(S_fresh, S_salty), colormap=:lajolla, nan_color=:gray)

    if simulation.model.clock.iteration == 0
        Colorbar(fig[1, 2], hm_u, label="m/s")
        Colorbar(fig[1, 4], hm_v, label="m/s")
        Colorbar(fig[1, 6], hm_w, label="m/s")
        Colorbar(fig[2, 2], hm_T, label="°C")
        Colorbar(fig[2, 4], hm_S, label="psu")
    end

    time = sim.model.clock.time
    title = "Time = $(@sprintf "%s" prettytime(time)) = $(@sprintf "%.3g" time/params.T_inertial) Inertial periods"
    ax_0.title = title

    recordframe!(io)
end
resize_to_layout!(fig) # Resize figure after everything is done to it, but before recording

io = VideoStream(fig, format="mp4", framerate=12, compression=20)
update_plot!(simulation)
add_callback!(simulation, update_plot!, TimeInterval(params.output_interval * minutes))
end # if params.animate
#---

#+++ Output setup
outputs = fields(model)
# Output + checkpoints go to --outdir (default <rundir>/output). Keep this OFF the git repo
# and on /glade/work or scratch for cluster runs.
outdir = isempty(params.outdir) ? joinpath(rundir, "output") : params.outdir
mkpath(outdir)

# Auto-resume: if a checkpoint for this simname already exists in outdir, pick up from it — so
# re-submitting the same cluster job continues instead of restarting at t=0. --pickup forces it.
ckpt_prefix = params.simname
pickup = params.pickup || any(startswith("$(ckpt_prefix)_iteration"), readdir(outdir))
pickup && @warn "Checkpoint for $(params.simname) found in $outdir — resuming."

using NCDatasets
simulation.output_writers[:fields] = NetCDFWriter(
    model, outputs;
    filename = joinpath(outdir, params.simname),
    schedule = TimeInterval(params.output_interval * minutes),
    overwrite_existing = !pickup             # fresh run overwrites; a resumed run appends
)

# Full-state checkpoints for resuming a long run. cleanup=true keeps only the latest (each is
# ~model-size; on the cluster you don't want to fill /glade/work with every checkpoint).
simulation.output_writers[:checkpointer] = Checkpointer(
    model;
    schedule = TimeInterval(params.checkpoint_interval * days),
    dir = outdir,
    prefix = ckpt_prefix,
    cleanup = true
)
#---

@info "Starting simulation..." pickup
run!(simulation; pickup)

if params.animate
    surface_animation_path = joinpath(animations_dir, "$(params.simname).mp4")
    save(surface_animation_path, io)
    @info "Wrote animation" surface_animation_path
end
@info "Done: $(params.simname)"
