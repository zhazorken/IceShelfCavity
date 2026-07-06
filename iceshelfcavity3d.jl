if ("PBS_JOBID" in keys(ENV))  @info "Job ID" ENV["PBS_JOBID"] end
# Set up the environment ONCE from the shell (do not uncomment here):
#     julia --project -e 'using Pkg; Pkg.instantiate()'
#
# =====================================================================================
#  3-D ice-shelf-cavity LES — the 2-D Pine Island cavity (x–z) EXTRUDED in a PERIODIC y.
#
#  Same geometry, stratification, 3-equation melt, shelf-front sponge, CG Poisson solver
#  and drag as `iceshelfcavity.jl`, but the y direction is a periodic channel (~10 km by
#  default) so 3-D instabilities and along-slope structure that the 2-D run suppresses can
#  develop (convective rolls, baroclinic eddies, tidal–topography interaction in 3-D).
#
#  The ice/sea-floor geometry is INDEPENDENT of y (extruded), so `cavity_mask(x,y,z)` ignores
#  y. A true 3-D LES of a 67 km × 10 km × ~0.74 km domain is huge, so the boundary layer is
#  necessarily coarser here than in the 2-D run — pick --dz / --dy for the cells you can afford
#  (the driver prints the total). Closure defaults to AnisotropicMinimumDissipation (SGS LES).
#
#  Orientation (same as 2-D): x=0 grounding line (closed, deep); x=Lx shelf front (open, warm).
# =====================================================================================
using ArgParse
using Oceananigans
using Oceananigans: on_architecture
using Oceananigans.Units
using Oceananigans.Fields: @compute
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans: prettytime

const CUDA_LOADED = try
    @eval import CUDA
    true
catch err
    @warn "CUDA not loaded — CPU only." exception=(err, catch_backtrace())
    false
end
gpu_functional() = CUDA_LOADED && CUDA.functional()

using Interpolations: LinearInterpolation, Flat as FlatBC
using Statistics: mean
using Printf

#+++ Parse initial arguments
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--simname";              default = "iceshelfcavity3d"; arg_type = String
        "--arch";                 help = "auto | cpu | gpu";    default = "auto"; arg_type = String
        "--aspect";  help = "Δx = aspect·Δz";                   default = 4.0;    arg_type = Float64
        "--dz";      help = "Vertical spacing [m] (3-D is coarser than the 2-D run)"; default = 8.0; arg_type = Float64
        "--Ly";      help = "Along-slope (periodic y) extent in KILOMETRES";         default = 10.0; arg_type = Float64
        "--dy";      help = "y grid spacing [m] (0 ⇒ isotropic horizontal, Δy = Δx)"; default = 0.0;  arg_type = Float64
        "--stop_time";            help = "Stop time in DAYS";   default = 5.0;    arg_type = Float64
        "--output_interval";      help = "Slice/average output cadence in MINUTES"; default = 30.0; arg_type = Float64
        "--fields3d_interval";    help = "FULL 3-D field output cadence in HOURS (big files!)"; default = 6.0; arg_type = Float64
        "--max_dt";               help = "Max time step [s]";   default = 5.0;    arg_type = Float64
        "--nu";      help = "Constant ν=κ [m²/s]; 0 ⇒ AnisotropicMinimumDissipation SGS closure"; default = 0.0; arg_type = Float64
        "--checkpoint_interval";  help = "Checkpoint cadence in DAYS"; default = 0.5; arg_type = Float64
        "--pickup";               help = "Force resume from latest checkpoint"; action = :store_true
        "--bathymetry";           default = "pineislandbath.csv"; arg_type = String
        "--outdir";  help = "Output + checkpoint dir (default <rundir>/output; use /glade/work for cluster)"; default = ""; arg_type = String
        "--wall_time_limit";      help = "Stop gracefully after this many HOURS (checkpoint to resume)"; default = Inf; arg_type = Float64
        "--cg_reltol";            default = 1e-5; arg_type = Float64
        "--cg_maxiter";           default = 30;   arg_type = Int
        "--tide";        help = "Barotropic tide amplitude [m/s] at the shelf front (0 = off)"; default = 0.0; arg_type = Float64
        "--tide_period"; help = "Tidal period in HOURS (12.42 = M2)"; default = 12.42; arg_type = Float64
        "--melt_Cd";     help = "Shear drag for the melt closure (Wild et al.; default 0.0022). Melt = max(shear, convective[Kerr])"; default = 0.0022; arg_type = Float64
    end
    return parse_args(settings, as_symbols=true)
end

params = (; parse_command_line_arguments()...)
rundir = @__DIR__
#---

#+++ Architecture
arch = if params.arch == "cpu"; CPU()
elseif params.arch == "gpu"; GPU()
elseif gpu_functional(); GPU()
else; CPU() end
@info "Starting 3-D simulation $(params.simname) with dz=$(params.dz), Ly=$(params.Ly) km on $arch\n"
#---

#+++ Geometry + helpers
locate(name) = (p = joinpath(@__DIR__, "scripts", name); isfile(p) ? p : joinpath(@__DIR__, name))
include(locate("read_bathymetry.jl"))
include(locate("utils.jl"))
include(locate("melt_parameterization.jl"))
include(locate("stratification.jl"))

bathymetry = get_bathymetry_from(params.bathymetry)
top_interp = LinearInterpolation(bathymetry.x, bathymetry.top, extrapolation_bc=FlatBC())
bottom_interp = LinearInterpolation(bathymetry.x, bathymetry.bottom, extrapolation_bc=FlatBC())

let
    Lx = maximum(bathymetry.x)
    Ly = params.Ly * 1e3meters
    zᵇ, zᵗ = (minimum(bathymetry.bottom), maximum(bathymetry.top))
    Lz = zᵗ - zᵇ

    Δx = params.aspect * params.dz
    dy = params.dy > 0 ? params.dy : Δx           # default: isotropic horizontal (Δy = Δx)

    Nx = max(ceil(Int, Lx / Δx), 5)
    Ny = max(ceil(Int, Ly / dy), 4)
    Nz = max(ceil(Int, Lz / params.dz), 2)
    Nx = closest_factor_number((2, 3, 5), Nx)
    Ny = closest_factor_number((2, 3, 5), Ny)     # factorable Ny keeps the FFT preconditioner fast
    Nz = closest_factor_number((2, 3, 5), Nz)
    N_total = Nx * Ny * Nz

    f = FPlane(latitude=-75).f
    N² = 1e-5/second^2
    T_inertial = 2π / abs(f)
    z₀ = 0.1meters

    global params = merge(params, Base.@locals)
end
@info "3-D grid" Nx=params.Nx Ny=params.Ny Nz=params.Nz N_total=params.N_total
@info "≈ memory: a NonhydrostaticModel needs very roughly $(round(params.N_total * 8 * 30 / 1e9, digits=1)) GB " *
      "(order-of-magnitude; ~30 arrays × 8 B/cell). Keep it under your GPU's memory."
#---

#+++ Base grid (PERIODIC in y)
grid_base = RectilinearGrid(arch; topology = (Bounded, Periodic, Bounded),
                            size = (params.Nx, params.Ny, params.Nz),
                            x = (0, params.Lx), y = (0, params.Ly), z = (params.zᵇ, params.zᵗ))
@info grid_base
#---

#+++ Immersed boundary (geometry is extruded in y ⇒ mask ignores y)
cavity_mask(x, y, z) = ifelse((z > bottom_interp(x)) & (z < top_interp(x)), 0, 1)
grid = ImmersedBoundaryGrid(grid_base, GridFittedBoundary(cavity_mask))
@info grid
#---

#+++ Drag (quadratic, all immersed faces, all three components, full speed)
z₁ = minimum_zspacing(grid_base, Center(), Center(), Center())/2
const κᵛᵏ = 0.4
params = (; params..., c_dz = (κᵛᵏ / log(z₁/params.z₀))^2)
@info "Cᴰ =" params.c_dz

@inline _spd(u, v, w) = √(u^2 + v^2 + w^2)
@inline τᵘ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * u * _spd(u, v, w)
@inline τᵛ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * v * _spd(u, v, w)
@inline τʷ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * w * _spd(u, v, w)
τᵘ = FluxBoundaryCondition(τᵘ_drag, field_dependencies=(:u, :v, :w), parameters=(; Cᴰ = params.c_dz))
τᵛ = FluxBoundaryCondition(τᵛ_drag, field_dependencies=(:u, :v, :w), parameters=(; Cᴰ = params.c_dz))
τʷ = FluxBoundaryCondition(τʷ_drag, field_dependencies=(:u, :v, :w), parameters=(; Cᴰ = params.c_dz))
#---

#+++ Ambient two-layer stratification + buoyancy
eos = LinearEquationOfState()
buoyancy_formulation = SeawaterBuoyancy(equation_of_state=eos)

let
    τ_nudge      = 30minutes
    sponge_width = 3000.0
    global params = merge(params, Base.@locals)
end

T₀(x, y, z) = T_profile(z)                 # 3-D signatures for set!
S₀(x, y, z) = S_profile(z)
Tᵉ(y, z, t) = T_profile(z)                 # eastern (shelf-front) open-boundary profile: (y,z,t)
Sᵉ(y, z, t) = S_profile(z)
#---

#+++ Boundary conditions
u_bcs = FieldBoundaryConditions(immersed = τᵘ,
                                east = OpenBoundaryCondition(0,
                                          scheme = PerturbationAdvection(inflow_timescale=10minutes)))
v_bcs = FieldBoundaryConditions(immersed = τᵛ)
w_bcs = FieldBoundaryConditions(immersed = τʷ)

# Melt closure = max(shear, convective) (Wild et al.; scripts/melt_parameterization.jl). The
# convective (Kerr) branch is velocity-independent, so melt is PIG-realistic even at low shear.
# --melt_Cd = shear drag; z₁ (first-cell height) makes the shear u★ resolution-independent.
melt_params = (; Cd = params.melt_Cd, z1 = z₁)
@info "Melt closure = max(shear[Cd], convective[Kerr])" melt_Cd=params.melt_Cd z1=z₁
T_melt = FluxBoundaryCondition(melt_heat_flux_3d, field_dependencies=(:u, :v, :w, :T, :S), parameters=melt_params)
S_melt = FluxBoundaryCondition(melt_salt_flux_3d, field_dependencies=(:u, :v, :w, :T, :S), parameters=melt_params)
T_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = T_melt), east = ValueBoundaryCondition(Tᵉ))
S_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(top = S_melt), east = ValueBoundaryCondition(Sᵉ))
#---

#+++ Shelf-front sponge (warm reservoir + absorber; optional barotropic tide on u)
front_sponge_mask = let xc = params.Lx, w = params.sponge_width
    (x, y, z) -> exp(-(x - xc)^2 / (2 * w^2))
end
T∞(x, y, z, t) = T_profile(z)
S∞(x, y, z, t) = S_profile(z)
utide = let A = params.tide, ω = 2π / (params.tide_period * hours)
    (x, y, z, t) -> A * sin(ω * t)
end
T_sponge = Relaxation(rate = 1/params.τ_nudge, mask = front_sponge_mask, target = T∞)
S_sponge = Relaxation(rate = 1/params.τ_nudge, mask = front_sponge_mask, target = S∞)
u_sponge = Relaxation(rate = 1/params.τ_nudge, mask = front_sponge_mask, target = utide)
#---

#+++ Model (CG Poisson solver for the immersed boundary; QAB2 = one pressure solve/step;
#    AnisotropicMinimumDissipation SGS closure for the 3-D LES unless --nu>0 is given)
coriolis = FPlane(; params.f)
closure = params.nu > 0 ? ScalarDiffusivity(ν=params.nu, κ=params.nu) : AnisotropicMinimumDissipation()
model = NonhydrostaticModel(grid;
                            timestepper = :QuasiAdamsBashforth2,
                            buoyancy = buoyancy_formulation,
                            coriolis,
                            advection = WENO(order=5),
                            tracers = (:T, :S),
                            closure,
                            forcing = (u = u_sponge, T = T_sponge, S = S_sponge),
                            pressure_solver = ConjugateGradientPoissonSolver(grid;
                                reltol = params.cg_reltol, maxiter = params.cg_maxiter),
                            boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs))

# Two-layer ambient + small 3-D noise to seed instabilities.
set!(model, T=T₀, S=S₀, w=(x, y, z) -> 1e-4 * randn())
#---

#+++ Simulation
stop_time = params.stop_time * days
wtl = params.wall_time_limit
simulation = Simulation(model; Δt = 0.5, stop_time,
                        wall_time_limit = isfinite(wtl) ? wtl * 3600 : Inf)
simulation.callbacks[:wizard] = Callback(TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=params.max_dt),
                                         IterationInterval(5))

using Oceanostics.ProgressMessengers
walltime_per_timestep = StepDuration(with_prefix=false)
walltime = Walltime()
progress(sim) = @info (PercentageProgress(with_prefix=false, with_units=false)
                       + walltime + MaxVelocities()
                       + TimeStep() + "CFL = " * AdvectiveCFLNumber(with_prefix=false)
                       + "step dur = " * walltime_per_timestep)(sim)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(40))
#---

#+++ Output setup
u, v, w = model.velocities
T, S = model.tracers
outputs = (; u, v, w, T, S)

outdir = isempty(params.outdir) ? joinpath(rundir, "output") : params.outdir
mkpath(outdir)
ckpt_prefix = params.simname
pickup = params.pickup || any(startswith("$(ckpt_prefix)_iteration"), readdir(outdir))
pickup && @warn "Checkpoint for $(params.simname) found in $outdir — resuming."

using NCDatasets
# FULL 3-D fields — big; write sparingly (hours). Kept on the cluster; not usually fetched.
simulation.output_writers[:fields3d] = NetCDFWriter(model, outputs;
    filename = joinpath(outdir, params.simname),
    schedule = TimeInterval(params.fields3d_interval * hours),
    overwrite_existing = !pickup)

# LIGHT x-z slice at mid-channel (comparable to the 2-D run) — the file to fetch + movie.
jmid = max(1, params.Ny ÷ 2)
simulation.output_writers[:midy] = NetCDFWriter(model, outputs;
    filename = joinpath(outdir, "$(params.simname)_midy"),
    schedule = TimeInterval(params.output_interval * minutes),
    indices = (:, jmid, :),
    overwrite_existing = !pickup)

# LIGHT y-AVERAGE x-z (the mean overturning across the channel). Wrapped defensively so an
# API hiccup in the reduction can never abort a multi-hour GPU run — you still get the 3-D
# field + the midy slice.
try
    ya = fld -> Field(Average(fld, dims=2))
    yavg_outputs = (u = ya(u), v = ya(v), w = ya(w), T = ya(T), S = ya(S))
    simulation.output_writers[:yavg] = NetCDFWriter(model, yavg_outputs;
        filename = joinpath(outdir, "$(params.simname)_yavg"),
        schedule = TimeInterval(params.output_interval * minutes),
        overwrite_existing = !pickup)
catch err
    @warn "y-average output writer disabled (reduction unsupported here); midy slice still written." exception=err
end

simulation.output_writers[:checkpointer] = Checkpointer(model;
    schedule = TimeInterval(params.checkpoint_interval * days),
    dir = outdir, prefix = ckpt_prefix, cleanup = true)
#---

@info "Starting 3-D run..." pickup
run!(simulation; pickup)
@info "Done: $(params.simname)"
