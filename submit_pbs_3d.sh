#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N iceshelfcavity3d
#PBS -o logs/iceshelfcavity3d.log
#PBS -e logs/iceshelfcavity3d.log
#PBS -l walltime=11:59:00
#PBS -q casper
#PBS -l select=1:ncpus=4:ngpus=1:gpu_type=a100
#PBS -M kenzhao@unc.edu
#PBS -m ae
#PBS -r n

# 3-D ice-shelf-cavity LES: the 2-D Pine Island cavity EXTRUDED ~10 km in a periodic y.
#     qsub submit_pbs_3d.sh                          # dz=8, Ly=10 km  (~60 M cells — an 80 GB A100)
#     qsub -v SIM=shake,DZ=16,LY=6 submit_pbs_3d.sh  # coarse shakedown (a few M cells)
#     qsub -v STOP=1 submit_pbs_3d.sh                # 1-day spin-up (auto-resumes on re-submit)
#     qsub -v TIDE=0.05 submit_pbs_3d.sh             # add an M2 barotropic tide
#     qsub -v JULIA=$HOME/.juliaup/bin/julia submit_pbs_3d.sh
#
# A 3-D LES of a 67 km × 10 km × ~0.74 km cavity is huge, so the boundary layer is coarser
# than the 2-D run: dz=8 m (Δx=Δy=32 m) → ~60 M cells, comfortable on an 80 GB A100. The
# driver PRINTS the cell count and a rough memory estimate at startup — check it before the
# big job. To refine, lower --dz/--dy but shrink --Ly or expect a multi-GPU / OOM situation.
# 3-D field files are large and written every --fields3d_interval hours; the *_midy and
# *_yavg x-z slices are the light files to fetch and plot.

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
JULIA="${JULIA:-julia}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ; Julia: $($JULIA --version 2>/dev/null)"

SIM="${SIM:-iceshelfcavity3d}"
DZ="${DZ:-10}"                     # vertical spacing (m); Δx=Δy = ASPECT·DZ
ASPECT="${ASPECT:-4}"              # horizontal:vertical aspect. Raise it to refine Δz WITHOUT adding
                                   # horizontal cells (e.g. DZ=5,ASPECT=8 keeps Δx=Δy=40 m, halves Δz)
LY="${LY:-10}"                     # periodic y extent in km
Z0="${Z0:-0.1}"                    # momentum roughness (m); lower ⇒ weaker drag ⇒ faster plume
DY="${DY:-0}"                      # y spacing [m]; 0 = isotropic horizontal (Δy = Δx)
STOP="${STOP:-5}"                  # stop time in DAYS
MAXDT="${MAXDT:-60}"               # max time step [s]. Buoyancy spin-up is slow, so 5 s wastes
                                   # steps (CFL~1e-4); 60 s lets the wizard run near CFL~0.3-0.4
                                   # once the flow develops. Lower it if you see CFL near 0.5.
TIDE="${TIDE:-0.0}"
TIDE_PERIOD="${TIDE_PERIOD:-12.42}"
MELT_CD="${MELT_CD:-0.0022}"       # shear drag for the melt closure (Wild et al.); melt is
                                   # max(shear[this Cd], convective[Kerr]) — PIG-realistic
BATHY="${BATHY:-pineislandbath.csv}"
OUTDIR="${OUTDIR:-/glade/work/$USER/cavity_runs}"
mkdir -p "$OUTDIR" logs

time $JULIA --project --pkgimages=no iceshelfcavity3d.jl \
    --arch=gpu --aspect="$ASPECT" --dz="$DZ" --Ly="$LY" --dy="$DY" --z0="$Z0" --bathymetry="$BATHY" \
    --stop_time="$STOP" --output_interval=30 --fields3d_interval=6 --checkpoint_interval=0.5 --max_dt="$MAXDT" \
    --tide="$TIDE" --tide_period="$TIDE_PERIOD" --melt_Cd="$MELT_CD" \
    --cg_reltol=1e-5 --cg_maxiter=30 \
    --wall_time_limit=11.5 --outdir="$OUTDIR" --simname="$SIM" \
    2>&1 | tee logs/${SIM}.out

qstat -f $PBS_JOBID >> logs/iceshelfcavity3d.log
