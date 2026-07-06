#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N iceshelfcavity
#PBS -o logs/iceshelfcavity.log
#PBS -e logs/iceshelfcavity.log
#PBS -l walltime=11:59:00
#PBS -q casper
#PBS -l select=1:ncpus=4:ngpus=1:gpu_type=a100
#PBS -M kenzhao@unc.edu
#PBS -m ae
#PBS -r n

# Ice-shelf-cavity LES (Oceananigans 0.109, CG Poisson solver). GPU production run.
#     qsub submit_pbs.sh                                    # Pine Island, production res
#     qsub -v SIM=coarse,DZ=4 submit_pbs.sh                 # quick GPU shakedown (~0.75 M cells)
#     qsub -v BATHY=othercavity.csv,SIM=other submit_pbs.sh # a different transect (add CSV to data/)
#     qsub -v JULIA=/path/to/julia submit_pbs.sh
#
# Two-phase spin-up → production (SAME SIM ⇒ the driver auto-resumes from its checkpoint):
#     qsub -v STOP=1 submit_pbs.sh              # phase 1: cheap 1-day spin-up, builds a checkpoint
#     qsub submit_pbs.sh                        # phase 2: resume, run to STOP=10 days
#     qsub -v STOP=15,TIDE=0.05 submit_pbs.sh   # phase 3: add an M2 tide, continue to 15 days
# (Each job also stops+checkpoints at the wall-time limit, so re-submitting always continues.)
#
# GPU-scale domain: Pine Island transect (67 km × ~0.74 km) at aspect=4, Δz=2 m → ~3 M cells,
# comfortable on an 80 GB A100. Drop to DZ=4 (~0.75 M cells) for a fast end-to-end shakedown.
#
# CUDA.jl bundles its own toolkit and uses the GPU node's driver (libcuda), so NO `module load
# cuda` is needed on Casper. If CUDA.jl ever can't find the driver, `module load cuda` (check the
# exact name with `module avail cuda`) — but do NOT force-purge / pin an ncarenv version.
#
# Run ./setup_casper.sh once on the cluster first (instantiates the env). Keep the Julia depot
# AND outputs on /glade/work (big quota), OFF the git repo.
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
JULIA="${JULIA:-julia}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ; Julia: $($JULIA --version 2>/dev/null)"

SIM="${SIM:-iceshelfcavity}"
DZ="${DZ:-2}"
STOP="${STOP:-10}"                 # stop time in DAYS (override for a spin-up phase, e.g. STOP=1)
TIDE="${TIDE:-0.0}"                # barotropic tide amplitude [m/s] at the shelf front (0 = off)
TIDE_PERIOD="${TIDE_PERIOD:-12.42}"
MELT_CD="${MELT_CD:-0.0022}"       # shear drag for the Wild et al. max(shear,convective) melt closure
BATHY="${BATHY:-pineislandbath.csv}"
OUTDIR="${OUTDIR:-/glade/work/$USER/cavity_runs}"
mkdir -p "$OUTDIR" logs

# STOP-day run; 30-min output; checkpoint every 0.5 d. The driver AUTO-RESUMES from the latest
# checkpoint in OUTDIR, so simply re-submitting this job continues where it left off.
#   --animate=false     : no inline CairoMakie on the GPU node (movies made post-hoc, see postprocess.sh)
#   --wall_time_limit   : stop + checkpoint at 11.5 h, before the 11:59 PBS walltime kills the job
time $JULIA --project --pkgimages=no iceshelfcavity.jl \
    --arch=gpu --aspect=4 --dz="$DZ" --bathymetry="$BATHY" \
    --stop_time="$STOP" --output_interval=30 --checkpoint_interval=0.5 --max_dt=5 \
    --tide="$TIDE" --tide_period="$TIDE_PERIOD" --melt_Cd="$MELT_CD" \
    --cg_reltol=1e-5 --cg_maxiter=30 --animate=false \
    --wall_time_limit=11.5 --outdir="$OUTDIR" --simname="$SIM" \
    2>&1 | tee logs/${SIM}.out

qstat -f $PBS_JOBID >> logs/iceshelfcavity.log
