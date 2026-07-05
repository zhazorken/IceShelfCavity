#!/bin/bash -l
# setup_casper.sh — one-time environment setup on Casper (NCAR), run from the repo directory.
# Resolves + precompiles the Oceananigans 0.109 Julia environment for the GPU cavity runs.
#
#   git clone <your-repo-url> IceShelfCavity && cd IceShelfCavity
#   JULIA=/path/to/julia ./setup_casper.sh
#   qsub submit_pbs.sh
set -e
cd "$(dirname "$0")"

# Ensure the PBS log directory exists BEFORE any qsub — the #PBS -o/-e directives write here at
# job launch, and a missing logs/ leaves the job stuck in the Held (H) state. (git doesn't clone
# empty dirs, so a fresh clone may not have it despite logs/.gitkeep on some setups.)
mkdir -p logs

# Keep the Julia depot on /glade/work ($HOME quota is small). Instantiating needs NO HPC modules
# (CUDA.jl bundles its own toolkit; the GPU driver is only needed at run time), so this just
# needs a working `julia`. If `which julia` is empty, install juliaup first:
#   curl -fsSL https://install.julialang.org | sh -s -- --yes && source ~/.bashrc
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/glade/work/$USER/.julia}"
JULIA="${JULIA:-julia}"
command -v "$JULIA" >/dev/null 2>&1 || { echo "ERROR: '$JULIA' not found — install Julia (juliaup) then re-run."; exit 1; }
echo "Julia:  $($JULIA --version)   depot: $JULIA_DEPOT_PATH"

# No Manifest is committed (it's .gitignored), so this resolves the Oceananigans 0.109 stack
# fresh for the cluster's Julia — Project.toml pins keep it on the tested majors (0.109 / 0.16).
$JULIA --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo
echo "Environment ready. Edit account/walltime in submit_pbs.sh, then submit:"
echo "    qsub submit_pbs.sh"
echo "Quick GPU sanity check first (a few minutes, coarse):"
echo "    $JULIA --project iceshelfcavity.jl --arch=gpu --dz=8 --stop_time=0.2 --animate=false --simname=gpucheck"
