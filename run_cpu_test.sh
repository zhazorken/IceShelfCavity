#!/bin/bash -l
# =====================================================================================
#  One-command CPU test for the ice-shelf-cavity model.
#
#    ./run_cpu_test.sh setup     # one-time: instantiate + precompile the Julia env
#    ./run_cpu_test.sh           # quick smoke test  (~3.9k cells, ~1-2 min after precompile)
#    ./run_cpu_test.sh check     # finer physics sanity (~21k cells, a few minutes)
#
#  Override the Julia binary with e.g.  JULIA=~/bin/julia-1.10.9/bin/julia ./run_cpu_test.sh
# =====================================================================================
set -e
cd "$(dirname "$0")"
JULIA=${JULIA:-julia}
mode=${1:-smoke}

# --- guard against the wrong Julia (the #1 setup failure) --------------------------------
check_julia_version() {
    local v major minor
    v=$("$JULIA" --version 2>/dev/null | awk '{print $NF}')
    major=${v%%.*}; minor=$(printf '%s' "$v" | cut -d. -f2)
    if [ -z "$major" ]; then
        echo "WARN: could not detect a Julia version from '$JULIA --version'. Is Julia installed / on PATH?"
        return
    fi
    if [ "$major" -lt 1 ] || { [ "$major" -eq 1 ] && [ "$minor" -lt 10 ]; }; then
        echo "ERROR: Julia $v detected ($JULIA). This model needs Julia >= 1.10 (>= 1.11 to use the"
        echo "       shipped Manifest.toml as-is, which was resolved with 1.12.6)."
        echo "  Install a modern Julia and make it the default (juliaup):"
        echo "     curl -fsSL https://install.julialang.org | sh    # then restart your shell"
        echo "     juliaup add 1.12 && juliaup default 1.12"
        echo "     julia --version     # must now be >= 1.11"
        echo "  Then re-run:  ./run_cpu_test.sh setup"
        exit 1
    fi
    # Manifest.toml is the tested 1.10.11 CG stack from Ovall26/newLES_cg (Oceananigans 0.109.2),
    # so it instantiates directly on Julia 1.10. (Manifest.shipped.toml is the older 1.12 backup.)
}
check_julia_version

if [ "$mode" = "setup" ]; then
    echo ">> instantiating + precompiling (first time can take 10-30 min)"
    "$JULIA" --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
    echo ">> environment ready. Now run:  ./run_cpu_test.sh"
    exit 0
fi

if [ "$mode" = "check" ]; then
    SIM=cpucheck; DZ=20; ASP=6; STOP=0.1; OUT=15      # ~21k cells
else
    SIM=cputest;  DZ=40; ASP=8; STOP=0.02; OUT=5      # ~3.9k cells
fi

echo ">> CPU $mode run: dz=$DZ  aspect=$ASP  stop_time=$STOP d  (simname=$SIM)"
"$JULIA" --project iceshelfcavity.jl --arch=cpu --dz=$DZ --aspect=$ASP \
         --stop_time=$STOP --output_interval=$OUT --max_dt=5 --simname=$SIM

PYHINT="install deps with:  python3 -m pip install numpy pandas xarray netcdf4 matplotlib"
echo ">> section plot"
python3 plot_cavity.py "$SIM" || echo "   (plot skipped — $PYHINT)"

echo ">> melt-flux sign check"
python3 check_melt_sign.py "$SIM" || echo "   (sign check skipped — $PYHINT)"

echo ">> done.  Open  output/${SIM}_section.png"
echo "   Check: (1) it stepped with CFL < 0.5,  (2) two-layer T/S + geometry look right,"
echo "          (3) the melt-flux sign check above says PASS."
