#!/bin/bash -l
# postprocess.sh — run ON CASPER (login node or a small job) to make quick-look PNGs + movies
# from the run outputs, WHERE THE DATA LIVES (so you never move big files just to plot).
#
#   ./postprocess.sh                              # default run, from $OUTDIR
#   OUTDIR=/glade/derecho/scratch/$USER/cavity ./postprocess.sh
#   ./postprocess.sh iceshelfcavity coarse        # specific runs
#
# Needs Python with xarray/netCDF4/matplotlib (+ffmpeg for mp4). NCAR provides these via conda:
set -u
cd "$(dirname "$0")"
OUTDIR="${OUTDIR:-/glade/work/$USER/cavity_runs}"

module load conda 2>/dev/null && conda activate npl 2>/dev/null || \
    echo "(couldn't load conda/npl — assuming a python with xarray/matplotlib is already on PATH)"
PY="${PY:-python3}"

RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(iceshelfcavity)
for name in "${RUNS[@]}"; do
    echo "=== $name ==="
    # 3-D run writes x-z slices (<name>_midy, <name>_yavg); plot those, not the 3-D field file.
    # 2-D run writes just <name>.nc.
    if [ -f "$OUTDIR/${name}_midy.nc" ]; then
        TARGETS=("${name}_midy" "${name}_yavg")
    else
        TARGETS=("$name")
    fi
    for t in "${TARGETS[@]}"; do
        [ -f "$OUTDIR/${t}.nc" ] || continue
        echo "  -- $t"
        $PY plot_cavity.py       "$t" --dir "$OUTDIR" || echo "     (section plot skipped)"
        $PY plot_melt_rate.py    "$t" --dir "$OUTDIR" || echo "     (melt-rate diagnostic skipped)"
        $PY make_cavity_movie.py "$t" --dir "$OUTDIR" || echo "     (movie skipped)"
    done
    $PY check_melt_sign.py "${TARGETS[0]}" --dir "$OUTDIR" || echo "  (sign check skipped)"
done
echo "Done. PNGs + movies are in $OUTDIR (fetch them to your laptop with fetch_results.sh)."
