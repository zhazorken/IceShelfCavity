#!/bin/bash
# fetch_results.sh — run ON YOUR LAPTOP to pull the LIGHT cavity results from Casper into
# ./output: section PNGs/PDFs, movies, and the small x-z slice files (_midy, _yavg). It does
# NOT pull the big field files by default (the 2-D <sim>.nc or the 3-D <sim>.nc can be tens–
# hundreds of GB) — postprocess.sh already turns those into light products on the cluster.
#
#   ./fetch_results.sh
#   REMOTE=kenzhao@data-access.ucar.edu:/glade/work/kenzhao/cavity_runs ./fetch_results.sh
#   SIM=iceshelfcavity ./fetch_results.sh          # just one run's files
#   FULL=1 ./fetch_results.sh                       # ALSO pull the big *.nc field files (careful)
#
# For very large / many-file transfers, prefer Globus over rsync.
set -u
REMOTE="${REMOTE:-kenzhao@data-access.ucar.edu:/glade/work/kenzhao/cavity_runs}"
DEST="${DEST:-output}"
SIM="${SIM:-}"
FULL="${FULL:-0}"
mkdir -p "$DEST"

P="${SIM:-}"; [ -n "$P" ] || P=""
INC=(--include="${P}*_midy.nc" --include="${P}*_yavg.nc" \
     --include="${P}*_section.png" --include="${P}*_section.pdf" \
     --include="${P}*.mp4" --include="${P}*.gif")
[ "$FULL" = "1" ] && INC+=(--include="${P}*.nc")

rsync -avh --progress "${INC[@]}" --exclude='*' "$REMOTE/" "$DEST/"

echo "Pulled light outputs to $DEST/. (Use FULL=1 to also pull the big *.nc field files.)"
echo "Re-plot locally with:  python3 plot_cavity.py <simname>[_midy] --dir $DEST"
