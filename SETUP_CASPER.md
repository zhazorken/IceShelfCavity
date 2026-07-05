# Getting the cavity runs onto Casper (NCAR) via GitHub

The code is self-contained (geometry CSV in `data/`, ambient profile baked into
`scripts/stratification.jl`), so a small GitHub repo is the easiest transfer. No Manifest,
outputs, or checkpoints are committed — the environment resolves fresh on the cluster.

## 1. Create the repo and push (from your laptop, in this folder)

`.gitignore` is already set up (outputs, checkpoints, `Manifest.toml`, movies are excluded).
Create an **empty** repo on GitHub (no README), then:

```bash
cd ~/Desktop/IGStalk26/IceShelfCavity
git init && git add -A && git commit -m "Ice-shelf-cavity LES (Oceananigans 0.109, CG solver)"
git branch -M main
git remote add origin git@github.com:<you>/IceShelfCavity.git    # or the https URL
git push -u origin main
```

(Or with the GitHub CLI, after `git init`/commit: `gh repo create IceShelfCavity --private --source=. --push`.)

## 2. On Casper: clone and set up the environment (once)

```bash
# from a login node — keep code + Julia depot on /glade/work (big, writable)
cd /glade/work/$USER
git clone git@github.com:<you>/IceShelfCavity.git
cd IceShelfCavity

# point JULIA at your Julia (juliaup: $HOME/.juliaup/bin/julia), then:
JULIA=/path/to/julia ./setup_casper.sh      # Pkg.instantiate + precompile (resolves 0.109 fresh)
```

Do this on a login node (or interactive session) — it needs network for `Pkg`.

## 3. Submit the production run

Edit the `submit_pbs.sh` header if needed (`#PBS -A <account>`, walltime), then:

```bash
qsub submit_pbs.sh                                  # Pine Island, aspect=4 Δz=2 m (~3 M cells)
qsub -v SIM=coarse,DZ=4 submit_pbs.sh               # quick GPU shakedown (~0.75 M cells)
qsub -v JULIA=$HOME/.juliaup/bin/julia submit_pbs.sh
```

One A100, 67 km × ~0.74 km domain. The job runs 10 model-days or stops at 11.5 h wall time,
writing a checkpoint. It **auto-resumes from the latest checkpoint in `OUTDIR`** — just
re-submit the same command to continue. Watch `logs/<sim>.out`.

## 4. Post-process on the cluster, then fetch the light files

Outputs + checkpoints land in **`/glade/work/$USER/cavity_runs/`** (set by `OUTDIR` in
`submit_pbs.sh`, off the git repo). Make plots/movies where the data lives, then pull them home:

```bash
# on Casper
./postprocess.sh iceshelfcavity        # section PNG + melt-sign check + mp4, into OUTDIR

# on your laptop, from the repo folder
./fetch_results.sh                     # rsync *.nc / *.png / *.mp4 into ./output
python3 plot_cavity.py iceshelfcavity --dir output
```

## Notes / gotchas

- **Julia**: NCAR doesn't always ship a Julia module; installing via
  [juliaup](https://github.com/JuliaLang/juliaup) into `/glade/work/$USER` is simplest. Use
  Julia ≥ 1.10 (1.10.11 is what the tested local Manifest used).
- **Depot on /glade/work**: `$HOME` has a small quota; `setup_casper.sh` and `submit_pbs.sh`
  default `JULIA_DEPOT_PATH=/glade/work/$USER/.julia`.
- **No `module load cuda` needed**: CUDA.jl bundles its own toolkit and uses the node driver.
- **First GPU compile is slow** (CUDA kernels); `--pkgimages=no` in `submit_pbs.sh` avoids a
  known cluster precompile issue.
- **Output size**: the 2-D cavity fields file grows with run length × resolution. If it gets
  unwieldy, raise `--output_interval` in `submit_pbs.sh`, or `ncks`-subset on GLADE before
  fetching.
- **Reproducibility**: `Manifest.toml` is gitignored so laptop/cluster Julia versions don't
  clash. To pin exactly later, commit the cluster-resolved Manifest on a branch.
