#!/bin/bash
# II.2  One weight-tuning round: every run_* folder of FOLDER in parallel, one core each.
# Usage:  OXDNA=/path/to/oxDNA ./run_round.sh FOLDER
#         STEPS=200000 REPLICAS="1 2" FORCE=1 OXDNA=... ./run_round.sh FOLDER     (optional settings)
# STEPS overrides the step count in the inputs; REPLICAS selects run folders (default: all).
# A run folder with an unprocessed energy.dat stops the round (run update_weights.py first);
# FORCE=1 renames that output to *_discarded_<date>.* and runs anyway.
# Every round starts from fragment_unbound.dat, which oxDNA never overwrites.
set -e
FOLDER=${1:?usage: OXDNA=/path/to/oxDNA ./run_round.sh FOLDER}
OXDNA=${OXDNA:?set OXDNA to the oxDNA executable}
cd "$FOLDER"
REPLICAS=${REPLICAS:-$(ls -d run_* | sed 's/run_//' | sort -n | tr '\n' ' ')}
export OMP_NUM_THREADS=1
for r in $REPLICAS; do
  d=run_$r
  if [ -f "$d/energy.dat" ]; then
    if [ "$FORCE" = "1" ]; then
      stamp=$(date +%Y%m%d_%H%M%S)
      for f in energy.dat last_hist.dat oxdna.log last_conf.dat; do
        [ -f "$d/$f" ] && mv "$d/$f" "$d/${f%.*}_discarded_$stamp.${f##*.}"
      done
    else
      echo "$d has an unprocessed energy.dat: run update_weights.py $FOLDER first (or FORCE=1)"; exit 1
    fi
  fi
  [ -n "$STEPS" ] && sed -i "s/^steps = .*/steps = $STEPS/" "$d/input"
  ( cd "$d" && "$OXDNA" input > oxdna.log 2>&1
    grep -q "END OF THE SIMULATION" oxdna.log && echo "$d finished" || echo "$d FAILED, see $d/oxdna.log" ) &
done
wait
