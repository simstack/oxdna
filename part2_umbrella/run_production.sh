#!/bin/bash
# II.3  Production: every run_* folder of FOLDER in parallel, one core each.
# Usage:  OXDNA=/path/to/oxDNA ./run_production.sh FOLDER
# A run folder that already has an energy.dat is skipped, so a resubmission never overwrites results.
# On a cluster, call this from a batch script that requests one core per replica.
FOLDER=${1:?usage: OXDNA=/path/to/oxDNA ./run_production.sh FOLDER}
OXDNA=${OXDNA:?set OXDNA to the oxDNA executable}
cd "$FOLDER" || exit 1
export OMP_NUM_THREADS=1
for d in run_*; do
  if [ -f "$d/energy.dat" ]; then echo "$d: energy.dat exists, skipped"; continue; fi
  ( cd "$d" && "$OXDNA" input > oxdna.log 2>&1
    grep -q "END OF THE SIMULATION" oxdna.log && echo "$d finished" || echo "$d FAILED, see $d/oxdna.log" ) &
done
wait
