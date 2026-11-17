# oxDNA workflow: thermodynamic features of a sticky-end pair (draft)

**Purpose.** For one pair of complementary sticky ends, one on a DNA nanostar arm and one on a
linker, compute with oxDNA:
- **Part I (MD):** the interaction energy between the two partners, split into the oxDNA2 terms and
  by location (sticky duplex, nicks, rest);
- **Part II (umbrella sampling, VMMC):** the free-energy profile of hybridisation, the bound
  fraction against temperature, and melting temperatures corrected for box volume and
  concentration.

Both parts are meant to become SimStack workflows. Each step below is written as one WaNo: its
inputs, what runs, its outputs, and the check that must pass before the next step.

## Input parameters (one file per system, e.g. `system.yaml`)

| parameter | example |
|---|---|
| name | `YLs6` |
| nanostar and linker structure (oxDNA `.top` / `.dat`), partners separated | `YLs6.top`, `YLs6.dat` |
| sticky-end sequences of both partners, and optional mutations | `GAGAGT` / `CGATTGACTCTC` |
| temperature of interest | 14.3 C |
| salt | 0.118 M Na+ |
| sticky-end concentration (for the bulk correction, Part II) | 0.6 mM each |
| run lengths, replicas | defaults below |
| oxDNA executable and sequence-parameter file | path, `oxDNA2_sequence_dependent_parameters.txt` |

**How the starting structure is made (manual step, outside the workflow).** The nanostar and linker
structure is currently built by hand: designed in cadnano, converted to oxDNA, and rearranged in
oxView so that the two sticky ends face each other without being paired. No available tool does
this reliably for the user, so the workflow takes the finished `.top/.dat` as input. Automating it
would be a separate project.

The designed base pairs between the two sticky ends follow from the topology and the sequences.
Every step that needs them (traps, order parameters, energy split) derives them from there.

## Part I: energies from MD

**I.1 Inputs.** The parameter file and the nanostar/linker structure with the partners not yet
joined.

**I.2 Fusion MD.** Join the partners with `mutual_trap` forces on every designed sticky-end pair,
in both directions (stiff 0.2, r0 1.2). The trap file is generated from the designed pairs.
CUDA MD, DNA2 with sequence dependence, 1e7 steps (currently run at 37 C).
*Output:* fused `.top/.dat`. *Check:* all designed pairs formed in the final configuration.

**I.3 Production MD.** Traps removed, temperature and salt from the parameters, 1e5 equilibration
+ 5e7 steps, dt 0.003, John thermostat. Frames every 5e4 steps. Live observables every 1e3 steps:
split potential energy, designed pairs formed (`hb_list`), and the branch-migration pair at the
junction. The nick contacts come from the pair energies of step I.4. A mutant uses the same
configuration with changed bases in the topology, and the mismatched pair is dropped from the
pair list.
*Check:* run finished, sequence-parameter file read, no unused input keys, first 10 % discarded as
equilibration, standard errors from the autocorrelation time.

**I.4 Energy extraction.** DNAnalysis writes the energy of every interacting pair, split into the
eight oxDNA2 terms, for each saved frame after the cut. The binding energy `E_bind` is the sum over
all pairs between the two partners, split by term (H-bonding, stacking, cross-stacking, coaxial
stacking, electrostatics, ...) and by location (sticky duplex, the two nicks, other contacts).
*Check:* the summed terms reproduce the run's own split energy in every frame.
*Output:* table of `E_bind` and its parts, mean +- standard error, plus pair occupancies and nick
stacking fractions.

## Part II: free energies from umbrella sampling

**II.1 Inputs.** A fragment cut from the fused structure: the sticky end plus 12 bp of duplex on
each partner, keeping both nicks and the junction region. The second partner is translated rigidly
away, so every run starts unbound. Box 30 oxDNA units.
Order parameter: number of designed pairs formed, plus a minimum-distance coordinate with an
interface at 1.5 (separates "unpaired but in contact" from "far apart").
Initial weights for each state (n, dist), from a reference system.
*Check:* fragment backbone continuous, partners separated by more than 1.5 at the start.

**II.2 Weight tuning (loop).** Repeat:
1. short VMMC rounds (CPU, energy line every 10 steps, 1e4 equilibration, 1e5-2e6 steps, 1-3
   replicas);
2. weight update: histogram of visited states, each entry weighted with 1/w, normalised per
   replica and averaged; suggested new weight 1/p, rescaled to the old scale and capped at a factor
   20 up or down; unvisited states keep their weight and are flagged;
3. the table is shown, single weights can be overridden by hand, and the weights, energies,
   histograms and logs of every round are archived.

*Stop:* every state visited, with full unbound <-> bound transitions in every replica.

**II.3 Production VMMC.** Fixed final weights, 6 replicas x 4e6 steps with different seeds,
temperature and salt from the parameters, histograms reweighted to 0-80 C in 2 C steps.
*Check:* number of full transitions per replica, every replica unbinds, no unvisited state, replica
spread of ln(phi). The oxDNA 3.6.0 release must not be used for VMMC with DNA2.

**II.4 Analysis.** From the unbiased histogram columns, per replica, then mean +- standard error:
- bound/unbound ratio phi(T), box bound fraction f = phi / (1 + phi);
- correction to concentration c: Phi = phi c / c_box, then f / (1 - f)^2 = Phi
  (c_box = 1/(N_A V), 0.1 mM for a 30-unit box);
- T(f = 0.5), and T at any target fraction (1/sqrt(2) = 0.7071 is the gel point for
  three- and two-armed partners), flagged if more than 6 C from the simulation temperature;
- free-energy profile F(n), the dissociation barrier F(1) - F(bound), the mean first-passage time
  of unbinding, and ln K.

*Output:* one table per system with these quantities and their errors, plus F(n).

## SimStack mapping (first sketch)

| WaNo | step | wraps | runs |
|---|---|---|---|
| `Fuse` | I.2 | trap file from the designed pairs + oxDNA MD | GPU |
| `ProductionMD` | I.3 | input from template + oxDNA MD + checks | GPU |
| `PairEnergies` | I.4 | DNAnalysis + decomposition | CPU |
| `Fragment` | II.1 | cut, unbound start, order parameter, initial weights | local |
| `UmbrellaRound` + `UpdateWeights` | II.2 | VMMC round, weight update | CPU, loop |
| `UmbrellaProduction` | II.3 | VMMC replicas + checks | CPU, parallel |
| `UmbrellaAnalysis` | II.4 | histograms -> fractions, T, F(n) | local |

A ForEach over systems runs the same chain for each sticky-end pair or mutant.

**Open questions for the SimStack team:**
1. Can a loop repeat until a convergence flag is set (weight tuning, II.2), or does each round have
   to be launched separately?
2. How does a manual step fit in (reviewing and overriding the suggested weights)? One option is a
   WaNo that writes the suggested weights and waits for an accepted weights file.
3. Can the starting structure (I.1) be passed in as a user-provided file, since it is built by hand?

## Notes for the implementation

- **Parameters, not constants:** sequences, temperature, salt, concentration and run lengths come
  only from the parameter file.
- **Recorded versions:** oxDNA git commit and parameter-file checksum written with every output.
- **Checks write a small `qc.json`** that the next WaNo reads, so a failed run stops the chain.
- **Known pitfalls:** the input key is `seq_dep_file_DNA` in some oxDNA versions and `seq_dep_file`
  in others; the live `hb_energy` observable counts each pair twice; oxDNA 3.6.0 VMMC with DNA2 is
  wrong (fixed in later versions).
- **NUPACK** (bound fraction and T_m of the free sticky ends) could be a separate, optional workflow
  if its licence allows; it takes seconds locally, so it is not needed here.

## Code folder

The scripts in this folder implement the steps above; `README.md` lists the commands and outputs.

| step | script |
|---|---|
| I.2 | `part1_md/prepare_fusion.py` |
| I.3 | `part1_md/prepare_md.py`, `part1_md/check_md.py` |
| I.4 | `part1_md/pair_energies.py`, `part1_md/decompose.py` |
| II.1 | `part2_umbrella/prepare_fragment.py` |
| II.2 | `part2_umbrella/run_round.sh`, `part2_umbrella/update_weights.py` |
| II.3 | `part2_umbrella/prepare_production.py`, `part2_umbrella/run_production.sh`, `part2_umbrella/check_umbrella.py` |
| II.4 | `part2_umbrella/analyse_umbrella.py` |

Each script takes `system.yaml` and a run folder and writes its outputs and a `qc.json`, so it can
become a WaNo command unchanged.
