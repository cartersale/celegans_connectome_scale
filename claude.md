# connectome_resolutions

Analysis code for a study of the resolution at which a connectome stops predicting signal
propagation. README.md has the layout, setup, data sources, and the map from each manuscript
figure and analysis to the script that produces it. Read it before working on the code.

## Layout and naming

- `src/conres/`: library (installed as `conres`). Data loaders, path operator, quotient,
  observables, partitions, scoring helpers.
- `scripts/`: run from the repo root (`python scripts/<name>.py`).
  - Panel scripts are named after the manuscript panel they draw: `figN<panel>_<what>.py`
    (`fig1b_roc.py` draws Fig. 1B) and write SVGs to `figures/figN/`.
  - Analysis scripts have descriptive names and write `results/<script name>*.csv|npz|log`.
  - `scripts/diagnostics/`: standalone checks that feed no table or figure. Each calls
    `conres.paths.tee_stdout(__file__)`, so its printed output is saved to
    `results/diagnostics/<name>.log`.
- `tests/`: regression tests (`pytest tests/`).
- `preregistration/`: the FlyWire partition rule. Never edit the rule; deviations go in its
  dated Amendments section, with the reason.
- `data/`, `results/`, `figures/` and `snapshots/` are not tracked. `data_manifest.sha256`
  lists the expected input files.
- PNG previews of figure panels are written only when `CONRES_CHECK_DIR` is set.

## Conventions the code relies on

- `A[i, j]` = synapse count from presynaptic i to postsynaptic j. `build_matrix("all")` sums
  chemical and electrical contacts as listed. The Cook release lists gap junctions in both
  orientations; Witvliet lists each once.
- Cook node set: 302 nodes = 300 neurons + the muscles `anal` and `sph`, in 119 classes.
  Witvliet common set: 166 cells = 161 neurons + 5 glia (`exclude_glia=True` drops them).
- `occ3[responder, stimulated]` counts stimulations of the stimulated neuron while the
  responder was identified; a pair is observed when it is > 0. "Unsynapsed" means
  `A[stim, resp] == 0`.
- Path operator: `randi.resolvent`, alpha = 0.1 everywhere, normalised by the spectral radius.
  The class-level operator `randi.class_leverage` uses the quotient's own spectral radius.
- Scores are performance above chance (AUC − 0.5, or Spearman rho). Losses are fractions of
  the full operator's own headroom.
- Size-matched random partitions are label permutations of the partition being tested.
- Cluster bootstrap resamples the cell class of the stimulated neuron
  (`observables.boot_indices`).
- Before reading any rank or overlap metric on sparse binary data, run `conres.metric_guard`
  (tie blocks, base rate, chance floor).

## Reproducibility

- Every reported number comes from a file in `results/` or from a script run.
- A script's defaults are the settings of the reported run. Where a reported run needs other
  flags, the README gives them.
- A script that depends on another script's output reads the result file. Do not copy a
  value from one script into another as a literal.
- When a change moves a number, rerun the script and diff the new outputs against the old
  ones to confirm that only the intended values moved.
- Run checks and comparisons in a scratch copy of the repo (symlink `data/`, copy
  `results/`), so that an exploratory run never overwrites a reported output.

## Documentation style

Code, comments and docs describe what is computed and how to reproduce it.

- Describe the code as it is. Do not mention earlier implementations, what was changed or
  by whom, reviews, conversations, or approaches that were not taken.
- Refer only to files, scripts, flags and outputs that exist in the repository.
- Keep scientific interpretation out of the code. Docstrings, comments and printed output
  state what is measured and how. They do not say what a result means, argue for the study
  design, or quote outcome numbers. Interpretation belongs in the manuscript.
- Docstrings say what a component does, what it reads, what it writes, its parameters and
  its assumptions. Script docstrings use short `Inputs:` / `Outputs:` / `Usage:` blocks.
- Comments explain only what is not obvious from the code: the reason for a non-obvious
  choice, invariants, mathematics, edge cases, external constraints, units, coordinate
  systems and data-format quirks. Do not translate code into English. Prefer deleting a
  comment to adding clutter.

      Bad:  # Previously this used cosine similarity, but we changed it to correlation.
      Bad:  # Classes survive here, which shows the fine structure is redundant.
      Good: # Fisher-z averaging avoids downward bias when aggregating correlations.

- README describes the current workflow only.
- Keep provenance: data sources, the preregistration, and the script that produces each
  reported result.
