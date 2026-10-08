# connectome_resolutions

Code for the manuscript in `docs/main.tex` (supplement: `docs/supplementary_methods_draft.tex`):
at what resolution does a connectome stop predicting signal propagation? README.md has the
layout, setup, and the map from every manuscript figure, analysis and supplementary table to
its script. Read it before working on the code.

## Layout and naming

- `src/conres/`: library (installed as `conres`). Data loaders, path operator, quotient,
  observables, partitions, scoring helpers.
- `scripts/`: run from the repo root (`python scripts/<name>.py`).
  - Panel scripts are named after the manuscript panel they draw: `figN<panel>_<what>.py`
    (`fig1b_roc.py` draws Fig. 1B). Panels are SVGs in `figures/figN/`, assembled by hand in
    `figures/figN/figure_N.ai`.
  - Analysis scripts have descriptive names and write `results/<script name>*.csv|npz|log`.
  - `scripts/diagnostics/`: standalone checks that support statements but feed no table or
    figure. Each calls `conres.paths.tee_stdout(__file__)`, so its printed output is saved to
    `results/diagnostics/<name>.log`.
- `archive/`: superseded or unused scripts and figures. It is gitignored, unmaintained and not
  part of the pipeline. If the manuscript turns out to cite a result produced only there,
  restore the script to `scripts/` with fixed imports and a descriptive name.
- `microns/`: exploratory MICrONS analyses, not in the manuscript.
- `preregistration/`: the FlyWire partition rule. Never edit the rule; deviations go in its
  dated Amendments section, with the reason.
- `data/`, `results/`, `figures/`, `docs/` are not tracked.
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
- Before interpreting any rank or overlap metric on sparse binary data, run
  `conres.metric_guard` (tie blocks, base rate, chance floor).

## Working on the manuscript

- Every number in the text must come from a file in `results/` or from a script run. Verify
  against the code; never trust old notes or prose.
- Mark revisions in `main.tex` and the supplement with `\rev{...}` (red).
- Supplementary tables name their source in `% Source: scripts/... -> results/...` comments.
  Keep these in sync when scripts or outputs are renamed.
- When a code fix changes a number, rerun the script, diff the new outputs against the old
  ones to confirm only the intended values moved, then update every place the number appears
  (main text, Methods, supplement, captions).
- For checks and comparisons, run in a scratch copy of the repo (symlink `data/`, copy
  `results/`). That way published outputs are never overwritten by an exploratory run.
- `main.tex` figures are not in `docs/`. Syntax-check with
  `pdflatex "\PassOptionsToPackage{draft}{graphicx}\input{main.tex}"`.

## Documentation style

Document the code as it exists now. Git history is for history.

- Never refer in comments or docs to previous implementations, earlier attempts, what was
  changed (by anyone, including Claude), conversations or prompts, or abandoned approaches.
- Docstrings say what a component does, what it reads, what it writes, its parameters and
  its assumptions. They do not argue for the study design or report results: outcome numbers
  go stale and belong in the manuscript.
- Script docstrings use short `Inputs:` / `Outputs:` / `Usage:` blocks.
- Comments explain only what is not obvious from the code: rationale for a non-obvious
  choice, invariants, mathematics, edge cases, external constraints, units, coordinate
  systems, and data-format quirks. Do not translate code into English. Prefer deleting a
  comment to adding clutter.

      Bad:  # Previously this used cosine similarity, but we changed it to correlation.
      Good: # Fisher-z averaging avoids downward bias when aggregating correlations.

- README describes the current workflow only.
- Keep scientifically meaningful provenance: data sources, preregistration, and the script
  that produces each reported result.
