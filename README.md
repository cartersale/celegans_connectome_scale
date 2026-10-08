# connectome_resolutions

Code for "At what resolution does a connectome stop predicting signal
propagation?" (manuscript: `docs/main.tex`; supplementary methods:
`docs/supplementary_methods_draft.tex`).

The path operator R = (I - αA/ρ)⁻¹ - I (α = 0.1) on the *C. elegans* connectome
is scored against measured signal propagation (Randi et al. 2023), activity
correlations and response amplitudes. Partitions of the neurons — biological cell
classes, connectivity-defined groups, anatomical subsystems, and size-matched
random controls — are scored by how much of that prediction the class-level
operator R_Q, built on the quotient matrix, retains. The same comparisons are run
across the Witvliet et al. 2021 developmental series and on the FlyWire
*Drosophila* connectome.

## Layout

```
src/conres/        library (installed as the `conres` package)
  celegans.py        Cook 2019 and Witvliet 2021 connectomes; neuron-class labels
  randi.py           Randi 2023 data; path operator, quotient, class-level operator;
                     observed-pair counts and activity correlations from raw recordings
  observables.py     the five functional observables and shared scoring/bootstrap helpers
  subsystems.py      annotation-based types / subsystems / divisions (WormAtlas)
  dvali.py           Dvali et al. 2025 community memberships
  flywire.py         FlyWire v783 scope, partition ladder, sparse path operator
  microns.py         MICrONS v1300 cells and graphs (exploratory arm, see below)
  worm.py            body outline and soma/neurite coordinates for drawing
  metric_guard.py    degeneracy diagnostics for sparse binary metrics
  figstyle.py        shared figure palette and SVG export
  equitability.py    equitability defect gamma of a partition
scripts/           analysis scripts and manuscript panel scripts (figN<panel>_*.py)
scripts/diagnostics/  standalone checks; each saves its printed output
microns/           exploratory MICrONS (mouse V1) analyses, not in the manuscript
archive/           superseded and unused scripts and figures (not tracked)
preregistration/   pre-registered FlyWire partition rule
data/raw/          input data (not tracked)
results/           CSV / NPZ outputs (not tracked)
figures/           SVG panels and assembled figures (not tracked)
docs/              manuscript sources and notes (not tracked)
```

Panel scripts are named after the manuscript panel they draw (`fig1b_roc.py`
draws Fig. 1B). Analysis scripts are named after the analysis and write
`results/<script>*.csv|npz`. Each script's docstring states what it computes,
its inputs, outputs and flags.

## Setup

```
pip install -e .                 # add [virtualworm] / [microns] as needed
```

Run every script from the repository root, e.g. `python scripts/complexity_curve.py`.
Scripts that import other scripts add `scripts/` to `sys.path` themselves.

Set `CONRES_CHECK_DIR=/some/dir` to also write PNG previews of figure panels;
otherwise none are written.

### Data (`data/raw/`)

| directory | contents |
|---|---|
| `celegans/` | Cook et al. 2019 `herm_full_edgelist.csv`; Witvliet et al. 2021 `witvliet_2020_<n> <stage>.xlsx`; Bentley et al. 2016 `edgelist_MA.csv`, `edgelist_NP.csv`; WormAtlas `all_cell_info.csv` |
| `randi/` | Randi et al. 2023 atlas `randi_funconn_<strain>.json` and per-recording exports `ex/exported_data[_<strain>]/` (OSF E2SYT); `cells.json` |
| `openworm_neuroml/` | OpenWorm NeuroML `<cell>.cell.nml` morphologies |
| `virtualworm/` | WormBase VirtualWorm `.blend` and the derived caches `anatomy_um.npz`, `soma_um.json` |
| `flywire/` | FlyWire v783 `neuron_annotations.tsv`, `proofread_connections_783.feather` |
| `wormneuroatlas/` | dump used by `verify_wormneuroatlas.py` |
| `microns/` | MICrONS tables and caches (exploratory arm only) |

## Main figures

Panels are written as SVG to `figures/figN/` and assembled by hand
(`figures/figN/figure_N.ai`). Panel scripts read the CSV/NPZ produced by the
analysis scripts listed with them, so run those first, in the order given.

| figure | panel scripts | analysis scripts they read |
|---|---|---|
| 1 Path structure predicts propagation | `fig1a_example_pair`, `fig1a_response_trace`, `fig1b_roc`, `fig1c_path_depth`, `fig1d_nosynapse_violin` | — (computed in place) |
| 2 Cell classes preserve prediction | `fig2_class_resolution` (A–F), `fig2_vd_anatomy` (VD inset) | `class_merge_structural_loss`, `two_observables` → `two_observable_trajectory` |
| 3 Connectivity identifies coarse-grainings | `fig3_coarse_grainings` (A–D) | `complexity_curve` → `observable_resolution` → `coarse_trajectory` → `block_decomposition`, `grouping_methods`, `coarse_transfer`; `prospective_merges` |
| 4 Development and transfer | `fig4a_dev_series`, `fig4b_l1_adult_prediction`, `fig4c_dev_loss`, `fig4d_transfer_matrix`, `fig4e_transfer_overlap` | `dev_sufficiency`, `cross_animal_transfer`, `functional_transfer`, `prospective_merges` |

Other analyses reported in the main text:

| analysis (Methods section) | script |
|---|---|
| degree-based predictors and degree-preserving surrogates (Path operator) | `degree_controls` |
| cell-class merge trajectories (Merge trajectories) | `class_merge_trajectory` |
| single-length operators on the quotient (Class-level path operator) | `path_length_quotient` |
| prospective merges, degree floors (Connectivity-defined classes) | `prospective_merges --min-degree {0,20,29,40}` |
| simulated annealing at k = 119 (Connectivity-defined classes) | `global_search` |
| class-member agreement, κ | `kappa_interchangeability`, `exchangeability` |
| γ of the cell classes vs 1,000 random partitions (Equitable partitions) | `class_gamma` |
| class-level loss for five observables vs 200 random partitions (Measured signal propagation) | `observable_class_loss` |
| pure vs mixed connectivity-defined classes (Connectivity-defined classes) | `heterogeneity_cost` |
| *Drosophila* partitions, spatial clusters | `fly_fragility`, `fly_complexity_curve` |

## Supplementary analyses

Each table in `docs/supplementary_methods_draft.tex` names its source script in a
`% Source:` comment.

| analysis | script |
|---|---|
| exposure control | `exposure_control` |
| dependence structure | `dependence_control`, `dependence_twoway_boundary` |
| sensitivity to α | `alpha_sensitivity` |
| spectral normalization | `normalization_control`, `normalization_fly`, `fly_fragility --own-rho` |
| unc-31 | `unc31_path_advantage` |
| lineage-only classes, γ calibration | `supp_checks` |
| stimulus-locked regression | `sta_regression` |
| stimulus excision | `two_observables --excise/--detrend`, `sham_excision` |
| grouping methods | `grouping_methods` |
| Dvali et al. communities | `dvali_communities` |

## Diagnostics

`scripts/diagnostics/` holds checks that support statements in the text but
produce no table or figure (unc-31 resolution, two-step-count robustness, fly
partition checks, metric parity between species, an independent cross-check
against the wormneuroatlas package, and others). Each writes everything it
prints to `results/diagnostics/<script>.log`; their docstrings say what they
check.

`export_soma_um.py` runs inside Blender
(`blender -b <file>.blend --python scripts/export_soma_um.py -- <spec.json>`) and
writes the `data/raw/virtualworm/soma_um.json` cache used by `conres.worm`.
