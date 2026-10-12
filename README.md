# connectome_resolutions

Code for "Connectome structure constrains neural activity across
scales". 

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
  worm.py            body outline and soma/neurite coordinates for drawing
  metric_guard.py    degeneracy diagnostics for sparse binary metrics
  figstyle.py        shared figure palette and SVG export
  equitability.py    equitability defect gamma of a partition
scripts/           analysis scripts and manuscript panel scripts (figN<panel>_*.py)
scripts/diagnostics/  standalone checks; each saves its printed output
preregistration/   pre-registered FlyWire partition rule
data/raw/          input data (not tracked)
results/           CSV / NPZ outputs (not tracked)
figures/           SVG panels and assembled figures (not tracked)
```

Panel scripts are named after the manuscript panel they draw (`fig1b_roc.py`
draws Fig. 1B). Analysis scripts are named after the analysis and write
`results/<script>*.csv|npz`. Each script's docstring states what it computes,
its inputs, outputs and flags.

## Setup

```
pip install -e .                 # add [virtualworm] / [test] as needed
```

Run every script from the repository root, e.g. `python scripts/complexity_curve.py`.
Scripts that import other scripts add `scripts/` to `sys.path` themselves.

`pytest tests/` (after `pip install -e .[test]`) pins the node and class counts, the
observed-pair counts, a zero equitability defect on a planted partition and the
quotient identity. Tests that need `data/raw/` are skipped without it.

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

#### Sources

`data_manifest.sha256` lists the SHA-256 of every manuscript input, relative to
`data/raw/` (`cd data/raw && shasum -a 256 -c ../../data_manifest.sha256`).

| dataset | publication | obtained from | files expected |
|---|---|---|---|
| Cook et al. 2019, adult hermaphrodite | *Nature* 571, 63–71; doi:10.1038/s41586-019-1352-7 | wormwiring.org; the edge list is the copy distributed with the OpenWorm ConnectomeToolbox (github.com/openworm/ConnectomeToolbox) | `celegans/herm_full_edgelist.csv` (columns Source, Target, Weight, Type) |
| Witvliet et al. 2021, eight developmental datasets | *Nature* 596, 257–261; doi:10.1038/s41586-021-03778-8 | wormwiring.org spreadsheets, which include electrical connections (the paper's supplementary matrices are chemical only) | `celegans/witvliet_2020_<1-8> <L1\|L2\|L3\|adult>.xlsx` |
| Randi et al. 2023, signal propagation atlas | *Nature* 623, 406–414; doi:10.1038/s41586-023-06683-4 | detected responses and cell table: funconn.princeton.edu; per-recording exports: OSF, doi:10.17605/OSF.IO/E2SYT (`wt.tar.gz`, `unc31.tar.gz`, unpacked to `randi/ex/`) | `randi/randi_funconn_wildty.json`, `randi/randi_funconn_unc31.json`, `randi/cells.json`, `randi/ex/exported_data/`, `randi/ex/exported_data_unc31/` |
| Bentley et al. 2016, monoamine and neuropeptide networks | *PLoS Comput. Biol.* 12, e1005283; doi:10.1371/journal.pcbi.1005283 | edge lists accompanying the paper | `celegans/edgelist_MA.csv`, `celegans/edgelist_NP.csv` |
| WormAtlas cell list | wormatlas.org | individual neuron pages, tabulated | `celegans/all_cell_info.csv` (columns Cell name, Type, Name details, Lineage, Classification) |
| FlyWire, materialization 783 | Dorkenwald et al. 2024, *Nature* 634, 124–138; Schlegel et al. 2024, *Nature* 634, 139–152 | connections: Zenodo, doi:10.5281/zenodo.10676866 (version 783.0); annotations: github.com/flyconnectome/flywire_annotations, `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | `flywire/proofread_connections_783.feather`, `flywire/neuron_annotations.tsv` |
| wormneuroatlas | github.com/francescorandi/wormneuroatlas | dump written by the snippet in `scripts/diagnostics/verify_wormneuroatlas.py` | `wormneuroatlas/wna_dump.npz` |

## Main figures

Panels are written as SVG to `figures/figN/` and assembled by hand
(`figures/figN/figure_N.ai`). Panel scripts read the CSV/NPZ produced by the
analysis scripts listed with them, so run those first, in the order given.

| figure | panel scripts | analysis scripts they read |
|---|---|---|
| 1 Path structure predicts propagation | `fig1a_example_pair`, `fig1a_response_trace`, `fig1b_roc`, `fig1c_path_depth`, `fig1d_nosynapse_violin` | — (computed in place) |
| 2 Cell classes preserve prediction | `fig2_class_resolution` (A–F), `fig2_vd_anatomy` (VD inset) | `class_merge_structural_loss`, `two_observables` → `two_observable_trajectory` |
| 3 Connectivity identifies coarse-grainings | `fig3_coarse_grainings` (A–D) | `complexity_curve` → `observable_resolution` → `coarse_trajectory` → `block_decomposition`, `grouping_methods`, `coarse_transfer`; `prospective_merges` |
| 4 Development and transfer | `fig4a_dev_series`, `fig4b_l1_adult_prediction`, `fig4c_dev_loss`, `fig4d_transfer_matrix`, `fig4e_transfer_overlap` | `dev_sufficiency`, `cross_animal_transfer` (30 merges, the default), `functional_transfer --pool all --n-merge 70`, `prospective_merges` |

Flags are given wherever a reported run does not use the script's defaults.

Other analyses reported in the main text:

| analysis (Methods section) | script |
|---|---|
| degree-based predictors and degree-preserving surrogates, with gap junctions rewired freely or kept symmetric (Path operator) | `degree_controls` |
| cell-class merge trajectories, 20 per arm (Merge trajectories; Fig. 2D, E) | `class_merge_structural_loss` |
| single-length operators on the quotient, 500 resamples with 50 random partitions each (Class-level path operator) | `path_length_quotient` |
| prospective merges, degree floors; 29 (the median) is the primary criterion (Connectivity-defined classes) | `prospective_merges --min-degree {0,20,29,40}` |
| simulated annealing at k = 119, 60,000 proposals per chain (Connectivity-defined classes) | `global_search` |
| predictive transfer of L1 merges to the adult, 161 candidates and 70 merges (Development and transfer) | `functional_transfer --pool all --n-merge 70` |
| class-member agreement, κ | `kappa_interchangeability`, `exchangeability` |
| γ of the cell classes vs 1,000 random partitions (Equitable partitions) | `class_gamma` |
| class-level loss for five observables vs 200 random partitions (Measured signal propagation) | `observable_class_loss` |
| pure vs mixed connectivity-defined classes (Connectivity-defined classes) | `heterogeneity_cost` |
| *Drosophila* partitions, spatial clusters | `fly_fragility`, `fly_complexity_curve` |

## Supplementary analyses

| analysis | script |
|---|---|
| exposure control | `exposure_control` |
| dependence structure | `dependence_control`, `dependence_twoway_boundary` |
| sensitivity to α | `alpha_sensitivity` |
| spectral normalization | `normalization_control`, `normalization_fly`, `fly_fragility --own-rho` |
| unc-31 | `unc31_path_advantage` |
| chemical synapses only, gap junctions only | `contact_type_control` |
| lineage-only classes, γ calibration | `supp_checks` |
| stimulus-locked regression | `sta_regression` |
| stimulus excision | `two_observables --excise {0,10,20,30,40,60}`, `--excise {0,20} --detrend`, `sham_excision` |
| developmental analyses without the five glia | `dev_sufficiency --exclude-glia`, `cross_animal_transfer --exclude-glia` |
| structural transfer at other merge counts | `cross_animal_transfer --n-merge {10,20,35}` |
| grouping methods | `grouping_methods` |
| Dvali et al. communities | `dvali_communities` |

## Diagnostics

`scripts/diagnostics/` holds checks that support statements in the text but
produce no table or figure (unc-31 resolution, two-step-count robustness, fly
partition checks, metric parity between species, an independent cross-check
against the wormneuroatlas package, and others). Three back specific sentences
of the Methods: `amplitude_validation` (amplitude against atlas detection and
magnitude), `class_assignment_check` (cell classes against the atlas cell
table) and `verify_metric_parity` (fly and worm operator code on the worm
connectome; fails outside tolerance). Each writes everything it
prints to `results/diagnostics/<script>.log`; their docstrings say what they
check.

## Submission snapshot

`python scripts/make_snapshot.py <name> --manuscript <files>` writes
`snapshots/<name>.tar.gz` with the code at the current commit, `results/`,
`figures/`, the data checksums, the Python environment and the manuscript
files given, plus a SHA-256 manifest. It refuses to run on an uncommitted tree
unless `--allow-dirty` is passed, in which case the uncommitted changes are
archived as a patch.

`export_soma_um.py` runs inside Blender
(`blender -b <file>.blend --python scripts/export_soma_um.py -- <spec.json>`) and
writes the `data/raw/virtualworm/soma_um.json` cache used by `conres.worm`.
