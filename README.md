# PathSnap: ML Map-Matching with HMM/Viterbi

Reconstructs vehicle routes from noisy 1 Hz GPS tracks by (1) generating candidate road arcs per GPS point,
(2) scoring candidates with a trained classifier, and (3) decoding a coherent route with an HMM/Viterbi over the road network.

Setup instructions: see [SETUP.md](SETUP.md).

## Layout
| Path | Purpose |
|---|---|
| `pathsnap/` | Core library: data loading, candidates, features, models, HMM/Viterbi, metrics |
| `scripts/` | Runnable entry points (EDA, feature build, training, Bayesian HPO, evaluation, maps) |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator-facing UI (Laptop 2) |
| `notebooks/` | Exploration notebooks |
| `artifacts/` | Saved model, preprocessing and HMM config used for inference |
| `results/` | Experiment logs, CSVs, figures, Folium HTML maps |
| `report/` | Technical report sources |
| `data/` | Raw data (git-ignored; see `data/README.md`) |

## Status
Scaffold only. Sections for training, HPO, HMM, API, UI and deployment will be added as they are built.
