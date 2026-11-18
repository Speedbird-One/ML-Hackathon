# `scripts/` — Runnable Entry Points

This directory contains standalone, runnable entry points for data processing, candidate generation, model training, evaluation, and map visualisations.

---

## 🗺️ Map Visualisation Guide (`scripts/make_map.py`)

`scripts/make_map.py` is the team's shared utility for generating both **interactive Folium HTML maps** and **high-resolution 300 DPI publication figures** for the Technical Report (satisfying Section 6 of the hackathon guidelines).

### 4 Mandatory Visual Layers
1. **Raw GPS Observations**: 1 Hz GPS track points (blue markers with timestamp & coordinate tooltips) plus connecting dashed trajectory.
2. **Road Network Geometry**: Candidate road network arcs near the vehicle's trajectory (grey line segments).
3. **Reference Route (Ground Truth)**: Ground truth road route (bold emerald green line, `#2ca02c`).
4. **Model Predicted Route**: Model top-1 candidate predictions (dashed crimson line, `#d62728`).
5. **Start & End Markers**: Distinct green play icon (Trip Start) and red stop icon (Trip End).
6. **HUD Dashboard & Legend**: Floating metadata box showing Ground-Truth Arc Match %, GPS count, and layer color guide.

---

### How Each Teammate Uses It

#### 1. Teammate 1: Logistic Regression
After running your training script (`scripts/train_logreg.py`), generate maps with:
```bash
python scripts/make_map.py --model logreg --trip-id 00000000
```

#### 2. Teammate 2: Gaussian Naive Bayes & HMM
After training Naive Bayes (`scripts/train_nb.py`) or generating HMM decoded routes (`results/scores/hmm_student_test.parquet`):
```bash
python scripts/make_map.py --model nb  --trip-id 00000000
python scripts/make_map.py --model hmm --trip-id 00000000
```
*(You can also pass a custom parquet/CSV score file via `--scores-path <path>`)*.

#### 3. Teammate 3: LightGBM & Random Forest
```bash
python scripts/make_map.py --model lgbm --trip-id 00000000
python scripts/make_map.py --model rf   --trip-id 00000000
```

#### 4. Tanmay: MLP (Neural Network)
```bash
python scripts/make_map.py --model mlp --trip-id 00000000
```

---

### Batch Mode (Generate All Models at Once)
To generate maps for every model that has scores in `results/scores/`:
```bash
python scripts/make_map.py --model all --trip-id 00000000
```

---

### Pure EDA / Ground Truth Map (No Model Predictions)
To inspect raw GPS and reference routes without running any model:
```bash
python scripts/make_map.py --model none --trip-id 00000000
```

---

### CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `lgbm` | Model key: `lgbm`, `rf`, `logreg`, `nb`, `mlp`, `hmm`, `none`, or `all` |
| `--partition` | str | `student_test` | Partition to map: `student_test` or `train` |
| `--trip-id` | str | `00000000` | 8-digit trip ID string (e.g. `00000000`, `00000007`) |
| `--scores-path` | str | `None` | Optional explicit path to a scores parquet or CSV file |
| `--model-name` | str | `None` | Custom display title on the map HUD and legend |
| `--no-png` | flag | `False` | Skip exporting the 300 DPI static `.png` figure |

### Outputs Generated
- **Interactive Map**: `results/maps/map_<model>_<partition>_<trip_id>.html` (Open in Chrome / Edge / Firefox)
- **Report Static Figure**: `results/figures/map_<model>_<partition>_<trip_id>.png` (300 DPI, embed directly in report)

---

## 🏃 Other Scripts in this Directory

| Script | Purpose |
|---|---|
| `build_candidates.py` | Generates `cache/candidates_train.parquet` and `cache/candidates_student_test.parquet` |
| `train_lgbm.py` | Trains LightGBM with 5-fold GroupKFold CV and exports scores |
| `train_rf.py` | Trains Random Forest with 5-fold GroupKFold CV and exports scores |
| `make_map.py` | Generates Folium interactive HTML maps and 300 DPI report figures |
