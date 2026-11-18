# `scripts/` – Runnables

This directory contains runnables for data processing, candidate generation, modeling, testing, and visualizing the maps.

---

## Map Visualization (`scripts/make_map.py`)

`scripts/make_map.py` is the collaborative tool provided by the team to create both interactive Folium HTML map and high-resolution maps at 300 DPI for Technical Report (meeting the requirements of Section 6 of hackathon).

### 4 Required Visual Layers
1. **Raw GPS observations**: 1 Hz GPS track points (blue dots with tooltips containing time & coordinates) along with dashed trajectory line.
2. **Road network geometry**: Candidate road networks arcs in the proximity of vehicle's trajectory (grey line segments).
3. **Reference Route**: Ground truth road route (bold emerald green line segment, `#2ca02c`).
4. **Model predicted route**: Top-1 candidates predicted by the model (dashed crimson line segment, `#d62728`).
5. **Trip start/end**: Distinct green play icon (Trip Start) and red stop icon (Trip End).
6. **HUD dashboard & legend**: Floating box for metadata including Ground-Truth Arc Match %, GPS count, and layers' color key.

---

### Usage by Each Teammate

#### 1. Teammate 1: Logistic Regression
After training the model using your training script (`scripts/train_logreg.py`), create maps using:
```bash
python scripts/make_map.py --model logreg --trip-id 00000000
```

#### 2. Teammate 2: Gaussian Naive Bayes & HMM
After training Naive Bayes (`scripts/train_nb.py`) or creating HMM decoded route maps (`results/scores/hmm_student_test.parquet`):
```bash
python scripts/make_map.py --model nb  --trip-id 00000000
python scripts/make_map.py --model hmm --trip-id 00000000
```
*(You may also provide a custom parquet/CSV score file using `--scores-path <path>`)*.

#### 3. Teammate 3: LightGBM & Random Forest
```bash
python scripts/make_map.py --model lgbm --trip-id 00000000
python scripts/make_map.py --model rf   --trip-id 00000000
```

#### 4. Teammate 4: MLP (Neural Network)
```bash
python scripts/make_map.py --model mlp --trip-id 00000000
```

---

### Batch Mode (Create All Maps at Once)
To create all possible models' maps which have scores in `results/scores/`:
```bash
python scripts/make_map.py --model all --trip-id 00000000
```

---

### Ground Truth Map (Without Any Model Prediction)
To check the raw GPS data and ground truth routes without any model predictions:
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

## Other Scripts in this Directory

| Script | Purpose |
|---|---|
| `build_candidates.py` | Generates `cache/candidates_train.parquet` and `cache/candidates_student_test.parquet` |
| `train_lgbm.py` | Trains LightGBM with 5-fold GroupKFold CV and exports scores |
| `train_rf.py` | Trains Random Forest with 5-fold GroupKFold CV and exports scores |
| `make_map.py` | Generates Folium interactive HTML maps and 300 DPI report figures |
