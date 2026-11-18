"""Central paths and constants. Override the data location with PATHSNAP_DATA."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))

PARTITIONS = {
    "train": DATA_DIR / "train",
    "student_test": DATA_DIR / "student_test",
    "evaluator": DATA_DIR / "evaluator",  # demo machine only
}

CACHE_DIR = ROOT / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

SEED = 42
# Dataset is around Darmstadt, Germany (lon ~8.6, lat ~49.85) -> UTM zone 32N
METRIC_CRS = "EPSG:32632"
