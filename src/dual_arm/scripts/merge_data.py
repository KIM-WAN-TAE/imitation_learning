#!/usr/bin/env python3
from pathlib import Path
import sys
import shutil
import logging

# ====== Dependency Check ======
try:
    import pyarrow
    import pandas
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please install required packages: pip install pyarrow pandas")
    sys.exit(1)

# ====== Setup logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Setup import path ======
# Path to the 'src' directory which contains the 'dual_arm' package
PROJECT_SRC = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from dual_arm.datasets.lerobot_dataset import LeRobotDataset
from dual_arm.datasets.dataset_tools import merge_datasets

# ====== Configuration ======
ROOT = Path("/home/wt/dual_arm/src/dual_arm/data")
OUT_REPO = "total_dataset"
OUT_DIR = ROOT / OUT_REPO

def main():
    # 1. Find all run_ folders
    repo_ids = sorted([
        d.name for d in ROOT.iterdir()
        if d.is_dir() and d.name.startswith("run_") and d.name != OUT_REPO
    ])

    if not repo_ids:
        logger.error(f"No 'run_' datasets found in {ROOT}")
        return

    logger.info(f"Found datasets to merge: {repo_ids}")

    # 2. Prepare output directory
    if OUT_DIR.exists():
        logger.info(f"Removing existing output directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    # 3. Initialize datasets with correct roots
    # LeRobotDataset expects 'root' to be the directory containing the 'meta' folder 
    # when repo_id is provided as a local identifier.
    datasets = []
    for rid in repo_ids:
        ds_path = ROOT / rid
        logger.info(f"Loading dataset {rid} from {ds_path}")
        datasets.append(LeRobotDataset(rid, root=ds_path))

    # 4. Merge datasets
    logger.info(f"Merging {len(datasets)} datasets into {OUT_DIR}...")
    merged = merge_datasets(datasets, output_repo_id=OUT_REPO, output_dir=OUT_DIR)

    logger.info("="*60)
    logger.info(f"[OK] Successfully merged into: {OUT_DIR}")
    logger.info(f"Total episodes: {merged.meta.total_episodes}")
    logger.info(f"Total frames: {merged.meta.total_frames}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
