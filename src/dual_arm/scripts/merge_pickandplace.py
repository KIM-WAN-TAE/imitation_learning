#!/usr/bin/env python3
from pathlib import Path
import sys
import shutil
import logging

# ====== Setup logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Setup import path ======
PROJECT_SRC = Path(__file__).resolve().parent.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from dual_arm.datasets.lerobot_dataset import LeRobotDataset
from dual_arm.datasets.dataset_tools import merge_datasets

# ====== Configuration ======
# 사용자가 요청한 pickandplace 경로
ROOT = Path("/home/wt/dual_arm/src/dual_arm/pickandplace")
OUT_REPO = "merged_dataset"
OUT_DIR = ROOT / OUT_REPO

def main():
    # 1. Find all sample_data folders
    # 'sample_data'로 시작하는 모든 폴더를 수집
    repo_ids = sorted([
        d.name for d in ROOT.iterdir()
        if d.is_dir() and d.name.startswith("sample_data") and d.name != OUT_REPO
    ])

    if not repo_ids:
        logger.error(f"No 'sample_data' datasets found in {ROOT}")
        return

    logger.info(f"Found datasets to merge in pickandplace: {repo_ids}")

    # 2. Prepare output directory
    if OUT_DIR.exists():
        logger.info(f"Removing existing output directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    # 3. Initialize datasets
    datasets = []
    for rid in repo_ids:
        ds_path = ROOT / rid
        # meta 폴더가 있는지 확인하여 유효한 데이터셋인지 체크
        if (ds_path / "meta").exists():
            logger.info(f"Loading dataset {rid} from {ds_path}")
            datasets.append(LeRobotDataset(rid, root=ds_path))
        else:
            logger.warning(f"Skipping {rid}: 'meta' folder not found.")

    if not datasets:
        logger.error("No valid LeRobot datasets found to merge.")
        return

    # 4. Merge datasets
    logger.info(f"Merging {len(datasets)} datasets into {OUT_DIR}...")
    merged = merge_datasets(datasets, output_repo_id=OUT_REPO, output_dir=OUT_DIR)

    logger.info("="*60)
    logger.info(f"[OK] Successfully merged pickandplace data into: {OUT_DIR}")
    logger.info(f"Total episodes: {merged.meta.total_episodes}")
    logger.info(f"Total frames: {merged.meta.total_frames}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
