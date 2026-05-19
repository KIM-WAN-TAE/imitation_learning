from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
import logging

logging.basicConfig(level=logging.INFO)

def main():
    root = Path("lerobot/cube_dataset")
    output_repo_id = "cube_dataset_merged"
    output_dir = Path("lerobot/cube_dataset_merged")

    # Find all sample_data folders that are complete (contain a 'data' directory)
    repo_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("sample_data") and (d / "data").is_dir()], 
                      key=lambda x: int(x.name.replace("sample_data", "")))
    
    print(f"Found {len(repo_dirs)} datasets to merge.")

    datasets = []
    for d in repo_dirs:
        # We use the folder name as repo_id and provide the full path as root
        datasets.append(LeRobotDataset(d.name, root=d))

    print("Merging datasets...")
    merged = merge_datasets(datasets, output_repo_id=output_repo_id, output_dir=output_dir)
    
    print(f"Successfully merged into {output_dir}")
    print(f"Total episodes: {merged.meta.total_episodes}")
    print(f"Total frames: {merged.meta.total_frames}")

if __name__ == "__main__":
    main()
# from datasets import load_dataset
# import os

# # 데이터셋 경로 지정
# dataset_path = "/home/temp_id/lerobot/src/lerobot/cube_dataset_merged"

# # 데이터셋 불러오기
# dataset = load_dataset(dataset_path, split="train")

# # 어떤 데이터(Feature)들이 있는지 출력
# print(dataset.features.keys())