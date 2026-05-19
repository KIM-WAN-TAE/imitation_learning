import torch
import numpy as np
from pathlib import Path
import json
from dual_arm.datasets.lerobot_dataset import LeRobotDataset

def inspect_dataset(repo_id, root):
    print(f"\n=== Inspecting Dataset: {repo_id} ===")
    dataset = LeRobotDataset(repo_id, root=root)
    
    # 1. Stats 확인
    stats_path = Path(root) / "meta" / "stats.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    print(f"\n[1] Statistics in stats.json:")
    for key in stats:
        if "pos" in key:
            print(f"  {key}: mean={stats[key]['mean']}, std={stats[key]['std']}")

    # 2. 실제 데이터 샘플링 분석
    num_samples = min(5000, len(dataset))
    print(f"\n[2] Analyzing {num_samples} random frames...")
    
    actions = []
    states = []
    
    for i in range(0, len(dataset), len(dataset) // num_samples):
        frame = dataset[i]
        actions.append([v for k, v in frame.items() if k.startswith("action")])
        states.append([v for k, v in frame.items() if k.startswith("observation.state")])
        
    actions = np.array(actions)
    
    print(f"Actual Data Range (Action):")
    print(f"  Min: {np.min(actions):.2f}")
    print(f"  Max: {np.max(actions):.2f}")
    print(f"  Mean: {np.mean(actions):.2f}")
    print(f"  Std: {np.std(actions):.2f}")

    # 3. 프레임 간 변화량 (Delta) 분석
    print(f"\n[3] Analyzing movement continuity (Deltas)...")
    deltas = np.abs(np.diff(actions, axis=0))
    max_delta = np.max(deltas)
    avg_delta = np.mean(deltas)
    
    print(f"  Average delta per frame: {avg_delta:.4f}")
    print(f"  Maximum jump found: {max_delta:.4f}")
    
    if max_delta > 50:
        print("  ! WARNING: Huge jumps detected! Episodes might be merged incorrectly.")
    if avg_delta < 0.001:
        print("  ! WARNING: Almost no movement detected. Dataset might be too static.")

    # 4. 에피소드 별 길이 확인
    print(f"\n[4] Episode lengths:")
    ep_lens = [len(dataset.get_episode_data(i)["index"]) for i in range(dataset.num_episodes)]
    print(f"  Total Episodes: {dataset.num_episodes}")
    print(f"  Avg Length: {np.mean(ep_lens):.1f} frames")
    print(f"  Shortest: {np.min(ep_lens)} | Longest: {np.max(ep_lens)}")

if __name__ == "__main__":
    inspect_dataset("merged_dataset", "/home/wt/dual_arm/src/dual_arm/pickandplace/merged_dataset")
