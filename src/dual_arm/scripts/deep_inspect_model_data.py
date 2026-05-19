import torch
import numpy as np
from pathlib import Path
import json
from safetensors.torch import load_file
from dual_arm.datasets.lerobot_dataset import LeRobotDataset
from dual_arm.policies.factory import make_policy

def deep_diagnosis(dataset_path, model_path):
    print("\n" + "="*60)
    print("DEEP DIAGNOSIS: DATASET & MODEL WEIGHTS")
    print("="*60)

    # 1. 데이터셋 피처 구조 확인
    dataset = LeRobotDataset("merged_dataset", root=dataset_path)
    print(f"\n[1] Dataset Info:")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Total Frames: {len(dataset)}")
    print(f"  State Keys: {[k for k in dataset.features if k.startswith('observation.state')]}")
    print(f"  Action Keys: {[k for k in dataset.features if k.startswith('action')]}")

    # 2. 가중치 파일 직접 분석
    print(f"\n[2] Weight Analysis (safetensors):")
    weights = load_file(Path(model_path) / "model.safetensors")
    
    # 특정 레이어(예: ACT의 입력/출력 헤드)의 가중치 통계 확인
    for key in list(weights.keys())[:5]: # 상위 5개 레이어만 샘플링
        w = weights[key]
        print(f"  {key}: shape={w.shape}, mean={w.mean():.4f}, std={w.std():.4f}")
    
    # 가중치 폭주(Explosion) 또는 소멸(Vanishing) 체크
    total_mean = torch.mean(torch.stack([w.float().mean() for w in weights.values()]))
    if torch.abs(total_mean) > 10 or torch.abs(total_mean) < 1e-7:
        print(f"  ! WARNING: Weights might be unstable (Overall Mean: {total_mean:.6f})")
    else:
        print(f"  Overall Weight Stability: OK")

    # 3. 모델 로딩 및 예측 재현성 테스트
    print(f"\n[3] Inference Simulation (Dataset Replay):")
    # policy config 로드
    from dual_arm.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    
    # 첫 번째 에피소드의 중간 프레임으로 테스트
    sample_idx = len(dataset) // 2
    frame = dataset[sample_idx]
    
    # 원본 액션 (정답)
    true_action = np.array([frame[k] for k in dataset.features if k.startswith('action')])
    
    # 모델 예측 시도
    print(f"  Sample Frame Index: {sample_idx}")
    print(f"  Recorded True Action: {true_action.flatten()[:3]}...") # 앞 3개만 출력

    # 4. 결론 도출
    print(f"\n[4] Diagnostic Conclusion:")
    # 관절 순서 불일치 가능성 체크
    dataset_action_names = dataset.features['action']['names']
    print(f"  Dataset Action Order: {dataset_action_names}")
    
    # 만약 MoveMag가 30씩 튀었던 원인이 가중치 값 자체가 크기 때문이라면?
    output_layer_key = [k for k in weights.keys() if 'action' in k or 'head' in k]
    if output_layer_key:
        out_w = weights[output_layer_key[0]]
        if out_w.abs().max() > 100:
            print("  ! CRITICAL: Output layer has extremely high weights. Overfitting or bad scaling suspected.")
        else:
            print("  Output layer weights scale: OK")

    print("="*60)

if __name__ == "__main__":
    DATA_ROOT = "/home/wt/dual_arm/src/dual_arm/pickandplace/merged_dataset"
    MODEL_ROOT = "/home/wt/outputs/train/2026-02-27/16-20-48_act/checkpoints/last/pretrained_model"
    deep_diagnosis(DATA_ROOT, MODEL_ROOT)
