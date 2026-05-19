#!/usr/bin/env python3
import time
import argparse
import sys
from pathlib import Path
from dual_arm.datasets.lerobot_dataset import LeRobotDataset

def replay_dataset(repo_id, root, episode_index=0, limit=None):
    print(f"데이터셋 로딩 중: {repo_id} (root: {root})")
    
    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"에러: 데이터셋을 불러올 수 없습니다. 경로를 확인해주세요.{e}")
        return

    # 실제 데이터셋의 FPS 가져오기 (기본값 30)
    fps = dataset.fps if hasattr(dataset, "fps") else 30
    delay = 1.0 / fps

    print(f"--- 리플레이 시작 (에피소드: {episode_index}, FPS: {fps}) ---")
    print("프레임 | 관절 상태 (Joint States)")
    print("-" * 60)

    # 전체 프레임 수 계산
    total_frames = dataset.num_frames
    if limit:
        total_frames = min(total_frames, limit)

    try:
        for i in range(total_frames):
            frame = dataset[i]
            
            # 관절 값(state) 추출
            state = frame["observation.state"]
            
            # 리스트로 변환 및 소수점 포맷팅
            state_list = state.tolist()
            formatted_state = [f"{s:6.3f}" for s in state_list]
            state_str = ", ".join(formatted_state)

            # 터미널 한 줄 업데이트 (사용)
            sys.stdout.write(f"[{i:04d}/{total_frames-1}] [{state_str}]")
            sys.stdout.flush()

            # 실제 녹화 속도에 맞게 대기
            time.sleep(delay)

    except KeyboardInterrupt:
        print("리플레이가 사용자에 의해 중단되었습니다.")
    except KeyError:
        print("에러: 'observation.state' 키를 찾을 수 없습니다. 데이터셋 구조를 확인하세요.")
    else:
        print("--- 리플레이 완료 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeRobot 데이터셋 관절 값 실시간 리플레이")
    parser.add_argument("--repo-id", type=str, default="sample_data", help="데이터셋 이름")
    parser.add_argument("--root", type=str, required=True, help="데이터셋 루트 경로")
    parser.add_argument("--episode", type=int, default=0, help="리플레이할 에피소드 인덱스")
    parser.add_argument("--limit", type=int, default=None, help="최대 프레임 제한")

    args = parser.parse_args()
    
    # PYTHONPATH 설정 없이 실행 가능하도록 src 추가
    sys.path.append(str(Path(__file__).parent / "src"))
    
    replay_dataset(args.repo_id, args.root, args.episode, args.limit)
