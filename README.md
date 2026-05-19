# Dual Arm Robot 프로젝트 가이드

이 프로젝트는 자체 제작 양팔 로봇 팔을 활용하여 데이터를 수집하고, ACT(Action Chunking Transformer) 모델을 학습시키며, 실시간 자연어 명령 제어를 통해 로봇을 동작시키는 시스템입니다.

---

## 1. 데이터 수집 (Data Collection)

로봇의 움직임과 카메라 영상을 기록하여 학습용 데이터셋을 생성합니다.

### 카메라 2개 사용 (Laptop + Realsense)
```bash
python -m dual_arm.scripts.lerobot_record \
--robot.type=so100_follower \
--robot.port=/dev/ttyUSB0 \
--robot.cameras="{laptop: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, realsense: {type: intelrealsense, serial_number_or_name: '048522072070', width: 640, height: 480, fps: 30}}" \
--teleop.type=so100_leader \
--teleop.port=/dev/ttyUSB1 \
--dataset.root=/home/roma/dual_arm/src/dual_arm/total_dataset/sample_data10 \
--dataset.repo_id=wt/run_010 \
--dataset.num_episodes=100 \
--dataset.single_task="Pick the cube" \
--dataset.fps=20 \
--display_data=true
```

### 카메라 3개 사용 (Laptop + Extra + Realsense)
```bash
python -m dual_arm.scripts.lerobot_record \
--robot.type=so100_follower \
--robot.port=/dev/ttyUSB0 \
--robot.cameras="{laptop: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, extra_cam: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, realsense: {type: intelrealsense, serial_number_or_name: '048522072070', width: 640, height: 480, fps: 30}}" \
--teleop.type=so100_leader \
--teleop.port=/dev/ttyUSB1 \
--dataset.root=/home/roma/dual_arm/src/dual_arm/total_dataset_blue/sample_data10 \
--dataset.repo_id=wt/run_010 \
--dataset.num_episodes=100 \
--dataset.single_task="Pick the cube" \
--dataset.fps=20 \
--display_data=true
```

---

## 2. 모델 학습 (Model Training)

수집된 데이터를 사용하여 ACT 모델을 학습시킵니다. 이미지 변형(Augmentation) 옵션이 포함되어 있습니다.

```bash
python3 src/dual_arm/scripts/lerobot_train.py \
  --dataset.repo_id=total_dataset_red_merged \
  --dataset.root=dual_arm/total_dataset_red_merged \
  --policy.type=act \
  --policy.device=cuda \
  --batch_size=16 \
  --num_workers=8 \
  --optimizer.lr=5e-5 \
  --steps=10000 \
  --save_freq=5000 \
  --job_name=red_0518_chunk100 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --dataset.image_transforms.enable=true
```

---

## 3. 추론 및 평가 (Inference & Evaluation)

학습된 모델 가중치를 사용하여 실시간으로 로봇을 제어합니다.

```bash
python -m dual_arm.scripts.lerobot_record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyUSB1 \
  --robot.cameras='{"laptop":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":480,"fps":30},"extra_cam":{"type":"opencv","index_or_path":"/dev/video8","width":640,"height":480,"fps":30},"realsense":{"type":"intelrealsense","serial_number_or_name":"048522072070","width":640,"height":480,"fps":30}}' \
  --policy.path=/home/roma/dual_arm/src/outputs/triple_camera/chunk80/010000/pretrained_model \
  --dataset.repo_id=TAEDX/eval_results_recorded \
  --policy.n_action_steps=80 \
  --dataset.fps=20
```

---

## 4. 명령 제어 (Command Control) 상세 가이드

`command_control.py`를 실행하면 별도의 입력창을 통해 실행 중인 로봇에게 실시간으로 명령을 내릴 수 있습니다.

### 🛠 실행 방법
1. **터미널 1**: 위의 **3. 추론 및 평가** 명령어를 실행하여 로봇을 대기 상태로 만듭니다.
2. **터미널 2**: 아래 명령어를 실행하여 제어 입력창을 엽니다.
   ```bash
   python3 src/dual_arm/scripts/command_control.py
   ```

### 💬 케이스별 입력 예시

| 케이스 | 입력 예시 (자유롭게 입력 가능) | 결과 |
| :--- | :--- | :--- |
| **🟢 초록 물체 조작** | `초록색 잡아줘`, `녹색`, `green pick` | **Green Policy** 로드 및 동작 실행 |
| **🔴 빨강 물체 조작** | `빨간색 이동해`, `적색`, `red` | **Red Policy** 로드 및 동작 실행 |
| **🔵 파랑 물체 조작** | `파란색 큐브 집어`, `청색`, `blue` | **Blue Policy** 로드 및 동작 실행 |
| **동작 일시 정지** | `멈춰`, `정지`, `stop`, `wait` | 로봇이 즉시 동작을 멈추고 **대기(Idle)** 상태로 전환 |
| **홈 위치 복귀** | `초기화`, `원위치`, `home`, `reset` | 로봇이 정의된 **Home 포지션**으로 안전하게 복귀 |
| **프로그램 종료** | `종료`, `끝내`, `exit`, `quit` | 추론 스크립트와 제어 스크립트 모두 **안전하게 종료** |

---

**주의사항**:
*   **장치 확인**: `/dev/video*` (카메라) 및 `/dev/ttyUSB*` (모터) 번호가 환경에 따라 다를 수 있으니 반드시 확인 후 명령어를 수정하세요.
*   **서버 연결**: `command_control.py` 실행 시 "전송 실패"가 뜬다면, `lerobot_record.py`가 먼저 실행 중인지 확인하세요.
*   **모델 경로**: 각 색상별 모델 경로는 `command_control.py` 상단의 `POLICY_REGISTRY`에서 수정할 수 있습니다.
