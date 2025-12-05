# Behavior Cloning (BC) Training

훈련된 모델을 사용하여 데이터를 수집하고, 그 데이터로 BC 모델을 학습시키는 코드입니다.
입력/출력 차원은 데이터셋에서 자동으로 감지됩니다.

## 파일 구조

- `model.py`: 신경망 모델 정의 (BCNetwork)
- `dataset.py`: 데이터셋 클래스 (.mpz 파일 로드)
- `train.py`: 학습 스크립트
- `README.md`: 사용 설명서

## 사용 방법

### 1. 데이터 수집 (Rollout)

훈련된 모델을 사용하여 BC 학습용 데이터를 수집합니다:

```bash
python mimickit/run_bc.py \
    --mode rollout \
    --num_workers 1 \
    --num_envs 4 \
    --env_config data/envs/deepmimic_dextra_lowerbody_env_30.yaml \
    --agent_config data/agents/deepmimic_dextra_lowerbody_ppo_agent.yaml \
    --model_file output/flat_foot/model.pt \
    --rollout_episodes 100 \
    --rollout_file output/bc_data/rollout_dataset.npz \
    --visualize false
```

**주요 파라미터:**
- `--mode rollout`: rollout 모드로 실행
- `--model_file`: 훈련된 모델 파일 경로 (필수)
- `--rollout_episodes`: 수집할 에피소드 개수 (기본값: 100)
- `--rollout_file`: 저장할 데이터셋 파일 경로 (기본값: output/rollout_dataset.npz)
- `--obs_indices`: 현실에서 관측 가능한 state 인덱스 (선택, 예: `0,1,2,5,6`)
- `--num_workers 1`: rollout 모드는 단일 워커 권장

### 2. BC 모델 학습

```bash
python bc/train.py \
    --data_path /path/to/data.npz \
    --output_dir output/bc_model \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --device cuda
```

### 3. 주요 파라미터

- `--data_path`: .npz 데이터 파일 경로 (필수)
- `--output_dir`: 모델 및 로그 저장 디렉토리 (기본값: output)
- `--batch_size`: 배치 크기 (기본값: 64)
- `--epochs`: 에폭 수 (기본값: 100)
- `--lr`: 학습률 (기본값: 1e-3)
- `--val_split`: 검증 데이터 비율 (기본값: 0.2)
- `--hidden_dims`: 은닉층 차원 리스트 (기본값: [256, 256, 128])
- `--device`: 사용할 디바이스 (cuda 또는 cpu, 기본값: cuda)

### 4. 전체 워크플로우 예시

```bash
# 1단계: 훈련된 모델로 데이터 수집
python mimickit/run_bc.py \
    --mode rollout \
    --num_workers 1 \
    --num_envs 4 \
    --env_config data/envs/deepmimic_dextra_lowerbody_env_30.yaml \
    --agent_config data/agents/deepmimic_dextra_lowerbody_ppo_agent.yaml \
    --model_file output/flat_foot/model.pt \
    --rollout_episodes 1000 \
    --rollout_file output/bc_data/rollout_dataset.npz \
    --visualize false

# 2단계: BC 모델 학습
python bc/train.py \
    --data_path output/bc_data/rollout_dataset.npz \
    --output_dir output/bc_model \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --device cuda
```

### 5. 출력 파일

학습 후 `output_dir`에 다음 파일들이 생성됩니다:

- `best_model.pt`: 검증 손실이 가장 낮은 모델
- `final_model.pt`: 최종 모델
- `checkpoint_epoch_N.pt`: 주기적으로 저장되는 체크포인트
- `training_history.json`: 학습 히스토리
- `logs/`: TensorBoard 로그 파일

## 데이터 형식

`rollout` 모드로 생성된 `.npz` 파일은 다음 형식을 가집니다:

- `obs`: 관측값 배열 (N, obs_dim) - flattened
- `actions`: 액션 배열 (N, act_dim) - flattened
- `episodes`: 에피소드 개수
- `episode_length`: 각 에피소드의 길이
- `obs_indices`: 관측 가능한 state 인덱스 (선택)

## 모델 구조

- 입력: 데이터셋에서 자동 감지 (기본적으로 관측값 차원)
- 은닉층: [256, 256, 128] (기본값, `--hidden_dims`로 변경 가능)
- 출력: 데이터셋에서 자동 감지 (기본적으로 액션 차원)
- 활성화 함수: ReLU
- 손실 함수: MSE (Mean Squared Error)

## 참고사항

- `rollout` 모드는 정확히 `target_ep_len` (150 steps) 길이의 에피소드만 수집합니다
- 에피소드 길이가 맞지 않으면 해당 에피소드는 버려집니다
- `num_workers=1`을 권장합니다 (멀티프로세스 지원은 제한적)


