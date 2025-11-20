# Behavior Cloning (BC) Training

72차원 입력과 12차원 출력을 가진 신경망을 학습시키는 코드입니다.

## 파일 구조

- `model.py`: 신경망 모델 정의 (BCNetwork)
- `dataset.py`: 데이터셋 클래스 (.mpz 파일 로드)
- `train.py`: 학습 스크립트
- `README.md`: 사용 설명서

## 사용 방법

### 1. 데이터 준비

`.mpz` 형식의 데이터 파일을 준비합니다.

### 2. 학습 실행

```bash
python bc/train.py \
    --data_path /path/to/data.mpz \
    --output_dir output/bc_model \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --device cuda
```

### 3. 주요 파라미터

- `--data_path`: .mpz 데이터 파일 경로 (필수)
- `--output_dir`: 모델 및 로그 저장 디렉토리 (기본값: output)
- `--batch_size`: 배치 크기 (기본값: 64)
- `--epochs`: 에폭 수 (기본값: 100)
- `--lr`: 학습률 (기본값: 1e-3)
- `--val_split`: 검증 데이터 비율 (기본값: 0.2)
- `--hidden_dims`: 은닉층 차원 리스트 (기본값: [256, 256, 128])
- `--device`: 사용할 디바이스 (cuda 또는 cpu, 기본값: cuda)

### 4. 출력 파일

학습 후 `output_dir`에 다음 파일들이 생성됩니다:

- `best_model.pt`: 검증 손실이 가장 낮은 모델
- `final_model.pt`: 최종 모델
- `checkpoint_epoch_N.pt`: 주기적으로 저장되는 체크포인트
- `training_history.json`: 학습 히스토리
- `logs/`: TensorBoard 로그 파일

## TODO: 전처리 구현

`dataset.py`의 `_load_mpz_data` 메서드와 전처리 부분을 구현해야 합니다:

1. `.mpz` 파일 로드
2. 데이터 정규화
3. 특징 엔지니어링
4. 데이터 증강 (필요시)

## 모델 구조

- 입력: 72차원
- 은닉층: [256, 256, 128] (기본값)
- 출력: 12차원
- 활성화 함수: ReLU
- 손실 함수: MSE (Mean Squared Error)

