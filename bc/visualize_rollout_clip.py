"""
Isaac Gym에서 BC rollout (obs, actions)을 재생하고
state-action pair(= obs_t, action_t, obs_{t+1})가 잘 저장됐는지 확인하는 스크립트.

npz 형식 (예시):
  - obs            : (T, obs_dim)          T = episodes * episode_length
  - actions        : (T, act_dim)
  - episodes       : scalar (int)
  - episode_length : scalar (int)
  - obs_indices    : (0,)  <-- 현재 사용 안 함

사용법 예:
  python visualize_bc_rollout.py \
    --npz_path bc/rollout_10000.npz \
    --env_config output/stiffness_30/env_config.yaml \
    --episode 0 \
    --speed 1.0
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np

# MimicKit 경로 추가 (mimickit 폴더를 Python path에 추가)
mimickit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mimickit'))
if mimickit_dir not in sys.path:
    sys.path.insert(0, mimickit_dir)

import envs.env_builder as env_builder
from util.logger import Logger

import torch

# ---------------------------------------------------------
# 1. npz 로드 + 에피소드 잘라내기
# ---------------------------------------------------------
def load_rollout_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    print("=" * 60)
    print(f"📥 Loaded rollout npz: {npz_path}")
    print("=" * 60)
    for k in data.keys():
        v = data[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v} (type={type(v)})")
    print("=" * 60)

    if "obs" not in data or "actions" not in data:
        raise ValueError(
            "npz에 'obs' 또는 'actions' 키가 없습니다.\n"
            f"  keys = {list(data.keys())}"
        )

    obs = data["obs"]        # (T, obs_dim)
    actions = data["actions"]  # (T, act_dim)

    # episodes / episode_length는 scalar라고 가정
    if "episodes" in data and "episode_length" in data:
        episodes = int(data["episodes"])
        episode_length = int(data["episode_length"])
    else:
        raise ValueError(
            "npz에 'episodes' 또는 'episode_length' 스칼라가 없습니다. "
            "현재 구조에 맞게 저장되어 있어야 합니다."
        )

    T = obs.shape[0]
    if T != episodes * episode_length:
        raise ValueError(
            f"obs 길이 T={T} 가 episodes * episode_length = "
            f"{episodes} * {episode_length} = {episodes * episode_length} 와 다릅니다."
        )
    if actions.shape[0] != T:
        raise ValueError(
            f"actions 길이 {actions.shape[0]} 이(가) obs 길이 {T} 와 다릅니다."
        )

    return obs, actions, episodes, episode_length


def slice_episode(obs, actions, episodes, episode_length, episode_idx):
    if episode_idx < 0 or episode_idx >= episodes:
        raise ValueError(
            f"episode_idx={episode_idx} 가 범위 [0, {episodes-1}] 를 벗어났습니다."
        )

    start = episode_idx * episode_length
    end = start + episode_length

    obs_ep = obs[start:end]        # (L, obs_dim)
    actions_ep = actions[start:end]  # (L, act_dim)

    print(f"\n🎬 Selected episode: {episode_idx}")
    print(f"   step range: [{start}, {end})")
    print(f"   obs_ep shape: {obs_ep.shape}")
    print(f"   actions_ep shape: {actions_ep.shape}\n")

    return obs_ep, actions_ep


# ---------------------------------------------------------
# 2. env 초기화
# ---------------------------------------------------------
def build_env(env_config_path, device, visualize=True):
    # env_config 로드 & ref character 끄기
    with open(env_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "env" in cfg:
        cfg["env"]["visualize_ref_char"] = False

    # 임시 설정 파일로 저장
    import tempfile
    temp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, temp_cfg)
    temp_cfg.close()
    tmp_cfg_path = temp_cfg.name

    Logger.print(f"Using env config: {env_config_path}")
    Logger.print(f"Temp env config with visualize_ref_char=False: {tmp_cfg_path}")

    env = env_builder.build_env(
        env_file=tmp_cfg_path,
        num_envs=1,          # 시각화용 env 1개
        device=device,
        visualize=visualize,
    )

    return env, tmp_cfg_path


# ---------------------------------------------------------
# 3. rollout 재생 + obs 일치도 체크
# ---------------------------------------------------------
def visualize_bc_rollout(
    npz_path,
    env_config_path=None,
    episode_idx=0,
    speed=1.0,
    print_interval=50,
):
    # 3-1. 데이터 로드
    obs_all, actions_all, episodes, episode_length = load_rollout_npz(npz_path)
    obs_ep, actions_ep = slice_episode(
        obs_all, actions_all, episodes, episode_length, episode_idx
    )

    L, obs_dim = obs_ep.shape
    _, act_dim = actions_ep.shape

    # 3-2. env_config 기본값
    if env_config_path is None:
        env_config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "stiffness_30",
            "env_config.yaml",
        )

    env_config_path = os.path.abspath(env_config_path)
    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config not found: {env_config_path}")

    # 3-3. device 설정
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"🖥️  Device: {device}\n")

    # 3-4. 환경 생성
    Logger.print("Initializing environment...")
    env, tmp_cfg_path = build_env(env_config_path, device=device, visualize=True)

    # 3-5. 환경 reset
    Logger.print("Resetting environment...")
    obs_env, info = env.reset()    # obs_env: (1, obs_dim_env)

    # obs 차원 체크
    try:
        obs_env_dim = obs_env.shape[-1]
    except Exception:
        obs_env_dim = None

    print(f"🔎 Dataset obs_dim = {obs_dim}")
    if obs_env_dim is not None:
        print(f"🔎 Env obs_dim      = {obs_env_dim}")
        if obs_env_dim != obs_dim:
            print("⚠️  Env obs_dim과 dataset obs_dim이 다릅니다. "
                  "같은 환경/설정에서 생성한 rollout인지 확인하세요.\n")
    else:
        print("⚠️  env.reset()에서 obs shape를 알 수 없습니다.\n")

    # 첫 obs 비교
    try:
        diff0 = np.linalg.norm(obs_env[0].cpu().numpy() - obs_ep[0])
        print(f"📏 ||obs_env_0 - obs_dataset_0|| = {diff0:.4e}")
    except Exception as e:
        print(f"⚠️  Initial obs diff 계산 실패: {e}")

    # 타임스텝 가져오기
    try:
        timestep = env._engine.get_timestep()
    except Exception:
        # 없으면 대충 60Hz 가정
        timestep = 1.0 / 60.0
    print(f"\n⏱  Env timestep: {timestep:.6f} s")
    print(f"⚡ Playback speed: {speed}x\n")

    # 3-6. 재생 루프
    Logger.print(f"Starting playback for episode {episode_idx} ({L} steps)...")
    Logger.print("Ctrl+C 또는 viewer 창을 닫으면 종료됩니다.\n")

    # 텐서 변환
    actions_tensor = torch.tensor(actions_ep, device=device, dtype=torch.float32)

    diff_list = []  # obs_t+1 vs dataset obs_{t+1} 차이 기록
    step_idx = 0

    try:
        while step_idx < L - 1:   # 마지막 step은 next obs가 없으니 L-1 까지만
            # 현재 step의 action
            action_t = actions_tensor[step_idx].unsqueeze(0)   # (1, act_dim)

            # env 한 스텝
            obs_env, reward, done, info = env.step(action_t)   # obs_env: (1, obs_dim_env)

            # 다음 step의 target obs (dataset)
            target_obs_next = obs_ep[step_idx + 1]  # (obs_dim,)

            # obs 차이 계산
            try:
                obs_env_np = obs_env[0].detach().cpu().numpy()
                diff = np.linalg.norm(obs_env_np - target_obs_next)
                diff_list.append(diff)

                if (step_idx + 1) % print_interval == 0 or step_idx + 1 == L - 1:
                    Logger.print(
                        f"[step {step_idx+1}/{L-1}] "
                        f"||obs_env - obs_dataset|| = {diff:.4e}"
                    )
            except Exception as e:
                Logger.print(f"⚠️  obs diff 계산 실패 (step {step_idx}): {e}")
                # 한 번 터졌다고 전체를 멈출 필요는 없으니 계속 진행

            # 재생 속도 조절
            if speed != 0:
                time.sleep(timestep / speed)

            step_idx += 1

            # done이면 env reset (dataset은 계속 이어질 수 있으니 참고용)
            try:
                if done[0].item() != 0:
                    Logger.print(f"Episode terminated in env at step {step_idx}, resetting env...")
                    obs_env, info = env.reset()
            except Exception:
                # done의 타입/shape이 다를 수 있으니, 에러 나면 그냥 무시
                pass

    except KeyboardInterrupt:
        Logger.print("\nPlayback interrupted by user.")
    except Exception as e:
        Logger.print(f"\nError during playback: {e}")
        import traceback
        traceback.print_exc()
    finally:
        Logger.print(f"\nPlayback finished at step {step_idx}/{L-1}")

        # diff 통계 출력
        if len(diff_list) > 0:
            diff_arr = np.array(diff_list)
            Logger.print(
                "State-action consistency (||obs_env_{t+1} - obs_dataset_{t+1}||):\n"
                f"  mean   = {diff_arr.mean():.4e}\n"
                f"  median = {np.median(diff_arr):.4e}\n"
                f"  max    = {diff_arr.max():.4e}\n"
                f"  min    = {diff_arr.min():.4e}\n"
                f"  count  = {len(diff_arr)}"
            )

        # 임시 env config 삭제
        if "tmp_cfg_path" in locals() and os.path.exists(tmp_cfg_path):
            os.unlink(tmp_cfg_path)


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize BC rollout (obs, actions) in Isaac Gym and "
                    "check state-action consistency."
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="bc/rollout_10000.npz",
        help="Path to rollout npz file "
             "(must contain obs, actions, episodes, episode_length)",
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default=None,
        help="Path to environment config YAML file "
             "(default: output/stiffness_30/env_config.yaml)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to visualize (0-based, default: 0)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=50,
        help="Print obs diff every N steps (default: 50)",
    )

    args = parser.parse_args()

    # npz_path 절대 경로 변환 (repo 루트 기준 ../)
    if not os.path.isabs(args.npz_path):
        args.npz_path = os.path.join(os.path.dirname(__file__), "..", args.npz_path)
    args.npz_path = os.path.abspath(args.npz_path)

    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"Rollout file not found: {args.npz_path}")

    visualize_bc_rollout(
        npz_path=args.npz_path,
        env_config_path=args.env_config,
        episode_idx=args.episode,
        speed=args.speed,
        print_interval=args.print_interval,
    )


if __name__ == "__main__":
    main()
