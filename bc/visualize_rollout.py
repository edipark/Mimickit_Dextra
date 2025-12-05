"""
Isaac Gym에서 rollout action을 시각화하는 스크립트
"""
import os
import sys
import numpy as np
import yaml
import argparse
import time

# MimicKit 경로 추가 (mimickit 폴더를 Python path에 추가)
# run.py는 mimickit/ 폴더 안에서 실행되므로, mimickit/ 폴더를 path에 추가
mimickit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mimickit'))
if mimickit_dir not in sys.path:
    sys.path.insert(0, mimickit_dir)

# Isaac Gym을 먼저 import해야 함 (torch보다 먼저)
import envs.env_builder as env_builder
from util.logger import Logger

import torch


def load_rollout_data(npz_path):
    """
    npz 파일에서 rollout 데이터 로드
    
    Args:
        npz_path: .npz 파일 경로
    
    Returns:
        Dictionary containing actions, observations, etc.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    print("=" * 60)
    print("📥 Loaded Rollout Data")
    print("=" * 60)
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  {key}: {type(data[key])}")
    print("=" * 60)
    
    return data


def visualize_rollout(npz_path, env_config_path=None, episode_idx=0, speed=1.0):
    """
    Rollout action을 Isaac Gym에서 시각화
    
    Args:
        npz_path: rollout .npz 파일 경로
        env_config_path: 환경 설정 파일 경로 (None이면 stiffness_30 설정 사용)
        episode_idx: 재생할 에피소드 인덱스
        speed: 재생 속도 배율 (1.0 = 정상 속도)
    """
    # Rollout 데이터 로드
    rollout_data = load_rollout_data(npz_path)
    
    # Action 데이터 확인 및 처리
    # action_list shape: (num_episodes, num_steps, num_envs, action_dim)
    if 'action_list' in rollout_data:
        action_list = rollout_data['action_list']
    elif 'actions' in rollout_data:
        action_list = rollout_data['actions']
    else:
        raise ValueError("Action data not found in npz file. Available keys: {}".format(list(rollout_data.keys())))
    
    # action_list shape 확인 및 에피소드 선택
    if action_list.ndim == 4:
        # Shape: (num_episodes, num_steps, num_envs, action_dim)
        num_episodes = action_list.shape[0]
        if episode_idx >= num_episodes:
            raise ValueError(f"Episode index {episode_idx} out of range. Available episodes: {num_episodes}")
        
        # 특정 에피소드 선택: (num_steps, num_envs, action_dim)
        actions = action_list[episode_idx]
        print(f"📦 Total episodes available: {num_episodes}")
    elif action_list.ndim == 3:
        # Shape: (num_steps, num_envs, action_dim) - 단일 에피소드
        actions = action_list
    elif action_list.ndim == 2:
        # Shape: (num_steps, action_dim) -> (num_steps, 1, action_dim)
        actions = action_list[:, np.newaxis, :]
    else:
        raise ValueError(f"Unexpected action_list shape: {action_list.shape}")
    
    num_steps, num_envs, action_dim = actions.shape
    print(f"\n📊 Action shape: ({num_steps}, {num_envs}, {action_dim})")
    print(f"🎬 Episode index: {episode_idx}")
    print(f"⚡ Playback speed: {speed}x\n")
    
    # Observation 데이터 확인 및 처리 (첫 state 설정용)
    # obs_list shape: (num_episodes, num_steps, num_envs, obs_dim)
    initial_obs = None
    if 'obs_list' in rollout_data:
        obs_list = rollout_data['obs_list']
        
        if obs_list.ndim == 4:
            # Shape: (num_episodes, num_steps, num_envs, obs_dim)
            num_episodes = obs_list.shape[0]
            if episode_idx < num_episodes:
                # 특정 에피소드의 첫 observation 선택: (num_envs, obs_dim)
                episode_obs = obs_list[episode_idx]  # (num_steps, num_envs, obs_dim)
                initial_obs = episode_obs[0]  # 첫 스텝의 observation: (num_envs, obs_dim)
                # 첫 번째 환경의 observation만 사용: (obs_dim,)
                if initial_obs.ndim == 2:
                    initial_obs = initial_obs[0]  # (obs_dim,)
            else:
                Logger.print(f"Warning: Episode {episode_idx} not found in obs_list, using default reset")
        elif obs_list.ndim == 3:
            # Shape: (num_steps, num_envs, obs_dim) - 단일 에피소드
            initial_obs = obs_list[0, 0]  # 첫 스텝, 첫 환경: (obs_dim,)
        elif obs_list.ndim == 2:
            # Shape: (num_steps, obs_dim)
            initial_obs = obs_list[0]  # 첫 스텝: (obs_dim,)
        
        if initial_obs is not None:
            print(f"📥 Loaded initial observation: shape={initial_obs.shape}")
    else:
        Logger.print("No obs_list found, using default reset state")
    
    # 환경 설정
    if env_config_path is None:
        # 기본값: stiffness_30 설정 사용
        env_config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'output', 
            'stiffness_30', 
            'env_config.yaml'
        )
    
    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config not found: {env_config_path}")
    
    # 환경 설정 로드 및 수정 (reference character 비활성화)
    original_config_path = env_config_path
    with open(env_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Action만 재생하므로 reference character 비활성화
    if "env" in config:
        config["env"]["visualize_ref_char"] = False
    
    # 임시 설정 파일 생성 (reference character 비활성화)
    import tempfile
    temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config_file)
    temp_config_file.close()
    env_config_path = temp_config_file.name
    temp_config_path = env_config_path  # 나중에 삭제하기 위해 저장
    
    # 디바이스 설정 (run.py와 동일하게 문자열로 전달)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"🖥️  Device: {device}\n")
    
    # 환경 초기화
    num_vis_envs = 1  # 시각화용 환경 1개
    visualize = True
    
    Logger.print("Initializing environment...")
    # env_builder를 사용하여 환경 생성 (run.py와 동일한 방식)
    env = env_builder.build_env(
        env_file=env_config_path,
        num_envs=num_vis_envs,
        device=device,
        visualize=visualize
    )
    
    # 환경 리셋
    Logger.print("Resetting environment...")
    obs, info = env.reset()
    
    # 첫 observation이 있으면 환경 state를 첫 observation에 맞게 설정
    if initial_obs is not None:
        Logger.print("Setting environment state to match initial observation...")
        # observation을 텐서로 변환
        initial_obs_tensor = torch.tensor(initial_obs, device=device, dtype=torch.float32)
        
        # observation shape 조정 (환경 수에 맞게)
        if initial_obs_tensor.ndim == 1:
            initial_obs_tensor = initial_obs_tensor.unsqueeze(0)  # (1, obs_dim)
        
        # observation으로부터 state를 복원하여 환경에 설정
        # observation은 state에서 계산되므로, observation을 그대로 사용해서
        # 환경의 observation buffer를 설정하는 것이 아니라
        # observation과 일치하도록 환경의 state를 조정해야 합니다
        
        # 현재 observation을 첫 observation으로 업데이트
        # 이는 환경이 첫 observation에 해당하는 state를 가지도록 하는 것입니다
        # 실제로는 observation을 역변환해서 state를 추출해야 하지만,
        # 이것은 매우 복잡하므로 환경이 observation과 일치하도록 조정합니다
        
        # 환경의 observation buffer 업데이트
        if hasattr(env, '_obs_buf'):
            current_obs_shape = env._obs_buf.shape
            if initial_obs_tensor.shape == current_obs_shape[1:]:
                env._obs_buf[0] = initial_obs_tensor[0]
                Logger.print("Updated observation buffer with initial observation")
            else:
                Logger.print(f"Warning: Observation shape mismatch. Expected {current_obs_shape[1:]}, got {initial_obs_tensor.shape}. Environment state may not match initial observation.")
        else:
            Logger.print("Warning: Environment does not have _obs_buf. Cannot set initial observation.")
    
    # Action을 텐서로 변환
    actions_tensor = torch.tensor(actions, device=device, dtype=torch.float32)
    
    # 재생 루프
    Logger.print(f"Starting playback ({num_steps} steps)...")
    Logger.print("Press 'Q' or close window to exit\n")
    
    step_idx = 0
    try:
        while step_idx < num_steps:
            # 현재 스텝의 action 가져오기
            # episode_idx가 num_envs보다 크면 첫 번째 환경 사용
            env_idx = min(episode_idx, num_envs - 1)
            current_action = actions_tensor[step_idx, env_idx:env_idx+1, :]  # (1, action_dim)
            
            # Action 적용 및 시뮬레이션 스텝
            obs, reward, done, info = env.step(current_action)
            
            # 속도 조절
            if speed != 1.0:
                timestep = env._engine.get_timestep()
                time.sleep(timestep / speed)
            
            step_idx += 1
            
            # Done이면 리셋
            if done[0].item() != 0:
                Logger.print(f"Episode done at step {step_idx}, resetting...")
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        Logger.print("\nPlayback interrupted by user")
    except Exception as e:
        Logger.print(f"\nError during playback: {e}")
        import traceback
        traceback.print_exc()
    finally:
        Logger.print(f"\nPlayback completed. Total steps: {step_idx}/{num_steps}")
        # 임시 설정 파일 삭제
        if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize rollout actions in Isaac Gym")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="bc/rollout_10000.npz",
        help="Path to rollout .npz file"
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default=None,
        help="Path to environment config YAML file (default: output/stiffness_30/env_config.yaml)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to visualize (default: 0)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    if not os.path.isabs(args.npz_path):
        args.npz_path = os.path.join(os.path.dirname(__file__), '..', args.npz_path)
    args.npz_path = os.path.abspath(args.npz_path)
    
    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"Rollout file not found: {args.npz_path}")
    
    visualize_rollout(
        npz_path=args.npz_path,
        env_config_path=args.env_config,
        episode_idx=args.episode,
        speed=args.speed
    )


if __name__ == "__main__":
    main()

