import numpy as np
import os
import shutil
import sys
import time

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util
import envs.base_env as base_env
import learning.base_agent as base_agent

import torch
import json

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize):
    env_file = args.parse_string("env_config")
    env = env_builder.build_env(env_file, num_envs, device, visualize)
    return env

def build_agent(agent_file, env, device):
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, out_model_file, int_output_dir, logger_type, log_file):
    agent.train_model(max_samples=max_samples, out_model_file=out_model_file, 
                      int_output_dir=int_output_dir, logger_type=logger_type,
                      log_file=log_file)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))
    return result

def collect_rollout(agent, env, num_episodes, out_dataset_file,
                    obs_indices=None):
    """
    훈련된 agent를 사용하여 rollout 데이터를 수집합니다.
    
    Args:
        agent: 이미 학습된 policy를 가진 agent (agent.load(...) 호출 이후 상태)
        env: build_env로 만든 환경 (vec env 가능)
        num_episodes: '길이가 target_ep_len인' 에피소드를 몇 개 모을지
        out_dataset_file: npz 저장 경로
        obs_indices: 현실에서 관측 가능한 state 인덱스 list (예: [0,1,2,5,...])
    """
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # 목표 에피소드 길이 (현재 환경: 5초 * 30Hz = 150 step)
    target_ep_len = 150

    # agent를 TEST 모드로 설정 (가능하면)
    try:
        agent.set_mode(base_agent.AgentMode.TEST)
    except Exception:
        pass
    agent.eval()

    # 환경 초기화
    obs, info = env.reset()
    if isinstance(obs, np.ndarray):
        num_envs = obs.shape[0]
    elif torch.is_tensor(obs):
        num_envs = obs.shape[0]
    else:
        raise RuntimeError("Unsupported obs type: {}".format(type(obs)))

    Logger.print(f"Collecting {num_episodes} episodes with length {target_ep_len} (num_envs={num_envs})")

    # env별 현재 episode 버퍼
    curr_obs_buf = [[] for _ in range(num_envs)]
    curr_act_buf = [[] for _ in range(num_envs)]
    step_counters = np.zeros(num_envs, dtype=np.int32)

    # 조건을 만족하는 episode들만 모아두는 리스트
    ep_obs_list = []
    ep_act_list = []

    with torch.no_grad():
        while len(ep_obs_list) < num_episodes:
            # 액션 결정
            action, action_info = agent._decide_action(obs, info)

            # 한 스텝 진행
            next_obs, reward, done, next_info = env.step(action)

            # per-env로 obs / action 기록 + step 카운트
            if torch.is_tensor(obs):
                obs_cpu = obs.detach().cpu()
            else:
                obs_cpu = obs

            if torch.is_tensor(action):
                act_cpu = action.detach().cpu()
            else:
                act_cpu = action

            for i in range(num_envs):
                o_i = to_numpy(obs_cpu[i])
                a_i = to_numpy(act_cpu[i])
                curr_obs_buf[i].append(o_i)
                curr_act_buf[i].append(a_i)
                step_counters[i] += 1

            # done 처리
            if torch.is_tensor(done):
                done_np = done.detach().cpu().numpy()
            else:
                done_np = to_numpy(done)

            # DoneFlags.NULL 이 아닌 env들 = episode 끝난 env들
            done_mask = (done_np != base_env.DoneFlags.NULL.value)
            if np.any(done_mask):
                term_envs = np.nonzero(done_mask)[0]

                # 각 종료 env에 대해 episode 길이 체크
                for idx in term_envs:
                    ep_len = step_counters[idx]
                    if ep_len == target_ep_len:
                        ep_obs_arr = np.stack(curr_obs_buf[idx], axis=0)   # (L, obs_dim)
                        ep_act_arr = np.stack(curr_act_buf[idx], axis=0)   # (L, act_dim)
                        ep_obs_list.append(ep_obs_arr)
                        ep_act_list.append(ep_act_arr)
                        Logger.print(f"Collected episode {len(ep_obs_list)} from env {idx} (length={ep_len})")

                        if len(ep_obs_list) >= num_episodes:
                            break  # while 루프 바깥에서 종료

                    # 길이가 안 맞으면 버리고 초기화
                    curr_obs_buf[idx] = []
                    curr_act_buf[idx] = []
                    step_counters[idx] = 0

                # episode 개수가 다 찼다면 루프 종료
                if len(ep_obs_list) >= num_episodes:
                    break

                # 종료된 env들만 reset
                if torch.is_tensor(done):
                    env_ids = torch.nonzero(done != base_env.DoneFlags.NULL.value, as_tuple=False).flatten()
                else:
                    env_ids = torch.from_numpy(np.nonzero(done_mask)[0].astype(np.int64))

                obs, info = env.reset(env_ids)

                # reset 후 obs 전체를 다시 받아오므로,
                # 나머지 env들에 대해서도 obs를 통일시키기 위해 next_obs 대신 reset 결과 사용
            else:
                # 아무도 안 끝났으면 그대로 다음 step으로
                obs, info = next_obs, next_info

    # 이제 ep_obs_list / ep_act_list 에는 길이 target_ep_len짜리 episode만 num_episodes개 들어 있음
    if len(ep_obs_list) == 0:
        Logger.print("Warning: No episodes of length {} were collected.".format(target_ep_len))
        return

    # (E, L, D...) 로 stack
    obs_stack = np.stack(ep_obs_list, axis=0)   # (E, L, obs_dim)
    act_stack = np.stack(ep_act_list, axis=0)   # (E, L, act_dim)

    # 현실 로봇에서 관측 가능한 부분만 골라내기
    if obs_indices is not None:
        obs_stack = obs_stack[..., obs_indices]

    # (E * L, D...) 로 flatten
    E, L = obs_stack.shape[0], obs_stack.shape[1]

    obs_flat = obs_stack.reshape(E * L, -1)
    act_flat = act_stack.reshape(E * L, -1)

    np.savez_compressed(
        out_dataset_file,
        obs=obs_flat,
        actions=act_flat,
        obs_indices=np.array(obs_indices if obs_indices is not None else [], dtype=np.int64),
        episodes=np.array(E, dtype=np.int64),
        episode_length=np.array(L, dtype=np.int64),
    )

    Logger.print(f"Saved rollout dataset to {out_dataset_file}")
    Logger.print(f"  - episodes: {E}")
    Logger.print(f"  - episode_length: {L}")
    Logger.print(f"  - obs shape (flattened): {obs_flat.shape}")
    Logger.print(f"  - actions shape (flattened): {act_flat.shape}")
    return

def create_output_dirs(out_model_file, int_output_dir):
    if (mp_util.is_root_proc()):
        output_dir = os.path.dirname(out_model_file)
        if (output_dir != "" and (not os.path.exists(output_dir))):
            os.makedirs(output_dir, exist_ok=True)
        
        if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
            os.makedirs(int_output_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)
        
    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def run(rank, num_procs, device, master_port, args):
    mode = args.parse_string("mode", "train")
    num_envs = args.parse_int("num_envs", 1)
    visualize = args.parse_bool("visualize", True)
    logger_type = args.parse_string("logger", "tb")
    log_file = args.parse_string("log_file", "output/log.txt")
    out_model_file = args.parse_string("out_model_file", "output/model.pt")
    int_output_dir = args.parse_string("int_output_dir", "")
    model_file = args.parse_string("model_file", "")

    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()

    create_output_dirs(out_model_file, int_output_dir)

    env = build_env(args, num_envs, device, visualize)
    
    out_model_dir = os.path.dirname(out_model_file)
    agent_file = args.parse_string("agent_config")
    agent = build_agent(agent_file, env, device)

    if (model_file != ""):
        agent.load(model_file)

    if (mode == "train"):
        env_file = args.parse_string("env_config")
        copy_file_to_dir(env_file, "env_config.yaml", out_model_dir)
        copy_file_to_dir(agent_file, "agent_config.yaml", out_model_dir)

        max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)
        train(agent=agent, max_samples=max_samples, out_model_file=out_model_file, 
              int_output_dir=int_output_dir, logger_type=logger_type, log_file=log_file)
    elif (mode == "test"):
        test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
        test_result = test(agent=agent, test_episodes=test_episodes)
        
        # Convert all values to numpy arrays (handle nested lists with tensors)
        def convert_to_numpy(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy()
            elif isinstance(obj, list):
                return [convert_to_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float)):
                return np.array(obj)
            else:
                return np.array(obj)
        
        Logger.print("Keys in test_result before conversion: {}".format(list(test_result.keys())))
        test_result = {k: convert_to_numpy(v) for k, v in test_result.items()}
        Logger.print("Keys in test_result after conversion: {}".format(list(test_result.keys())))
        np.savez(os.path.join(out_model_dir, "test_result.npz"), **test_result)
        Logger.print("Saved test_result.npz with keys: {}".format(list(test_result.keys())))
    
    elif (mode == "rollout"):
        # rollout 모드는 훈련된 모델을 사용하여 BC 학습용 데이터를 수집
        # 멀티프로세스 처리까지 신경 쓰기 복잡하니까, 가능하면 num_workers=1 로 쓰는 걸 추천
        if num_procs != 1:
            Logger.print("Warning: rollout mode is intended for num_workers=1. Current num_procs = {}".format(num_procs))

        # 몇 개의 episode 를 수집할지
        rollout_episodes = args.parse_int("rollout_episodes", 100)

        # 저장 파일 경로
        rollout_file = args.parse_string("rollout_file", "output/rollout_dataset.npz")

        # 현실에서 관측 가능한 state index 설정
        # 예: --obs_indices 0,1,2,5,6
        obs_indices_str = args.parse_string("obs_indices", "")
        if obs_indices_str == "":
            obs_indices = None
        else:
            obs_indices = [int(x) for x in obs_indices_str.split(",")]

        if model_file == "":
            Logger.print("Error: rollout mode requires --model_file to load a trained policy.")
        else:
            # root 프로세스에서만 저장
            if mp_util.is_root_proc():
                collect_rollout(
                    agent=agent,
                    env=env,
                    num_episodes=rollout_episodes,
                    out_dataset_file=rollout_file,
                    obs_indices=obs_indices
                )
    
    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def main(argv):
    root_rank = 0
    args = load_args(argv)
    master_port = args.parse_int("master_port", None)
    num_workers = args.parse_int("num_workers", 1)
    device = args.parse_string("device", "cuda:0")
    assert(num_workers > 0)
    
    # if master port is not specified, then pick a random one
    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for i in range(num_workers - 1):
        rank = i + 1
        if ("cuda" in device):
            curr_device = "cuda:" + str(rank)
        else:
            curr_device = device

        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)

    
    if (num_workers > 1 and "cuda" in device):
        curr_device = "cuda:" + str(root_rank)
    else:
        curr_device = device
    
    run(root_rank, num_workers, curr_device, master_port, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)
