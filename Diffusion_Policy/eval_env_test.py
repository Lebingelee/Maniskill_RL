from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.entropy_estimate import estimate_policy_entropy_gaussian, estimate_policy_entropy_knn
from diffusion_policy.visualization import EntropyVisualizer
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import tyro
from dataclasses import dataclass ,field
from typing import List, Optional
import matplotlib.pyplot as plt
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack
from mani_skill.utils import common
import gymnasium as gym
from CFG_dp import Agent, Args
import torch


@dataclass
class Args_eval(Args):
    env_id: str = "StackCube-v1"
    """the id of the environment"""

    num_eval_envs: int = 1
    """the number of parallel environments to evaluate the agent on"""

    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""

    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    obs_mode: str = "rgb"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""

    ckpt_path: Optional[str] = None
    """The path of Policy Checkpoint if you need to load the Checkpoints (New lebinge write)"""


def test_env(args, env_kwargs, other_kwargs):
    envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            
            wrappers=[FlattenRGBDObservationWrapper],
        )
    obs, info = envs.reset()

    print(obs.keys())
    print(obs['rgb'].shape)
    first_fig = obs['rgb'][0,0,:,:,0:3]
    plt.imshow(first_fig)
    plt.show()
    envs.close()

def test_obs():

    args = tyro.cli(Args_eval)
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    wrappers=[FlattenRGBDObservationWrapper]

    test = True
    if test == True:
        test_env(args, env_kwargs, other_kwargs)
    env = gym.make(args.env_id, reconfiguration_freq=1,
                   #max_episode_steps=15, 
                   **env_kwargs)

    obs, info = env.reset()
    for wrapper in wrappers:
        env = wrapper(env)
    env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])

    obs,info = env.reset()
    print(obs['rgb'].shape)
    
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    while True:
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward,terminated,truncated)


def load_checkpoint(ckpt_path, agent, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 加载agent参数
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
        print(f"Loaded agent parameters")

    return agent

def _main():
    # ---------------------------------------------------------------------------- #
    # Creating Envs.
    # ---------------------------------------------------------------------------- #
    args = tyro.cli(Args_eval)
    args.ckpt_path = "runs/checkpoint/256_rgb_basecam_plugcharger/70000.pt"
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )

    other_kwargs = dict(obs_horizon=args.obs_horizon)
    wrappers=[FlattenRGBDObservationWrapper]

    env = gym.make(args.env_id, reconfiguration_freq=1, **env_kwargs)

    for wrapper in wrappers:
        env = wrapper(env)
    env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])

    env_vec = CPUGymWrapper(env)
    # ---------------------------------------------------------------------------- #
    # Creating agent.
    # ---------------------------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = Agent(env_vec, args).to(device)
    agent = load_checkpoint(args.ckpt_path, agent,device)
    visualizer = EntropyVisualizer(
        video_path="policy_entropy_demo.mp4",
        fps=15,  # 根据环境速度调整
        history_len=100,  # 保留100步历史
        )



    # ---------------------------------------------------------------------------- #
    # Evaluation.
    # ---------------------------------------------------------------------------- #
    obs, info = env.reset()
    print(obs['rgb'].shape)
    break_sign = False
    k= 0
    act_smaple = 32
    
    while True:
        if break_sign == True:
            break
        obs = common.to_tensor(obs, device)
        action_seq = agent.get_action(obs, sample_num=act_smaple)
        action_seq_imp = action_seq[[0]]
        #action_entropy, entropy_list = estimate_policy_entropy_gaussian(action_seq)
        action_entropy, entropy_list = estimate_policy_entropy_knn(action_seq)
        for i in range(action_seq.shape[1]):
            
            obs, rew, terminated, truncated, info = env.step(action_seq_imp[:, i])
            if truncated or terminated:
                break_sign = True
                break
            visualizer.update(obs['rgb'], entropy_list[i].item())
        #print(action_entropy)
        #print(action_seq_imp[0,0,:])
        k = k+1
        env.render()

    #print(k)
    env.close()

if __name__ == "__main__":
    #_main()
    test_obs()
    