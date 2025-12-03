import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import VisualEncoder,VisualDistributionalValue
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)
@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    #数据的导入与训练部分
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    ckpt_path: Optional[str] = None
    """The path of Policy Checkpoint if you need to load the Checkpoints (New lebinge write)"""
    ckpt_num: int = 0
    """The checkpoint number of iteration training"""
    ##Class free guidence部分
    guidance_scale: float = 1.1
    """推理时 CFG 强度 """
    CFG_alpha: float = 0.2
    """训练时 conditional loss 权重"""
    use_cfg_loss: bool= True
    """是否启用 E[-logπ_uncond - alpha*logπ_cond]作为CFG损失函数"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )
    use_res: bool= False
    """input True if you want to use ResNet as vision encoder"""


    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 10000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = 10000
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]


        #*#相关模块设计

        # ============ CFG 相关超参数 ============
        self.guidance_scale = args.guidance_scale      # 推理时 CFG 强度 
        self.alpha = args.CFG_alpha                        # 训练时 conditional loss 权重
        self.use_cfg_loss = args.use_cfg_loss          # 是否启用你提出的 E[-logπ_uncond - α logπ_cond]
        self.p_uncond = getattr(args, "p_uncond", 0.1)  #设置dropout参数

        # ============ condition embedding ============
        # obs_feat 约 300 维，cond 升维到 32 维以避免被淹没
        self.cond_embed_dim = 32
        self.cond_embed = nn.Sequential(
            nn.Linear(1, self.cond_embed_dim),
            nn.SiLU(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
        )


        #视觉编码器模块
        visual_feature_dim = 256
        self.visual_encoder = VisualEncoder(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )

        #添加去噪器模块
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim+self.cond_embed_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        #添加离散value估计
        self.value_index = np.linspace(1,-1,200)
        self.value_distribution = VisualDistributionalValue(
            value_index=self.value_index, #待添加
            obs_horizon = self.obs_horizon,
            obs_state_dim = obs_state_dim,
            use_res=args.use_res
        )
        
        #加噪&去噪设计为100步
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )
        

    def encode_obs(self, obs_seq, eval_mode, use_value = False):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss_actor(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]
        device = action_seq.device

        # ========= 1. 编码观测 =========
        obs_feat = self.encode_obs(obs_seq, eval_mode=False)  # (B, F)

        # ========= 2. 读取 cond =========
        assert "cond" in obs_seq

        cond = obs_seq["cond"].float().to(device)   # (B,1)
        

        # ========= 3. 随机构造 uncond（关键！！） =========
        # 无论 use_cfg_loss 是否开启，都做这一步
        drop_mask = (torch.rand(B, device=device) < self.p_uncond).unsqueeze(1)

        train_cond = cond.clone()
        train_cond[drop_mask] = 0.0     # 人工构造 unconditional

        # ========= 4. cond embedding =========
        cond_feat = self.cond_embed(train_cond)
        global_cond = torch.cat([obs_feat, cond_feat], dim=1)

        # 真正的 uncond 分支（用于 CFG loss）
        uncond = torch.zeros_like(cond)
        uncond_feat = self.cond_embed(uncond)
        global_uncond = torch.cat([obs_feat, uncond_feat], dim=1)

        # ========= 5. 扩散加噪 =========
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device
        ).long()

        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps
        )

        # ========= 6. 噪声预测 =========
        noise_pred_train = self.noise_pred_net(
            noisy_action_seq,
            timesteps,
            global_cond=global_cond
        )

        # ========= 7. loss 形式切换 =========
        # ---------- (A) 普通 diffusion loss ----------
        if not self.use_cfg_loss:
            loss = F.mse_loss(noise_pred_train, noise)
            return loss

        # ---------- (B) 你的 CFG 形式 ----------
        noise_pred_uncond = self.noise_pred_net(
            noisy_action_seq,
            timesteps,
            global_cond=global_uncond
        )

        loss_uncond = F.mse_loss(noise_pred_uncond, noise)
        loss_cond   = F.mse_loss(noise_pred_train, noise)

        loss = loss_uncond + self.alpha * loss_cond
        return loss
    

    
    def get_action(self, obs_seq, sample_num=1, cond_override=1):
        """
        cond_override: None / -1 / 1
        - 若 obs_seq 中存在 'cond'：优先生效
        - 否则使用 cond_override
        - 若两者都没有：默认使用 1
        """
        device = obs_seq["state"].device
        B = obs_seq["state"].shape[0]

        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            # ========= 1. 编码观测 =========
            obs_feat = self.encode_obs(obs_seq, eval_mode=True)

            # ========= 2. condition 选择逻辑（你要求的重点） =========
            if "cond" in obs_seq:
                cond = obs_seq["cond"].float().to(device)
            elif cond_override is not None:
                cond = torch.full((B, 1), float(cond_override), device=device)
            else:
                cond = torch.zeros((B, 1), device=device)   # 默认 cond = 1

            cond_feat = self.cond_embed(cond)

            uncond = torch.zeros_like(cond)
            uncond_feat = self.cond_embed(uncond)

            global_cond = torch.cat([obs_feat, cond_feat], dim=1)
            global_uncond = torch.cat([obs_feat, uncond_feat], dim=1)

            if sample_num > 1:
                global_cond = global_cond.repeat(sample_num, 1)
                global_uncond = global_uncond.repeat(sample_num, 1)
                B = B * sample_num

            # ========= 3. 初始噪声 =========
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim),
                device=device
            )

            # ========= 4. CFG 反向扩散 =========
            for k in self.noise_scheduler.timesteps:

                noise_pred_uncond = self.noise_pred_net(
                    noisy_action_seq, k, global_cond=global_uncond
                )

                noise_pred_cond = self.noise_pred_net(
                    noisy_action_seq, k, global_cond=global_cond
                )

                noise_pred = (
                    noise_pred_uncond
                    + self.guidance_scale
                    * (noise_pred_cond - noise_pred_uncond)
                )

                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")