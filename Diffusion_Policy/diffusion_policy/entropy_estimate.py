import torch
import numpy as np
from scipy.special import digamma
from scipy.spatial import KDTree

def estimate_policy_entropy_gaussian(rollout_action):
    """
    基于高斯分布假设估算 Policy 熵值。
    
    Args:
        rollout_action: torch.Tensor, shape [sample_num, action_chunk, action_dim]
                        例如: [32, 8, 7]
    
    Returns:
        avg_entropy: float, 整个动作序列的平均熵值
    """
    # 确保输入是 Tensor
    if isinstance(rollout_action, np.ndarray):
        rollout_action = torch.from_numpy(rollout_action)
        
    sample_num, action_chunk, action_dim = rollout_action.shape
    
    entropy_per_step = []

    # 我们需要在 dim=0 (sample_num) 上统计分布，因此针对每个时间步 t 进行循环
    for t in range(action_chunk):
        # 获取当前时间步的所有样本: shape [sample_num, action_dim]
        current_step_actions = rollout_action[:, t, :]
        
        # 1. 计算协方差矩阵 (Covariance Matrix)
        # rowvar=False 表示每一列是一个变量 (action_dim)，每一行是一个样本
        cov_matrix = torch.cov(current_step_actions.T)
        
        # 2. 处理数值稳定性 (防止行列式为 0 或负数)
        # 向对角线添加微小的噪声 (jitter)
        cov_matrix += torch.eye(action_dim, device=rollout_action.device) * 1e-9
        
        # 3. 计算多变量高斯分布的微分熵
        # 公式: H = 0.5 * log((2 * pi * e)^k * |Sigma|)
        # 其中 k 是维度 (action_dim), |Sigma| 是协方差矩阵的行列式
        # 使用 logdet 计算行列式的对数更稳定：log(|Sigma|)
        sign, logdet = torch.slogdet(cov_matrix)
        
        if sign <= 0:
            # 如果行列式非正（极少发生），说明样本坍缩到一个子空间，熵视为 -inf 或极小值
            step_entropy = torch.tensor(-100.0,device=rollout_action.device )
        else:
            # Constant term: k/2 * (1 + log(2*pi))
            # 简化写法: 0.5 * (k * (1 + ln(2*pi)) + ln(|Sigma|))
            k = action_dim
            const_term = 0.5 * k * (1 + torch.log(torch.tensor(2 * torch.pi)))
            step_entropy = const_term + 0.5 * logdet
            
        entropy_per_step.append(step_entropy)

    # 计算所有时间步的平均熵
    avg_entropy = torch.stack(entropy_per_step).mean().item()
    
    return avg_entropy, entropy_per_step


def estimate_policy_entropy_knn(rollout_action, k=1):
    """
    使用 K-近邻 Kozachenko-Leonenko 估计器估算微分熵 (假设 L_inf 距离)。
    
    Args:
        rollout_action: torch.Tensor, shape [sample_num, action_chunk, action_dim]
        k: 寻找的近邻数 (通常 k=1 或 k=2)。
    """
    
    sample_num, action_chunk, D = rollout_action.shape
    entropy_per_step = []

    # 预计算常数项
    psi_N = digamma(sample_num)
    psi_k = digamma(k)
    
    # 针对每个时间步计算熵
    for t in range(action_chunk):
        # 1. 提取当前时间步的样本 (N, D)
        current_step_actions = rollout_action[:, t, :].cpu().numpy()
        
        # 2. 构建 KDTree 以高效查找近邻
        # L_inf (Chebyshev) 距离用于简化 c_D 项的计算，但 L_2 也可以
        tree = KDTree(current_step_actions)
        
        # 3. 查询 k+1 个近邻 (因为第一个是自身，距离为 0)
        # distances shape: (N, k+1), indices shape: (N, k+1)
        distances, _ = tree.query(current_step_actions, k=k + 1, p=float('inf'))
        
        # 4. 提取到第 k 个近邻的距离 (epsilon_i)
        # distances[:, k] 是到第 k+1 个查询结果 (即第 k 个不同点) 的距离
        epsilon_i = distances[:, k]
        
        # 防止 log(0)，用一个极小的数替换 0 距离
        epsilon_i[epsilon_i == 0] = 1e-10 
        
        # 5. 计算 KL 估计器的核心求和项
        sum_log_epsilon = np.sum(np.log(epsilon_i))
        
        # 6. 计算最终熵值 (L_inf 距离下，c_D=2^D)
        # H = -psi(k) + psi(N) + log(c_D) + D/N * sum(log(epsilon_i))
        # log(c_D) = D * log(2)
        H_t = -psi_k + psi_N + D * np.log(2) + (D / sample_num) * sum_log_epsilon
        
        entropy_per_step.append(H_t)

    return np.mean(entropy_per_step), entropy_per_step
# 示例调用
# dummy_action = torch.randn(32, 8, 7) 
# entropy_score = estimate_policy_entropy_gaussian(dummy_action)
# print(f"Estimated Entropy: {entropy_score}")