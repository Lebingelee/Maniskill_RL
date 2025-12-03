import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


import torch
import torch.nn as nn
from torchvision import models


class VisualEncoder(nn.Module):
    """
    统一视觉编码器：
    - 支持 plain CNN 和 resnet 两种 backbone
    - 输入输出接口与原 PlainConv / ResPlainConv 完全一致
    - 适配 VLA 使用场景
    """

    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        pool_feature_map=True,
        last_act=True,
        use_res=False,   # "plain" or "resnet"

    ):
        super().__init__()

        self.out_dim = out_dim
        self.use_res = use_res

        # ==================================================
        # =============== 1. Plain CNN =====================
        # ==================================================
        if use_res == False:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(16, 32, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 128, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            )

            if pool_feature_map:
                self.pool = nn.AdaptiveMaxPool2d((1, 1))
                self.fc = make_mlp(128, [out_dim], last_act=last_act)
            else:
                self.pool = None
                self.fc = make_mlp(128 * 4 * 4 * 4, [out_dim], last_act=last_act)

        # ==================================================
        # =============== 2. ResNet Backbone ================
        # ==================================================
        elif use_res == True:
            backbone = models.resnet18(pretrained=True)

            # 多通道输入适配
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(
                    in_channels, 64,
                    kernel_size=7, stride=2, padding=3, bias=False
                )

            # 去掉 avgpool 和 fc
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            # 输出: [B, 512, H/32, W/32]

            if pool_feature_map:
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = make_mlp(512, [out_dim], last_act=last_act)
            else:
                self.pool = None
                self.fc = make_mlp(512 * 4 * 4, [out_dim], last_act=last_act)

        else:
            raise ValueError(f"Unknown backbone_type")

        self.reset_parameters()

    # ==================================================
    # =============== 参数初始化规则 ====================
    # ==================================================
    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ==================================================
    # =============== Forward（接口一致） ================
    # ==================================================
    def forward(self, image):
        x = self.cnn(image)        # Plain: [B, 128, H', W']
                                   # ResNet: [B, 512, H', W']
        if self.pool is not None:
            x = self.pool(x)       # → [B, C, 1, 1]
        x = x.flatten(1)          # → [B, C]
        x = self.fc(x)            # → [B, out_dim]
        return x


class VisualDistributionalValue(VisualEncoder):
    """
    基于 VisualEncoder 的离散分布 Value Function：
    - 输入: observation (image)
    - 输出: value 的离散概率分布
    - 支持快速计算期望 E[V]
    """

    def __init__(
        self,
        value_index,            # ① 你给出的离散 value support，如 [0.1, 0.4, 0.8, ...]
        obs_horizon,             # 时间步长度 T
        obs_state_dim,           # 关节状态维度
        in_channels=3,
        pool_feature_map=True,
        use_res=False,  # "plain" or "resnet"
        
    ):
        # --------------------------------------------------
        # 1. 先初始化父类 VisualEncoder（只作为图像编码器）
        #    注意：out_dim = 单帧视觉特征维度 D
        # --------------------------------------------------
        self.num_bins = len(value_index)
        self.visual_dim = 256  # 你原来 VisualEncoder 的输出维度 D
        super().__init__(
            in_channels=in_channels,
            out_dim=self.visual_dim,    # ★ 单帧视觉特征维度
            pool_feature_map=pool_feature_map,
            use_res=use_res,
            
            last_act=True,              # ★ 不能在最后用激活，否则会破坏 logit
        )

         # --------------------------------------------------
        # 2. value support（不参与训练）
        # --------------------------------------------------
        value_index = torch.tensor(value_index, dtype=torch.float32)
        self.register_buffer("value_index", value_index)

        fused_dim = obs_horizon * (self.visual_dim + obs_state_dim)

        self.value_head = make_mlp(
            fused_dim,
            [fused_dim, self.num_bins],   # 你也可以自行改更深
            last_act=False                # ★ 输出 logits，不能加激活
        )

    def forward(self, obs_seq):
        """
        obs_seq["rgb"]   : [B, T, C, H, W]
        obs_seq["state"] : [B, T, obs_state_dim]
        """

        rgb = obs_seq["rgb"]        # [B, T, C, H, W]
        state = obs_seq["state"]   # [B, T, obs_state_dim]

        B, T, C, H, W = rgb.shape
        assert T == self.obs_horizon, "obs_horizon 与输入时间维不一致"

        # --------------------------------------------------
        # (1) 逐帧编码图像
        #     先拉平成 [B*T, C, H, W]
        # --------------------------------------------------
        rgb = rgb.view(B * T, C, H, W)

        visual_feat = super().forward(rgb)    # [B*T, D]
        visual_feat = visual_feat.view(
            B, T, self.visual_dim
        )                                      # [B, T, D]

        # --------------------------------------------------
        # (2) 与关节状态拼接
        # --------------------------------------------------
        feature = torch.cat(
            (visual_feat, state), dim=-1
        )  # [B, T, D + obs_state_dim]

        # --------------------------------------------------
        # (3) 展平成单个向量
        # --------------------------------------------------
        feature = feature.flatten(1)   # [B, T*(D+obs_state_dim)]

        # --------------------------------------------------
        # (4) Value Head：输出 logits & prob
        # --------------------------------------------------
        value_logits = self.value_head(feature)         # [B, num_bins]
        value_prob = F.softmax(value_logits, dim=-1)   # [B, num_bins]

        return value_prob, value_logits

    # ==========================================================
    # 5. 快速计算期望 E[V]
    # ==========================================================
    @torch.no_grad()
    def expectation(self, value_prob):
        """
        value_prob: [B, num_bins]
        return:     [B, 1]  期望 value
        """
        # E[V] = sum_i p_i * v_i
        exp_value = torch.sum(
            value_prob * self.value_index.unsqueeze(0),
            dim=-1,
            keepdim=True
        )
        return exp_value