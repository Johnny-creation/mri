import torch
import torch.nn as nn
from torchvision import models
import snntorch as snn
from snntorch import surrogate

import math

class SCAEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SE) 的 ECA-Net/SECA 实现。
    使用 1-D 卷积替换全连接降维，实现局部跨通道交互。
    """

    def __init__(self, channel, gamma=2, b=1):
        # 兼容性考虑：原 SEBlock 使用 reduction 参数，这里我们采用 SECA 的参数
        super().__init__()

        # 1. 计算自适应卷积核 k
        # k = |log2(channel) / gamma + b|_odd
        k = int(abs((math.log2(channel) / gamma) + b))
        k = k if k % 2 else k + 1  # 保证 k 是奇数

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 (GAP)

        # 2. 核心：用 1-D 卷积替换 FC 层
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k,
                                padding=k // 2, bias=False)  # 1-D 卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)

        # 1. Squeeze: GAP
        y = self.avg_pool(x)  # (B, C, 1, 1)

        # 2. Excitation Preparation: 调整维度以适应 1-D 卷积 (输入要求 B, 1, C)
        # (B, C, 1, 1) -> (B, C, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(1, 2)

        # 3. Excitation: 1-D 卷积
        y = self.conv1d(y)  # (B, 1, C)
        y = self.sigmoid(y)

        # 4. Scale: 调整回 (B, C, 1, 1) 形式
        y = y.transpose(1, 2).unsqueeze(-1)

        return x * y  # 逐通道加权

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn

class DropBlock2D(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        return x * (1 - mask)

class EfficientNetB0_2Channel_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.reduce_mid = nn.Conv2d(40, 128, 1)
        self.reduce_high = nn.Conv2d(1280, 128, 1)
        self.se = SCAEBlock(256)
        self.spatial_attn = SpatialAttention()
        self.dropblock = DropBlock2D(block_size=7, drop_prob=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if i == 3:
                mid_feat = self.reduce_mid(x)
        high_feat = self.reduce_high(x)
        # 统一空间尺寸
        mid_feat = nn.functional.adaptive_avg_pool2d(mid_feat, high_feat.shape[2:])
        x = torch.cat([mid_feat, high_feat], dim=1)  # (B, 256, H, W)
        x = self.se(x)
        x = self.spatial_attn(x)
        x = self.dropblock(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, dropout_rate=0.3, use_snn_head=False, T=10, beta=0.9):
        super(MultiTaskModel, self).__init__()
        self.base_model = EfficientNetB0_2Channel_Fusion()
        num_features = 256
        self.use_snn_head = use_snn_head
        self.T = T
        if self.use_snn_head:
            # LIF 脉冲单元，使用替代梯度以支持反传
            spike_grad = surrogate.fast_sigmoid()
            self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=False)
            
        
        self.lesion_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
        
        for head in [self.lesion_head, self.time_head]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.base_model.model.features):
            x = layer(x)
            if i == 3:
                mid_feat = x
        high_feat = x
        
        mid_feat = self.base_model.reduce_mid(mid_feat)
        high_feat = self.base_model.reduce_high(high_feat)
        
        mid_feat = nn.functional.adaptive_avg_pool2d(mid_feat, high_feat.shape[2:])
        features = torch.cat([mid_feat, high_feat], dim=1)
        
        features = self.base_model.se(features)
        features = self.base_model.spatial_attn(features)
        features = self.base_model.dropblock(features)
        
        # features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # lesion_out = self.lesion_head(features)
        # time_out = self.time_head(features)

        features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)  # [B,256]

        if self.use_snn_head:
            # 在读出层进行 T 步脉冲积分放电
            mem = self.lif.init_leaky()   # 初始化膜电位
            spk_sum = 0
            for _ in range(self.T):
                spk, mem = self.lif(features, mem)  # spk: [B,256] 的0/1脉冲
                spk_sum = spk_sum + spk
            feats = spk_sum / float(self.T)         # 时间平均的脉冲率
        else:
            feats = features

        lesion_out = self.lesion_head(feats)
        time_out = self.time_head(feats)
        
        return lesion_out, time_out