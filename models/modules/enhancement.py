import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleResidualFusion(nn.Module):
    """
    Fuse layer3 detail cues into the final layer4 feature map with a residual gate.
    The gate is initialized small, so training starts close to the original
    backbone representation and learns to use the multi-scale path when it helps.
    """

    def __init__(self, low_channels=1024, high_channels=2048):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, low_feat, high_feat):
        low_feat = self.low_proj(low_feat)
        low_feat = F.interpolate(low_feat, size=high_feat.shape[-2:], mode='bilinear', align_corners=False)
        return high_feat + self.fusion_scale * low_feat


class LightweightChannelSpatialAttention(nn.Module):
    """
    A residual CBAM-style attention block for video ReID features.
    It refines channel saliency and coarse pedestrian regions without changing the
    feature tensor shape.
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden_channels = max(channels // reduction, 64)
        padding = spatial_kernel // 2

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.spatial_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        channel_weight = self.channel_att(x)
        x = x * (1.0 + self.channel_scale * channel_weight)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_att(torch.cat([avg_map, max_map], dim=1))
        x = x * (1.0 + self.spatial_scale * spatial_weight)

        return x


class PartGuidedAggregation(nn.Module):
    """
    Extract horizontal local body-part descriptors from the final feature map.
    Each part is GeM pooled, softly reweighted, projected with a shared lightweight
    reducer, and concatenated as a local branch.
    """

    def __init__(self, channels=2048, part_num=4, out_channels=2048, reduction=16, gem_p=3.0):
        super().__init__()
        if out_channels % part_num != 0:
            raise ValueError('out_channels must be divisible by part_num.')

        self.part_num = part_num
        self.gem_p = gem_p
        hidden_channels = max(channels // reduction, 64)
        part_channels = out_channels // part_num

        self.part_score = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1, bias=True),
        )
        self.part_reduce = nn.Sequential(
            nn.Linear(channels, part_channels, bias=False),
            nn.LayerNorm(part_channels),
            nn.ReLU(inplace=True),
        )

    def _gem_pool(self, x, eps=1e-6):
        x = x.clamp(min=eps).pow(self.gem_p)
        x = x.mean(dim=(-2, -1)).pow(1.0 / self.gem_p)
        return x

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        if h < self.part_num:
            x = F.interpolate(x, size=(self.part_num, w), mode='bilinear', align_corners=False)

        part_feats = []
        for stripe in torch.tensor_split(x, self.part_num, dim=2):
            part_feats.append(self._gem_pool(stripe))
        part_feats = torch.stack(part_feats, dim=1)

        part_weights = torch.softmax(self.part_score(part_feats).squeeze(-1), dim=1)
        part_feats = part_feats * (1.0 + part_weights.unsqueeze(-1))

        reduced_parts = [self.part_reduce(part_feats[:, i, :]) for i in range(self.part_num)]
        return torch.cat(reduced_parts, dim=1)


class TemporalEmbeddingRefinement(nn.Module):
    """
    Low-rank temporal residual refinement for frame-level video embeddings.

    It is intentionally zero-initialized at the output projection so a warm-started
    checkpoint begins with the same retrieval behavior and learns temporal context
    only when the training losses support it.
    """

    def __init__(self, in_channels, hidden_channels=256, dropout=0.0):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError('hidden_channels must be positive.')
        self.norm = nn.LayerNorm(in_channels)
        self.down = nn.Linear(in_channels, hidden_channels, bias=False)
        self.temporal_conv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1,
            groups=hidden_channels, bias=False,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(hidden_channels, in_channels, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        residual = x
        y = self.down(self.norm(x))
        y = self.temporal_conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.act(y)
        y = y * self.gate(y)
        y = self.dropout(y)
        return residual + self.scale * self.up(y)


class AdaptiveFusionGate(nn.Module):
    """
    Predict a per-track local-feature weight for dual-fusion inference.

    The final layer is initialized to produce init_alpha for every sample, so a
    warm-started fixed-alpha checkpoint begins close to its known retrieval
    behavior and can learn sample-specific deviations during fine-tuning.
    """

    def __init__(self, global_dim, part_dim, hidden_dim=256, init_alpha=0.01,
                 min_alpha=0.0, max_alpha=0.12):
        super().__init__()
        if not 0.0 <= min_alpha < max_alpha <= 1.0:
            raise ValueError('Adaptive gate requires 0 <= min_alpha < max_alpha <= 1.')

        min_alpha_float = float(min_alpha)
        max_alpha_float = float(max_alpha)
        self.register_buffer('min_alpha', torch.tensor(min_alpha_float))
        self.register_buffer('max_alpha', torch.tensor(max_alpha_float))
        self.global_norm = nn.LayerNorm(global_dim)
        self.part_norm = nn.LayerNorm(part_dim)
        self.net = nn.Sequential(
            nn.Linear(global_dim + part_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=True),
        )

        relative_alpha = (float(init_alpha) - min_alpha_float) / (max_alpha_float - min_alpha_float)
        relative_alpha = min(max(relative_alpha, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(relative_alpha / (1.0 - relative_alpha))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, global_embed, part_embed):
        gate_input = torch.cat([self.global_norm(global_embed), self.part_norm(part_embed)], dim=1)
        gate = torch.sigmoid(self.net(gate_input))
        return self.min_alpha + (self.max_alpha - self.min_alpha) * gate
