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


class GatedResidualFusion(nn.Module):
    """
    Inject local part cues into the global embedding through a tightly bounded
    residual path. The residual projection starts at zero so a warm-started
    checkpoint begins from its original global retrieval behavior.
    """

    def __init__(self, global_dim, part_dim, hidden_dim=256, init_scale=0.05, max_scale=0.10):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive.')
        if not 0.0 < max_scale <= 1.0:
            raise ValueError('max_scale must be in (0, 1].')

        self.global_norm = nn.LayerNorm(global_dim)
        self.part_norm = nn.LayerNorm(part_dim)
        self.part_proj = nn.Linear(part_dim, global_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(global_dim + part_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.max_scale = float(max_scale)

        nn.init.zeros_(self.part_proj.weight)
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

    def forward(self, global_embed, part_embed):
        gate_input = torch.cat([self.global_norm(global_embed), self.part_norm(part_embed)], dim=1)
        gate = self.gate(gate_input)
        scale = torch.clamp(self.scale, min=0.0, max=self.max_scale)
        residual = self.part_proj(part_embed)
        return global_embed + scale * gate * residual, gate


class PartAwareTokenFusion(nn.Module):
    """
    Use the global descriptor as a query over local body-part tokens, then inject
    the attended local context through a zero-initialized residual projection.
    """

    def __init__(self, global_dim, part_dim, part_num=4, token_dim=256,
                 init_scale=0.05, max_scale=0.10):
        super().__init__()
        if part_num <= 0 or part_dim % part_num != 0:
            raise ValueError('part_dim must be divisible by part_num.')
        if token_dim <= 0:
            raise ValueError('token_dim must be positive.')
        if not 0.0 < max_scale <= 1.0:
            raise ValueError('max_scale must be in (0, 1].')

        part_channels = part_dim // part_num
        self.part_num = part_num
        self.part_channels = part_channels
        self.token_dim = token_dim

        self.global_norm = nn.LayerNorm(global_dim)
        self.part_norm = nn.LayerNorm(part_channels)
        self.query = nn.Linear(global_dim, token_dim, bias=False)
        self.key = nn.Linear(part_channels, token_dim, bias=False)
        self.value = nn.Linear(part_channels, token_dim, bias=False)
        self.out = nn.Linear(token_dim, global_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.max_scale = float(max_scale)

        nn.init.zeros_(self.out.weight)

    def forward(self, global_embed, part_embed):
        parts = part_embed.reshape(part_embed.size(0), self.part_num, self.part_channels)
        parts = self.part_norm(parts)
        query = self.query(self.global_norm(global_embed)).unsqueeze(1)
        key = self.key(parts)
        attn = torch.softmax((query * key).sum(dim=-1) / math.sqrt(self.token_dim), dim=1)
        value = self.value(parts)
        context = (attn.unsqueeze(-1) * value).sum(dim=1)
        residual = self.out(context)
        scale = torch.clamp(self.scale, min=0.0, max=self.max_scale)
        return global_embed + scale * residual, attn


class ReliabilityCalibratedPartFusion(nn.Module):
    """
    Conservative global-local adapter for warm-started VI-ReID checkpoints.

    The module keeps the original global descriptor as the main route, then adds
    two bounded zero-initialized corrections: a low-rank global calibration
    adapter and a reliability-weighted local part residual. The local residual is
    projected away from the current global direction so it complements the
    baseline representation instead of overwriting it.
    """

    def __init__(self, global_dim, part_dim, part_num=4, token_dim=256,
                 adapter_dim=256, init_scale=0.04, max_scale=0.08):
        super().__init__()
        if part_num <= 0 or part_dim % part_num != 0:
            raise ValueError('part_dim must be divisible by part_num.')
        if token_dim <= 0 or adapter_dim <= 0:
            raise ValueError('token_dim and adapter_dim must be positive.')
        if not 0.0 < max_scale <= 1.0:
            raise ValueError('max_scale must be in (0, 1].')

        part_channels = part_dim // part_num
        reliability_hidden = max(part_channels // 8, 32)
        self.part_num = part_num
        self.part_channels = part_channels
        self.token_dim = token_dim
        self.max_scale = float(max_scale)

        self.global_norm = nn.LayerNorm(global_dim)
        self.part_norm = nn.LayerNorm(part_channels)

        self.global_down = nn.Linear(global_dim, adapter_dim, bias=False)
        self.global_up = nn.Linear(adapter_dim, global_dim, bias=False)
        self.global_act = nn.GELU()
        self.global_scale = nn.Parameter(torch.tensor(float(init_scale)))

        self.query = nn.Linear(global_dim, token_dim, bias=False)
        self.key = nn.Linear(part_channels, token_dim, bias=False)
        self.value = nn.Linear(part_channels, token_dim, bias=False)
        self.local_out = nn.Linear(token_dim, global_dim, bias=False)
        self.local_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.part_reliability = nn.Sequential(
            nn.Linear(part_channels, reliability_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reliability_hidden, 1, bias=True),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.global_up.weight)
        nn.init.zeros_(self.local_out.weight)

    def _orthogonalize(self, residual, reference):
        reference = F.normalize(reference.detach(), dim=1)
        projection = (residual * reference).sum(dim=1, keepdim=True) * reference
        return residual - projection

    def forward(self, global_embed, part_embed):
        global_norm = self.global_norm(global_embed)
        global_delta = self.global_up(self.global_act(self.global_down(global_norm)))

        parts = part_embed.reshape(part_embed.size(0), self.part_num, self.part_channels)
        parts = self.part_norm(parts)
        reliability = self.part_reliability(parts).squeeze(-1)
        query = self.query(global_norm).unsqueeze(1)
        key = self.key(parts)
        attn_logits = (query * key).sum(dim=-1) / math.sqrt(self.token_dim)
        attn = torch.softmax(attn_logits + torch.log(reliability.clamp_min(1e-6)), dim=1)
        value = self.value(parts)
        context = (attn.unsqueeze(-1) * value).sum(dim=1)
        local_delta = self._orthogonalize(self.local_out(context), global_embed)

        global_scale = torch.clamp(self.global_scale, min=0.0, max=self.max_scale)
        local_scale = torch.clamp(self.local_scale, min=0.0, max=self.max_scale)
        fused = global_embed + global_scale * global_delta + local_scale * local_delta
        reliability_score = (attn * reliability).sum(dim=1, keepdim=True)
        return fused, reliability_score


class BidirectionalModalityCalibration(nn.Module):
    """
    Direction-aware residual calibration for IR/RGB clips.

    The router predicts whether a clip is closer to IR or RGB statistics from the
    global and part descriptors. Two zero-initialized residual adapters then
    specialize to the two modalities while a shared calibration path keeps the
    output anchored to the warm-started baseline embedding.
    """

    def __init__(self, global_dim, part_dim, router_hidden=128, adapter_dim=128,
                 init_scale=0.03, max_scale=0.08):
        super().__init__()
        if router_hidden <= 0 or adapter_dim <= 0:
            raise ValueError('router_hidden and adapter_dim must be positive.')
        if not 0.0 < max_scale <= 1.0:
            raise ValueError('max_scale must be in (0, 1].')

        self.global_norm = nn.LayerNorm(global_dim)
        self.part_norm = nn.LayerNorm(part_dim)
        self.router = nn.Sequential(
            nn.Linear(global_dim + part_dim, router_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, 2, bias=True),
        )
        self.shared_down = nn.Linear(global_dim, adapter_dim, bias=False)
        self.shared_up = nn.Linear(adapter_dim, global_dim, bias=False)
        self.ir_down = nn.Linear(global_dim, adapter_dim, bias=False)
        self.ir_up = nn.Linear(adapter_dim, global_dim, bias=False)
        self.rgb_down = nn.Linear(global_dim, adapter_dim, bias=False)
        self.rgb_up = nn.Linear(adapter_dim, global_dim, bias=False)
        self.act = nn.GELU()
        self.shared_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.ir_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.rgb_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.max_scale = float(max_scale)

        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.shared_up.weight)
        nn.init.zeros_(self.ir_up.weight)
        nn.init.zeros_(self.rgb_up.weight)

    def _orthogonalize(self, residual, reference):
        reference = F.normalize(reference.detach(), dim=1)
        projection = (residual * reference).sum(dim=1, keepdim=True) * reference
        return residual - projection

    def _adapter(self, down, up, x):
        return up(self.act(down(x)))

    def forward(self, global_embed, part_embed):
        global_norm = self.global_norm(global_embed)
        part_norm = self.part_norm(part_embed)
        router_logits = self.router(torch.cat([global_norm, part_norm], dim=1))
        router_prob = torch.softmax(router_logits, dim=1)
        ir_prob = router_prob[:, 0:1]
        rgb_prob = router_prob[:, 1:2]

        shared_delta = self._orthogonalize(
            self._adapter(self.shared_down, self.shared_up, global_norm), global_embed
        )
        ir_delta = self._orthogonalize(
            self._adapter(self.ir_down, self.ir_up, global_norm), global_embed
        )
        rgb_delta = self._orthogonalize(
            self._adapter(self.rgb_down, self.rgb_up, global_norm), global_embed
        )

        shared_scale = torch.clamp(self.shared_scale, min=0.0, max=self.max_scale)
        ir_scale = torch.clamp(self.ir_scale, min=0.0, max=self.max_scale)
        rgb_scale = torch.clamp(self.rgb_scale, min=0.0, max=self.max_scale)
        fused = (
            global_embed
            + shared_scale * shared_delta
            + ir_prob * ir_scale * ir_delta
            + rgb_prob * rgb_scale * rgb_delta
        )
        rgb_gate = rgb_prob
        return fused, router_logits, rgb_gate
