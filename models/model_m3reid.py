# ------------------------------------------------------------------------------
# File:    M3-ReID/models/model_m3reid.py
#
# Description:
#    This module defines the complete M3-ReID network architecture.
#    It integrates the ResNet-50 backbone with Spatio-temporal Non-Local blocks
#    and the Multi-View Learning (MVL) module to extract robust video representations.
#
# Key Features:
# - ResNet-50 Backbone with configurable Non-Local block insertion.
# - Integration of Multi-View Learning Attention for spatio-temporal feature mining.
# - Multi-Granularity feature extraction (Frame-level and Video-level).
#
# Classes:
# - M3ReID
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.backbones.resnet import resnet50
from models.modules.non_local import NonLocal
from models.modules.mvl_attention import MultiViewLearningAttention
from models.modules.normalize import Normalize
from models.modules.enhancement import LightweightChannelSpatialAttention
from models.modules.enhancement import MultiScaleResidualFusion
from models.modules.enhancement import AdaptiveFusionGate
from models.modules.enhancement import GatedResidualFusion
from models.modules.enhancement import PartGuidedAggregation
from models.modules.enhancement import PartAwareTokenFusion
from models.modules.enhancement import ReliabilityCalibratedPartFusion
from models.modules.enhancement import BidirectionalModalityCalibration
from models.modules.enhancement import ModalityInvariantSpecificCalibration
from models.modules.enhancement import AnchorPreservingProjectionFusion
from models.modules.enhancement import AdaptiveProjectionCalibrationGate
from models.modules.enhancement import TemporalEmbeddingRefinement


class M3ReID(nn.Module):
    """
    The main M3-ReID model architecture.
    It unifies Multi-View, Multi-Granularity, and Multi-Modality components into
    an end-to-end trainable framework.

    'M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification'
    by Liang et al. See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
    """

    def __init__(self, sample_seq_num, class_num, use_enhancements=False, m3plus_mode='full', part_num=4,
                 mvl_num_heads=2, part_dim=2048, feature_dropout=0.0, fusion_alpha=0.2,
                 calibration_alpha=0.0, projection_alpha=0.0, grad_checkpoint_head=False,
                 temporal_dim=256, temporal_dropout=0.0,
                 adaptive_gate_min=0.0, adaptive_gate_max=0.12):
        """
        Initialize the M3-ReID model.

        Process:
        1. Load pre-trained ResNet-50 backbone.
        2. Define indices for inserting Non-Local blocks into ResNet layers.
        3. Initialize Non-Local modules for intermediate layers (mainly Layer 2 and 3).
        4. Initialize the Multi-View Learning (MVL) attention module.
        5. Setup the BNNeck (BatchNorm1d).
        6. Setup the Classification heads (Linear layers) for both frame-level and video-level supervision.

        Args:
            sample_seq_num (int): Number of frames in input video clips.
            class_num (int): Number of identity classes for classification.
        """

        super(M3ReID, self).__init__()

        self.embedding_dim = 2048  # ResNet
        self.sample_seq_num = sample_seq_num
        self.class_num = class_num
        self.use_enhancements = use_enhancements
        if m3plus_mode not in (
            'full', 'part_only', 'local_residual', 'dual_fusion',
            'temporal_dual_fusion', 'adaptive_dual_fusion',
            'supervised_dual_fusion', 'gated_residual_fusion', 'part_token_fusion',
            'reliability_part_fusion', 'bidirectional_calibration',
            'invariant_specific_calibration', 'anchor_projection_fusion',
            'dual_calibrated_fusion', 'temporal_dual_calibrated_fusion',
            'projection_calibrated_fusion', 'quad_calibrated_fusion',
            'adaptive_projection_calibrated_fusion'
        ):
            raise ValueError(f'Unsupported m3plus_mode: {m3plus_mode}')
        self.m3plus_mode = m3plus_mode
        self.use_feature_enhancers = use_enhancements and m3plus_mode == 'full'
        self.use_part_branch = use_enhancements and m3plus_mode in (
            'full', 'part_only', 'local_residual', 'dual_fusion',
            'temporal_dual_fusion', 'adaptive_dual_fusion',
            'supervised_dual_fusion', 'gated_residual_fusion', 'part_token_fusion',
            'reliability_part_fusion', 'bidirectional_calibration',
            'invariant_specific_calibration', 'anchor_projection_fusion',
            'dual_calibrated_fusion', 'temporal_dual_calibrated_fusion',
            'projection_calibrated_fusion', 'quad_calibrated_fusion',
            'adaptive_projection_calibrated_fusion'
        )
        self.use_local_residual = use_enhancements and m3plus_mode == 'local_residual'
        self.use_dual_fusion = use_enhancements and m3plus_mode in (
            'dual_fusion', 'temporal_dual_fusion', 'adaptive_dual_fusion',
            'supervised_dual_fusion'
        )
        self.use_supervised_dual_fusion = use_enhancements and m3plus_mode == 'supervised_dual_fusion'
        self.use_part_supervision = use_enhancements and m3plus_mode in (
            'dual_fusion', 'temporal_dual_fusion', 'adaptive_dual_fusion',
            'supervised_dual_fusion', 'gated_residual_fusion', 'part_token_fusion',
            'reliability_part_fusion', 'bidirectional_calibration',
            'invariant_specific_calibration', 'anchor_projection_fusion',
            'dual_calibrated_fusion', 'temporal_dual_calibrated_fusion',
            'projection_calibrated_fusion', 'quad_calibrated_fusion',
            'adaptive_projection_calibrated_fusion'
        )
        self.use_gated_residual_fusion = use_enhancements and m3plus_mode == 'gated_residual_fusion'
        self.use_part_token_fusion = use_enhancements and m3plus_mode == 'part_token_fusion'
        self.use_reliability_part_fusion = use_enhancements and m3plus_mode == 'reliability_part_fusion'
        self.use_bidirectional_calibration = use_enhancements and m3plus_mode == 'bidirectional_calibration'
        self.use_invariant_specific_calibration = (
            use_enhancements and m3plus_mode == 'invariant_specific_calibration'
        )
        self.use_anchor_projection_fusion = use_enhancements and m3plus_mode == 'anchor_projection_fusion'
        self.use_dual_calibrated_fusion = (
            use_enhancements and m3plus_mode in (
                'dual_calibrated_fusion', 'temporal_dual_calibrated_fusion'
            )
        )
        self.use_projection_calibrated_fusion = (
            use_enhancements and m3plus_mode == 'projection_calibrated_fusion'
        )
        self.use_quad_calibrated_fusion = (
            use_enhancements and m3plus_mode == 'quad_calibrated_fusion'
        )
        self.use_adaptive_projection_calibrated_fusion = (
            use_enhancements and m3plus_mode == 'adaptive_projection_calibrated_fusion'
        )
        self.use_temporal_refine = (
            use_enhancements and m3plus_mode in (
                'temporal_dual_fusion', 'temporal_dual_calibrated_fusion'
            )
        )
        self.use_adaptive_dual_fusion = use_enhancements and m3plus_mode == 'adaptive_dual_fusion'
        self.fusion_alpha = float(fusion_alpha)
        self.calibration_alpha = float(calibration_alpha)
        self.projection_alpha = float(projection_alpha)
        self.grad_checkpoint_head = grad_checkpoint_head

        self.backbone = resnet50(pretrained=True)

        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList(
            [NonLocal(256, mode='THW') for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [NonLocal(512, mode='THW') for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [NonLocal(1024, mode='THW') for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [NonLocal(2048, mode='THW') for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        num_heads = mvl_num_heads
        self.mvl_attention = MultiViewLearningAttention(self.embedding_dim, num_heads=num_heads, mode='gem')

        self.embedding_dim = self.embedding_dim * (num_heads * 3)
        if self.use_enhancements:
            if part_dim <= 0:
                raise ValueError('part_dim must be positive when M3Plus enhancements are enabled.')
        if self.use_feature_enhancers:
            self.multi_scale_fusion = MultiScaleResidualFusion(low_channels=1024, high_channels=2048)
            self.feature_attention = LightweightChannelSpatialAttention(2048)
        if self.use_part_branch:
            self.part_aggregation = PartGuidedAggregation(channels=2048, part_num=part_num, out_channels=part_dim)
            if self.use_local_residual:
                self.part_residual = nn.Linear(part_dim, self.embedding_dim, bias=False)
                nn.init.zeros_(self.part_residual.weight)
                self.part_residual_scale = nn.Parameter(torch.tensor(0.1))
            elif self.use_gated_residual_fusion:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.gated_residual_fusion = GatedResidualFusion(self.embedding_dim, part_dim)
            elif self.use_part_token_fusion:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.part_token_fusion = PartAwareTokenFusion(
                    self.embedding_dim, part_dim, part_num=part_num
                )
            elif self.use_reliability_part_fusion:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.reliability_part_fusion = ReliabilityCalibratedPartFusion(
                    self.embedding_dim, part_dim, part_num=part_num
                )
            elif self.use_bidirectional_calibration:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.bidirectional_calibration = BidirectionalModalityCalibration(
                    self.embedding_dim, part_dim
                )
            elif self.use_invariant_specific_calibration:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.invariant_specific_calibration = ModalityInvariantSpecificCalibration(
                    self.embedding_dim, part_dim
                )
            elif self.use_anchor_projection_fusion:
                self.projection_dim = 512
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.projection_bn_neck = nn.BatchNorm1d(self.projection_dim)
                nn.init.constant_(self.projection_bn_neck.bias, 0)
                self.projection_bn_neck.bias.requires_grad_(False)
                self.projection_classifier = nn.Linear(self.projection_dim, class_num, bias=False)
                self.anchor_projection_fusion = AnchorPreservingProjectionFusion(
                    self.embedding_dim, part_dim, projection_dim=self.projection_dim
                )
            elif self.use_dual_calibrated_fusion:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.calibration_bn_neck = nn.BatchNorm1d(self.embedding_dim)
                nn.init.constant_(self.calibration_bn_neck.bias, 0)
                self.calibration_bn_neck.bias.requires_grad_(False)
                self.calibration_classifier_frame = nn.Linear(self.embedding_dim, class_num, bias=False)
                self.calibration_classifier = nn.Linear(self.embedding_dim, class_num, bias=False)
                self.dual_calibration_fusion = GatedResidualFusion(
                    self.embedding_dim, part_dim, init_scale=0.03, max_scale=0.08
                )
            elif (
                self.use_projection_calibrated_fusion
                or self.use_adaptive_projection_calibrated_fusion
                or self.use_quad_calibrated_fusion
            ):
                self.projection_dim = 512
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                self.projection_bn_neck = nn.BatchNorm1d(self.projection_dim)
                nn.init.constant_(self.projection_bn_neck.bias, 0)
                self.projection_bn_neck.bias.requires_grad_(False)
                self.projection_classifier = nn.Linear(self.projection_dim, class_num, bias=False)
                self.anchor_projection_fusion = AnchorPreservingProjectionFusion(
                    self.embedding_dim, part_dim, projection_dim=self.projection_dim
                )
                self.calibration_bn_neck = nn.BatchNorm1d(self.embedding_dim)
                nn.init.constant_(self.calibration_bn_neck.bias, 0)
                self.calibration_bn_neck.bias.requires_grad_(False)
                self.calibration_classifier_frame = nn.Linear(self.embedding_dim, class_num, bias=False)
                self.calibration_classifier = nn.Linear(self.embedding_dim, class_num, bias=False)
                calibration_init_scale = 0.03 if self.use_quad_calibrated_fusion else 0.02
                calibration_max_scale = 0.08 if self.use_quad_calibrated_fusion else 0.06
                self.dual_calibration_fusion = GatedResidualFusion(
                    self.embedding_dim, part_dim,
                    init_scale=calibration_init_scale,
                    max_scale=calibration_max_scale,
                )
                if self.use_adaptive_projection_calibrated_fusion:
                    adaptive_output_dim = self.embedding_dim + self.projection_dim + self.embedding_dim
                    self.adaptive_projection_bn_neck = nn.BatchNorm1d(adaptive_output_dim)
                    nn.init.constant_(self.adaptive_projection_bn_neck.bias, 0)
                    self.adaptive_projection_bn_neck.bias.requires_grad_(False)
                    self.adaptive_projection_classifier = nn.Linear(adaptive_output_dim, class_num, bias=False)
                    self.adaptive_projection_calibration_gate = AdaptiveProjectionCalibrationGate(
                        self.embedding_dim, self.projection_dim,
                        init_projection_alpha=fusion_alpha,
                        init_calibration_alpha=calibration_alpha
                    )
            elif self.use_dual_fusion:
                self.part_bn_neck = nn.BatchNorm1d(part_dim)
                nn.init.constant_(self.part_bn_neck.bias, 0)
                self.part_bn_neck.bias.requires_grad_(False)
                self.part_classifier_frame = nn.Linear(part_dim, class_num, bias=False)
                self.part_classifier = nn.Linear(part_dim, class_num, bias=False)
                if self.use_adaptive_dual_fusion:
                    self.adaptive_fusion_gate = AdaptiveFusionGate(
                        self.embedding_dim, part_dim, init_alpha=fusion_alpha,
                        min_alpha=adaptive_gate_min, max_alpha=adaptive_gate_max
                    )
                if self.use_adaptive_dual_fusion or self.use_supervised_dual_fusion:
                    self.fusion_bn_neck = nn.BatchNorm1d(self.embedding_dim + part_dim)
                    nn.init.constant_(self.fusion_bn_neck.bias, 0)
                    self.fusion_bn_neck.bias.requires_grad_(False)
                    self.fusion_classifier = nn.Linear(self.embedding_dim + part_dim, class_num, bias=False)
            else:
                self.embedding_dim += part_dim

        self.bn_neck = nn.BatchNorm1d(self.embedding_dim)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.use_temporal_refine:
            self.temporal_refine = TemporalEmbeddingRefinement(
                self.embedding_dim, hidden_channels=temporal_dim, dropout=temporal_dropout
            )
        self.feature_dropout = nn.Dropout(p=feature_dropout) if feature_dropout > 0 else nn.Identity()

        self.classifier_frame = nn.Linear(self.embedding_dim, class_num, bias=False)
        self.classifier = nn.Linear(self.embedding_dim, class_num, bias=False)

        self.l2_norm = Normalize(power=2)
        if self.use_dual_fusion:
            self.output_dim = self.embedding_dim + part_dim
        elif self.use_anchor_projection_fusion:
            self.output_dim = self.embedding_dim + self.projection_dim
        elif self.use_dual_calibrated_fusion:
            self.output_dim = self.embedding_dim + part_dim + self.embedding_dim
        elif self.use_projection_calibrated_fusion or self.use_adaptive_projection_calibrated_fusion:
            self.output_dim = self.embedding_dim + self.projection_dim + self.embedding_dim
        else:
            self.output_dim = self.embedding_dim

    def _checkpoint_if_enabled(self, fn, *args):
        if self.training and self.grad_checkpoint_head:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _mvl_forward(self, global_feat):
        return self.mvl_attention(global_feat)

    def _adaptive_fusion(self, global_embed, part_embed):
        alpha = self.adaptive_fusion_gate(global_embed, part_embed)
        global_eval = self.l2_norm(global_embed) * torch.sqrt(1.0 - alpha)
        part_eval = self.l2_norm(part_embed) * torch.sqrt(alpha)
        return torch.cat([global_eval, part_eval], dim=1), alpha

    def _fixed_dual_fusion(self, global_embed, part_embed):
        alpha = min(max(self.fusion_alpha, 0.0), 1.0)
        alpha_tensor = global_embed.new_full((global_embed.size(0), 1), alpha)
        global_eval = self.l2_norm(global_embed) * ((1.0 - alpha) ** 0.5)
        part_eval = self.l2_norm(part_embed) * (alpha ** 0.5)
        return torch.cat([global_eval, part_eval], dim=1), alpha_tensor

    def _dual_calibrated_eval(self, global_embed, part_embed, calibration_embed):
        part_alpha = min(max(self.fusion_alpha, 0.0), 1.0)
        calibration_alpha = min(max(self.calibration_alpha, 0.0), 1.0 - part_alpha)
        global_alpha = max(1.0 - part_alpha - calibration_alpha, 0.0)
        global_eval = self.l2_norm(global_embed) * (global_alpha ** 0.5)
        part_eval = self.l2_norm(part_embed) * (part_alpha ** 0.5)
        calibration_eval = self.l2_norm(calibration_embed) * (calibration_alpha ** 0.5)
        return torch.cat([global_eval, part_eval, calibration_eval], dim=1)

    def _projection_calibrated_eval(self, global_embed, projection_embed, calibration_embed):
        projection_alpha = min(max(self.fusion_alpha, 0.0), 1.0)
        calibration_alpha = min(max(self.calibration_alpha, 0.0), 1.0 - projection_alpha)
        global_alpha = max(1.0 - projection_alpha - calibration_alpha, 0.0)
        global_eval = self.l2_norm(global_embed) * (global_alpha ** 0.5)
        projection_eval = self.l2_norm(projection_embed) * (projection_alpha ** 0.5)
        calibration_eval = self.l2_norm(calibration_embed) * (calibration_alpha ** 0.5)
        return torch.cat([global_eval, projection_eval, calibration_eval], dim=1)

    def _quad_calibrated_eval(self, global_embed, part_embed, projection_embed, calibration_embed):
        part_alpha = min(max(self.fusion_alpha, 0.0), 1.0)
        projection_alpha = min(max(self.projection_alpha, 0.0), 1.0 - part_alpha)
        calibration_alpha = min(max(self.calibration_alpha, 0.0), 1.0 - part_alpha - projection_alpha)
        global_alpha = max(1.0 - part_alpha - projection_alpha - calibration_alpha, 0.0)
        global_eval = self.l2_norm(global_embed) * (global_alpha ** 0.5)
        part_eval = self.l2_norm(part_embed) * (part_alpha ** 0.5)
        projection_eval = self.l2_norm(projection_embed) * (projection_alpha ** 0.5)
        calibration_eval = self.l2_norm(calibration_embed) * (calibration_alpha ** 0.5)
        return torch.cat([global_eval, part_eval, projection_eval, calibration_eval], dim=1)

    def _adaptive_projection_calibrated_eval(self, global_embed, projection_embed, calibration_embed):
        projection_alpha, calibration_alpha = self.adaptive_projection_calibration_gate(
            global_embed, projection_embed, calibration_embed
        )
        global_alpha = (1.0 - projection_alpha - calibration_alpha).clamp_min(1e-6)
        global_eval = self.l2_norm(global_embed) * torch.sqrt(global_alpha)
        projection_eval = self.l2_norm(projection_embed) * torch.sqrt(projection_alpha.clamp_min(1e-6))
        calibration_eval = self.l2_norm(calibration_embed) * torch.sqrt(calibration_alpha.clamp_min(1e-6))
        return torch.cat([global_eval, projection_eval, calibration_eval], dim=1), projection_alpha, calibration_alpha

    def set_scheme_d_base_trainable(self, trainable):
        if not self.use_part_supervision:
            return
        part_prefixes = (
            'part_aggregation', 'part_bn_neck', 'part_classifier', 'temporal_refine',
            'adaptive_fusion_gate', 'fusion_bn_neck', 'fusion_classifier',
            'gated_residual_fusion', 'part_token_fusion', 'reliability_part_fusion',
            'bidirectional_calibration', 'invariant_specific_calibration',
            'anchor_projection_fusion', 'projection_bn_neck', 'projection_classifier',
            'dual_calibration_fusion', 'calibration_bn_neck', 'calibration_classifier',
            'adaptive_projection_calibration_gate', 'adaptive_projection_bn_neck',
            'adaptive_projection_classifier',
        )
        for name, param in self.named_parameters():
            param.requires_grad_(trainable or name.startswith(part_prefixes))

        base_modules = [
            self.backbone, self.NL_1, self.NL_2, self.NL_3, self.NL_4,
            self.mvl_attention, self.bn_neck, self.classifier_frame, self.classifier,
        ]
        for module in base_modules:
            module.train(trainable)

    def forward(self, inputs):
        """
        Forward pass of the M3-ReID network.

        Process:
        1. Stem: Reshape input [B, T, C, H, W] to [B*T, C, H, W] and pass through ResNet stem.
        2. Backbone & Non-Local: Iterate through ResNet layers. At specific indices, reshape features
           to include the temporal dimension [B, T, C, H, W], apply the Spatiotemporal Non-Local block,
           and reshape back to [B*T, C, H, W].
        3. Multi-View Learning: Pass global features to the MVL module to obtain multi-view aggregated
           embeddings and attention masks.
        4. Bottleneck: Apply BNNeck to the embeddings.
        5. Multi-Granularity:
           - Frame-level: Keep features as [B, T, C].
           - Video-level: Average pool frame features across time to get [B, C].
        6. Heads: Compute logits for both frame and video representations.

        Args:
            inputs (Tensor): Input video tensor of shape [B, T, C, H, W].

        Returns:
            If training:
                tuple: (x_embed, x_embed_mean, x_logits, x_logits_mean, mvl_att_masks)
            If evaluating:
                Tensor: Normalized video-level embedding [B, C].
        """

        b, t, c, h, w = inputs.shape
        inputs = inputs.reshape(-1, c, h, w)

        inputs = self.backbone.conv1(inputs)
        inputs = self.backbone.bn1(inputs)
        inputs = self.backbone.relu(inputs)
        inputs = self.backbone.maxpool(inputs)

        x = inputs
        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.backbone.layer1)):
            x = self.backbone.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = x.reshape(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.backbone.layer2)):
            x = self.backbone.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = x.reshape(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.backbone.layer3)):
            x = self.backbone.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = x.reshape(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Layer 4
        layer3_feat = x
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.backbone.layer4)):
            x = self.backbone.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = x.reshape(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        global_feat = x

        if self.use_feature_enhancers:
            global_feat = self.multi_scale_fusion(layer3_feat, global_feat)
            global_feat = self._checkpoint_if_enabled(self.feature_attention, global_feat)

        _, C, H, W = global_feat.shape
        global_feat = global_feat.reshape(b, t, C, H, W)
        x_pool, mvl_att_masks = self._checkpoint_if_enabled(self._mvl_forward, global_feat)
        part_pool = None
        fusion_gate = None
        projection_gate = None
        calibration_gate = None
        router_logits = None
        invariant_pool = None
        modality_logits = None
        projection_pool = None
        calibration_pool = None
        if self.use_part_branch:
            part_pool = self._checkpoint_if_enabled(self.part_aggregation, global_feat)
            if self.use_local_residual:
                x_pool = x_pool + self.part_residual_scale * self.part_residual(part_pool)
            elif self.use_gated_residual_fusion:
                x_pool, fusion_gate = self._checkpoint_if_enabled(
                    self.gated_residual_fusion, x_pool, part_pool
                )
            elif self.use_part_token_fusion:
                x_pool, fusion_gate = self._checkpoint_if_enabled(
                    self.part_token_fusion, x_pool, part_pool
                )
            elif self.use_reliability_part_fusion:
                x_pool, fusion_gate = self._checkpoint_if_enabled(
                    self.reliability_part_fusion, x_pool, part_pool
                )
            elif self.use_bidirectional_calibration:
                x_pool, router_logits, fusion_gate = self._checkpoint_if_enabled(
                    self.bidirectional_calibration, x_pool, part_pool
                )
            elif self.use_invariant_specific_calibration:
                x_pool, router_logits, fusion_gate, invariant_pool, modality_logits = self._checkpoint_if_enabled(
                    self.invariant_specific_calibration, x_pool, part_pool
                )
            elif self.use_anchor_projection_fusion:
                projection_pool, fusion_gate = self._checkpoint_if_enabled(
                    self.anchor_projection_fusion, x_pool, part_pool
                )
            elif self.use_dual_calibrated_fusion:
                calibration_pool, calibration_gate = self._checkpoint_if_enabled(
                    self.dual_calibration_fusion, x_pool, part_pool
                )
                fusion_gate = calibration_gate
            elif (
                self.use_projection_calibrated_fusion
                or self.use_adaptive_projection_calibrated_fusion
                or self.use_quad_calibrated_fusion
            ):
                projection_pool, projection_gate = self._checkpoint_if_enabled(
                    self.anchor_projection_fusion, x_pool, part_pool
                )
                calibration_pool, calibration_gate = self._checkpoint_if_enabled(
                    self.dual_calibration_fusion, x_pool, part_pool
                )
                fusion_gate = projection_gate
            elif not self.use_dual_fusion:
                x_pool = torch.cat([x_pool, part_pool], dim=1)

        x_embed = self.bn_neck(x_pool)

        b, c = x_pool.shape
        x_pool = x_pool.reshape(-1, t, c)
        x_pool_mean = torch.mean(x_pool, dim=1)
        x_embed = x_embed.reshape(-1, t, c)
        if self.use_temporal_refine:
            x_embed = self._checkpoint_if_enabled(self.temporal_refine, x_embed)
        x_embed_mean = torch.mean(x_embed, dim=1)

        part_aux = None
        if self.use_part_supervision:
            part_embed = self.part_bn_neck(part_pool)
            part_embed = part_embed.reshape(-1, t, part_embed.shape[-1])
            part_embed_mean = torch.mean(part_embed, dim=1)
            part_aux = (part_embed, part_embed_mean)

        if self.training:
            b, t, c = x_embed.shape
            x_logits = self.classifier_frame(self.feature_dropout(x_embed.reshape(b * t, c))).reshape(b, t, -1)
            x_logits_mean = self.classifier(self.feature_dropout(x_embed_mean))
            if self.use_part_supervision:
                part_embed, part_embed_mean = part_aux
                part_b, part_t, part_c = part_embed.shape
                part_logits = self.part_classifier_frame(
                    self.feature_dropout(part_embed.reshape(part_b * part_t, part_c))
                ).reshape(part_b, part_t, -1)
                part_logits_mean = self.part_classifier(self.feature_dropout(part_embed_mean))
                aux = {
                    'part_embed': part_embed,
                    'part_embed_mean': part_embed_mean,
                    'part_logits': part_logits,
                    'part_logits_mean': part_logits_mean,
                }
                if fusion_gate is not None:
                    aux['fusion_gate'] = fusion_gate.reshape(-1, t, fusion_gate.shape[-1]).mean(dim=1)
                if (
                    (self.use_bidirectional_calibration or self.use_invariant_specific_calibration)
                    and router_logits is not None
                ):
                    aux['router_logits'] = router_logits.reshape(-1, t, router_logits.shape[-1]).mean(dim=1)
                if self.use_invariant_specific_calibration:
                    aux['invariant_embed_mean'] = invariant_pool.reshape(-1, t, invariant_pool.shape[-1]).mean(dim=1)
                    aux['modality_logits'] = modality_logits.reshape(-1, t, modality_logits.shape[-1]).mean(dim=1)
                if (
                    self.use_anchor_projection_fusion
                    or self.use_projection_calibrated_fusion
                    or self.use_quad_calibrated_fusion
                    or self.use_adaptive_projection_calibrated_fusion
                ):
                    projection_embed = self.projection_bn_neck(projection_pool)
                    projection_embed = projection_embed.reshape(-1, t, projection_embed.shape[-1])
                    projection_embed_mean = torch.mean(projection_embed, dim=1)
                    aux['projection_embed'] = projection_embed
                    aux['projection_embed_mean'] = projection_embed_mean
                    aux['projection_logits_mean'] = self.projection_classifier(
                        self.feature_dropout(projection_embed_mean)
                    )
                    if projection_gate is not None:
                        aux['projection_gate'] = projection_gate.reshape(
                            -1, t, projection_gate.shape[-1]
                        ).mean(dim=1)
                if (
                    self.use_dual_calibrated_fusion
                    or self.use_projection_calibrated_fusion
                    or self.use_quad_calibrated_fusion
                    or self.use_adaptive_projection_calibrated_fusion
                ):
                    calibration_embed = self.calibration_bn_neck(calibration_pool)
                    calibration_embed = calibration_embed.reshape(-1, t, calibration_embed.shape[-1])
                    calibration_embed_mean = torch.mean(calibration_embed, dim=1)
                    aux['calibration_embed'] = calibration_embed
                    aux['calibration_embed_mean'] = calibration_embed_mean
                    aux['calibration_logits'] = self.calibration_classifier_frame(
                        self.feature_dropout(calibration_embed.reshape(-1, calibration_embed.shape[-1]))
                    ).reshape(calibration_embed.size(0), calibration_embed.size(1), -1)
                    aux['calibration_logits_mean'] = self.calibration_classifier(
                        self.feature_dropout(calibration_embed_mean)
                    )
                    if calibration_gate is not None:
                        aux['calibration_gate'] = calibration_gate.reshape(
                            -1, t, calibration_gate.shape[-1]
                        ).mean(dim=1)
                    if self.use_adaptive_projection_calibrated_fusion:
                        adaptive_eval, projection_alpha, calibration_alpha = self._adaptive_projection_calibrated_eval(
                            x_embed_mean, aux['projection_embed_mean'], calibration_embed_mean
                        )
                        aux['fusion_embed_mean'] = adaptive_eval
                        adaptive_bn = self.adaptive_projection_bn_neck(adaptive_eval)
                        aux['fusion_logits_mean'] = self.adaptive_projection_classifier(
                            self.feature_dropout(adaptive_bn)
                        )
                        aux['projection_alpha'] = projection_alpha
                        aux['calibration_alpha'] = calibration_alpha
                if self.use_adaptive_dual_fusion or self.use_supervised_dual_fusion:
                    if self.use_adaptive_dual_fusion:
                        fusion_embed_mean, fusion_gate_mean = self._adaptive_fusion(x_embed_mean, part_embed_mean)
                    else:
                        fusion_embed_mean, fusion_gate_mean = self._fixed_dual_fusion(x_embed_mean, part_embed_mean)
                    fusion_bn = self.fusion_bn_neck(fusion_embed_mean)
                    aux.update({
                        'fusion_embed_mean': fusion_embed_mean,
                        'fusion_logits_mean': self.fusion_classifier(self.feature_dropout(fusion_bn)),
                        'fusion_gate': fusion_gate_mean,
                    })
                return (x_embed, x_embed_mean, x_logits, x_logits_mean, mvl_att_masks, aux)
            return x_embed, x_embed_mean, x_logits, x_logits_mean, mvl_att_masks
        else:
            if self.use_dual_fusion:
                if self.use_adaptive_dual_fusion:
                    return self._adaptive_fusion(x_embed_mean, part_aux[1])[0]
                return self._fixed_dual_fusion(x_embed_mean, part_aux[1])[0]
            if self.use_anchor_projection_fusion:
                projection_embed = self.projection_bn_neck(projection_pool)
                projection_embed = projection_embed.reshape(-1, t, projection_embed.shape[-1]).mean(dim=1)
                global_eval = self.l2_norm(x_embed_mean) * ((1.0 - self.fusion_alpha) ** 0.5)
                projection_eval = self.l2_norm(projection_embed) * (self.fusion_alpha ** 0.5)
                return torch.cat([global_eval, projection_eval], dim=1)
            if self.use_dual_calibrated_fusion:
                calibration_embed = self.calibration_bn_neck(calibration_pool)
                calibration_embed = calibration_embed.reshape(-1, t, calibration_embed.shape[-1]).mean(dim=1)
                return self._dual_calibrated_eval(x_embed_mean, part_aux[1], calibration_embed)
            if (
                self.use_projection_calibrated_fusion
                or self.use_adaptive_projection_calibrated_fusion
                or self.use_quad_calibrated_fusion
            ):
                projection_embed = self.projection_bn_neck(projection_pool)
                projection_embed = projection_embed.reshape(-1, t, projection_embed.shape[-1]).mean(dim=1)
                calibration_embed = self.calibration_bn_neck(calibration_pool)
                calibration_embed = calibration_embed.reshape(-1, t, calibration_embed.shape[-1]).mean(dim=1)
                if self.use_quad_calibrated_fusion:
                    return self._quad_calibrated_eval(
                        x_embed_mean, part_aux[1], projection_embed, calibration_embed
                    )
                if self.use_adaptive_projection_calibrated_fusion:
                    return self._adaptive_projection_calibrated_eval(
                        x_embed_mean, projection_embed, calibration_embed
                    )[0]
                return self._projection_calibrated_eval(x_embed_mean, projection_embed, calibration_embed)
            return self.l2_norm(x_embed_mean)
