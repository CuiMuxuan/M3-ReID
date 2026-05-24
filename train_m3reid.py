# ------------------------------------------------------------------------------
# File:    M3-ReID/train_m3reid.py
#
# Description:
#    The main training script for the M3-ReID framework.
#    It orchestrates the entire pipeline including data loading, model initialization,
#    loss computation (ID, MMA, OFR, DAC), optimization, and evaluation.
#
# Key Features:
# - Supports training on HITSZ-VCM and BUPTCampus datasets.
# - Implements the full M3-ReID training loop with mixed precision (AMP) support.
# - Handles evaluation on both Visible-to-Infrared and Infrared-to-Visible modes.
# - Logs metrics to TensorBoard and text files.
# - Save checkpoints based on specified intervals.
#
# Paper:
#     M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-
#     Infrared Person Re-Identification by Liang et al.
#     See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
# ------------------------------------------------------------------------------

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.manager import HITSZVCMDataManager
from data.manager import BUPTCampusDataManager
from data.sampler import NormTripletSampler
from data.sampler import CrossModalityTripletSampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import CrossModalityIdentitySampler
from data.sampler import IdentityCrossModalitySampler
from data.dataset import VideoVIDataset
from data.transform import SyncTrackTransform
from data.transform import WeightedGrayscale
from data.transform import StyleVariation
from data.transform import WeakLowLight
from data.transform import RandomBlockOcclusion
from models.model_m3reid import M3ReID
from losses.mma_loss import MultiModalityAlignmentLoss
from losses.metric_loss import CosFaceProxyLoss
from losses.metric_loss import CrossModalityBatchHardTripletLoss
from losses.prototype_loss import CrossModalityPrototypeTripletLoss
from losses.prototype_loss import PrototypeMemoryLoss
from losses.sep_loss import SeparationLoss
from tools.eval_metrics import get_cmc_mAP_mINP
from tools.utils import set_seed, time_str, Logger


def build_loader_kwargs(args):
    kwargs = {
        'num_workers': args.workers,
        'pin_memory': args.pin_memory,
    }
    if args.workers > 0:
        kwargs['persistent_workers'] = args.persistent_workers
        if args.prefetch_factor is not None and args.prefetch_factor > 0:
            kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs


def get_m3plus_aug_probs(strength):
    if strength == 'standard':
        return {
            'ir_lowlight': 0.25,
            'rgb_lowlight': 0.35,
            'ir_occlusion': 0.20,
            'rgb_occlusion': 0.30,
        }
    if strength == 'mild':
        return {
            'ir_lowlight': 0.10,
            'rgb_lowlight': 0.15,
            'ir_occlusion': 0.05,
            'rgb_occlusion': 0.10,
        }
    if strength == 'none':
        return {
            'ir_lowlight': 0.0,
            'rgb_lowlight': 0.0,
            'ir_occlusion': 0.0,
            'rgb_occlusion': 0.0,
        }
    raise ValueError(f'Unknown M3Plus augmentation strength: {strength}')


def build_optimizer(args, params):
    if args.optimizer == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == 'adamw':
        return optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == 'adam8bit':
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                'optimizer=adam8bit requires bitsandbytes on the training server. '
                'Install it with: pip install bitsandbytes'
            ) from exc
        return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.wd)
    raise ValueError(f'Unsupported optimizer: {args.optimizer}')


def build_train_params(args, model):
    part_prefixes = (
        'part_aggregation', 'part_bn_neck', 'part_classifier',
        'adaptive_fusion_gate', 'fusion_bn_neck', 'fusion_classifier',
        'gated_residual_fusion', 'part_token_fusion', 'reliability_part_fusion',
        'bidirectional_calibration', 'invariant_specific_calibration',
        'anchor_projection_fusion', 'projection_bn_neck', 'projection_classifier',
        'dual_calibration_fusion', 'calibration_bn_neck', 'calibration_classifier',
        'adaptive_projection_calibration_gate', 'adaptive_projection_bn_neck',
        'adaptive_projection_classifier',
    )
    temporal_prefixes = ('temporal_refine',)
    base_params, part_params, temporal_params = [], [], []
    for name, param in model.named_parameters():
        if name.startswith(temporal_prefixes):
            temporal_params.append(param)
        elif name.startswith(part_prefixes):
            part_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{'params': base_params, 'lr': args.lr}]
    if part_params:
        param_groups.append({'params': part_params, 'lr': args.lr * args.part_lr_mult})
    if temporal_params:
        param_groups.append({'params': temporal_params, 'lr': args.lr * args.temporal_lr_mult})
    if len(param_groups) == 1:
        return model.parameters()
    return param_groups


def parse_lr_milestones(milestones):
    if milestones is None or milestones.strip() == '':
        return []
    return [int(item.strip()) for item in milestones.split(',') if item.strip()]


def cross_modality_identity_consistency(features, labels, modality_labels):
    losses = []
    for pid in labels.unique():
        pid_mask = labels == pid
        ir_mask = pid_mask & (modality_labels == 1)
        rgb_mask = pid_mask & (modality_labels == 2)
        if ir_mask.any() and rgb_mask.any():
            ir_center = F.normalize(features[ir_mask].mean(dim=0, keepdim=True), p=2, dim=1)
            rgb_center = F.normalize(features[rgb_mask].mean(dim=0, keepdim=True), p=2, dim=1)
            losses.append(1.0 - (ir_center * rgb_center).sum(dim=1))
    if not losses:
        return features.new_zeros(())
    return torch.cat(losses).mean()


if __name__ == '__main__':

    # Arguments --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Video-based visible-infrared cross-modality ReID')

    # -- Data Arguments ------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', default='HITSZVCM', help='Dataset name')
    parser.add_argument('--dataset_dir', default='../Datasets/HITSZ-VCM', help='Directory of dataset')
    parser.add_argument('--img_h', default=288, type=int, help='Height of input images')
    parser.add_argument('--img_w', default=144, type=int, help='Width of input images')
    parser.add_argument('--p_num', default=4, type=int, help='Num of identities')
    parser.add_argument('--k_num', default=8, type=int, help='Num of samples per identity')
    parser.add_argument('--t', default=6, type=int, help='Number of sampled frames per video track')
    parser.add_argument('--test_batch_size', default=None, type=int,
                        help='Batch size for evaluation. Defaults to p_num * k_num')
    parser.add_argument('--workers', default=4, type=int, help='Num of dataloader workers')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Pin dataloader memory for faster host-to-GPU copies')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep dataloader workers alive between epochs when workers > 0')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Dataloader prefetch factor when workers > 0. Set <=0 to disable')
    parser.add_argument('--non_blocking', action=argparse.BooleanOptionalAction, default=True,
                        help='Use non-blocking CUDA transfers when pin_memory is enabled')
    parser.add_argument('--torch_sharing_strategy', default=None,
                        choices=['file_descriptor', 'file_system'],
                        help='Torch multiprocessing sharing strategy for DataLoader workers')

    # -- Optim Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate for adam optimizer')
    parser.add_argument('--wd', default=0.0005, type=float, help='Weight decay for adam optimizer')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'adam8bit'],
                        help='Optimizer. adam8bit requires bitsandbytes on Linux')
    parser.add_argument('--accum_steps', default=1, type=int,
                        help='Gradient accumulation steps for memory-limited GPUs')
    parser.add_argument('--lr_milestones', default='80,120',
                        help='Comma-separated epochs for MultiStepLR decay. Empty string disables milestone decay')

    # -- Other Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--fp16', action='store_true', default=False, help='Whether to use AMP')
    parser.add_argument('--eval_fp16', action='store_true', default=False,
                        help='Use AMP autocast during validation feature extraction')
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable cuDNN benchmark for fixed image sizes')
    parser.add_argument('--resume', default=None, type=str, help='Resume from path of checkpoint')
    parser.add_argument('--drop_classifier_on_resume', action='store_true', default=False,
                        help='Drop ID classifier heads from a resumed state_dict. Useful when dataset relabel order changed.')
    parser.add_argument('--use_m3plus', action='store_true', default=False,
                        help='Enable enhanced M3-ReID with multi-scale, local part, attention, hard triplet, and robust augmentation')
    parser.add_argument('--m3plus_mode', default='full',
                        choices=['full', 'part_only', 'local_residual', 'dual_fusion',
                                 'temporal_dual_fusion', 'adaptive_dual_fusion',
                                 'supervised_dual_fusion', 'gated_residual_fusion',
                                 'part_token_fusion', 'reliability_part_fusion',
                                 'bidirectional_calibration',
                                 'invariant_specific_calibration',
                                 'anchor_projection_fusion',
                                 'dual_calibrated_fusion',
                                 'temporal_dual_calibrated_fusion',
                                 'projection_calibrated_fusion',
                                 'adaptive_projection_calibrated_fusion'],
                        help='M3Plus architecture mode. dual_fusion keeps the baseline global head and adds a supervised local fusion branch')
    parser.add_argument('--m3plus_aug_strength', default='none',
                        choices=['standard', 'mild', 'none'],
                        help='Strength of extra M3Plus low-light and block-occlusion augmentation')
    parser.add_argument('--disable_random_erasing', action=argparse.BooleanOptionalAction, default=True,
                        help='Disable RandomErasing augmentation for experiments that must exclude BoT-style tricks.')
    parser.add_argument('--part_num', default=4, type=int, help='Number of horizontal local parts for M3Plus')
    parser.add_argument('--part_dim', default=2048, type=int,
                        help='Output dimension of the M3Plus local part branch')
    parser.add_argument('--mvl_num_heads', default=2, type=int,
                        help='Number of MVL attention heads per view')
    parser.add_argument('--feature_dropout', default=0.0, type=float,
                        help='Dropout applied before ID classifiers during training')
    parser.add_argument('--fusion_alpha', default=0.2, type=float,
                        help='Local feature weight used by dual_fusion inference, or initial local weight for adaptive_dual_fusion')
    parser.add_argument('--calibration_alpha', default=0.0, type=float,
                        help='Calibrated global feature weight used by dual_calibrated_fusion inference')
    parser.add_argument('--adaptive_gate_min', default=0.0, type=float,
                        help='Minimum local feature weight predicted by adaptive_dual_fusion')
    parser.add_argument('--adaptive_gate_max', default=0.12, type=float,
                        help='Maximum local feature weight predicted by adaptive_dual_fusion')
    parser.add_argument('--part_lr_mult', default=1.0, type=float,
                        help='Learning-rate multiplier for dual_fusion local branch parameters')
    parser.add_argument('--temporal_dim', default=256, type=int,
                        help='Hidden dimension of temporal_dual_fusion refinement head')
    parser.add_argument('--temporal_dropout', default=0.0, type=float,
                        help='Dropout inside temporal_dual_fusion refinement head')
    parser.add_argument('--temporal_lr_mult', default=1.0, type=float,
                        help='Learning-rate multiplier for temporal_dual_fusion refinement parameters')
    parser.add_argument('--freeze_base_epochs', default=0, type=int,
                        help='For dual_fusion, train only local branch for this many initial epochs')
    parser.add_argument('--grad_checkpoint_head', action='store_true', default=False,
                        help='Checkpoint M3Plus attention/local heads to save activation memory')
    parser.add_argument('--sample_method', default=None, type=str,
                        choices=['norm_triplet', 'cross_modality_triplet', 'cross_modality_random',
                                 'cross_modality_identity', 'identity_cross_modality'],
                        help='Sampler strategy. Defaults to identity_cross_modality for M3Plus and norm_triplet otherwise')
    parser.add_argument('--triplet_weight', default=0.5, type=float, help='Weight of cross-modality batch-hard triplet loss')
    parser.add_argument('--triplet_frame_weight', default=0.25, type=float, help='Relative frame-level triplet loss weight')
    parser.add_argument('--triplet_margin', default=0.3, type=float, help='Margin for hard triplet loss when soft margin is disabled')
    parser.add_argument('--id_label_smoothing', default=None, type=float,
                        help='Cross entropy label smoothing. Defaults to 0.1 for M3Plus and 0.0 for baseline')
    parser.add_argument('--part_id_weight', default=0.0, type=float,
                        help='Weight of dual_fusion local branch ID loss')
    parser.add_argument('--part_triplet_weight', default=0.0, type=float,
                        help='Weight of dual_fusion local branch cross-modality triplet loss')
    parser.add_argument('--part_mma_weight', default=0.0, type=float,
                        help='Weight of dual_fusion local branch modality alignment loss')
    parser.add_argument('--fusion_id_weight', default=0.0, type=float,
                        help='Weight of adaptive_dual_fusion fused video-level ID loss')
    parser.add_argument('--fusion_triplet_weight', default=0.0, type=float,
                        help='Weight of adaptive_dual_fusion fused video-level cross-modality triplet loss')
    parser.add_argument('--cosface_weight', default=0.0, type=float,
                        help='Weight of global video-level CosFace proxy loss')
    parser.add_argument('--cosface_frame_weight', default=0.0, type=float,
                        help='Relative frame-level CosFace proxy loss weight')
    parser.add_argument('--part_cosface_weight', default=0.0, type=float,
                        help='Weight of local branch video-level CosFace proxy loss')
    parser.add_argument('--cosface_margin', default=0.2, type=float,
                        help='Cosine margin for CosFace proxy loss')
    parser.add_argument('--cosface_scale', default=32.0, type=float,
                        help='Logit scale for CosFace proxy loss')
    parser.add_argument('--cosface_start_epoch', default=1, type=int,
                        help='First 1-based epoch to enable CosFace proxy loss')
    parser.add_argument('--proto_weight', default=0.0, type=float,
                        help='Weight of global EMA class-prototype loss')
    parser.add_argument('--part_proto_weight', default=0.0, type=float,
                        help='Weight of dual_fusion local branch EMA class-prototype loss')
    parser.add_argument('--proto_temperature', default=0.07, type=float,
                        help='Temperature for EMA class-prototype logits')
    parser.add_argument('--proto_momentum', default=0.2, type=float,
                        help='EMA momentum for updating class prototypes')
    parser.add_argument('--proto_start_epoch', default=1, type=int,
                        help='First 1-based epoch to enable prototype loss')
    parser.add_argument('--cross_proto_weight', default=0.0, type=float,
                        help='Weight of global cross-modality EMA prototype triplet loss')
    parser.add_argument('--part_cross_proto_weight', default=0.0, type=float,
                        help='Weight of local branch cross-modality EMA prototype triplet loss')
    parser.add_argument('--cross_proto_margin', default=0.1, type=float,
                        help='Similarity margin for cross-modality prototype triplet loss')
    parser.add_argument('--cross_proto_start_epoch', default=1, type=int,
                        help='First 1-based epoch to enable cross-modality prototype triplet loss')
    parser.add_argument('--router_weight', default=0.05, type=float,
                        help='Weight of bidirectional_calibration modality-router supervision')
    parser.add_argument('--modality_adv_weight', default=0.0, type=float,
                        help='Weight of SchemeU modality-adversarial invariant supervision')
    parser.add_argument('--invariant_consistency_weight', default=0.0, type=float,
                        help='Weight of SchemeU same-ID cross-modality invariant consistency loss')
    parser.add_argument('--projection_id_weight', default=0.0, type=float,
                        help='Weight of SchemeV projection subspace ID loss')
    parser.add_argument('--projection_triplet_weight', default=0.0, type=float,
                        help='Weight of SchemeV projection subspace cross-modality triplet loss')
    parser.add_argument('--projection_mma_weight', default=0.0, type=float,
                        help='Weight of SchemeV projection subspace modality alignment loss')
    parser.add_argument('--projection_gate_weight', default=0.0, type=float,
                        help='Weight of SchemeV modality-aware projection gate supervision')
    parser.add_argument('--projection_gate_ir_target', default=0.7, type=float,
                        help='SchemeV target projection gate value for IR tracks')
    parser.add_argument('--projection_gate_rgb_target', default=0.3, type=float,
                        help='SchemeV target projection gate value for RGB tracks')
    parser.add_argument('--calibration_id_weight', default=0.0, type=float,
                        help='Weight of SchemeW calibrated global branch ID loss')
    parser.add_argument('--calibration_triplet_weight', default=0.0, type=float,
                        help='Weight of SchemeW calibrated global branch cross-modality triplet loss')
    parser.add_argument('--calibration_mma_weight', default=0.0, type=float,
                        help='Weight of SchemeW calibrated global branch modality alignment loss')
    parser.add_argument('--calibration_gate_weight', default=0.0, type=float,
                        help='Weight of bounded calibration gate regularization for calibrated fusion modes')
    parser.add_argument('--calibration_gate_target', default=0.60, type=float,
                        help='Target gate value for bounded calibration gate regularization')
    parser.add_argument('--branch_gate_weight', default=0.0, type=float,
                        help='Weight of SchemeY adaptive projection/calibration branch gate anchoring')
    parser.add_argument('--projection_alpha_target', default=None, type=float,
                        help='Target projection branch weight for SchemeY adaptive gate. Defaults to fusion_alpha')
    parser.add_argument('--calibration_alpha_target', default=None, type=float,
                        help='Target calibration branch weight for SchemeY adaptive gate. Defaults to calibration_alpha')
    parser.add_argument('--log_interval', default=10, type=int, help='Interval of logging')
    parser.add_argument('--test_interval', default=1, type=int, help='Interval of testing. Set 0 to disable')
    parser.add_argument('--eval_start_epoch', default=1, type=int,
                        help='First epoch allowed to run evaluation')
    parser.add_argument('--early_stop_patience', default=0, type=int,
                        help='Stop after this many evaluated rounds without avg Rank-1 improvement. Set 0 to disable')
    parser.add_argument('--save_interval', default=1, type=int, help='Interval of saving checkpoints. Set 0 to disable')
    parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
    parser.add_argument('--max_train_batches', default=None, type=int,
                        help='Optional maximum training batches per epoch for smoke tests')

    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--desc', type=str, default=None, help='Description for this training process')

    args = parser.parse_args()

    # Env  -------------------------------------------------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for M3-ReID training.')
    if args.torch_sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(args.torch_sharing_strategy)
    torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('high')  # highest high medium

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if args.cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
    suffix = f'Time-{time_str()}' if (args.desc is None) else f'Time-{time_str()}_{args.desc}'

    ckptlog_dir = os.path.join('ckptlog', args.dataset, suffix)
    os.makedirs(ckptlog_dir, exist_ok=True)

    tensorboard_dir = os.path.join(ckptlog_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    log_path = os.path.join(ckptlog_dir, f'log_{suffix}.txt')
    sys.stdout = Logger(log_path)

    modelckpt_dir = os.path.join(ckptlog_dir, 'modelckpt')
    os.makedirs(modelckpt_dir, exist_ok=True)

    print(f'Args: {args}')
    device_props = torch.cuda.get_device_properties(args.gpu)
    print(f'CUDA device: id={args.gpu}, name={device_props.name}, '
          f'total_memory={device_props.total_memory / (1024 ** 3):.1f}GB, '
          f'cudnn_benchmark={torch.backends.cudnn.benchmark}, '
          f'cudnn_deterministic={torch.backends.cudnn.deterministic}')
    print(f'Torch multiprocessing sharing strategy: {torch.multiprocessing.get_sharing_strategy()}')

    # Data -------------------------------------------------------------------------------------------------------------
    sample_seq_num = args.t
    train_batch_size = args.p_num * args.k_num
    test_batch_size = args.test_batch_size or train_batch_size  # Set Appropriate Values Based on GPU Memory
    loader_kwargs = build_loader_kwargs(args)
    print(f'Dataloader setting: {loader_kwargs}, non_blocking_cuda={args.non_blocking}')

    # -- DataManager ---------------------------------------------------------------------------------------------------
    if args.dataset == 'HITSZVCM':
        data_manager = HITSZVCMDataManager(args.dataset_dir)
    elif args.dataset == 'BUPTCampus':
        data_manager = BUPTCampusDataManager(args.dataset_dir)
    else:
        raise RuntimeError(f'Dataset {args.dataset} is not supported for now.')

    num_train_class = data_manager.train_num_pids
    num_test_class = data_manager.test_num_pids
    num_query = len(data_manager.query_track_pids)
    num_gallery = len(data_manager.gallery_track_pids)

    # -- Dataset & Dataloader ------------------------------------------------------------------------------------------
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    random_erasing = [] if args.disable_random_erasing else [T.RandomErasing()]

    if args.use_m3plus:
        aug_probs = get_m3plus_aug_probs(args.m3plus_aug_strength)
        print(f'M3Plus augmentation strength={args.m3plus_aug_strength}, probs={aug_probs}')
        transform_train_ir = SyncTrackTransform(T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_h, args.img_w)),
            WeakLowLight(p=aug_probs['ir_lowlight']),
            T.RandomCrop((args.img_h, args.img_w), padding=5, fill=0),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            RandomBlockOcclusion(p=aug_probs['ir_occlusion']),
            normalize,
            *random_erasing,
            StyleVariation(mode='one', p=1.0),
        ]))
        transform_train_rgb = SyncTrackTransform(T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_h, args.img_w)),
            WeightedGrayscale(p=0.5),
            WeakLowLight(p=aug_probs['rgb_lowlight']),
            T.RandomCrop((args.img_h, args.img_w), padding=5, fill=0),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            RandomBlockOcclusion(p=aug_probs['rgb_occlusion']),
            normalize,
            *random_erasing,
            StyleVariation(mode='all', p=1.0),
        ]))
    else:
        transform_train_ir = SyncTrackTransform(T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_h, args.img_w)),
            T.RandomCrop((args.img_h, args.img_w), padding=5, fill=0),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
            *random_erasing,
            StyleVariation(mode='one', p=1.0),
        ]))
        transform_train_rgb = SyncTrackTransform(T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_h, args.img_w)),
            WeightedGrayscale(p=0.5),
            T.RandomCrop((args.img_h, args.img_w), padding=5, fill=0),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
            *random_erasing,
            StyleVariation(mode='all', p=1.0),
        ]))
    transform_train = (transform_train_ir, transform_train_rgb)

    transform_test = SyncTrackTransform(T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_h, args.img_w)),
        T.ToTensor(),
        normalize,
    ]))

    train_loader = None  # Get the train_loader in Each Epoch of Training for Random Selection

    query_dataset = VideoVIDataset(data_manager, transform=transform_test,
                                   sample_seq_num=sample_seq_num, sample_mode='evenly', dataset_mode='query')
    gallery_dataset = VideoVIDataset(data_manager, transform=transform_test,
                                     sample_seq_num=sample_seq_num, sample_mode='evenly', dataset_mode='gallery')
    query_loader = DataLoader(query_dataset, batch_size=test_batch_size,
                              shuffle=False, **loader_kwargs)
    gallery_loader = DataLoader(gallery_dataset, batch_size=test_batch_size,
                                shuffle=False, **loader_kwargs)

    # Model ------------------------------------------------------------------------------------------------------------
    model = M3ReID(sample_seq_num, num_train_class,
                   use_enhancements=args.use_m3plus, m3plus_mode=args.m3plus_mode, part_num=args.part_num,
                   mvl_num_heads=args.mvl_num_heads, part_dim=args.part_dim,
                   feature_dropout=args.feature_dropout, fusion_alpha=args.fusion_alpha,
                   calibration_alpha=args.calibration_alpha,
                   grad_checkpoint_head=args.grad_checkpoint_head,
                   temporal_dim=args.temporal_dim, temporal_dropout=args.temporal_dropout,
                   adaptive_gate_min=args.adaptive_gate_min,
                   adaptive_gate_max=args.adaptive_gate_max).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
        dropped_classifier_keys = []
        if args.drop_classifier_on_resume:
            for key in list(checkpoint.keys()):
                if 'classifier' in key:
                    dropped_classifier_keys.append(key)
                    checkpoint.pop(key)
        removed_mismatch_keys = []
        for key in list(checkpoint.keys()):
            model_state_dict = model.state_dict()
            if key in model_state_dict:
                if torch.is_tensor(checkpoint[key]) and checkpoint[key].shape != model_state_dict[key].shape:
                    print(f'Warning during loading weights - Auto remove mismatch key: {key}')
                    removed_mismatch_keys.append(key)
                    checkpoint.pop(key)
        load_result = model.load_state_dict(checkpoint, strict=False)
        print(f'Loaded checkpoint: {args.resume}')
        print(f'Checkpoint load summary: loaded_keys={len(checkpoint)}, '
              f'dropped_classifier_keys={len(dropped_classifier_keys)}, '
              f'removed_mismatch_keys={len(removed_mismatch_keys)}, '
              f'missing_keys={len(load_result.missing_keys)}, '
              f'unexpected_keys={len(load_result.unexpected_keys)}')
        if dropped_classifier_keys:
            print(f'Dropped classifier keys preview: {dropped_classifier_keys[:12]}')
        if removed_mismatch_keys:
            print(f'Removed mismatch keys preview: {removed_mismatch_keys[:12]}')
        if load_result.missing_keys:
            print(f'Missing keys preview: {load_result.missing_keys[:12]}')
        if load_result.unexpected_keys:
            print(f'Unexpected keys preview: {load_result.unexpected_keys[:12]}')
        if args.m3plus_mode in (
            'dual_calibrated_fusion',
            'temporal_dual_calibrated_fusion',
            'projection_calibrated_fusion',
            'adaptive_projection_calibrated_fusion',
        ):
            missing = set(load_result.missing_keys)
            with torch.no_grad():
                if 'calibration_bn_neck.weight' in missing:
                    model.calibration_bn_neck.weight.copy_(model.bn_neck.weight)
                if 'calibration_bn_neck.bias' in missing:
                    model.calibration_bn_neck.bias.copy_(model.bn_neck.bias)
                if 'calibration_bn_neck.running_mean' in missing:
                    model.calibration_bn_neck.running_mean.copy_(model.bn_neck.running_mean)
                if 'calibration_bn_neck.running_var' in missing:
                    model.calibration_bn_neck.running_var.copy_(model.bn_neck.running_var)
                if 'calibration_classifier.weight' in missing:
                    model.calibration_classifier.weight.copy_(model.classifier.weight)
                if 'calibration_classifier_frame.weight' in missing:
                    model.calibration_classifier_frame.weight.copy_(model.classifier_frame.weight)
            print('SchemeW/X init: copied global BN/classifier into calibration branch for warm start.')

    # Loss -------------------------------------------------------------------------------------------------------------
    label_smoothing = args.id_label_smoothing
    if label_smoothing is None:
        label_smoothing = 0.0
    criterion_ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda()
    criterion_mma_loss = MultiModalityAlignmentLoss().cuda()
    criterion_triplet_loss = CrossModalityBatchHardTripletLoss(margin=args.triplet_margin).cuda()
    criterion_cosface_loss = None
    if args.use_m3plus and (args.cosface_weight > 0 or args.part_cosface_weight > 0):
        criterion_cosface_loss = CosFaceProxyLoss(
            scale=args.cosface_scale,
            margin=args.cosface_margin,
        ).cuda()
    criterion_proto_loss = None
    criterion_part_proto_loss = None
    criterion_cross_proto_loss = None
    criterion_part_cross_proto_loss = None
    if args.use_m3plus and args.proto_weight > 0:
        criterion_proto_loss = PrototypeMemoryLoss(
            num_train_class, model.embedding_dim,
            temperature=args.proto_temperature,
            momentum=args.proto_momentum,
        ).cuda()
    if args.use_m3plus and args.cross_proto_weight > 0:
        criterion_cross_proto_loss = CrossModalityPrototypeTripletLoss(
            num_train_class, model.embedding_dim,
            margin=args.cross_proto_margin,
            momentum=args.proto_momentum,
        ).cuda()
    part_supervision_modes = (
        'dual_fusion', 'temporal_dual_fusion', 'adaptive_dual_fusion',
        'supervised_dual_fusion', 'gated_residual_fusion', 'part_token_fusion',
        'reliability_part_fusion', 'bidirectional_calibration',
        'invariant_specific_calibration', 'anchor_projection_fusion',
        'dual_calibrated_fusion', 'temporal_dual_calibrated_fusion',
        'projection_calibrated_fusion',
        'adaptive_projection_calibrated_fusion'
    )
    if args.use_m3plus and args.m3plus_mode in part_supervision_modes and args.part_cross_proto_weight > 0:
        criterion_part_cross_proto_loss = CrossModalityPrototypeTripletLoss(
            num_train_class, args.part_dim,
            margin=args.cross_proto_margin,
            momentum=args.proto_momentum,
        ).cuda()
    if args.use_m3plus and args.m3plus_mode in part_supervision_modes and args.part_proto_weight > 0:
        criterion_part_proto_loss = PrototypeMemoryLoss(
            num_train_class, args.part_dim,
            temperature=args.proto_temperature,
            momentum=args.proto_momentum,
        ).cuda()
    criterion_ofr_loss = SeparationLoss().cuda()
    criterion_dac_loss = SeparationLoss().cuda()

    # Optimizer --------------------------------------------------------------------------------------------------------
    optimizer = build_optimizer(args, build_train_params(args, model))
    lr_milestones = parse_lr_milestones(args.lr_milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_milestones, gamma=0.1)

    # Iteration --------------------------------------------------------------------------------------------------------
    total_epoch_num = args.epochs
    sample_method = args.sample_method
    if sample_method is None:
        sample_method = 'identity_cross_modality' if args.use_m3plus else 'norm_triplet'
    enable_triplet_loss = args.use_m3plus and args.triplet_weight > 0
    print(f'Effective setting: sample_seq_num={sample_seq_num}, use_m3plus={args.use_m3plus}, '
          f'm3plus_mode={args.m3plus_mode}, '
          f'sample_method={sample_method}, label_smoothing={label_smoothing:.3f}, '
          f'disable_random_erasing={args.disable_random_erasing}, '
          f'enable_triplet_loss={enable_triplet_loss}, accum_steps={args.accum_steps}, '
          f'train_batch_size={train_batch_size}, test_batch_size={test_batch_size}, '
          f'fp16={args.fp16}, eval_fp16={args.eval_fp16}, '
          f'mvl_num_heads={args.mvl_num_heads}, part_dim={args.part_dim}, '
          f'feature_dropout={args.feature_dropout}, fusion_alpha={args.fusion_alpha}, '
          f'calibration_alpha={args.calibration_alpha}, '
          f'adaptive_gate_min={args.adaptive_gate_min}, adaptive_gate_max={args.adaptive_gate_max}, '
          f'part_lr_mult={args.part_lr_mult}, freeze_base_epochs={args.freeze_base_epochs}, '
          f'temporal_dim={args.temporal_dim}, temporal_dropout={args.temporal_dropout}, '
          f'temporal_lr_mult={args.temporal_lr_mult}, '
          f'part_id_weight={args.part_id_weight}, part_triplet_weight={args.part_triplet_weight}, '
          f'part_mma_weight={args.part_mma_weight}, '
          f'fusion_id_weight={args.fusion_id_weight}, fusion_triplet_weight={args.fusion_triplet_weight}, '
          f'cosface_weight={args.cosface_weight}, cosface_frame_weight={args.cosface_frame_weight}, '
          f'part_cosface_weight={args.part_cosface_weight}, '
          f'cosface_margin={args.cosface_margin}, cosface_scale={args.cosface_scale}, '
          f'cosface_start_epoch={args.cosface_start_epoch}, '
          f'proto_weight={args.proto_weight}, part_proto_weight={args.part_proto_weight}, '
          f'proto_temperature={args.proto_temperature}, proto_momentum={args.proto_momentum}, '
          f'proto_start_epoch={args.proto_start_epoch}, '
          f'cross_proto_weight={args.cross_proto_weight}, '
          f'part_cross_proto_weight={args.part_cross_proto_weight}, '
          f'cross_proto_margin={args.cross_proto_margin}, '
          f'cross_proto_start_epoch={args.cross_proto_start_epoch}, '
          f'router_weight={args.router_weight}, '
          f'modality_adv_weight={args.modality_adv_weight}, '
          f'invariant_consistency_weight={args.invariant_consistency_weight}, '
          f'projection_id_weight={args.projection_id_weight}, '
          f'projection_triplet_weight={args.projection_triplet_weight}, '
          f'projection_mma_weight={args.projection_mma_weight}, '
          f'projection_gate_weight={args.projection_gate_weight}, '
          f'projection_gate_ir_target={args.projection_gate_ir_target}, '
          f'projection_gate_rgb_target={args.projection_gate_rgb_target}, '
          f'calibration_id_weight={args.calibration_id_weight}, '
          f'calibration_triplet_weight={args.calibration_triplet_weight}, '
          f'calibration_mma_weight={args.calibration_mma_weight}, '
          f'calibration_gate_weight={args.calibration_gate_weight}, '
          f'calibration_gate_target={args.calibration_gate_target}, '
          f'branch_gate_weight={args.branch_gate_weight}, '
          f'grad_checkpoint_head={args.grad_checkpoint_head}, optimizer={args.optimizer}, '
          f'lr_milestones={lr_milestones}')

    best_score = -1
    best_result = None
    best_epoch = -1
    stale_eval_count = 0

    if args.fp16: amp_scaler = torch.amp.GradScaler('cuda')
    for epoch in range(total_epoch_num):
        train_dataset = VideoVIDataset(data_manager, transform=transform_train,
                                       sample_seq_num=sample_seq_num, sample_mode='evenly', dataset_mode='train')

        shuffle = False
        if sample_method == 'norm_triplet':
            sampler = NormTripletSampler(train_dataset, train_batch_size, args.k_num)
        elif sample_method == 'cross_modality_triplet':
            ratio = 0.5  # Set to a custom value
            sampler = CrossModalityTripletSampler(train_dataset, train_batch_size, args.k_num, modal_ratio=ratio)
        elif sample_method == 'cross_modality_random':
            sampler = CrossModalityRandomSampler(train_dataset, train_batch_size)
        elif sample_method == 'cross_modality_identity':
            sampler = CrossModalityIdentitySampler(train_dataset, args.p_num, args.k_num)
        elif sample_method == 'identity_cross_modality':
            sampler = IdentityCrossModalitySampler(train_dataset, train_batch_size, args.k_num)
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(train_dataset, train_batch_size, sampler=sampler,
                                  shuffle=shuffle, drop_last=True, **loader_kwargs)

        # -- Train -----------------------------------------------------------------------------------------------------
        model.train()
        if args.m3plus_mode in part_supervision_modes:
            base_trainable = epoch >= args.freeze_base_epochs
            model.set_scheme_d_base_trainable(base_trainable)
            if epoch == 0 and not base_trainable:
                print(f'Scheme D warmup: freezing baseline global branch for {args.freeze_base_epochs} epoch(s).')
            if epoch == args.freeze_base_epochs and args.freeze_base_epochs > 0:
                print('Scheme D warmup complete: unfreezing baseline global branch.')
        s_time = time.time()
        optimizer.zero_grad()
        for batch_idx, batch_data in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            track_data, track_pid, track_cid, track_mid = batch_data
            inputs = track_data.cuda(non_blocking=args.non_blocking)
            labels = track_pid.cuda(non_blocking=args.non_blocking)

            with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
                model_outputs = model(inputs)
                if len(model_outputs) == 5:
                    x_embed, x_embed_m, x_logits, x_logits_m, mvl_att_masks = model_outputs
                    part_aux = None
                else:
                    x_embed, x_embed_m, x_logits, x_logits_m, mvl_att_masks, part_aux = model_outputs
                id_labels = labels
                m_labels = track_mid.cuda(non_blocking=args.non_blocking)

                loss_ofr = criterion_ofr_loss(x_embed)

                loss_dac = sum([criterion_dac_loss(mvl_att_masks[i]) for i in range(len(mvl_att_masks))])

                loss_mma = criterion_mma_loss(x_embed_m, id_labels, m_labels)
                loss_id = criterion_ce_loss(x_logits_m, labels)

                b, t, c = x_embed.shape
                id_labels_all = id_labels.repeat_interleave(t)
                m_labels_all = m_labels.repeat_interleave(t)
                loss_mma_frames = criterion_mma_loss(x_embed.reshape(b * t, c), id_labels_all, m_labels_all)
                loss_id_frames = criterion_ce_loss(x_logits.reshape(b * t, -1), id_labels_all)
                if enable_triplet_loss:
                    loss_triplet = criterion_triplet_loss(x_embed_m, id_labels, m_labels)
                    loss_triplet_frames = criterion_triplet_loss(x_embed.reshape(b * t, c), id_labels_all, m_labels_all)
                else:
                    loss_triplet = x_embed_m.new_zeros(())
                    loss_triplet_frames = x_embed_m.new_zeros(())

                loss_part_id = x_embed_m.new_zeros(())
                loss_part_mma = x_embed_m.new_zeros(())
                loss_part_triplet = x_embed_m.new_zeros(())
                loss_cosface = x_embed_m.new_zeros(())
                loss_part_cosface = x_embed_m.new_zeros(())
                loss_fusion_id = x_embed_m.new_zeros(())
                loss_fusion_triplet = x_embed_m.new_zeros(())
                fusion_gate_mean = x_embed_m.new_zeros(())
                loss_proto = x_embed_m.new_zeros(())
                loss_part_proto = x_embed_m.new_zeros(())
                loss_cross_proto = x_embed_m.new_zeros(())
                loss_part_cross_proto = x_embed_m.new_zeros(())
                loss_router = x_embed_m.new_zeros(())
                loss_modality_adv = x_embed_m.new_zeros(())
                loss_invariant_consistency = x_embed_m.new_zeros(())
                loss_projection_id = x_embed_m.new_zeros(())
                loss_projection_triplet = x_embed_m.new_zeros(())
                loss_projection_mma = x_embed_m.new_zeros(())
                loss_projection_gate = x_embed_m.new_zeros(())
                loss_calibration_id = x_embed_m.new_zeros(())
                loss_calibration_triplet = x_embed_m.new_zeros(())
                loss_calibration_mma = x_embed_m.new_zeros(())
                loss_calibration_gate = x_embed_m.new_zeros(())
                loss_branch_gate = x_embed_m.new_zeros(())
                proto_enabled = epoch + 1 >= args.proto_start_epoch
                cross_proto_enabled = epoch + 1 >= args.cross_proto_start_epoch
                cosface_enabled = epoch + 1 >= args.cosface_start_epoch
                if cosface_enabled and criterion_cosface_loss is not None and args.cosface_weight > 0:
                    loss_cosface = criterion_cosface_loss(x_embed_m, model.classifier.weight, labels)
                    if args.cosface_frame_weight > 0:
                        loss_cosface_frames = criterion_cosface_loss(
                            x_embed.reshape(b * t, c), model.classifier_frame.weight, id_labels_all
                        )
                        loss_cosface = loss_cosface + args.cosface_frame_weight * loss_cosface_frames
                if proto_enabled and criterion_proto_loss is not None:
                    loss_proto = criterion_proto_loss(x_embed_m, labels)
                if cross_proto_enabled and criterion_cross_proto_loss is not None:
                    loss_cross_proto = criterion_cross_proto_loss(x_embed_m, labels, m_labels)
                if part_aux is not None:
                    part_embed = part_aux['part_embed']
                    part_embed_m = part_aux['part_embed_mean']
                    part_logits = part_aux['part_logits']
                    part_logits_m = part_aux['part_logits_mean']
                    part_b, part_t, part_c = part_embed.shape
                    loss_part_id = criterion_ce_loss(part_logits_m, labels)
                    loss_part_id = loss_part_id + criterion_ce_loss(
                        part_logits.reshape(part_b * part_t, -1), id_labels.repeat_interleave(part_t)
                    )
                    loss_part_mma = criterion_mma_loss(part_embed_m, id_labels, m_labels)
                    loss_part_mma = loss_part_mma + criterion_mma_loss(
                        part_embed.reshape(part_b * part_t, part_c),
                        id_labels.repeat_interleave(part_t),
                        m_labels.repeat_interleave(part_t),
                    )
                    loss_part_triplet = criterion_triplet_loss(part_embed_m, id_labels, m_labels)
                    loss_part_triplet_frames = criterion_triplet_loss(
                        part_embed.reshape(part_b * part_t, part_c),
                        id_labels.repeat_interleave(part_t),
                        m_labels.repeat_interleave(part_t),
                    )
                    loss_part_triplet = loss_part_triplet + args.triplet_frame_weight * loss_part_triplet_frames
                    if proto_enabled and criterion_part_proto_loss is not None:
                        loss_part_proto = criterion_part_proto_loss(part_embed_m, labels)
                    if cross_proto_enabled and criterion_part_cross_proto_loss is not None:
                        loss_part_cross_proto = criterion_part_cross_proto_loss(part_embed_m, labels, m_labels)
                    if cosface_enabled and criterion_cosface_loss is not None and args.part_cosface_weight > 0:
                        loss_part_cosface = criterion_cosface_loss(
                            part_embed_m, model.part_classifier.weight, labels
                        )
                    if 'fusion_embed_mean' in part_aux:
                        fusion_embed_m = part_aux['fusion_embed_mean']
                        fusion_logits_m = part_aux['fusion_logits_mean']
                        loss_fusion_id = criterion_ce_loss(fusion_logits_m, labels)
                        loss_fusion_triplet = criterion_triplet_loss(fusion_embed_m, id_labels, m_labels)
                    if 'fusion_gate' in part_aux:
                        fusion_gate = part_aux['fusion_gate']
                        fusion_gate_mean = fusion_gate.detach().mean()
                    if 'router_logits' in part_aux:
                        router_logits = part_aux['router_logits']
                        router_targets = (m_labels == 2).long()
                        loss_router = criterion_ce_loss(router_logits, router_targets)
                    if 'modality_logits' in part_aux:
                        modality_logits = part_aux['modality_logits']
                        modality_targets = (m_labels == 2).long()
                        loss_modality_adv = criterion_ce_loss(modality_logits, modality_targets)
                    if 'invariant_embed_mean' in part_aux:
                        loss_invariant_consistency = cross_modality_identity_consistency(
                            part_aux['invariant_embed_mean'], id_labels, m_labels
                        )
                    if 'projection_embed_mean' in part_aux:
                        projection_embed = part_aux['projection_embed']
                        projection_embed_m = part_aux['projection_embed_mean']
                        projection_logits_m = part_aux['projection_logits_mean']
                        proj_b, proj_t, proj_c = projection_embed.shape
                        loss_projection_id = criterion_ce_loss(projection_logits_m, labels)
                        loss_projection_mma = criterion_mma_loss(projection_embed_m, id_labels, m_labels)
                        loss_projection_mma = loss_projection_mma + criterion_mma_loss(
                            projection_embed.reshape(proj_b * proj_t, proj_c),
                            id_labels.repeat_interleave(proj_t),
                            m_labels.repeat_interleave(proj_t),
                        )
                        loss_projection_triplet = criterion_triplet_loss(projection_embed_m, id_labels, m_labels)
                        loss_projection_triplet_frames = criterion_triplet_loss(
                            projection_embed.reshape(proj_b * proj_t, proj_c),
                            id_labels.repeat_interleave(proj_t),
                            m_labels.repeat_interleave(proj_t),
                        )
                        loss_projection_triplet = (
                            loss_projection_triplet + args.triplet_frame_weight * loss_projection_triplet_frames
                        )
                        if 'projection_gate' in part_aux:
                            projection_gate = part_aux['projection_gate']
                        elif 'fusion_gate' in part_aux:
                            projection_gate = part_aux['fusion_gate']
                        else:
                            projection_gate = None
                        if projection_gate is not None:
                            ir_targets = projection_gate.new_full(
                                projection_gate.shape, args.projection_gate_ir_target
                            )
                            rgb_targets = projection_gate.new_full(
                                projection_gate.shape, args.projection_gate_rgb_target
                            )
                            gate_targets = torch.where(m_labels.unsqueeze(1) == 1, ir_targets, rgb_targets)
                            loss_projection_gate = F.mse_loss(projection_gate, gate_targets)
                    if 'calibration_embed_mean' in part_aux:
                        calibration_embed = part_aux['calibration_embed']
                        calibration_embed_m = part_aux['calibration_embed_mean']
                        calibration_logits = part_aux['calibration_logits']
                        calibration_logits_m = part_aux['calibration_logits_mean']
                        cal_b, cal_t, cal_c = calibration_embed.shape
                        loss_calibration_id = criterion_ce_loss(calibration_logits_m, labels)
                        loss_calibration_id = loss_calibration_id + criterion_ce_loss(
                            calibration_logits.reshape(cal_b * cal_t, -1), id_labels.repeat_interleave(cal_t)
                        )
                        loss_calibration_mma = criterion_mma_loss(calibration_embed_m, id_labels, m_labels)
                        loss_calibration_mma = loss_calibration_mma + criterion_mma_loss(
                            calibration_embed.reshape(cal_b * cal_t, cal_c),
                            id_labels.repeat_interleave(cal_t),
                            m_labels.repeat_interleave(cal_t),
                        )
                        loss_calibration_triplet = criterion_triplet_loss(calibration_embed_m, id_labels, m_labels)
                        loss_calibration_triplet_frames = criterion_triplet_loss(
                            calibration_embed.reshape(cal_b * cal_t, cal_c),
                            id_labels.repeat_interleave(cal_t),
                            m_labels.repeat_interleave(cal_t),
                        )
                        loss_calibration_triplet = (
                            loss_calibration_triplet
                            + args.triplet_frame_weight * loss_calibration_triplet_frames
                        )
                    if 'calibration_gate' in part_aux:
                        calibration_gate = part_aux['calibration_gate']
                        calibration_targets = calibration_gate.new_full(
                            calibration_gate.shape, args.calibration_gate_target
                        )
                        loss_calibration_gate = F.mse_loss(calibration_gate, calibration_targets)
                    if 'projection_alpha' in part_aux and 'calibration_alpha' in part_aux:
                        projection_target = args.projection_alpha_target
                        if projection_target is None:
                            projection_target = args.fusion_alpha
                        calibration_target = args.calibration_alpha_target
                        if calibration_target is None:
                            calibration_target = args.calibration_alpha
                        projection_targets = part_aux['projection_alpha'].new_full(
                            part_aux['projection_alpha'].shape, projection_target
                        )
                        calibration_targets = part_aux['calibration_alpha'].new_full(
                            part_aux['calibration_alpha'].shape, calibration_target
                        )
                        loss_branch_gate = F.mse_loss(part_aux['projection_alpha'], projection_targets)
                        loss_branch_gate = loss_branch_gate + F.mse_loss(
                            part_aux['calibration_alpha'], calibration_targets
                        )

            _, predicted = x_logits_m.max(dim=1)
            cls_acc = (predicted.eq(labels).sum().item()) / len(labels)

            loss_mid = loss_id + loss_id_frames
            loss_mma = loss_mma + loss_mma_frames
            loss_triplet = loss_triplet + args.triplet_frame_weight * loss_triplet_frames

            loss = loss_mid + loss_mma + loss_ofr + loss_dac
            if enable_triplet_loss:
                loss = loss + args.triplet_weight * loss_triplet
            loss = loss + args.part_id_weight * loss_part_id
            loss = loss + args.part_mma_weight * loss_part_mma
            loss = loss + args.part_triplet_weight * loss_part_triplet
            loss = loss + args.fusion_id_weight * loss_fusion_id
            loss = loss + args.fusion_triplet_weight * loss_fusion_triplet
            loss = loss + args.cosface_weight * loss_cosface
            loss = loss + args.part_cosface_weight * loss_part_cosface
            loss = loss + args.proto_weight * loss_proto
            loss = loss + args.part_proto_weight * loss_part_proto
            loss = loss + args.cross_proto_weight * loss_cross_proto
            loss = loss + args.part_cross_proto_weight * loss_part_cross_proto
            if args.use_m3plus and args.m3plus_mode in (
                'bidirectional_calibration', 'invariant_specific_calibration'
            ):
                loss = loss + args.router_weight * loss_router
            if args.use_m3plus and args.m3plus_mode == 'invariant_specific_calibration':
                loss = loss + args.modality_adv_weight * loss_modality_adv
                loss = loss + args.invariant_consistency_weight * loss_invariant_consistency
            if args.use_m3plus and args.m3plus_mode in (
                'anchor_projection_fusion', 'projection_calibrated_fusion',
                'adaptive_projection_calibrated_fusion'
            ):
                loss = loss + args.projection_id_weight * loss_projection_id
                loss = loss + args.projection_triplet_weight * loss_projection_triplet
                loss = loss + args.projection_mma_weight * loss_projection_mma
                loss = loss + args.projection_gate_weight * loss_projection_gate
            if args.use_m3plus and args.m3plus_mode in (
                'dual_calibrated_fusion', 'temporal_dual_calibrated_fusion',
                'projection_calibrated_fusion',
                'adaptive_projection_calibrated_fusion'
            ):
                loss = loss + args.calibration_id_weight * loss_calibration_id
                loss = loss + args.calibration_triplet_weight * loss_calibration_triplet
                loss = loss + args.calibration_mma_weight * loss_calibration_mma
                loss = loss + args.calibration_gate_weight * loss_calibration_gate
            if args.use_m3plus and args.m3plus_mode == 'adaptive_projection_calibrated_fusion':
                loss = loss + args.branch_gate_weight * loss_branch_gate

            backward_loss = loss / args.accum_steps
            if args.fp16:
                amp_scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            is_update_step = ((batch_idx + 1) % args.accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if args.max_train_batches is not None:
                is_update_step = is_update_step or ((batch_idx + 1) == args.max_train_batches)
            if is_update_step:
                if args.fp16:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]['lr']

            if (batch_idx + 1) % args.log_interval == 0:
                batch_num = len(train_loader)
                iter_num = epoch * batch_num + batch_idx + 1
                e_time = time.time()
                print(f'Epoch: [{epoch + 1}][{batch_idx + 1}/{batch_num}] '
                      f'Time: {e_time - s_time:.4f}s '
                      f'lr:{current_lr:.8f} '
                      f'cls_acc: {cls_acc:.4f} '
                      f'loss_mid: {loss_mid.data:.4f} '
                      f'loss_mma: {loss_mma.data:.4f} '
                      f'loss_triplet: {loss_triplet.data:.4f} '
                      f'loss_part_id: {loss_part_id.data:.4f} '
                      f'loss_part_triplet: {loss_part_triplet.data:.4f} '
                      f'loss_fusion_id: {loss_fusion_id.data:.4f} '
                      f'loss_fusion_triplet: {loss_fusion_triplet.data:.4f} '
                      f'fusion_gate: {fusion_gate_mean.data:.4f} '
                      f'loss_cosface: {loss_cosface.data:.4f} '
                      f'loss_part_cosface: {loss_part_cosface.data:.4f} '
                      f'loss_proto: {loss_proto.data:.4f} '
                      f'loss_part_proto: {loss_part_proto.data:.4f} '
                      f'loss_cross_proto: {loss_cross_proto.data:.4f} '
                      f'loss_part_cross_proto: {loss_part_cross_proto.data:.4f} '
                      f'loss_router: {loss_router.data:.4f} '
                      f'loss_modality_adv: {loss_modality_adv.data:.4f} '
                      f'loss_invariant_consistency: {loss_invariant_consistency.data:.4f} '
                      f'loss_projection_id: {loss_projection_id.data:.4f} '
                      f'loss_projection_triplet: {loss_projection_triplet.data:.4f} '
                      f'loss_projection_mma: {loss_projection_mma.data:.4f} '
                      f'loss_projection_gate: {loss_projection_gate.data:.4f} '
                      f'loss_calibration_id: {loss_calibration_id.data:.4f} '
                      f'loss_calibration_triplet: {loss_calibration_triplet.data:.4f} '
                      f'loss_calibration_mma: {loss_calibration_mma.data:.4f} '
                      f'loss_calibration_gate: {loss_calibration_gate.data:.4f} '
                      f'loss_branch_gate: {loss_branch_gate.data:.4f} '
                      f'loss_ofr: {loss_ofr.data:.4f} '
                      f'loss_dac: {loss_dac.data:.4f} '
                      )
                s_time = time.time()
                writer.add_scalar('metric/cls_acc', cls_acc, iter_num)
                writer.add_scalar('metric/loss_mid', loss_mid.data, iter_num)
                writer.add_scalar('metric/loss_mma', loss_mma.data, iter_num)
                writer.add_scalar('metric/loss_triplet', loss_triplet.data, iter_num)
                writer.add_scalar('metric/loss_part_id', loss_part_id.data, iter_num)
                writer.add_scalar('metric/loss_part_mma', loss_part_mma.data, iter_num)
                writer.add_scalar('metric/loss_part_triplet', loss_part_triplet.data, iter_num)
                writer.add_scalar('metric/loss_fusion_id', loss_fusion_id.data, iter_num)
                writer.add_scalar('metric/loss_fusion_triplet', loss_fusion_triplet.data, iter_num)
                writer.add_scalar('metric/fusion_gate_mean', fusion_gate_mean.data, iter_num)
                writer.add_scalar('metric/loss_cosface', loss_cosface.data, iter_num)
                writer.add_scalar('metric/loss_part_cosface', loss_part_cosface.data, iter_num)
                writer.add_scalar('metric/loss_proto', loss_proto.data, iter_num)
                writer.add_scalar('metric/loss_part_proto', loss_part_proto.data, iter_num)
                writer.add_scalar('metric/loss_cross_proto', loss_cross_proto.data, iter_num)
                writer.add_scalar('metric/loss_part_cross_proto', loss_part_cross_proto.data, iter_num)
                writer.add_scalar('metric/loss_router', loss_router.data, iter_num)
                writer.add_scalar('metric/loss_modality_adv', loss_modality_adv.data, iter_num)
                writer.add_scalar('metric/loss_invariant_consistency', loss_invariant_consistency.data, iter_num)
                writer.add_scalar('metric/loss_projection_id', loss_projection_id.data, iter_num)
                writer.add_scalar('metric/loss_projection_triplet', loss_projection_triplet.data, iter_num)
                writer.add_scalar('metric/loss_projection_mma', loss_projection_mma.data, iter_num)
                writer.add_scalar('metric/loss_projection_gate', loss_projection_gate.data, iter_num)
                writer.add_scalar('metric/loss_calibration_id', loss_calibration_id.data, iter_num)
                writer.add_scalar('metric/loss_calibration_triplet', loss_calibration_triplet.data, iter_num)
                writer.add_scalar('metric/loss_calibration_mma', loss_calibration_mma.data, iter_num)
                writer.add_scalar('metric/loss_calibration_gate', loss_calibration_gate.data, iter_num)
                writer.add_scalar('metric/loss_branch_gate', loss_branch_gate.data, iter_num)
                writer.add_scalar('metric/loss_ofr', loss_ofr.data, iter_num)
                writer.add_scalar('metric/loss_dac', loss_dac.data, iter_num)

        lr_scheduler.step()

        should_eval = (args.test_interval > 0 and (epoch + 1) >= args.eval_start_epoch
                       and (epoch + 1) % args.test_interval == 0)
        if should_eval:
            # -- Test --------------------------------------------------------------------------------------------------
            model.eval()

            s_time = time.time()

            eval_embedding_dim = getattr(model, 'output_dim', model.embedding_dim)
            query_embeddings = torch.zeros((num_query, eval_embedding_dim)).cuda()
            gallery_embeddings = torch.zeros((num_gallery, eval_embedding_dim)).cuda()
            query_ptr, gallery_ptr = 0, 0
            q_pids, q_cids, q_mids = [], [], []
            g_pids, g_cids, g_mids = [], [], []

            with torch.no_grad():

                for track_data, pids, cids, mids in query_loader:
                    inputs = track_data.cuda(non_blocking=args.non_blocking)
                    pids = pids.cuda(non_blocking=args.non_blocking)
                    cids = cids.cuda(non_blocking=args.non_blocking)
                    mids = mids.cuda(non_blocking=args.non_blocking)
                    batch_num = inputs.shape[0]
                    with torch.amp.autocast(device_type='cuda', enabled=args.eval_fp16):
                        embeddings = model(inputs)
                    query_embeddings[query_ptr:query_ptr + batch_num, :] = embeddings.detach()
                    query_ptr = query_ptr + batch_num
                    q_pids.extend(pids)
                    q_cids.extend(cids)
                    q_mids.extend(mids)
                q_pids = torch.stack(q_pids, dim=0)
                q_cids = torch.stack(q_cids, dim=0)
                q_mids = torch.stack(q_mids, dim=0)

                for track_data, pids, cids, mids in gallery_loader:
                    inputs = track_data.cuda(non_blocking=args.non_blocking)
                    pids = pids.cuda(non_blocking=args.non_blocking)
                    cids = cids.cuda(non_blocking=args.non_blocking)
                    mids = mids.cuda(non_blocking=args.non_blocking)
                    batch_num = inputs.shape[0]
                    with torch.amp.autocast(device_type='cuda', enabled=args.eval_fp16):
                        embeddings = model(inputs)
                    gallery_embeddings[gallery_ptr:gallery_ptr + batch_num, :] = embeddings.detach()
                    gallery_ptr = gallery_ptr + batch_num
                    g_pids.extend(pids)
                    g_cids.extend(cids)
                    g_mids.extend(mids)
                g_pids = torch.stack(g_pids, dim=0)
                g_cids = torch.stack(g_cids, dim=0)
                g_mids = torch.stack(g_mids, dim=0)

            e_time_1 = time.time()

            if args.dataset == 'HITSZVCM':
                i2v_dist_mat = -torch.matmul(query_embeddings, gallery_embeddings.t())
                i2v_sorted_indices = torch.argsort(i2v_dist_mat, dim=1)
                i2v_cmc, i2v_mAP, i2v_mINP = get_cmc_mAP_mINP(i2v_sorted_indices, q_pids, q_cids, g_pids, g_cids)
                v2i_dist_mat = i2v_dist_mat.t()
                v2i_sorted_indices = torch.argsort(v2i_dist_mat, dim=1)
                v2i_cmc, v2i_mAP, v2i_mINP = get_cmc_mAP_mINP(v2i_sorted_indices, g_pids, g_cids, q_pids, q_cids)
            elif args.dataset == 'BUPTCampus':
                i2v_query_embeddings = query_embeddings[q_mids == 1]
                i2v_gallery_embeddings = gallery_embeddings[g_mids == 2]
                i2v_q_pids, i2v_q_cids = q_pids[q_mids == 1], q_cids[q_mids == 1]
                i2v_g_pids, i2v_g_cids = g_pids[g_mids == 2], g_cids[g_mids == 2]
                i2v_dist_mat = -torch.matmul(i2v_query_embeddings, i2v_gallery_embeddings.t())
                i2v_sorted_indices = torch.argsort(i2v_dist_mat, dim=1)
                i2v_cmc, i2v_mAP, i2v_mINP = get_cmc_mAP_mINP(i2v_sorted_indices,
                                                              i2v_q_pids, i2v_q_cids, i2v_g_pids, i2v_g_cids)
                v2i_query_embeddings = query_embeddings[q_mids == 2]
                v2i_gallery_embeddings = gallery_embeddings[g_mids == 1]
                v2i_q_pids, v2i_q_cids = q_pids[q_mids == 2], q_cids[q_mids == 2]
                v2i_g_pids, v2i_g_cids = g_pids[g_mids == 1], g_cids[g_mids == 1]
                v2i_dist_mat = -torch.matmul(v2i_query_embeddings, v2i_gallery_embeddings.t())
                v2i_sorted_indices = torch.argsort(v2i_dist_mat, dim=1)
                v2i_cmc, v2i_mAP, v2i_mINP = get_cmc_mAP_mINP(v2i_sorted_indices,
                                                              v2i_q_pids, v2i_q_cids, v2i_g_pids, v2i_g_cids)
            else:
                raise RuntimeError(f'Dataset {args.dataset} is not supported for now.')

            e_time_2 = time.time()

            info_str = (f'EVAL - Epoch: [{epoch + 1}] '
                        f'Time: {e_time_2 - s_time:.4f} ({e_time_1 - s_time:.4f} + {e_time_2 - e_time_1:.4f})s \n'
                        f'Mode - i2v  '
                        f'r1: {i2v_cmc[0]:.2%} '
                        f'r5: {i2v_cmc[4]:.2%} '
                        f'r10: {i2v_cmc[9]:.2%} '
                        f'r20: {i2v_cmc[19]:.2%} '
                        f'mAP: {i2v_mAP:.2%} '
                        f'mINP: {i2v_mINP:.2%} \n'
                        f'Mode - v2i  '
                        f'r1: {v2i_cmc[0]:.2%} '
                        f'r5: {v2i_cmc[4]:.2%} '
                        f'r10: {v2i_cmc[9]:.2%} '
                        f'r20: {v2i_cmc[19]:.2%} '
                        f'mAP: {v2i_mAP:.2%} '
                        f'mINP: {v2i_mINP:.2%} ')

            info_str = '~' * 100 + '\n' + info_str + '\n' + '~' * 100
            print(info_str)

            # Log Overall Performance
            writer.add_scalar('eval/i2v_r1', i2v_cmc[0], epoch + 1)
            writer.add_scalar('eval/i2v_r5', i2v_cmc[4], epoch + 1)
            writer.add_scalar('eval/i2v_r10', i2v_cmc[9], epoch + 1)
            writer.add_scalar('eval/i2v_r20', i2v_cmc[19], epoch + 1)
            writer.add_scalar('eval/i2v_mAP', i2v_mAP, epoch + 1)
            writer.add_scalar('eval/i2v_mINP', i2v_mINP, epoch + 1)
            writer.add_scalar('eval/v2i_r1', v2i_cmc[0], epoch + 1)
            writer.add_scalar('eval/v2i_r5', v2i_cmc[4], epoch + 1)
            writer.add_scalar('eval/v2i_r10', v2i_cmc[9], epoch + 1)
            writer.add_scalar('eval/v2i_r20', v2i_cmc[19], epoch + 1)
            writer.add_scalar('eval/v2i_mAP', v2i_mAP, epoch + 1)
            writer.add_scalar('eval/v2i_mINP', v2i_mINP, epoch + 1)

            avg_r1 = (i2v_cmc[0] + v2i_cmc[0]) / 2
            if avg_r1 > best_score:
                best_score = avg_r1
                stale_eval_count = 0
                best_epoch = epoch + 1
                best_result = {
                    'i2v_cmc': i2v_cmc.detach().clone(),
                    'i2v_mAP': i2v_mAP.detach().clone(),
                    'i2v_mINP': i2v_mINP.detach().clone(),
                    'v2i_cmc': v2i_cmc.detach().clone(),
                    'v2i_mAP': v2i_mAP.detach().clone(),
                    'v2i_mINP': v2i_mINP.detach().clone(),
                }
                torch.save(model.state_dict(), os.path.join(modelckpt_dir, 'model_best.pth'))
            else:
                stale_eval_count += 1
                print(f'No avg Rank-1 improvement for {stale_eval_count} evaluated round(s). '
                      f'Best avg Rank-1: {best_score:.2%} @ epoch {best_epoch}.')

            if args.early_stop_patience > 0 and stale_eval_count >= args.early_stop_patience:
                print(f'Early stopping at epoch {epoch + 1}: no avg Rank-1 improvement for '
                      f'{stale_eval_count} evaluated rounds.')
                break

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            # -- Save --------------------------------------------------------------------------------------------------
            torch.save(model.state_dict(), os.path.join(modelckpt_dir, f'model_epoch-{epoch + 1}.pth'))

    if best_result is not None:
        info_str = (f'BEST-RESULT @ Epoch [{best_epoch}]\n'
                    f'Mode - i2v  '
                    f'r1: {best_result["i2v_cmc"][0]:.2%} '
                    f'r5: {best_result["i2v_cmc"][4]:.2%} '
                    f'r10: {best_result["i2v_cmc"][9]:.2%} '
                    f'r20: {best_result["i2v_cmc"][19]:.2%} '
                    f'mAP: {best_result["i2v_mAP"]:.2%} '
                    f'mINP: {best_result["i2v_mINP"]:.2%}\n'
                    f'Mode - v2i  '
                    f'r1: {best_result["v2i_cmc"][0]:.2%} '
                    f'r5: {best_result["v2i_cmc"][4]:.2%} '
                    f'r10: {best_result["v2i_cmc"][9]:.2%} '
                    f'r20: {best_result["v2i_cmc"][19]:.2%} '
                    f'mAP: {best_result["v2i_mAP"]:.2%} '
                    f'mINP: {best_result["v2i_mINP"]:.2%}')
        info_str = '~' * 100 + '\n' + info_str + '\n' + '~' * 100
        print(info_str)

    writer.close()

