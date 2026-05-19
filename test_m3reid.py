# ------------------------------------------------------------------------------
# File:    M3-ReID/test_m3reid.py
#
# Description:
#    The main testing script for the M3-ReID framework.
#    It evaluates the trained model checkpoint.
#
# Key Features:
# - Loads trained weights and initializes the M3-ReID model.
# - Extracts features for Query and Gallery sets using video-level inference.
# - Computes Cosine Distance matrices for cross-modality matching.
# - Calculates and reports standard metrics: Rank-1, Rank-5, Rank-10, mAP, and mINP.
# - Evaluates both Infrared-to-Visible (I2V) and Visible-to-Infrared (V2I) modes.
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
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.manager import HITSZVCMDataManager
from data.manager import BUPTCampusDataManager
from data.dataset import VideoVIDataset
from data.transform import SyncTrackTransform
from models.model_m3reid import M3ReID
from tools.eval_metrics import get_cmc_mAP_mINP
from tools.utils import time_str, Logger


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


def collate_all_tracks(batch):
    track_clips, pids, cids, mids = zip(*batch)
    return track_clips, pids, cids, mids


def extract_embeddings(model, dataset, loader, num_tracks, embedding_dim, args):
    embeddings = torch.zeros((num_tracks, embedding_dim)).cuda()
    ptr = 0
    all_pids, all_cids, all_mids = [], [], []

    if args.eval_sample_mode == 'evenly':
        for track_data, pids, cids, mids in loader:
            inputs = track_data.cuda(non_blocking=args.non_blocking)
            pids = pids.cuda(non_blocking=args.non_blocking)
            cids = cids.cuda(non_blocking=args.non_blocking)
            mids = mids.cuda(non_blocking=args.non_blocking)
            batch_num = inputs.shape[0]
            with torch.amp.autocast(device_type='cuda', enabled=args.eval_fp16):
                batch_embeddings = model(inputs)
            embeddings[ptr:ptr + batch_num, :] = batch_embeddings.detach()
            ptr += batch_num
            all_pids.extend(pids)
            all_cids.extend(cids)
            all_mids.extend(mids)
        return embeddings, torch.stack(all_pids, dim=0), torch.stack(all_cids, dim=0), torch.stack(all_mids, dim=0)

    if args.eval_sample_mode != 'all':
        raise ValueError(f'Unsupported eval_sample_mode: {args.eval_sample_mode}')

    track_index = 0
    for track_clips_batch, pids, cids, mids in loader:
        for track_clips, pid, cid, mid in zip(track_clips_batch, pids, cids, mids):
            if args.max_eval_clips is not None and args.max_eval_clips > 0:
                track_clips = track_clips[:args.max_eval_clips]
            clip_tensor = torch.stack(track_clips, dim=0)

            clip_embeddings = []
            for start in range(0, clip_tensor.size(0), args.batch_size):
                inputs = clip_tensor[start:start + args.batch_size].cuda(non_blocking=args.non_blocking)
                with torch.amp.autocast(device_type='cuda', enabled=args.eval_fp16):
                    clip_embeddings.append(model(inputs).detach())
            track_embedding = torch.cat(clip_embeddings, dim=0).mean(dim=0, keepdim=True)
            embeddings[track_index:track_index + 1, :] = F.normalize(track_embedding, p=2, dim=1)
            track_index += 1
            all_pids.append(pid)
            all_cids.append(cid)
            all_mids.append(mid)

    return (embeddings,
            torch.tensor(all_pids, device='cuda'),
            torch.tensor(all_cids, device='cuda'),
            torch.tensor(all_mids, device='cuda'))


def compute_part_match_similarity(query_embeddings, gallery_embeddings, global_dim, part_dim, part_num,
                                  neighbor_radius=1, symmetric=True):
    if part_dim <= 0 or part_num <= 0 or part_dim % part_num != 0:
        raise ValueError('part_dim must be positive and divisible by part_num for part matching.')

    expected_dim = global_dim + part_dim
    if query_embeddings.size(1) < expected_dim or gallery_embeddings.size(1) < expected_dim:
        raise ValueError('Part matching requires dual-fusion embeddings with a global segment and a part segment.')

    part_channels = part_dim // part_num
    q_parts = query_embeddings[:, global_dim:expected_dim].reshape(-1, part_num, part_channels)
    g_parts = gallery_embeddings[:, global_dim:expected_dim].reshape(-1, part_num, part_channels)
    q_parts = F.normalize(q_parts.float(), p=2, dim=2)
    g_parts = F.normalize(g_parts.float(), p=2, dim=2)

    radius = max(0, int(neighbor_radius))
    score = q_parts.new_zeros((q_parts.size(0), g_parts.size(0)))
    for q_idx in range(part_num):
        g_start = max(0, q_idx - radius)
        g_end = min(part_num, q_idx + radius + 1)
        best = None
        for g_idx in range(g_start, g_end):
            sim = torch.matmul(q_parts[:, q_idx, :], g_parts[:, g_idx, :].t())
            best = sim if best is None else torch.maximum(best, sim)
        score += best
    score = score / part_num

    if symmetric:
        reverse_score = q_parts.new_zeros((q_parts.size(0), g_parts.size(0)))
        for g_idx in range(part_num):
            q_start = max(0, g_idx - radius)
            q_end = min(part_num, g_idx + radius + 1)
            best = None
            for q_idx in range(q_start, q_end):
                sim = torch.matmul(q_parts[:, q_idx, :], g_parts[:, g_idx, :].t())
                best = sim if best is None else torch.maximum(best, sim)
            reverse_score += best
        score = 0.5 * (score + reverse_score / part_num)

    return score


def resolve_part_match_weight(args, direction):
    direction_weight = getattr(args, f'part_match_weight_{direction}', None)
    return args.part_match_weight if direction_weight is None else direction_weight


def compute_distance_matrix(query_embeddings, gallery_embeddings, args, global_dim, part_match_weight=None):
    match_weight = args.part_match_weight if part_match_weight is None else part_match_weight
    similarity = torch.matmul(query_embeddings, gallery_embeddings.t())
    if match_weight > 0:
        part_similarity = compute_part_match_similarity(
            query_embeddings, gallery_embeddings,
            global_dim=global_dim,
            part_dim=args.part_dim,
            part_num=args.part_num,
            neighbor_radius=args.part_match_neighbor_radius,
            symmetric=args.part_match_symmetric,
        )
        similarity = similarity + match_weight * part_similarity
    return -similarity


if __name__ == '__main__':

    # Arguments --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Video-based visible-infrared cross-modality ReID')

    # -- Data Arguments ------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', default='HITSZVCM', help='Dataset name')
    parser.add_argument('--dataset_dir', default='../Datasets/HITSZ-VCM', help='Directory of dataset')
    parser.add_argument('--img_h', default=288, type=int, help='Height of input images')
    parser.add_argument('--img_w', default=144, type=int, help='Width of input images')
    parser.add_argument('--t', default=6, type=int, help='Number of sampled frames per video track')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
    parser.add_argument('--workers', default=4, type=int, help='Num of dataloader workers')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Pin dataloader memory for faster host-to-GPU copies')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep dataloader workers alive when workers > 0')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Dataloader prefetch factor when workers > 0. Set <=0 to disable')
    parser.add_argument('--non_blocking', action=argparse.BooleanOptionalAction, default=True,
                        help='Use non-blocking CUDA transfers when pin_memory is enabled')
    parser.add_argument('--eval_sample_mode', default='evenly', choices=['evenly', 'all'],
                        help='Use one evenly sampled clip or aggregate all non-overlapping clips per track')
    parser.add_argument('--max_eval_clips', default=None, type=int,
                        help='Optional cap on clips per track when eval_sample_mode=all')

    # -- Other Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--resume', default=None, type=str, help='Resume from path of checkpoint')
    parser.add_argument('--eval_fp16', action='store_true', default=False,
                        help='Use AMP autocast during feature extraction')
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable cuDNN benchmark for fixed image sizes')
    parser.add_argument('--use_m3plus', action='store_true', default=False,
                        help='Enable enhanced M3-ReID architecture used by M3Plus checkpoints')
    parser.add_argument('--m3plus_mode', default='full',
                        choices=['full', 'part_only', 'local_residual', 'dual_fusion',
                                 'temporal_dual_fusion', 'adaptive_dual_fusion'],
                        help='M3Plus architecture mode used by the checkpoint')
    parser.add_argument('--part_num', default=4, type=int, help='Number of horizontal local parts for M3Plus')
    parser.add_argument('--part_dim', default=2048, type=int,
                        help='Output dimension of the M3Plus local part branch')
    parser.add_argument('--mvl_num_heads', default=2, type=int,
                        help='Number of MVL attention heads per view')
    parser.add_argument('--feature_dropout', default=0.0, type=float,
                        help='Dropout value used by the checkpoint architecture')
    parser.add_argument('--temporal_dim', default=256, type=int,
                        help='Hidden dimension of temporal_dual_fusion refinement head')
    parser.add_argument('--temporal_dropout', default=0.0, type=float,
                        help='Dropout inside temporal_dual_fusion refinement head')
    parser.add_argument('--fusion_alpha', default=0.2, type=float,
                        help='Local feature weight used by dual_fusion inference, or initial local weight for adaptive_dual_fusion')
    parser.add_argument('--adaptive_gate_min', default=0.0, type=float,
                        help='Minimum local feature weight predicted by adaptive_dual_fusion')
    parser.add_argument('--adaptive_gate_max', default=0.12, type=float,
                        help='Maximum local feature weight predicted by adaptive_dual_fusion')
    parser.add_argument('--part_match_weight', default=0.0, type=float,
                        help='Additive weight for Scheme L explicit part-level matching during evaluation')
    parser.add_argument('--part_match_weight_i2v', default=None, type=float,
                        help='Optional Scheme L part-match weight override for i2v evaluation')
    parser.add_argument('--part_match_weight_v2i', default=None, type=float,
                        help='Optional Scheme L part-match weight override for v2i evaluation')
    parser.add_argument('--part_match_neighbor_radius', default=1, type=int,
                        help='Part index radius for Scheme L matching. 0 matches only aligned parts')
    parser.add_argument('--part_match_symmetric', action=argparse.BooleanOptionalAction, default=True,
                        help='Average query-to-gallery and gallery-to-query local-window part scores')
    parser.add_argument('--grad_checkpoint_head', action='store_true', default=False,
                        help='Accepted for architecture parity; checkpointing is only active during training')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--desc', type=str, default=None, help='Description for this testing process')

    args = parser.parse_args()

    # Env  -------------------------------------------------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for M3-ReID testing.')
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.set_float32_matmul_precision('high')  # highest high medium

    suffix = f'Time-{time_str()}' if (args.desc is None) else f'Time-{time_str()}_{args.desc}'

    ckptlog_dir = os.path.join('ckptlog', args.dataset, suffix)
    os.makedirs(ckptlog_dir, exist_ok=True)

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

    # Data -------------------------------------------------------------------------------------------------------------
    sample_seq_num = args.t
    test_batch_size = args.batch_size  # Set Appropriate Values Based on GPU Memory
    loader_kwargs = build_loader_kwargs(args)
    print(f'Dataloader setting: {loader_kwargs}, non_blocking_cuda={args.non_blocking}, '
          f'eval_fp16={args.eval_fp16}, eval_sample_mode={args.eval_sample_mode}, '
          f'max_eval_clips={args.max_eval_clips}')
    i2v_part_match_weight = resolve_part_match_weight(args, 'i2v')
    v2i_part_match_weight = resolve_part_match_weight(args, 'v2i')
    print(f'SchemeL part matching: weight={args.part_match_weight}, '
          f'i2v_weight={i2v_part_match_weight}, v2i_weight={v2i_part_match_weight}, '
          f'neighbor_radius={args.part_match_neighbor_radius}, symmetric={args.part_match_symmetric}')

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

    transform_test = SyncTrackTransform(T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_h, args.img_w)),
        T.ToTensor(),
        normalize,
    ]))

    query_dataset = VideoVIDataset(data_manager, transform=transform_test,
                                   sample_seq_num=sample_seq_num, sample_mode=args.eval_sample_mode,
                                   dataset_mode='query')
    gallery_dataset = VideoVIDataset(data_manager, transform=transform_test,
                                     sample_seq_num=sample_seq_num, sample_mode=args.eval_sample_mode,
                                     dataset_mode='gallery')
    if args.eval_sample_mode == 'evenly':
        query_loader = DataLoader(query_dataset, batch_size=test_batch_size,
                                  shuffle=False, **loader_kwargs)
        gallery_loader = DataLoader(gallery_dataset, batch_size=test_batch_size,
                                    shuffle=False, **loader_kwargs)
    else:
        query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False,
                                  collate_fn=collate_all_tracks, **loader_kwargs)
        gallery_loader = DataLoader(gallery_dataset, batch_size=1, shuffle=False,
                                    collate_fn=collate_all_tracks, **loader_kwargs)

    # Model ------------------------------------------------------------------------------------------------------------
    model = M3ReID(sample_seq_num, num_train_class,
                   use_enhancements=args.use_m3plus, m3plus_mode=args.m3plus_mode, part_num=args.part_num,
                   mvl_num_heads=args.mvl_num_heads, part_dim=args.part_dim,
                   feature_dropout=args.feature_dropout, fusion_alpha=args.fusion_alpha,
                   grad_checkpoint_head=args.grad_checkpoint_head,
                   temporal_dim=args.temporal_dim, temporal_dropout=args.temporal_dropout,
                   adaptive_gate_min=args.adaptive_gate_min,
                   adaptive_gate_max=args.adaptive_gate_max).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
        for key in list(checkpoint.keys()):
            model_state_dict = model.state_dict()
            if key in model_state_dict:
                if torch.is_tensor(checkpoint[key]) and checkpoint[key].shape != model_state_dict[key].shape:
                    print(f'Warning during loading weights - Auto remove mismatch key: {key}')
                    checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=False)

    # Test --------------------------------------------------------------------------------------------------------
    model.eval()

    s_time = time.time()

    eval_embedding_dim = getattr(model, 'output_dim', model.embedding_dim)
    global_embedding_dim = model.embedding_dim
    if max(i2v_part_match_weight, v2i_part_match_weight) > 0:
        dual_modes = ('dual_fusion', 'temporal_dual_fusion', 'adaptive_dual_fusion')
        if not (args.use_m3plus and args.m3plus_mode in dual_modes):
            raise ValueError('Scheme L part matching requires a dual-fusion M3Plus checkpoint.')
        if eval_embedding_dim < global_embedding_dim + args.part_dim:
            raise ValueError('Scheme L part matching could not find the local part segment in embeddings.')
    with torch.no_grad():
        query_embeddings, q_pids, q_cids, q_mids = extract_embeddings(
            model, query_dataset, query_loader, num_query, eval_embedding_dim, args
        )
        gallery_embeddings, g_pids, g_cids, g_mids = extract_embeddings(
            model, gallery_dataset, gallery_loader, num_gallery, eval_embedding_dim, args
        )

    e_time_1 = time.time()

    if args.dataset == 'HITSZVCM':
        i2v_dist_mat = compute_distance_matrix(
            query_embeddings, gallery_embeddings, args, global_embedding_dim,
            part_match_weight=i2v_part_match_weight,
        )
        i2v_sorted_indices = torch.argsort(i2v_dist_mat, dim=1)
        i2v_cmc, i2v_mAP, i2v_mINP = get_cmc_mAP_mINP(i2v_sorted_indices, q_pids, q_cids, g_pids, g_cids)
        if v2i_part_match_weight != i2v_part_match_weight or not args.part_match_symmetric:
            v2i_dist_mat = compute_distance_matrix(
                gallery_embeddings, query_embeddings, args, global_embedding_dim,
                part_match_weight=v2i_part_match_weight,
            )
        else:
            v2i_dist_mat = i2v_dist_mat.t()
        v2i_sorted_indices = torch.argsort(v2i_dist_mat, dim=1)
        v2i_cmc, v2i_mAP, v2i_mINP = get_cmc_mAP_mINP(v2i_sorted_indices, g_pids, g_cids, q_pids, q_cids)
    elif args.dataset == 'BUPTCampus':
        i2v_query_embeddings = query_embeddings[q_mids == 1]
        i2v_gallery_embeddings = gallery_embeddings[g_mids == 2]
        i2v_q_pids, i2v_q_cids = q_pids[q_mids == 1], q_cids[q_mids == 1]
        i2v_g_pids, i2v_g_cids = g_pids[g_mids == 2], g_cids[g_mids == 2]
        i2v_dist_mat = compute_distance_matrix(
            i2v_query_embeddings, i2v_gallery_embeddings, args, global_embedding_dim,
            part_match_weight=i2v_part_match_weight,
        )
        i2v_sorted_indices = torch.argsort(i2v_dist_mat, dim=1)
        i2v_cmc, i2v_mAP, i2v_mINP = get_cmc_mAP_mINP(i2v_sorted_indices,
                                                      i2v_q_pids, i2v_q_cids, i2v_g_pids, i2v_g_cids)
        v2i_query_embeddings = query_embeddings[q_mids == 2]
        v2i_gallery_embeddings = gallery_embeddings[g_mids == 1]
        v2i_q_pids, v2i_q_cids = q_pids[q_mids == 2], q_cids[q_mids == 2]
        v2i_g_pids, v2i_g_cids = g_pids[g_mids == 1], g_cids[g_mids == 1]
        v2i_dist_mat = compute_distance_matrix(
            v2i_query_embeddings, v2i_gallery_embeddings, args, global_embedding_dim,
            part_match_weight=v2i_part_match_weight,
        )
        v2i_sorted_indices = torch.argsort(v2i_dist_mat, dim=1)
        v2i_cmc, v2i_mAP, v2i_mINP = get_cmc_mAP_mINP(v2i_sorted_indices,
                                                      v2i_q_pids, v2i_q_cids, v2i_g_pids, v2i_g_cids)
    else:
        raise RuntimeError(f'Dataset {args.dataset} is not supported for now.')

    e_time_2 = time.time()

    info_str = (f'EVAL'
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

    # Save -------------------------------------------------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(modelckpt_dir, f'model.pth'))

