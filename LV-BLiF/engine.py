from __future__ import annotations
import logging
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
import sys
import random

from metrics import calculate_metrics, plcc_loss
from utils import apply_logistic_fitting
from torch.amp.grad_scaler import GradScaler

logger = logging.getLogger(__name__)

def train_one_step(
        model: nn.Module,
        batch: dict,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg,
        scaler: GradScaler
) -> Tuple[float, float, float, float]:
    model.train()
    F_lmm = batch["lmm_features"].to(device, non_blocking=True)
    mos_gt = batch["mos_true"].to(device, non_blocking=True).view(-1, 1)
    spa_gt = batch["s_true_spatial"].to(device, non_blocking=True)
    ang_gt = batch["s_true_angular"].to(device, non_blocking=True)

    visual_patch_batch = None
    if cfg.visual_branch_config.get('enabled', False):
        visual_patch_batch = batch.get("visual_patch")
        if visual_patch_batch is not None:
            visual_patch_batch = visual_patch_batch.to(device, non_blocking=True)
            if random.random() < 0.5:
                visual_patch_batch = torch.flip(visual_patch_batch, dims=[-2])
            if random.random() < 0.5:
                visual_patch_batch = torch.flip(visual_patch_batch, dims=[-1])

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        (mos_pred, spa_pred, ang_pred, _) = model(F_lmm_views=F_lmm, visual_patch=visual_patch_batch)

        if cfg.label_smoothing.enabled:
            alpha = cfg.label_smoothing.alpha
            center_value = (cfg.mos_min_overall + cfg.mos_max_overall) / 2.0
            mos_gt_smoothed = mos_gt * (1.0 - alpha) + alpha * center_value
        else:
            mos_gt_smoothed = mos_gt

        mos_gt_smoothed = mos_gt_smoothed.view_as(mos_pred)
        mse_loss = criterion(mos_pred, mos_gt_smoothed)
        plc_loss = plcc_loss(mos_pred, mos_gt_smoothed)
        mos_loss_composite = cfg.w_mse * mse_loss + cfg.w_plcc * plc_loss
        total_loss = cfg.w_main * mos_loss_composite
        spa_loss_val, ang_loss_val = 0.0, 0.0

        if spa_pred is not None:
            spa_gt = spa_gt.view_as(spa_pred)
            spa_loss = criterion(spa_pred, spa_gt)
            total_loss = total_loss + cfg.w_spa_aux * spa_loss
            spa_loss_val = spa_loss.item() * cfg.w_spa_aux

        if ang_pred is not None:
            ang_gt = ang_gt.view_as(ang_pred)
            ang_loss = criterion(ang_pred, ang_gt)
            total_loss = total_loss + cfg.w_ang_aux * ang_loss
            ang_loss_val = ang_loss.item() * cfg.w_ang_aux

    scaler.scale(total_loss).backward()
    if cfg.grad_clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()

    return total_loss.item(), mos_loss_composite.item(), spa_loss_val, ang_loss_val


@torch.no_grad()
def evaluate_on_test_set(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        cfg,
        epoch_num: int
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    instance_predictions = defaultdict(list)
    instance_gts = {}

    progress_bar_eval = tqdm(
        test_loader,
        desc=f"Epoch {epoch_num} [Evaluation]",
        file=sys.stdout
    )

    for batch in progress_bar_eval:
        F_lmm = batch["lmm_features"].to(device, non_blocking=True)
        mos_gt_batch = batch["mos_true"].to(device, non_blocking=True)
        visual_patch_batch = batch.get("visual_patch")
        if visual_patch_batch is not None:
            visual_patch_batch = visual_patch_batch.to(device, non_blocking=True)

        instance_ids_batch = batch["instance_ids"]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            mos_pred_batch, _, _, _ = model(F_lmm_views=F_lmm, visual_patch=visual_patch_batch)

        mos_pred_cpu = mos_pred_batch.cpu().numpy().flatten()
        mos_gt_cpu = mos_gt_batch.cpu().numpy().flatten()

        for i, instance_id in enumerate(instance_ids_batch):
            instance_predictions[instance_id].append(mos_pred_cpu[i])
            if instance_id not in instance_gts:
                instance_gts[instance_id] = mos_gt_cpu[i]

    if not instance_predictions:
        logger.warning("Evaluation resulted in no predictions. Returning NaN metrics.")
        return np.nan, {'plcc': np.nan, 'srcc': np.nan, 'mae': np.nan, 'rmse': np.nan}

    final_preds = []
    final_gts = []
    sorted_instance_ids = sorted(instance_predictions.keys())

    for instance_id in sorted_instance_ids:
        avg_prediction = np.mean(instance_predictions[instance_id])
        final_preds.append(avg_prediction)
        final_gts.append(instance_gts[instance_id])

    preds_np = np.array(final_preds)
    gts_np = np.array(final_gts)

    eval_loss_mae = np.mean(np.abs(preds_np - gts_np))

    if cfg.get('apply_logistic_fitting_on_eval', False):
        preds_np_fitted = apply_logistic_fitting(preds_np, gts_np)
        metric_dict = calculate_metrics(gts_np, preds_np_fitted)
    else:
        metric_dict = calculate_metrics(gts_np, preds_np)

    return eval_loss_mae, metric_dict