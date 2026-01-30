import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.amp.grad_scaler import GradScaler
import logging
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from collections import defaultdict
from omegaconf import OmegaConf
from args import parse_args
from dataset import LFIQADataset
from model import QualityAssessmentModel
from engine import train_one_step, evaluate_on_test_set  
from utils import setup_logging, set_seed, save_best_checkpoint, plot_loss_curves, LFIQA_collate_fn
from data_splits import get_split_for_fold

logger = logging.getLogger(__name__)


def discover_h5_files_and_map_to_scenes(h5_dir: str) -> dict:
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"H5 dataset directory not found at: {h5_dir}")
    scene_to_h5_list_map = defaultdict(list)
    h5_files = [f for f in os.listdir(h5_dir) if f.lower().endswith('.h5')]
    logger.info(f"Discovering scenes from {len(h5_files)} H5 files in '{h5_dir}'...")
    for h5_filename in tqdm(h5_files, desc="Discovering Scenes"):
        h5_path = os.path.join(h5_dir, h5_filename)
        try:
            with h5py.File(h5_path, 'r') as hf:
                scene_name_data = hf['scene_name'][()]
                if isinstance(scene_name_data, bytes):
                    scene_name = scene_name_data.decode('utf-8')
                elif isinstance(scene_name_data, np.ndarray):
                    scene_name_val = scene_name_data.item(0)
                    if isinstance(scene_name_val, bytes):
                        scene_name = scene_name_val.decode('utf-8')
                    else:
                        scene_name = str(scene_name_val)
                else:
                    scene_name = str(scene_name_data)
                scene_to_h5_list_map[scene_name.strip()].append(h5_filename)
        except Exception as e:
            logger.error(f"Failed to read scene_name from {h5_path}: {e}")
            continue
    return scene_to_h5_list_map


def main(cfg):
    set_seed(cfg.seed)
    run_base_name = f"cv_on_{cfg.active_dataset}"
    log_dir = os.path.join(cfg.log_dir, run_base_name)
    output_dir = os.path.join(cfg.output_dir, run_base_name)
    setup_logging(log_dir, log_name=f"train_log_all_folds.log")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(
        f"--- Starting {cfg.cross_validation.num_folds}-Fold Cross-Validation for Dataset: {cfg.active_dataset} ---")
    logger.info(
        f"Device: {device}, Total Train Epochs: {cfg.epochs}")

    dataset_cfg = cfg.datasets[cfg.active_dataset]
    lmm_feature_dir_for_dataset = os.path.join(cfg.lmm_feature_dir, dataset_cfg.lmm_feature_subdir)
    scene_to_h5_list_map = discover_h5_files_and_map_to_scenes(dataset_cfg.h5_dataset_dir)
    all_content_scenes = sorted(list(scene_to_h5_list_map.keys()))

    all_folds_metrics = []

    for fold in range(1, cfg.cross_validation.num_folds + 1):
        logger.info(f"\n{'=' * 25} FOLD {fold}/{cfg.cross_validation.num_folds} {'=' * 25}")

        train_content_scenes, test_content_scenes = get_split_for_fold(cfg.active_dataset, all_content_scenes, fold)
        train_h5_files = [h5 for scene in train_content_scenes for h5 in scene_to_h5_list_map[scene]]
        test_h5_files = [h5 for scene in test_content_scenes for h5 in scene_to_h5_list_map[scene]]
        logger.info(
            f"Fold {fold}: {len(train_h5_files)} H5 files for training, {len(test_h5_files)} H5 files for testing.")

        train_dataset = LFIQADataset(dataset_cfg.h5_dataset_dir, lmm_feature_dir_for_dataset, train_h5_files, cfg)
        test_dataset = LFIQADataset(dataset_cfg.h5_dataset_dir, lmm_feature_dir_for_dataset, test_h5_files, cfg)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True, drop_last=True, collate_fn=LFIQA_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True, collate_fn=LFIQA_collate_fn)

        model = QualityAssessmentModel(cfg).to(device)

        if cfg.differential_lr.enabled:
            pass
        else:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        criterion = nn.MSELoss().to(device)
        steps_per_epoch = len(train_loader)
        total_train_steps_for_scheduler = steps_per_epoch * cfg.epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps_for_scheduler,
                                                   eta_min=cfg.lr_scheduler.eta_min)
        logger.info(f"Scheduler `CosineAnnealingLR` initialized with T_max = {total_train_steps_for_scheduler} steps.")

        scaler = GradScaler(enabled=(device.type == "cuda"))
        best_composite_metric_for_fold = -float('inf')
        best_metrics_for_fold = {}
        train_losses_per_epoch = []
        val_maes_per_epoch = []

        for epoch in range(1, cfg.epochs + 1):
            model.train() 
            epoch_train_losses = []
            progress_bar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{cfg.epochs} [Training]",
                                file=sys.stdout)

            for batch in progress_bar:
                total_loss, mos_loss, _, _ = train_one_step(model, batch, criterion, optimizer, device, cfg, scaler)

                epoch_train_losses.append(mos_loss)
                progress_bar.set_postfix(
                    avg_loss=f"{np.mean(epoch_train_losses):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}"
                )
                if scheduler:
                    scheduler.step()

            avg_epoch_train_loss = np.mean(epoch_train_losses)
            train_losses_per_epoch.append(avg_epoch_train_loss)
            logger.info(f"Fold {fold} Epoch {epoch} - Average Training MOS Loss: {avg_epoch_train_loss:.4f}")
            eval_loss, test_metrics = evaluate_on_test_set(
                model, test_loader, device, cfg, epoch_num=epoch
            )
            val_maes_per_epoch.append(eval_loss)

            logger.info(
                f"--- Epoch [{epoch}/{cfg.epochs}] Eval Summary: "
                f"PLCC = {test_metrics['plcc']:.4f}, SRCC = {test_metrics['srcc']:.4f}, "
                f"RMSE = {test_metrics['rmse']:.4f}, MAE = {test_metrics['mae']:.4f} ---"
            )

            current_composite_metric = (test_metrics.get('plcc', -1) + test_metrics.get('srcc', -1)) / 2.0
            if current_composite_metric > best_composite_metric_for_fold:
                best_composite_metric_for_fold = current_composite_metric
                best_metrics_for_fold = test_metrics
                fold_output_dir = os.path.join(output_dir, f"fold_{fold}")

                save_best_checkpoint({
                    'epoch': epoch,  
                    'state_dict': model.state_dict(),
                    'best_composite_metric': best_composite_metric_for_fold,
                    'test_metrics_at_best': best_metrics_for_fold,
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'config': OmegaConf.to_container(cfg, resolve=True)
                }, output_dir=fold_output_dir)


        if train_losses_per_epoch and val_maes_per_epoch:
            fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
            plot_filename = f"loss_curves_fold_{fold}_epochs.png"
            plot_loss_curves(
                train_losses=train_losses_per_epoch,
                val_losses=val_maes_per_epoch,
                output_dir=fold_output_dir,
                filename=plot_filename,
                title=f"Loss Curves for Fold {fold} on {cfg.active_dataset}",
                x_label_train="Epochs",  
                x_label_val="Epochs"
            )

        logger.info(
            f"*** Fold {fold} Best Metrics: PLCC={best_metrics_for_fold.get('plcc', -1):.4f}, SRCC={best_metrics_for_fold.get('srcc', -1):.4f} ***")
        all_folds_metrics.append(best_metrics_for_fold)

    logger.info(f"\n{'=' * 25} CROSS-VALIDATION SUMMARY FOR {cfg.active_dataset} {'=' * 25}")
    if all_folds_metrics:
        metrics_df = pd.DataFrame(all_folds_metrics)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        logger.info("Metrics for each fold:")
        logger.info("\n" + metrics_df.to_string())

        logger.info("\nAverage metrics across all folds (mean ± std):")
        for metric in mean_metrics.index:
            logger.info(f"  - Average {metric.upper()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    logger.info(f"--- Run for Active Dataset: {cfg.active_dataset} COMPLETED ---")
    logger.info(f"All logs and models saved in: {output_dir}")


if __name__ == '__main__':
    config_from_args = parse_args()

    main(config_from_args)
