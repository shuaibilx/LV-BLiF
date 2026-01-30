#utils.py

import random
import numpy as np
import torch
import logging
import sys
import os
import matplotlib.pyplot as plt
from metrics import calculate_metrics  # Import from metrics.py
import torch
from torch.utils.data.dataloader import default_collate # Import default_collate
import logging # Ensure logging is imported
import scipy.optimize as opt


def setup_logging(log_dir, log_name='train.log'):
    """ Sets up logging to file and console. """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')  # Simpler format for console
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def save_best_checkpoint(state: dict, output_dir: str):

    try:
        os.makedirs(output_dir, exist_ok=True)
        best_filepath = os.path.join(output_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        logging.info(f"Saved new best model to {best_filepath}")
    except Exception as e:
        logging.error(f"Failed to save best checkpoint in {output_dir}: {e}")


def plot_loss_curves(train_losses: list,
                     val_losses: list,
                     output_dir: str,
                     filename: str = 'loss_curves.png',
                     title: str = 'Training and Validation Loss',
                     y_label: str = 'Loss',
                     x_label_train: str = 'Epochs',
                     x_label_val: str = 'Epochs'
                     ):

    try:
        os.makedirs(output_dir, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(12, 7))

        color = 'tab:blue'
        ax1.set_xlabel(x_label_train, fontsize=12)
        ax1.set_ylabel(y_label, color=color, fontsize=12)
        ax1.plot(train_losses, color=color, marker='.', linestyle='-', label='Train Loss (MOS Loss)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, axis='y')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Validation MAE', color=color, fontsize=12)

        ax2.plot(val_losses, color=color, marker='o', linestyle='--', label='Validation MAE')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(title, fontsize=16)
        fig.tight_layout()

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved loss curves to {save_path}")
    except Exception as e:
        logging.error(f"Failed to plot loss curves: {e}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):  # Added scheduler
    if os.path.isfile(checkpoint_path):
        logging.info(f"=> Loading checkpoint '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        start_epoch = checkpoint.get('epoch', 0)  # Default to 0 if epoch not found
        best_metric = checkpoint.get('best_composite_metric', -float('inf'))  # Use the new metric name

        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)  # Simpler, assuming saved without module. prefix or handled by model itself

        if optimizer and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("  Loaded optimizer state.")
            except Exception as e:
                logging.warning(f"  Could not load optimizer state: {e}")

        if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                logging.info("  Loaded scheduler state.")
            except Exception as e:
                logging.warning(f"  Could not load scheduler state: {e}")

        logging.info(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_metric: {best_metric:.4f})")
        return start_epoch, best_metric
    else:
        logging.info(f"=> No checkpoint found at '{checkpoint_path}'")
        return 0, -float('inf')  # Default for best_composite_metric (higher is better)

logger_utils = logging.getLogger(__name__ + ".utils") # Specific logger for utils

import torch
import logging
from torch.utils.data import default_collate

logger_collate = logging.getLogger(__name__ + ".collate_fn")

def LFIQA_collate_fn(batch):

    first_sample = batch[0]
    has_visual_patch = first_sample.get('visual_patch') is not None
    has_instance_id = first_sample.get('instance_id') is not None

    instance_ids = None
    if has_instance_id:
        instance_ids = [sample.pop('instance_id') for sample in batch]

    visual_patches = None
    if has_visual_patch:
        first_patch_shape = first_sample['visual_patch'].shape
        if not all(sample['visual_patch'].shape == first_patch_shape for sample in batch):
            logger_collate.error("FATAL SHAPE MISMATCH in collate_fn for 'visual_patch'.")
            raise RuntimeError("Shape mismatch for 'visual_patch'.")
        visual_patches = [sample.pop('visual_patch') for sample in batch]

    collated_batch = default_collate(batch)

    if has_visual_patch:
        collated_batch['visual_patch'] = default_collate(visual_patches)
    else:
        collated_batch['visual_patch'] = None

    if has_instance_id:
        collated_batch['instance_ids'] = instance_ids

    return collated_batch


def apply_logistic_fitting(y_pred_np, y_true_np):

    if opt is None:
        logging.warning("Scipy is not available, skipping logistic fitting.")
        return y_pred_np

    try:

        def logistic_curve(x, beta1, beta2, beta3, beta4):
            logistic_part = 0.5 - 1.0 / (1.0 + np.exp((x - beta3) / beta4))
            return beta1 + (beta2 - beta1) * logistic_part
        beta1_initial = np.max(y_true_np)
        beta2_initial = np.min(y_true_np)
        beta3_initial = np.mean(y_pred_np)
        beta4_initial = np.std(y_pred_np) / 4.0
        initial_params = [beta1_initial, beta2_initial, beta3_initial, beta4_initial]
        popt, _ = opt.curve_fit(
            logistic_curve,
            y_pred_np,
            y_true_np,
            p0=initial_params,
            maxfev=10000,
            bounds=([-np.inf, -np.inf, -np.inf, 1e-7], [np.inf, np.inf, np.inf, np.inf])
        )
        y_pred_calibrated = logistic_curve(y_pred_np, *popt)

        return y_pred_calibrated
    except Exception as e:

        logging.warning(f"Logistic fitting failed with error: {e}. Returning original predictions.")
        return y_pred_np

logger_utils = logging.getLogger(__name__)

