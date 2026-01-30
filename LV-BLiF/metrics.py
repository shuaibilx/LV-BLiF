import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
def calculate_metrics(y_true, y_pred):

    metrics = {'plcc': np.nan, 'srcc': np.nan, 'mae': np.nan, 'rmse': np.nan}
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if len(y_true) != len(y_pred):
            print("Error: Length mismatch between y_true and y_pred.")
            return metrics
        if len(y_true) < 2:
            print("Error: Need at least 2 samples to calculate correlations.")
            return metrics

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print("Warning: NaNs or Infs found in predictions. Metrics might be unreliable.")

        # PLCC
        plcc, p_plcc = pearsonr(y_true, y_pred)
        metrics['plcc'] = plcc

        # SRCC
        srcc, p_srcc = spearmanr(y_true, y_pred)
        metrics['srcc'] = srcc

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = rmse

    except Exception as e:
        print(f"Error calculating metrics: {e}")

    return metrics

import torch


def plcc_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:


    x = y_pred.view(-1)
    y = y_true.view(-1)
    vx = x - x.mean()
    vy = y - y.mean()
    denominator = torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8)
    corr = torch.sum(vx * vy) / denominator

    return 1.0 - corr

