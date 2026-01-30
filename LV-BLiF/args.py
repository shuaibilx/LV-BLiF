import argparse
from omegaconf import OmegaConf
import os
import sys
from data_splits import CROSS_VALIDATION_SPLITS


def parse_args():

    parser = argparse.ArgumentParser(description='Light Field IQA Training')

    parser.add_argument('--config', type=str, default='configs/combined.yaml',
                        help="Path to the config file (default: configs/combined.yaml)")
    parser.add_argument('--active_dataset', type=str, required=True,
                        choices=['NBU', 'SHU', 'Win5LID'],
                        help="Which dataset to run 5-fold cross-validation on.")
    parser.add_argument('--device', type=str, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--lr', type=float, dest='learning_rate', help="Learning rate")
    parser.add_argument('--wd', type=float, dest='weight_decay', help="Weight decay")
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--run_name_suffix', type=str, default='',
                        help="Optional suffix for the run name (e.g., '_exp1')")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    cfg = OmegaConf.load(args.config)
    cfg.active_dataset = args.active_dataset
    cfg.config_file_path = args.config
    cli_args_dict = vars(args)
    cli_override_keys = {k: v for k, v in cli_args_dict.items()
                         if v is not None and k not in ['config', 'active_dataset']}
    cfg_cli_overrides = OmegaConf.create(cli_override_keys)
    cfg = OmegaConf.merge(cfg, cfg_cli_overrides)

    if cfg.active_dataset not in cfg.datasets:
        print(
            f"Error: active_dataset '{cfg.active_dataset}' is not defined in the 'datasets' section of {args.config}.")
        sys.exit(1)
    if cfg.active_dataset not in CROSS_VALIDATION_SPLITS:
        print(f"Error: active_dataset '{cfg.active_dataset}' does not have defined splits in data_splits.py.")
        sys.exit(1)

    run_name = f"cv_on_{cfg.active_dataset}{args.run_name_suffix}"
    cfg.output_dir = os.path.join(cfg.output_dir, run_name)
    cfg.log_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    return cfg

