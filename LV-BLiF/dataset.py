import torch
from torch.utils.data import Dataset
import os
import numpy as np
import logging
import h5py
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)


class LFIQADataset(Dataset):
    def __init__(self,
                 h5_dir: str,
                 lmm_dir: str,
                 h5_filename_list: list,
                 cfg):
        super().__init__()
        self.cfg = cfg
        self.h5_dir = h5_dir
        self.lmm_dir = lmm_dir


        self.lmm_feature_cache = {}


        self.data_index = []
        self._build_index(h5_filename_list)

    def _build_index(self, h5_filename_list):
        logger.info(f"Indexing patches from {len(h5_filename_list)} H5 files...")
        for h5_filename in tqdm(h5_filename_list, desc="Building dataset index"):
            h5_path = os.path.join(self.h5_dir, h5_filename)


            scene_id = os.path.splitext(h5_filename)[0]
            lmm_feature_path = os.path.join(self.lmm_dir, f"scene_{scene_id}_features.pt")

            if not os.path.exists(h5_path):
                logger.warning(f"H5 file not found at '{h5_path}'. Skipping instance.")
                continue
            if not os.path.exists(lmm_feature_path):
                logger.warning(f"LMM feature file not found at '{lmm_feature_path}'. Skipping instance.")
                continue

            try:
                with h5py.File(h5_path, 'r') as hf:
                    num_patches = hf['visual_patches'].shape[0]
            except Exception as e:
                logger.error(f"Failed to read num_patches from {h5_path}: {e}")
                continue

            for patch_idx in range(num_patches):
                self.data_index.append({
                    "h5_path": h5_path,
                    "lmm_feature_path": lmm_feature_path,
                    "patch_index": patch_idx,
                    "instance_id": h5_filename
                })
        logger.info(f"Finished indexing. Total patches available: {len(self.data_index)}")

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        sample_info = self.data_index[idx]
        h5_path = sample_info["h5_path"]
        lmm_feature_path = sample_info["lmm_feature_path"]
        patch_index = sample_info["patch_index"]
        instance_id = sample_info["instance_id"]

        try:

            if lmm_feature_path in self.lmm_feature_cache:
                lmm_features = self.lmm_feature_cache[lmm_feature_path]
            else:
                lmm_features = torch.load(lmm_feature_path, map_location='cpu',weights_only=True).to(torch.float32)
                self.lmm_feature_cache[lmm_feature_path] = lmm_features


            with h5py.File(h5_path, 'r') as hf:
                visual_patch_np = hf['visual_patches'][patch_index]
                mos_score = np.array(hf['mos'][()], dtype=np.float32)
                spatial_label = np.array(hf['spa_mos'][()], dtype=np.float32)
                angular_label = np.array(hf['ang_mos'][()], dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to load data (h5:'{h5_path}', lmm:'{lmm_feature_path}') at index {patch_index}: {e}")
            return self._get_placeholder_sample(instance_id)


        visual_patch = torch.from_numpy(visual_patch_np).to(torch.float32)
        if random.random() < 0.5: visual_patch = torch.flip(visual_patch, dims=[-2])
        if random.random() < 0.5: visual_patch = torch.flip(visual_patch, dims=[-1])

        sample_data = {
            'visual_patch': visual_patch,
            'lmm_features': lmm_features,
            'mos_true': torch.from_numpy(mos_score).unsqueeze(0),
            's_true_spatial': torch.from_numpy(spatial_label).unsqueeze(0),
            's_true_angular': torch.from_numpy(angular_label).unsqueeze(0),
            'instance_id': instance_id,
        }
        return sample_data

    def _get_placeholder_sample(self, instance_id="placeholder.h5"):
        vp_shape = (1, 576, 576)
        lmm_shape = (81, self.cfg.get('lmm_feature_dim', 4096))
        return {
            'visual_patch': torch.zeros(vp_shape, dtype=torch.float32),
            'lmm_features': torch.zeros(lmm_shape, dtype=torch.float32),
            'mos_true': torch.zeros(1, dtype=torch.float32),
            's_true_spatial': torch.zeros(1, dtype=torch.float32),
            's_true_angular': torch.zeros(1, dtype=torch.float32),
            'instance_id': instance_id,
        }