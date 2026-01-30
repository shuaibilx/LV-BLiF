import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from einops import rearrange
from torch.utils.checkpoint import checkpoint
logger = logging.getLogger(__name__)
import torchvision.models as models


import torchvision.models as models

class ResNetBlock(nn.Module):

    def __init__(self, in_chans, out_chans, resnet_version='resnet18', patch_size=64, angRes=9):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        if resnet_version == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            resnet_out_channels = 256
        elif resnet_version == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            resnet_out_channels = 256
        else:
            raise ValueError(f"Unsupported resnet_version: {resnet_version}")

        original_conv1 = resnet.conv1
        self.modified_conv1 = nn.Conv2d(in_chans, original_conv1.out_channels,
                                        kernel_size=original_conv1.kernel_size, stride=original_conv1.stride,
                                        padding=original_conv1.padding, bias=False)
        with torch.no_grad():
            mean_weight = original_conv1.weight.data.mean(dim=1, keepdim=True)
            self.modified_conv1.weight.data.fill_(0)
            for i in range(in_chans):
                self.modified_conv1.weight.data[:, i:i + 1, :, :] = mean_weight

        self.feature_extractor = nn.Sequential(
            self.modified_conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )

        target_upsample_size = angRes * patch_size

        self.decoder = nn.Sequential(
            nn.Conv2d(resnet_out_channels, out_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Upsample(size=(target_upsample_size, target_upsample_size), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output_feature_map = self.decoder(features)
        return output_feature_map
class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        y = x.view(b, c, h, w * self.factor)
        return y


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels // 4, channels // 2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.BatchNorm2d(SpaChannel), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.BatchNorm2d(SpaChannel), nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.BatchNorm2d(AngChannel), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(angRes * angRes * AngChannel), nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes],
                      padding=[0, angRes * (angRes - 1) // 2], bias=False),
            nn.BatchNorm2d(EpiChannel), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(angRes * EpiChannel), nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(channels), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        return buffer + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.Block = nn.Sequential(*[DisentgBlock(angRes, channels) for _ in range(n_block)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        return self.conv(self.Block(x)) + x


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.Group = nn.ModuleList([DisentgGroup(n_block, angRes, channels) for _ in range(n_group)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for group in self.Group:
            buffer = group(buffer)
        return self.conv(buffer) + x


def SAI2MacPI(x, angRes):
    return rearrange(x, 'b c (u h) (v w) -> b c (h u) (w v)', u=angRes, v=angRes)


class TransformerAggregator(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)
        logger.info(f"Initialized TransformerAggregator with {num_layers} layers and {num_heads} heads.")

    def forward(self, x):
        output = self.transformer_encoder(x)
        output = self.norm(output)
        return torch.mean(output, dim=1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_S_from_A = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_A_from_S = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2)
        )
        self.norm3 = nn.LayerNorm(embed_dim * 2)

    def forward(self, feat_spatial, feat_angular):
        s_seq = feat_spatial.unsqueeze(1)
        a_seq = feat_angular.unsqueeze(1)

        s_enhanced, _ = self.cross_attn_S_from_A(query=s_seq, key=a_seq, value=a_seq)
        s_out = self.norm1(s_seq + s_enhanced)

        a_enhanced, _ = self.cross_attn_A_from_S(query=a_seq, key=s_seq, value=s_seq)
        a_out = self.norm2(a_seq + a_enhanced)

        fused_vec = torch.cat([s_out.squeeze(1), a_out.squeeze(1)], dim=1)
        fused_vec = self.norm3(fused_vec + self.ffn(fused_vec))
        return fused_vec


class AuxBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, F_agg):
        intermediate_feat = self.relu1(self.fc1(F_agg))
        intermediate_feat = self.dropout(intermediate_feat)
        score_pred_raw = self.fc2(intermediate_feat)
        score_pred_activated = self.output_activation(score_pred_raw)
        return intermediate_feat, score_pred_activated

class FiLMLayer(nn.Module):
    def __init__(self, context_dim, feature_channels):
        super().__init__()
        self.generator = nn.Linear(context_dim, feature_channels * 2)

    def forward(self, feature_map, context_vector):
        params = self.generator(context_vector).view(context_vector.shape[0], -1, 1, 1)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return gamma * feature_map + beta


class VisualBranch(nn.Module):

    def __init__(self, cfg_visual, cfg, cfg_semantic, use_film_layer: bool = True):
        super().__init__()
        channels = cfg_visual.channels
        context_dim = cfg.aux_hidden_dim * 2
        self.angRes = cfg_visual.angRes
        resnet_version = cfg_visual.get('resnet_version', 'resnet18')
        patch_size = cfg_visual.get('patch_size', 64)

        self.use_film = use_film_layer

        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=self.angRes, padding=self.angRes,
                                   bias=False)
        self.disentangler = ParallelDisentangler(cfg_visual.n_group, cfg_visual.n_block, self.angRes, channels)

        self.resnet_spa = ResNetBlock(in_chans=channels, out_chans=channels, resnet_version=resnet_version,
                                      patch_size=patch_size, angRes=self.angRes)

        self.resnet_ang = ResNetBlock(in_chans=channels, out_chans=channels, resnet_version=resnet_version,
                                      patch_size=patch_size, angRes=self.angRes)
        self.resnet_epi = ResNetBlock(in_chans=channels , out_chans=channels, resnet_version=resnet_version,
                                      patch_size=patch_size, angRes=self.angRes)

        if self.use_film:
            self.film_layer = FiLMLayer(context_dim, channels)
        else:
            self.film_layer = None

        gate_input_dim = channels * 3
        self.gate = nn.Sequential(nn.Linear(gate_input_dim, 3), nn.Softmax(dim=-1))
        self.regression_head = nn.Sequential(nn.Linear(channels, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, visual_patch, context_vector):
        macpi_patch = SAI2MacPI(visual_patch, self.angRes)
        init_features = self.init_conv(macpi_patch)
        map_spa_local, map_ang_local, map_epi_local = self.disentangler(init_features)
        map_spa_global = self.resnet_spa(map_spa_local)
        map_ang_global = self.resnet_ang(map_ang_local)
        map_epi_global = self.resnet_epi(map_epi_local)
        if self.use_film and self.film_layer is not None and context_vector is not None:
            map_spa_mod = self.film_layer(map_spa_global, context_vector)
            map_ang_mod = self.film_layer(map_ang_global, context_vector)
            map_epi_mod = self.film_layer(map_epi_global, context_vector)
        else:
            map_spa_mod = map_spa_global
            map_ang_mod = map_ang_global
            map_epi_mod = map_epi_global
        v_spa = F.adaptive_avg_pool2d(map_spa_mod, 1).flatten(1)
        v_ang = F.adaptive_avg_pool2d(map_ang_mod, 1).flatten(1)
        v_epi = F.adaptive_avg_pool2d(map_epi_mod, 1).flatten(1)

        gate_input = torch.cat([v_spa, v_ang, v_epi], dim=1)
        gate_input = F.layer_norm(gate_input, (gate_input.shape[-1],))
        gating_weights = self.gate(gate_input).unsqueeze(-1)

        all_vectors = torch.stack([v_spa, v_ang, v_epi], dim=1)
        v_fused = torch.sum(all_vectors * gating_weights, dim=1)

        score_visual = self.regression_head(v_fused)
        return score_visual

class SemanticBranch(nn.Module):
    def __init__(self, cfg, use_auxiliary_learning: bool = True):
        super().__init__()
        self.use_aux = use_auxiliary_learning
        if self.use_aux:
            self.spatial_branch = AuxBranch(cfg.lmm_feature_dim, cfg.aux_hidden_dim, 1, cfg.model.aux_branch_dropout)
            self.angular_branch = AuxBranch(cfg.lmm_feature_dim, cfg.aux_hidden_dim, 1, cfg.model.aux_branch_dropout)
            self.cross_attention = CrossAttentionFusion(cfg.aux_hidden_dim, cfg.cross_attn.num_heads,
                                                        cfg.cross_attn.dropout)

            self.regression_head = nn.Sequential(nn.Linear(cfg.aux_hidden_dim * 2, 128), nn.GELU(), nn.Linear(128, 1))
        else:

            self.spatial_branch = None
            self.angular_branch = None
            self.cross_attention = None
            self.regression_head = nn.Sequential(nn.Linear(cfg.lmm_feature_dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, F_lmm_views):
        F_agg = torch.mean(F_lmm_views, dim=1)

        if self.use_aux:
            feat_spatial, score_spatial_pred = self.spatial_branch(F_agg)
            feat_angular, score_angular_pred = self.angular_branch(F_agg)

            context_vector = self.cross_attention(feat_spatial, feat_angular)
            score_semantic = self.regression_head(context_vector)

            return context_vector, score_semantic, score_spatial_pred, score_angular_pred
        else:
            score_semantic = self.regression_head(F_agg)

            return None, score_semantic, None, None


class QualityAssessmentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        switches = cfg.model.get('ablation_switches', {})
        self.use_semantic = switches.get('use_semantic_branch', True)
        self.use_aux = switches.get('use_auxiliary_learning', True)
        self.use_film = switches.get('use_film_layer', True)

        logger.info("--- Ablation Switches Status ---")
        if not self.use_semantic:
            if self.use_aux or self.use_film:
                logger.warning("`use_semantic_branch` is False. Forcing Auxiliary Learning and FiLM to be disabled.")
                self.use_aux = False
                self.use_film = False
        elif not self.use_aux and self.use_film:
            logger.warning("`use_auxiliary_learning` is False. FiLM requires it, so forcing FiLM to be disabled.")
            self.use_film = False

        logger.info(f"Semantic Branch       : {'ENABLED' if self.use_semantic else 'DISABLED'}")
        logger.info(f"Auxiliary Learning    : {'ENABLED' if self.use_aux else 'DISABLED'}")
        logger.info(f"FiLM Layer Modulation : {'ENABLED' if self.use_film else 'DISABLED'}")
        logger.info("--------------------------------")

        if self.use_semantic:
            self.semantic_branch = SemanticBranch(cfg, use_auxiliary_learning=self.use_aux)
        else:
            self.semantic_branch = None

        self.use_visual_branch = cfg.visual_branch_config.get('enabled', False)
        if self.use_visual_branch:
            self.visual_branch = VisualBranch(cfg.visual_branch_config, cfg, cfg, use_film_layer=self.use_film)
            self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.visual_branch = None

        self.final_output_activation = nn.Sigmoid()
        self.mos_min_overall = cfg.get('mos_min_overall', 1.0)
        self.mos_max_overall = cfg.get('mos_max_overall', 5.0)

    def forward(self, F_lmm_views, visual_patch=None, **kwargs):
        context_vector, score_semantic, spa_pred, ang_pred = None, None, None, None
        score_visual = None
        if self.use_semantic and self.semantic_branch is not None:
            context_vector, score_semantic, spa_pred, ang_pred = self.semantic_branch(F_lmm_views)
        if self.use_visual_branch and self.visual_branch is not None and visual_patch is not None:
            score_visual = self.visual_branch(visual_patch, context_vector)
        if score_semantic is not None and score_visual is not None:
            w_vis = torch.sigmoid(self.fusion_logit)
            final_raw_score = (1 - w_vis) * score_semantic + w_vis * score_visual
        elif score_semantic is not None:
            final_raw_score = score_semantic
        elif score_visual is not None:
            final_raw_score = score_visual
        else:
            raise ValueError("Both semantic and visual branches are disabled. Cannot compute a score.")

        score_0_to_1 = self.final_output_activation(final_raw_score)
        mos_range = self.mos_max_overall - self.mos_min_overall
        score_final_predicted = self.mos_min_overall + score_0_to_1 * mos_range

        return score_final_predicted, spa_pred, ang_pred, None

class ParallelDisentangler(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super().__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels // 4, channels // 2
        self.groups = nn.ModuleList([DisentgGroup(n_block, angRes, channels) for _ in range(n_group)])
        self.final_spa_conv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.BatchNorm2d(SpaChannel), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.BatchNorm2d(SpaChannel), nn.LeakyReLU(0.1, inplace=True),
        )

        self.final_ang_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.BatchNorm2d(channels), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * (angRes ** 2), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels * (angRes ** 2)), nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )

        self.final_epi_conv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes],
                      padding=[0, angRes * (angRes - 1) // 2], bias=False),
            nn.BatchNorm2d(EpiChannel), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(angRes * EpiChannel), nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )

    def forward(self, x):
        buffer = x
        for group in self.groups:
            buffer = group(buffer)
        map_spa = self.final_spa_conv(buffer)
        map_ang = self.final_ang_conv(buffer)
        epi_h = self.final_epi_conv(buffer)
        epi_v = self.final_epi_conv(buffer.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        map_epi = torch.cat((epi_h,epi_v), dim=1)

        return map_spa, map_ang, map_epi
