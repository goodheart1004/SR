from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

DEPTH_INPUT_CHANNELS = 1
FEATURE_DIM = 64


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class LocalRefinementNet(nn.Module):

    def __init__(self, in_channels, channels=FEATURE_DIM, num_blocks=4, boundary_refinement=False):
        super().__init__()
        self.boundary_refinement = boundary_refinement
        self.in_projection = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.upsample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        if boundary_refinement:
            self.flat_residual_head = nn.Conv2d(channels, DEPTH_INPUT_CHANNELS, kernel_size=3, padding=1)
            self.edge_residual_head = nn.Conv2d(channels, DEPTH_INPUT_CHANNELS, kernel_size=3, padding=1)
            self.boundary_gate = nn.Sequential(
                nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, DEPTH_INPUT_CHANNELS, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.residual_head = nn.Conv2d(channels, DEPTH_INPUT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, features, initial_dsm, boundary_prior=None):
        x = self.in_projection(features)
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        if x.shape[-2:] != initial_dsm.shape[-2:]:
            x = F.interpolate(x, size=initial_dsm.shape[-2:], mode='bilinear', align_corners=False)

        if not self.boundary_refinement:
            residual = self.residual_head(x)
            return initial_dsm + residual, residual, {}

        has_boundary_prior = boundary_prior is not None
        boundary_prior = self._resize_boundary_prior(boundary_prior, x)
        flat_residual = self.flat_residual_head(x)
        edge_residual = self.edge_residual_head(x)
        learned_boundary_gate = self.boundary_gate(torch.cat([x, boundary_prior], dim=1))
        boundary_gate = (
            0.5 * (learned_boundary_gate + boundary_prior)
            if has_boundary_prior
            else learned_boundary_gate
        )
        residual = (1.0 - boundary_gate) * flat_residual + boundary_gate * edge_residual
        aux = {
            'flat_residual': flat_residual,
            'edge_residual': edge_residual,
            'boundary_gate': boundary_gate,
            'learned_boundary_gate': learned_boundary_gate,
            'boundary_prior': boundary_prior,
        }
        return initial_dsm + residual, residual, aux

    @staticmethod
    def _resize_boundary_prior(boundary_prior, reference):
        if boundary_prior is None:
            return reference.new_zeros(reference.shape[0], 1, reference.shape[-2], reference.shape[-1])
        boundary_prior = boundary_prior.to(device=reference.device, dtype=reference.dtype)
        if boundary_prior.ndim == 3:
            boundary_prior = boundary_prior.unsqueeze(1)
        if boundary_prior.shape[1] != 1:
            boundary_prior = boundary_prior.mean(dim=1, keepdim=True)
        if boundary_prior.shape[-2:] != reference.shape[-2:]:
            boundary_prior = F.interpolate(
                boundary_prior,
                size=reference.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        return boundary_prior.clamp(0.0, 1.0)


class GateConv2D(nn.Module):
    """Apply a learned, per-location gate before cross-modal interaction."""

    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=channels),
        )

    def forward(self, x):
        return self.attention(x) * self.feature(x)


class EITF(nn.Module):
    """DSM-guided local RGB similarity modulation.

    This is the efficient equivalent of the diagonal-only correlation used by
    DEMSR: it compares normalized 3x3 RGB and DSM patches at the same spatial
    location, rather than materializing an unused N-by-N similarity matrix.
    """

    def __init__(self, channels):
        super().__init__()
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=channels),
        )

    def forward(self, rgb_features, dsm_features):
        target_size = dsm_features.shape[-2:]
        rgb_lr = F.interpolate(
            rgb_features,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )

        rgb_patches = F.unfold(rgb_lr, kernel_size=3, padding=1)
        dsm_patches = F.unfold(dsm_features, kernel_size=3, padding=1)
        similarity = (F.normalize(rgb_patches, dim=1) * F.normalize(dsm_patches, dim=1)).sum(dim=1)
        similarity = similarity.view(rgb_lr.shape[0], 1, *target_size)

        rgb_modulated = self.rgb_projection(rgb_lr * (1.0 + similarity))
        rgb_modulated = F.interpolate(
            rgb_modulated,
            size=rgb_features.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return rgb_modulated, similarity


class MMAB(nn.Module):
    """Joint-statistics channel gating for HR RGB and upsampled DSM features."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        if reduction <= 0:
            raise ValueError('cross-modal reduction must be positive')
        hidden_channels = max(8, (2 * channels) // reduction)
        self.squeeze = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=hidden_channels),
        )
        self.depth_excitation = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rgb_excitation = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, dsm_features, rgb_features):
        dsm_hr = F.interpolate(
            dsm_features,
            size=rgb_features.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        fused_features = self.squeeze(torch.cat([dsm_hr, rgb_features], dim=1))
        mean = F.adaptive_avg_pool2d(fused_features, output_size=1)
        variance = (fused_features - mean).square().mean(dim=(2, 3), keepdim=True)
        joint_statistics = mean + variance

        depth_gate = self.depth_excitation(joint_statistics)
        rgb_gate = self.rgb_excitation(joint_statistics)
        return 0.5 * depth_gate * dsm_hr, 0.5 * rgb_gate * rgb_features, depth_gate, rgb_gate


class RGBDSMFeatureFusion(nn.Module):
    """Fuse a local RGB--DSM interaction into existing full-resolution features."""

    def __init__(self, feature_channels, reduction=2):
        super().__init__()
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, feature_channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=feature_channels),
        )
        self.dsm_stem = nn.Sequential(
            nn.Conv2d(DEPTH_INPUT_CHANNELS, feature_channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=feature_channels),
        )
        self.rgb_gate = GateConv2D(feature_channels)
        self.dsm_gate = GateConv2D(feature_channels)
        self.eitf = EITF(feature_channels)
        self.mmab = MMAB(feature_channels, reduction=reduction)
        self.cross_projection = nn.Sequential(
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=feature_channels),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, base_features, rgb, dsm_lr):
        rgb_features = self.rgb_gate(self.rgb_stem(rgb))
        dsm_features = self.dsm_gate(self.dsm_stem(dsm_lr))
        rgb_features, similarity = self.eitf(rgb_features, dsm_features)
        dsm_features, rgb_features, depth_gate, rgb_gate = self.mmab(dsm_features, rgb_features)

        cross_features = self.cross_projection(torch.cat([dsm_features, rgb_features], dim=1))
        if cross_features.shape[-2:] != base_features.shape[-2:]:
            cross_features = F.interpolate(
                cross_features,
                size=base_features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        fusion_gate = self.fusion_gate(torch.cat([base_features, cross_features], dim=1))
        fused_features = base_features + fusion_gate * cross_features

        aux = {
            'rgb_dsm_similarity': similarity.detach(),
            'rgb_dsm_depth_gate': depth_gate.detach(),
            'rgb_dsm_rgb_gate': rgb_gate.detach(),
            'rgb_dsm_fusion_gate': fusion_gate.mean(dim=1, keepdim=True).detach(),
        }
        return fused_features, aux


class GADBase(nn.Module):
    
    def __init__(
            self, feature_extractor='UNet',
            Npre=8000, Ntrain=1024,
            guide_channels=4,
            use_refinement_net=True,
            refinement_channels=FEATURE_DIM,
            refinement_blocks=4,
            refinement_only=False,
            boundary_refinement=False,
            use_cross_modal_fusion=False,
            cross_modal_reduction=2,
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.guide_channels = guide_channels
        self.refinement_only = refinement_only
        self.boundary_refinement = boundary_refinement
        self.refinement_net = None
        self.cross_modal_fusion = None
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.Ntrain = 0
            self.register_buffer('logk', torch.log(torch.tensor(0.03)))
            feature_channels = guide_channels + DEPTH_INPUT_CHANNELS

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            self.feature_extractor =  torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=1, mode='bicubic'),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=guide_channels + DEPTH_INPUT_CHANNELS),
                torch.nn.Identity())
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))
            feature_channels = FEATURE_DIM

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

        if use_cross_modal_fusion:
            if guide_channels < 3:
                raise ValueError('Cross-modal RGB--DSM fusion requires at least three RGB guide channels')
            self.cross_modal_fusion = RGBDSMFeatureFusion(
                feature_channels,
                reduction=cross_modal_reduction,
            )

        if use_refinement_net:
            self.refinement_net = LocalRefinementNet(
                in_channels=feature_channels,
                channels=refinement_channels,
                num_blocks=refinement_blocks,
                boundary_refinement=boundary_refinement,
            )
             

    def forward(self, sample, train=False):
        guide = sample['guide']
        y_bicubic = sample['y_bicubic'].clone()
        guide_feats = self.extract_features(y_bicubic, guide.clone())

        aux = {}
        if self.cross_modal_fusion is not None:
            guide_feats, cross_modal_aux = self.cross_modal_fusion(
                base_features=guide_feats,
                rgb=guide[:, :3],
                dsm_lr=sample['source'],
            )
            aux.update(cross_modal_aux)
        if self.refinement_net is not None:
            y_init, refinement_residual, refinement_aux = self.refinement_net(
                guide_feats,
                y_bicubic,
                boundary_prior=sample.get('boundary_prior'),
            )
            aux['refinement_residual'] = refinement_residual
            aux.update(refinement_aux)
            if 'boundary_gate' in refinement_aux:
                boundary_gate = refinement_aux['boundary_gate']
                aux['fusion_gates'] = torch.cat([1.0 - boundary_gate, boundary_gate], dim=1)
                aux['fusion_gate_modalities'] = ('flat_residual', 'edge_residual')
        else:
            y_init = y_bicubic

        if self.refinement_only:
            return {**{'y_pred': y_init}, **aux}

        y_pred, diffusion_aux = self.diffuse(
            y_init.clone(),
            guide.clone(),
            guide_feats=guide_feats,
            K=torch.exp(self.logk),
            train=train,
        )
        aux.update(diffusion_aux)

        if self.refinement_net is not None:
            aux['y_refined'] = y_init

        # return {'y_pred': y_pred} | aux
        return {**{'y_pred': y_pred}, **aux}

    def extract_features(self, img, guide):
        if self.feature_extractor is None:
            return torch.cat([guide, img], 1)
        return self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1))

    def diffuse(self, img, guide, guide_feats=None, l=0.24, K=0.01, train=False):

        # Deep Learning version or RGB version to calucalte the coefficients
        if guide_feats is None:
            guide_feats = self.extract_features(img, guide)
        
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide_feats, K=K)

        # Iterations without gradient
        if self.Npre>0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                for t in range(Npre):                     
                    img = diffuse_step(cv, ch, img, l=l)

        # Iterations with gradient
        if self.Ntrain>0: 
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)

        return img, {"cv": cv, "ch": ch}


# @torch.jit.script
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

# @torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I
