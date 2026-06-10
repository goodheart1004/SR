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

    def __init__(self, in_channels, channels=FEATURE_DIM, num_blocks=4):
        super().__init__()
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
        self.residual_head = nn.Conv2d(channels, DEPTH_INPUT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, features, initial_dsm):
        x = self.in_projection(features)
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        if x.shape[-2:] != initial_dsm.shape[-2:]:
            x = F.interpolate(x, size=initial_dsm.shape[-2:], mode='bilinear', align_corners=False)
        residual = self.residual_head(x)
        return initial_dsm + residual, residual


class GADBase(nn.Module):
    
    def __init__(
            self, feature_extractor='UNet',
            Npre=8000, Ntrain=1024,
            guide_channels=4,
            use_refinement_net=True,
            refinement_channels=FEATURE_DIM,
            refinement_blocks=4,
            refinement_only=False,
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.guide_channels = guide_channels
        self.refinement_only = refinement_only
        self.refinement_net = None
 
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

        if use_refinement_net:
            self.refinement_net = LocalRefinementNet(
                in_channels=feature_channels,
                channels=refinement_channels,
                num_blocks=refinement_blocks,
            )
             

    def forward(self, sample, train=False):
        guide = sample['guide']
        y_bicubic = sample['y_bicubic'].clone()
        guide_feats = self.extract_features(y_bicubic, guide.clone())

        aux = {}
        if self.refinement_net is not None:
            y_init, refinement_residual = self.refinement_net(guide_feats, y_bicubic)
            aux['refinement_residual'] = refinement_residual
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
