from random import randrange

import torch
from torch import nn
import segmentation_models_pytorch as smp

DEPTH_INPUT_CHANNELS = 1
FEATURE_DIM = 64

class GADBase(nn.Module):
    
    def __init__(
            self, feature_extractor='UNet',
            Npre=8000, Ntrain=1024,
            guide_channels=4,
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.guide_channels = guide_channels
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.Ntrain = 0
            self.register_buffer('logk', torch.log(torch.tensor(0.03)))

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            self.feature_extractor =  torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=1, mode='bicubic'),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=guide_channels + DEPTH_INPUT_CHANNELS),
                torch.nn.Identity())
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')
             

    def forward(self, sample, train=False):
        guide = sample['guide']

        y_pred, aux = self.diffuse(
            sample['y_bicubic'].clone(),
            guide.clone(),
            K=torch.exp(self.logk),
            train=train,
        )

        # return {'y_pred': y_pred} | aux
        return {**{'y_pred': y_pred}, **aux}


    def diffuse(self, img, guide, l=0.24, K=0.01, train=False):

        # Deep Learning version or RGB version to calucalte the coefficients
        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1) 
        else:
            guide_feats = self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True) ], 1))
        
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
