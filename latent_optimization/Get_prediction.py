import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from image_utils import *
import torch
import networks
from DegradProcess import batch_degrade, MotionDegrad
from torch import optim
import torch.nn.functional as F
import torch



def get_prediction_latent_optimization(seg_LR, model, z_dim, epochs, device):

    new_D = seg_LR.shape[2]
    MotionLayer = MotionDegrad(newD=new_D, mode='nearest')
    MotionLayer.to(device)

    # LATENT OPTIMISATION
    z0 = torch.zeros((1, z_dim)).to(device)
    seg_map = torch.argmax(seg_LR, axis=1)
    z_recall = z0.clone().detach().requires_grad_(True)

    # optimizer for z
    optimizer1 = optim.Adam([{'params': z_recall}], lr=0.2)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.5)
    # optimizer for motion
    optimizer2 = optim.Adam(MotionLayer.parameters(), lr=0.1)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.5)

    for k in (range(0, epochs)):
        # E - step, estimate motion
        optimizer2.zero_grad()
        recon_x = model.decode(z_recall)
        recon_x = MotionLayer(recon_x)
        loss = F.cross_entropy(recon_x, seg_map,reduction='mean')
        loss.backward()
        optimizer2.step()

        # M - step, estimate z
        optimizer1.zero_grad()
        recon_x = model.decode(z_recall)
        recon_x = MotionLayer(recon_x)
        loss = F.cross_entropy(recon_x, seg_map, reduction='mean')
        loss.backward()
        optimizer1.step()

        scheduler1.step()
        scheduler2.step()

    SR_data = onehot2label(model.decode(z_recall).squeeze().detach().cpu().numpy())
    return SR_data