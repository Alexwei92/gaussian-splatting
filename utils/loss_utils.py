#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization_new._C import fusedssim, fusedssim_backward
except:
    pass

try:
    from pytorch3d.ops import knn_points
    # PYTORCH3D_AVAILABLE = True
    PYTORCH3D_AVAILABLE = False
except:
    PYTORCH3D_AVAILABLE = False

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def loss_cls_3d(features, predictions, k=5, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    
    device = features.device
    N, D = features.shape
    C = predictions.shape[1]
    
    # Conditionally downsample if points exceed max_points
    if N > max_points:
        indices = torch.randperm(features.size(0), device=device)[:max_points]
        features = features[indices]
        predictions = predictions[indices]
        N = max_points

    # Randomly sample points for which we'll compute the loss
    sample_indices = torch.randperm(features.size(0), device=device)[:sample_size]
    sample_features = features[sample_indices]  # (sample_size, D)
    sample_preds = predictions[sample_indices]  # (sample_size, C)

    if PYTORCH3D_AVAILABLE:
        # Prepare for knn_points: (batch_size, N, D)
        sample_features_batch = sample_features.unsqueeze(0)  # (1, sample_size, D)
        features_batch = features.unsqueeze(0)  # (1, N, D)
    
        # Perform KNN search
        knn = knn_points(sample_features_batch, features_batch, K=k)
        neighbor_indices = knn.idx[0]  # (sample_size, k)
        
    else:
        # Compute top-k nearest neighbors directly in PyTorch
        dists = torch.cdist(sample_features, features)  # Compute pairwise distances
        _, neighbor_indices = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices]  # (sample_size, k, C)

    # Compute KL divergence
    log_P = torch.log(sample_preds.clamp(min=1e-10, max=1.0)).unsqueeze(1)  # (sample_size, 1, C)
    log_Q = torch.log(neighbor_preds.clamp(min=1e-10, max=1.0))  # (sample_size, k, C)
    kl = sample_preds.unsqueeze(1) * (log_P - log_Q)  # (sample_size, k, C)
    loss = kl.sum(dim=-1).sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    normalized_loss = loss / C

    return normalized_loss