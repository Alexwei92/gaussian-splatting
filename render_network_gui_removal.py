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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import numpy as np
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import objects_to_rgb
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, network_gui
try:
    from diff_gaussian_rasterization_new import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
    
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay


# def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         results = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
#         rendering = results["render"]
#         gt = view.original_image[0:3, :, :]
#         invdepthmap = view.invdepthmap

#         if args.train_test_exp:
#             rendering = rendering[..., rendering.shape[-1] // 2:]
#             gt = gt[..., gt.shape[-1] // 2:]

#         # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
#         rendering_cpu = rendering.cpu().numpy()
#         rendering_cpu = (rendering_cpu.transpose(1, 2, 0) * 255).astype('uint8')
        
#         gt_cpu = gt.cpu().numpy()
#         gt_cpu = (gt_cpu.transpose(1, 2, 0) * 255).astype('uint8')
        
#         depth_cpu = results['depth'].squeeze(0).cpu().numpy()
        
#         invdepthmap_cpu = invdepthmap.squeeze(0).cpu().numpy()
        
#         plt.subplot(1, 4, 1)
#         plt.imshow(rendering_cpu)
#         plt.subplot(1, 4, 2)
#         plt.imshow(gt_cpu)
#         plt.subplot(1, 4, 3)
#         plt.imshow(depth_cpu, cmap='gray')
#         plt.subplot(1, 4, 4)
#         plt.imshow(invdepthmap_cpu, cmap='gray')
#         plt.show()
#         break


def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask


def removal_setup(opt, gaussians, selected_obj_ids, removal_thresh):
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    with torch.no_grad():
        logits3d = gaussians.classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.squeeze()
        # try:
        #     mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        #     mask3d = torch.logical_or(mask3d,mask3d_convex)
        # except:
        #     pass

        mask3d = mask3d.float()[:,None,None]

    # fix some gaussians
    gaussians.removal_setup(opt,mask3d)
    
    print(f"Removed {mask3d.sum()} gaussians with id = {selected_obj_ids} from original gaussians")

    return gaussians

def render_gui(dataset : ModelParams, opt : OptimizationParams, pipeline : PipelineParams, iteration : int, render_type: str, select_obj_id: int, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        chkpnt_path = os.path.join(scene.model_path, "chkpnt" + str(scene.loaded_iter) + ".pth")
        (model_params, _) = torch.load(chkpnt_path)
        
        gaussians.training_setup(opt)
        gaussians.classifier.load_state_dict(model_params[12])
        # gaussians.restore(model_params, opt)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for i in range(7, 140):
        # for i in [0,3,4]:
        for i in range(7):
            gaussians = removal_setup(opt, gaussians, i, 0.3)

        while network_gui.conn == None:
            network_gui.try_connect()
            
            if network_gui.conn != None:
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipeline.convert_SHs_python, pipeline.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, gaussians, pipeline, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                            
                            if render_type == "rgb":
                                net_image = render_pkg["render"]
                                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                            
                            elif render_type == "depth":
                                net_image = render_pkg["depth"]
                                net_image = net_image.repeat(3, 1, 1)
                                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                            
                            elif render_type == "objects":
                                logits = gaussians.classifier(render_pkg["objects"])
                                net_image = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
                                # net_image = torch.argmax(render_pkg["objects"], dim=0).cpu().numpy().astype(np.uint8)
                                net_image = objects_to_rgb(net_image)
                                net_image_bytes = memoryview(net_image)
                        network_gui.send(net_image_bytes, dataset.source_path)
                    except Exception as e:
                        print(e)
                        network_gui.conn = None
                


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_type", default="rgb", type=str, choices=["rgb", "depth", "objects"])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.optimizer_type = "sparse_adam"
    args.depths = os.path.join("../", "depth")
    args.objects = os.path.join("../", "segmentation")
    
    args.num_classes = 140
    args.reg3d_lambda = 2.0
    args.reg3d_interval = 2
    args.reg3d_k = 5
    args.reg3d_max_points = 100000
    args.reg3d_sample_size = 500
    
    args.removal_thresh = 0.3
    args.select_obj_id = 6
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)

    render_gui(model.extract(args), opt.extract(args), pipeline.extract(args), args.iteration, args.render_type, args.select_obj_id, SPARSE_ADAM_AVAILABLE)