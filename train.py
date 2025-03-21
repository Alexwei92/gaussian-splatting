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

import os
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import datetime
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr, objects_to_rgb
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from utils.wandb_utils import WandbLogger
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization_new import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def get_memory_usage_recursive(obj):
    cpu_memory = 0
    gpu_memory = 0

    if isinstance(obj, torch.Tensor):  # Directly a tensor
        memory = obj.element_size() * obj.numel()
        if obj.is_cuda:
            gpu_memory += memory
        else:
            cpu_memory += memory
    elif isinstance(obj, list) or isinstance(obj, dict):  # Collections
        items = obj if isinstance(obj, list) else obj.values()
        for item in items:
            sub_cpu, sub_gpu = get_memory_usage_recursive(item)
            cpu_memory += sub_cpu
            gpu_memory += sub_gpu
    elif hasattr(obj, "__dict__"):  # Check nested objects (class instances)
        for _, attr in vars(obj).items():
            sub_cpu, sub_gpu = get_memory_usage_recursive(attr)
            cpu_memory += sub_cpu
            gpu_memory += sub_gpu

    return cpu_memory / 1024**2, gpu_memory / 1024**2

def training(dataset, opt, pipe, args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    logger = prepare_output_and_logger(dataset, opt, args.use_wandb)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_objects_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                    net_image = render_pkg["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                    # logits = gaussians.classifier(render_pkg["objects"])
                    # net_image = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
                    # net_image = torch.argmax(render_pkg["objects"], dim=0).cpu().numpy().astype(np.uint8)
                    # net_image = objects_to_rgb(net_image)
                    # net_image_bytes = memoryview(net_image)
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        Ll1depth = None
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth    
            
        # Objects Classification loss
        objects_cls_loss = None
        if viewpoint_cam.objects is not None:  
            gt_objects = viewpoint_cam.objects.cuda().long()
            logits = gaussians.classifier(render_pkg["objects"])
            objects_cls_loss = gaussians.classifier_criternion(logits.unsqueeze(0), gt_objects).squeeze(0).mean()
            # objects_cls_loss = gaussians.classifier_criternion(render_pkg["objects"].unsqueeze(0), gt_objects).squeeze(0).mean()
            objects_cls_loss /= torch.log(torch.tensor(opt.num_classes))
            loss += objects_cls_loss
            
        # 3D Objects regularization
        objects_reg3d_loss = None
        # if iteration % opt.reg3d_interval == 0:
        #     logits3d = gaussians.classifier(gaussians._objects_dc.permute(2,0,1))
        #     # logits3d = gaussians._objects_dc.permute(2,0,1)
        #     prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1,0)
        #     objects_reg3d_loss = opt.reg3d_lambda * loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_max_points, opt.reg3d_sample_size)
        #     loss += objects_reg3d_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth.item() + 0.6 * ema_Ll1depth_for_log if Ll1depth else 0.0
            ema_objects_loss_for_log = 0.4 * objects_cls_loss.item() + 0.6 * ema_objects_loss_for_log if objects_cls_loss else 0.0

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Objects Loss": f"{ema_objects_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            loss_dict = {
                "l1_loss": Ll1.item(),
                "ssim_loss": ssim_value.item(),
                "l1_depth_loss": Ll1depth.item() if Ll1depth else None,
                "objects_cls_loss": objects_cls_loss.item() if objects_cls_loss else None,
                "objects_reg3d_loss": objects_reg3d_loss.item() if objects_reg3d_loss else None,
                "total_loss": loss.item(),
            }
            
            training_report(logger, iteration, loss_dict, iter_start.elapsed_time(iter_end), args.test_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
                gaussians.classifier_optimizer.step()
                gaussians.classifier_optimizer.zero_grad(set_to_none=True)

            # if (iteration in args.save_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, opt, use_wandb):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        # args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join(args.source_path, "output", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Wandb logger
    if use_wandb:
        if WANDB_FOUND:
            logger = WandbLogger(
                project="gaussian_splatting",
                config=opt,
                group=(args.source_path).split("/")[-1],
                name=(args.model_path).split("/")[-1],
            )
            return logger
        else:
            print(f"Wandb not available")
    
    # Create Tensorboard writer
    logger = None
    if TENSORBOARD_FOUND:
        logger = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    return logger

def training_report(logger, iteration, loss_dict, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if logger:
        for loss_name, loss_value in loss_dict.items():
            if loss_value is not None:
                logger.add_scalar(f'training/{loss_name}', loss_value, iteration)
        logger.add_scalar('training/iter_time', elapsed, iteration)
        logger.add_scalar('training/scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in [0]]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if viewpoint.depth_reliable:
                        gt_invdepth = viewpoint.invdepthmap.to("cuda")
                        invdepth = render_pkg["depth"]
                    
                    if viewpoint.objects is not None:
                        gt_objects = viewpoint.objects.cpu().numpy().astype(np.uint8)
                        gt_objects = objects_to_rgb(gt_objects)
                        logits = scene.gaussians.classifier(render_pkg["objects"])
                        pred_objects = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
                        # pred_objects = torch.argmax(render_pkg["objects"], dim=0).cpu().numpy().astype(np.uint8)
                        pred_objects = objects_to_rgb(pred_objects)
                    
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if logger and (idx < 5):
                        logger.add_image("validation/render/" + config['name'] + "_view_{}".format(viewpoint.image_name), image, global_step=iteration)
                        if viewpoint.depth_reliable:
                            logger.add_image("validation/render/" + config['name'] + "_view_depth_{}".format(viewpoint.image_name), invdepth, global_step=iteration)
                        if viewpoint.objects is not None:
                            logger.add_image("validation/render/" + config['name'] + "_view_objects_{}".format(viewpoint.image_name), pred_objects, global_step=iteration)
                        if iteration == testing_iterations[0]:
                            logger.add_image("validation/ground_truth/" + config['name'] + "_view_{}".format(viewpoint.image_name), gt_image, global_step=iteration)
                            if viewpoint.depth_reliable:
                                logger.add_image("validation/ground_truth/" + config['name'] + "_view_depth_{}".format(viewpoint.image_name), gt_invdepth, global_step=iteration)
                            if viewpoint.objects is not None:
                                logger.add_image("validation/ground_truth/" + config['name'] + "_view_objects_{}".format(viewpoint.image_name), gt_objects, global_step=iteration)
                        
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if logger:
                    logger.add_scalar("validation/" + config['name'] + '_l1_loss', l1_test, iteration)
                    logger.add_scalar("validation/" + config['name'] + '_psnr', psnr_test, iteration)

        if logger:
            logger.add_histogram("validation/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            # logger.add_scalar('validation/scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_wandb", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    args.test_iterations = [3_000, 5_000, 7_000, 10_000]
    args.save_iterations = [3_000, 5_000, 7_000, 10_000]
    args.optimizer_type = "sparse_adam"
    # args.depths = os.path.join("../", "depth")
    # args.objects = os.path.join("../", "segmentation")
    args.use_wandb = True
    
    args.num_classes = 140
    args.reg3d_lambda = 2.0
    args.reg3d_interval = 4
    args.reg3d_k = 5
    args.reg3d_max_points = 200000
    args.reg3d_sample_size = 1000
    
    # for key, value in args.__dict__.items():
    #     print(f"{key}: {value}")
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
