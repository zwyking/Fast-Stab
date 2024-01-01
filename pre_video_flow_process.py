import cv2
import argparse
import os
import glob
import torch
import numpy as np
import math
import pickle
import torch.nn.functional as F

from matplotlib import pyplot as plt
from models.PDCNet.PDCNet import PDCNet_vgg16
from models.PDCNet.mod_uncertainty import estimate_probability_of_confidence_interval_of_mixture_density, estimate_average_variance_of_mixture_density
from utils_data.geometric_transformation_sampling.homography_parameters_sampling import RandomHomography

def get_grid(H, ccrop):
    # top-left corner of the central crop
    X_CCROP, Y_CCROP = ccrop[0], ccrop[1]

    W_FULL, H_FULL = (1280, 720)

    # inverse homography matrix
    Hinv = np.linalg.inv(H)
    # Hscale = np.eye(3)
    # Hscale[0,0] = Hscale[1,1] = self.ratio_homography
    # Hinv = Hscale @ Hinv @ np.linalg.inv(Hscale)

    # estimate the grid for the whole image
    X, Y = np.meshgrid(np.linspace(0, W_FULL - 1, W_FULL),
                        np.linspace(0, H_FULL - 1, H_FULL))
    X_, Y_ = X, Y
    X, Y = X.flatten(), Y.flatten()

    # create matrix representation
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

    # multiply Hinv to XYhom to find the warped grid
    XYwarpHom = np.dot(Hinv, XYhom)

    # vector representation
    XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
    YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
    ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

    X_grid_pivot = (XwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)
    Y_grid_pivot = (YwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)

    # normalize XwarpHom and YwarpHom and cast to [-1, 1] range
    Xwarp = (2 * X_grid_pivot / (W_FULL - 1) - 1)
    Ywarp = (2 * Y_grid_pivot / (H_FULL - 1) - 1)
    grid_full = torch.stack([Xwarp, Ywarp], dim=-1)

    return grid_full.unsqueeze(0)

def random_crop(img, size, top_left):
    if not isinstance(size, tuple):
        size = (size, size)
        #load_size is W,H

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.int(np.ceil((size[0] - w) / 2))
    if h < size[1]:
        pad_h = np.int(np.ceil((size[1] - h) / 2))
    img_pad = cv2.copyMakeBorder(img,
                                pad_h,
                                pad_h,
                                pad_w,
                                pad_w,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    h, w = img_pad.shape[:2]

    x1 = top_left[0]
    y1 = top_left[1]

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1

def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    """
    Opencv remap
    map_x contains the index of the matching horizontal position of each pixel [i,j] while map_y contains the
    index of the matching vertical position of each pixel [i,j]

    All arrays are numpy
    args:
        image: image to remap, HxWxC
        disp_x: displacement in the horizontal direction to apply to each pixel. must be float32. HxW
        disp_y: displacement in the vertical direction to apply to each pixel. must be float32. HxW
        interpolation
        border_mode
    output:
        remapped image. HxWxC
    """
    h_scale, w_scale=disp_x.shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    return remapped_image

def pre_process_data(source_img, target_img, device, mean_vector=[0.485, 0.456, 0.406],
                            std_vector=[0.229, 0.224, 0.225], apply_flip=False):
    """

    Args:
        source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
        target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
        device:
        mean_vector:
        std_vector:
        apply_flip: bool, flip the target image in horizontal direction ?

    Returns:
        source_img_copy: source torch tensor, in range [0, 1], resized so that its size is dividable by 8
                         and normalized by imagenet weights
        target_img_copy: target torch tensor, in range [0, 1], resized so that its size is dividable by 8
                         and normalized by imagenet weights
        source_img_256: source torch tensor, in range [0, 1], resized to 256x256 and normalized by imagenet weights
        target_img_256: target torch tensor, in range [0, 1], resized to 256x256 and normalized by imagenet weights
        ratio_x: scaling ratio in horizontal dimension from source_img_copy and original (input) source_img
        ratio_y: scaling ratio in vertical dimension from source_img_copy and original (input) source_img
    """
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape

    # original resolution
    if h_scale < 256:
        int_preprocessed_height = 256
    else:
        int_preprocessed_height = int(math.floor(int(h_scale / 8.0) * 8.0))

    if w_scale < 256:
        int_preprocessed_width = 256
    else:
        int_preprocessed_width = int(math.floor(int(w_scale / 8.0) * 8.0))

    if apply_flip:
        # flip the target image horizontally
        target_img_original = target_img
        target_img = []
        for i in range(b):
            transformed_image = np.fliplr(target_img_original[i].cpu().permute(1, 2, 0).numpy())
            target_img.append(transformed_image)

        target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)

    source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                      size=(int_preprocessed_height, int_preprocessed_width),
                                                      mode='area')
    target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                      size=(int_preprocessed_height, int_preprocessed_width),
                                                      mode='area')
    source_img_copy = source_img_copy.div(255.0)
    target_img_copy = target_img_copy.div(255.0)
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device), size=(256, 256), mode='area')
    target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device), size=(256, 256), mode='area')
    source_img_256 = source_img_256.div(255.0)
    target_img_256 = target_img_256.div(255.0)
    source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

    ratio_x = float(w_scale) / float(int_preprocessed_width)
    ratio_y = float(h_scale) / float(int_preprocessed_height)
    return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), \
        target_img_256.to(device), ratio_x, ratio_y

def load_network(net, checkpoint_path=None, **kwargs):
    """Loads a network checkpoint file.
    args:
        net: network architecture
        checkpoint_path
    outputs:
        net: loaded network
    """

    if not os.path.isfile(checkpoint_path):
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_path))

    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    try:
        net.load_state_dict(checkpoint_dict['state_dict'])
    except:
        net.load_state_dict(checkpoint_dict)
    return net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess a video sequence of flow')
    parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/PDCNet_megadepth.pth.tar',
                        help='Directory containing the pre-trained-models.')
    parser.add_argument('--video_path', type=str,
                        help='Path to the video.', default="/data3/zhaoweiyue/data/stable_video_dataset/warp_video_datasets/videos")
    parser.add_argument('--save_path', type=str,
                        help='Path to the save.', default="/data3/zhaoweiyue/data/stable_video_dataset/warp_video_datasets/videos")
    parser.add_argument('--save_file_name', type=str,
                        help='Name of save file.', default="video_data_gap2_len5")
    args = parser.parse_args()

    ######### prepare the PDCNet #########
    estimate_uncertainty = True
    # for global gocor, we apply L_r and L_q within the optimizer module
    global_gocor_arguments = {'optim_iter': 3, 'steplength_reg': 0.1, 'train_label_map': False,
                                'apply_query_loss': True,
                                'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}

    # for global gocor, we apply L_r only
    local_gocor_arguments = {'optim_iter': 3, 'steplength_reg': 0.1}
    network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                            normalize='leakyrelu', same_local_corr_at_all_levels=True,
                            local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                            local_decoder_type='OpticalFlowEstimatorResidualConnection',
                            global_decoder_type='CMDTopResidualConnection',
                            corr_for_corr_uncertainty_decoder='corr',
                            give_layer_before_flow_to_uncertainty_decoder=True,
                            var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_fname = args.pre_trained_models_dir
    if not os.path.exists(checkpoint_fname):
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_fname))

    network = load_network(network, checkpoint_path=checkpoint_fname)
    network.eval()
    network = network.to(device)


    ######### read video data ##########
    out_file_name = os.path.join(args.save_path, args.save_file_name) + ".pkl"
    all_videos = sorted(glob.glob(os.path.join(args.video_path, '*_unstable.mp4')))
    save_data = {}
    for video_id in range(len(all_videos)):
        vc = cv2.VideoCapture(all_videos[video_id])
        video_number = all_videos[video_id].split("/")[-1].split('_')[0]
        transform_info = np.load(os.path.join(args.video_path, video_number + "_stable.npy"))
        rot = transform_info[0] / 50
        scale = (transform_info[1] - 1.0) / 50
        tx = transform_info[3] / 50
        ty = transform_info[4] / 50
        topleft_x = transform_info[5]
        topleft_y = transform_info[6]

        unstable_information = np.load(os.path.join(args.video_path, video_number + "_unstable.npy"))
        video_patch = {}
        rval = True
        patch_gap = 2
        sequence_len = 5
        frame_count = 0
        while rval :
            patch_id = frame_count // (patch_gap * sequence_len) * patch_gap + (frame_count % patch_gap)
            rval, frame = vc.read()
            if rval:
                if patch_id not in video_patch.keys():
                    video_patch[patch_id] = []
                if video_number + '_' + str(patch_id) not in save_data.keys():
                    save_data[video_number + '_' + str(patch_id)] = {}
                    save_data[video_number + '_' + str(patch_id)]["flow_map"] = []
                    save_data[video_number + '_' + str(patch_id)]["conf_map"] = []
                    save_data[video_number + '_' + str(patch_id)]["gt_trace"] = []
                    save_data[video_number + '_' + str(patch_id)]["gt_unstable_trace"] = []
                    save_data[video_number + '_' + str(patch_id)]["gt_unstable"] = []    

                video_patch[patch_id].append(frame)
                save_data[video_number + '_' + str(patch_id)]["gt_trace"].append(np.array([rot * frame_count, 1.0 + scale * frame_count, tx * frame_count, ty * frame_count]))
                save_data[video_number + '_' + str(patch_id)]["gt_unstable_trace"].append(unstable_information[frame_count, :])
            frame_count += 1

        with torch.no_grad():
            for id in range(len(video_patch)):
                input_images = torch.from_numpy(np.stack(video_patch[id])).permute(0,3,1,2)
                source_img = input_images[1:, ...]
                source_img_original = torch.cat([source_img, input_images[0, ...].unsqueeze(0)], dim=0)
                target_img_original = input_images

                w_scale = target_img_original.shape[3]
                h_scale = target_img_original.shape[2]
                output_shape = (h_scale, w_scale)

                source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y \
                    = pre_process_data(source_img_original, target_img_original, device = device)            
                output_256, output = network(target_img, source_img, target_img_256, source_img_256)
                
                flow_est_list = output['flow_estimates']
                flow_est = flow_est_list[-1]
                uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

                # get the flow field
                flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                        align_corners=False)
                flow_est[:, 0, :, :] *= ratio_x
                flow_est[:, 1, :, :] *= ratio_y

                

                # get the confidence value
                if isinstance(uncertainty_list[0], list):
                    # estimate multiple uncertainty maps per level
                    log_var_map = torch.nn.functional.interpolate(input=uncertainty_list[0], size=output_shape,
                                                                mode='bilinear', align_corners=False)
                    weight_map = torch.nn.functional.interpolate(input=uncertainty_list[1], size=output_shape,
                                                                mode='bilinear', align_corners=False)
                else:
                    log_var_map = torch.nn.functional.interpolate(input=uncertainty_list[0], size=output_shape,
                                                                mode='bilinear', align_corners=False)
                    weight_map = torch.nn.functional.interpolate(input=uncertainty_list[1], size=output_shape,
                                                                mode='bilinear', align_corners=False)
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map,
                                                                                    R=1.0)
                variance = estimate_average_variance_of_mixture_density(weight_map, log_var_map)

                W = flow_est.shape[3]
                H = flow_est.shape[2]
                B = flow_est.shape[0]
                xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
                yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
                grid = torch.cat((xx, yy),1).float()
                grid = grid.to(flow_est.device) 
                flow_warp = grid + flow_est
                flow_x = flow_warp[:, 0, ...].unsqueeze(1)
                flow_y = flow_warp[:, 1, ...].unsqueeze(1)
                flow_x_flag = torch.logical_and(flow_x > 0, flow_x < W)
                flow_y_flag = torch.logical_and(flow_y > 0, flow_y < H)
                flow_flag = torch.logical_and(flow_x_flag, flow_y_flag)
                flow_flag = torch.logical_and(p_r > 0.5, flow_flag)
                p_r[flow_flag] = 1.0
                p_r[~flow_flag] = 0.0
                p_r = p_r.squeeze(1)

                ##### plot images #######
                # fig, axis = plt.subplots(5, 5, figsize=(20, 20), squeeze=False)
                weight_map_softmax = torch.nn.functional.softmax(weight_map.detach(), dim=1).permute(0,2,3,1)
                log_var_map = log_var_map.permute(0,2,3,1)
                source_img = source_img_original.permute(0,2,3,1).detach().cpu().numpy()
                target_img = target_img_original.permute(0,2,3,1).detach().cpu().numpy()
                flow_est = flow_est.permute(0,2,3,1).detach().cpu().numpy()
                save_data[video_number + '_' + str(id)]["flow_map"].append(flow_est)
                

                # homography transformation
                for k_id in range(len(save_data[video_number + '_' + str(id)]["gt_unstable_trace"]) - 1):                   
                    rot_frame = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][0] - save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][0] 
                    scale_frame = (1.0 + (save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][1] - 1.0) ) / (1.0 + (save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][1] - 1.0))

                    theta0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][0]
                    theta1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][0]
                    scale0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][1]
                    scale1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][1]
                    tx_0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][2]
                    ty_0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][3]
                    tx_1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][2]
                    ty_1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][3]
                    perspec_0_x = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][4]
                    perspec_0_y = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id][5]
                    perspec_1_x = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][4]
                    perspec_1_y = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][k_id+1][5]

                    homo_sampling_module = RandomHomography(p_flip=0.0, max_rotation=12.0, max_shear=0.0,
                                        max_scale=0.8, max_ar_factor=0.0,
                                        min_perspective=0.0, max_perspective=0.0,
                                        max_translation=100, pad_amount=0)
                    H_0 = homo_sampling_module._construct_t_mat((1248, 1664), False, theta0,
                                                        (0.0, 0.0), (scale0, scale0), tx=tx_0, ty=ty_0,
                                                        perspective_factor=(perspec_0_x, perspec_0_y))
                    H_1 = homo_sampling_module._construct_t_mat((1248, 1664), False, theta1,
                                                        (0.0, 0.0), (scale1, scale1), tx=tx_1, ty=ty_1,
                                                        perspective_factor=(perspec_1_x, perspec_1_y))
                    Hscale = np.eye(3)
                    Hscale[0,0] = Hscale[1,1] = 2.166667
                    H0_inv = Hscale @ np.linalg.inv(H_0) @ np.linalg.inv(Hscale)
                    p_center = np.array([[topleft_x + 640, topleft_y + 360, 1]]).T
                    P_ori = np.dot(H0_inv, p_center)
                    P_n = P_ori / P_ori[2]
                    H_1 =  Hscale @ H_1 @ np.linalg.inv(Hscale)
                    P_proj = np.dot(H_1, P_n)
                    P_proj = P_proj / P_proj[2]
                    tx_frame = P_proj[0] - p_center[0]
                    ty_frame = P_proj[1] - p_center[1]

                    H_temp = homo_sampling_module._construct_t_mat((720, 1280), False, rot_frame,
                                                        (0.0, 0.0), (scale_frame, scale_frame), tx=0.0, ty=0.0,
                                                        perspective_factor=(0.0, 0.0))
                    p_center = np.array([[640, 360, 1]]).T
                    P_new = np.dot(H_temp, p_center)
                    tx_frame_new = 640 + tx_frame - P_new[0]
                    ty_frame_new = 360 + ty_frame - P_new[1]
                    gt_unstable = np.array([rot_frame, scale_frame, tx_frame_new[0], ty_frame_new[0]])
                    save_data[video_number + '_' + str(id)]["gt_unstable"].append(gt_unstable)

                rot_frame = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][0] - save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][0] 
                scale_frame = (1.0 + (save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][1] - 1.0) ) / (1.0 + (save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][1] - 1.0))

                theta0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][0]
                theta1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][0]
                scale0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][1]
                scale1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][1]
                tx_0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][2]
                ty_0 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][3]
                tx_1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][2]
                ty_1 = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][3]
                perspec_0_x = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][4]
                perspec_0_y = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][4][5]
                perspec_1_x = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][4]
                perspec_1_y = save_data[video_number + '_' + str(id)]["gt_unstable_trace"][0][5]

                homo_sampling_module = RandomHomography(p_flip=0.0, max_rotation=12.0, max_shear=0.0,
                                    max_scale=0.8, max_ar_factor=0.0,
                                    min_perspective=0.0, max_perspective=0.0,
                                    max_translation=100, pad_amount=0)
                H_0 = homo_sampling_module._construct_t_mat((1248, 1664), False, theta0,
                                                    (0.0, 0.0), (scale0, scale0), tx=tx_0, ty=ty_0,
                                                    perspective_factor=(perspec_0_x, perspec_0_y))
                H_1 = homo_sampling_module._construct_t_mat((1248, 1664), False, theta1,
                                                    (0.0, 0.0), (scale1, scale1), tx=tx_1, ty=ty_1,
                                                    perspective_factor=(perspec_1_x, perspec_1_y))
                Hscale = np.eye(3)
                Hscale[0,0] = Hscale[1,1] = 2.166667
                H0_inv = Hscale @ np.linalg.inv(H_0) @ np.linalg.inv(Hscale)
                p_center = np.array([[topleft_x + 640, topleft_y + 360, 1]]).T
                P_ori = np.dot(H0_inv, p_center)
                P_n = P_ori / P_ori[2]
                H_1 =  Hscale @ H_1 @ np.linalg.inv(Hscale)
                P_proj = np.dot(H_1, P_n)
                P_proj = P_proj / P_proj[2]
                tx_frame = P_proj[0] - p_center[0]
                ty_frame = P_proj[1] - p_center[1]

                H_temp = homo_sampling_module._construct_t_mat((720, 1280), False, rot_frame,
                                                    (0.0, 0.0),  (scale_frame, scale_frame), tx=0.0, ty=0.0,
                                                    perspective_factor=(0.0, 0.0))
                p_center = np.array([[640, 360, 1]]).T
                P_new = np.dot(H_temp, p_center)
                tx_frame_new = 640 + tx_frame - P_new[0]
                ty_frame_new = 360 + ty_frame - P_new[1]
                gt_unstable = np.array([rot_frame, scale_frame, tx_frame_new[0], ty_frame_new[0]])
                save_data[video_number + '_' + str(id)]["gt_unstable"].append(gt_unstable)
                

                ####### plot crop image #########
                H_temp[0][2] = H_temp[0][2] + tx_frame_new
                H_temp[1][2] = H_temp[1][2] + ty_frame_new
                grid_fullstable = get_grid(H_temp, ccrop=(0, 0))

                img_src_orig = source_img[3, ...]
                img_src_orig = torch.Tensor(img_src_orig.astype(np.float32))
                img_src_orig = img_src_orig.permute(2, 0, 1)
                if float(torch.__version__[:3]) >= 1.3:
                    img_orig_target_vrbl_stable = F.grid_sample(img_src_orig.unsqueeze(0),
                                                            grid_fullstable, align_corners=True)
                else:
                    img_orig_target_vrbl_stable = F.grid_sample(img_src_orig.unsqueeze(0),
                                                            grid_fullstable)
                img_orig_target_vrbl_stable = \
                    img_orig_target_vrbl_stable.squeeze().permute(1, 2, 0)
                img_orig_target_vrbl_stable = img_orig_target_vrbl_stable.numpy()
                img_target_crop, _, _ = random_crop(img_orig_target_vrbl_stable, (1280, 720), (0, 0))
                img_target_crop = img_target_crop.astype(np.uint8)
                
                
                p_r = p_r.squeeze().detach().cpu().numpy()
                p_r_save = p_r.copy()
                for idx in range(p_r.shape[0] - 1, 0, -1):
                    pr_fron = p_r[idx]
                    flow = flow_est[idx - 1, ...]
                    warp_pr_fron = remap_using_flow_fields(pr_fron, flow[...,0], flow[...,1]) > 0.4
                    pr_back = p_r[idx - 1] > 0.4
                    pr_new = np.logical_and(pr_back, warp_pr_fron)
                    p_r_save[idx - 1] = pr_new
                    p_r[idx - 1] = warp_pr_fron

                save_data[video_number + '_' + str(id)]["conf_map"].append(p_r_save)

    with open(out_file_name, "wb") as ofp:
        pickle.dump(save_data, ofp)
