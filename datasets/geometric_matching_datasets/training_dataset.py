"""
Extracted from DGC-Net https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
"""
from os import path as osp
import os
from re import L
import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.semantic_matching_datasets.semantic_keypoints_datasets import random_crop
from ..util import center_crop
from utils_flow.flow_and_mapping_operations import unormalise_and_convert_mapping_to_flow
from datasets.util import define_mask_zero_borders
from utils_data.geometric_transformation_sampling.homography_parameters_sampling import RandomHomography
from datasets.object_augmented_dataset.base_video_dataset import BaseVideoDataset
from datasets.object_augmented_dataset.bounding_box_utils import masks_to_bboxes


class HomoAffTpsDataset(BaseVideoDataset):

    def __init__(self, image_path, csv_file, transforms, transforms_target=None, get_flow=False,
                 compute_mask_zero_borders=False, pyramid_param=[520], output_image_size=(520, 520),
                 original_DGC_transfo=True, foreground_image_dataset = None, foreground_transform=None,
                 number_of_objects=4, object_proba=0.8, save_dir=None):
        """
        Args:
            image_path: filepath to the dataset
            csv_file: csv file with ground-truth transformation parameters and name of original images
            transforms: image transformations for the source image (data preprocessing)
            transforms_target: image transformations for the target image (data preprocessing), if different than that of
            the source image
            get_flow: bool, whether to get flow or normalized mapping
            compute_mask_zero_borders: bool mask removing the black image borders
            pyramid_param: spatial resolution of the feature maps at each level
                of the feature pyramid (list)
            output_image_size: load_size (tuple) of the output images
        Output:
            if get_flow:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                flow_map: corresponding ground-truth flow field, shape 2xHxW
                correspondence_mask: mask of valid flow, shape HxW
            else:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                correspondence_map: correspondence_map, normalized to [-1,1], shape HxWx2,
                                    should correspond to correspondence_map_pyro[-1]
                correspondence_map_pyro: pixel correspondence map for each feature pyramid level
                mask_x: X component of the mask (valid/invalid correspondences)
                mask_y: Y component of the mask (valid/invalid correspondences)
                correspondence_mask: mask of valid flow, shape HxW, equal to mask_x and mask_y
        """
        super().__init__(foreground_image_dataset.get_name() + '_syn_vid_blend', foreground_image_dataset.root,
                         foreground_image_dataset.image_loader)

        self.foreground_image_dataset = foreground_image_dataset
        self.foreground_transform = foreground_transform
        self.number_of_objects = number_of_objects
        self.object_proba = object_proba


        self.img_path = image_path
        self.mask_zero_borders = compute_mask_zero_borders
        if not os.path.isdir(self.img_path):
            raise ValueError("The image path that you indicated does not exist!")

        self.transform_dict = {0: 'aff', 1: 'tps', 2: 'homo'}
        self.transforms_source = transforms
        if transforms_target is None:
            self.transforms_target = transforms
        else:
            self.transforms_target = transforms_target
        self.pyramid_param = pyramid_param
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            if len(self.df) == 0:
                raise ValueError("The csv file that you indicated is empty !")
        else:
            raise ValueError("The path to the csv file that you indicated does not exist !")
        self.get_flow = get_flow
        self.H_OUT, self.W_OUT = output_image_size

        # changed compared to version from DGC-Net
        self.ratio_cropping = 1.5
        # this is a scaling to apply to the homographies, usually applied to get 240x240 images
        if original_DGC_transfo:
            self.ratio_TPS = self.H_OUT / 240.0
            self.ratio_homography = self.H_OUT / 240.0

            self.H_AFF_TPS, self.W_AFF_TPS = (int(480*self.ratio_TPS), int(640*self.ratio_TPS))
            self.H_HOMO, self.W_HOMO = (int(576*self.ratio_homography), int(768*self.ratio_homography))
        else:
            self.ratio_TPS = 950/520
            self.ratio_homography = 950.0/520.0
            self.H_AFF_TPS, self.W_AFF_TPS = (int(480*self.ratio_TPS), int(640*self.ratio_TPS))
            self.H_HOMO, self.W_HOMO = (int(950), int(950))

        self.THETA_IDENTITY = \
            torch.Tensor(np.expand_dims(np.array([[1, 0, 0],
                                                  [0, 1, 0]]),
                                        0).astype(np.float32))
        self.gridGen = TpsGridGen(self.H_OUT, self.W_OUT)
        self.save_dir = save_dir

    def transform_image(self,
                        image,
                        out_h,
                        out_w,
                        padding_factor=1.0,
                        crop_factor=1.0,
                        theta=None):
        sampling_grid = self.generate_grid(out_h, out_w, theta)
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image

        if float(torch.__version__[:3]) >= 1.3:
            warped_image_batch = F.grid_sample(image, sampling_grid, align_corners=True)
        else:
            warped_image_batch = F.grid_sample(image, sampling_grid)

        return warped_image_batch

    def generate_grid(self, out_h, out_w, theta=None):
        out_size = torch.Size((1, 3, out_h, out_w))
        if theta is None:
            theta = self.THETA_IDENTITY
            theta = theta.expand(1, 2, 3).contiguous()
            return F.affine_grid(theta, out_size)
        elif (theta.shape[1] == 2):
            return F.affine_grid(theta, out_size)
        else:
            return self.gridGen(theta)

    def get_grid(self, H, ccrop):
        # top-left corner of the central crop
        X_CCROP, Y_CCROP = ccrop[0], ccrop[1]

        W_FULL, H_FULL = (self.W_HOMO, self.H_HOMO)
        W_SCALE, H_SCALE = (self.W_OUT, self.H_OUT)

        # inverse homography matrix
        Hinv = np.linalg.inv(H)
        Hscale = np.eye(3)
        Hscale[0,0] = Hscale[1,1] = self.ratio_homography
        Hinv = Hscale @ Hinv @ np.linalg.inv(Hscale)

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

        # getting the central patch from the pivot
        Xwarp_crop = X_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        Ywarp_crop = Y_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        X_crop = X_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]
        Y_crop = Y_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]

        # crop grid
        Xwarp_crop_range = \
            2 * (Xwarp_crop - X_crop.min()) / (X_crop.max() - X_crop.min()) - 1
        Ywarp_crop_range = \
            2 * (Ywarp_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min()) - 1
        grid_crop = torch.stack([Xwarp_crop_range,
                                 Ywarp_crop_range], dim=-1)
        return grid_full.unsqueeze(0), grid_crop.unsqueeze(0)

    def random_crop(self, img, size, top_left):
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

    def has_class_info(self):
        return self.foreground_image_dataset.has_class_info()

    def get_num_sequences(self):
        return self.foreground_image_dataset.get_num_images()
    
    def get_sequence_info(self, seq_id):
        image_info = self.foreground_image_dataset.get_image_info(seq_id)

        image_info = {k: v.unsqueeze(0) for k, v in image_info.items()}
        return image_info

    def get_class_name(self, seq_id):
        return self.foreground_image_dataset.get_class_name(seq_id)


    @staticmethod
    def symmetric_image_pad(image_batch, padding_factor):
        """
        Pad an input image mini-batch symmetrically
        Args:
            image_batch: an input image mini-batch to be pre-processed
            padding_factor: padding factor
        Output:
            image_batch: padded image mini-batch
        """
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))

        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left),
                                 image_batch,
                                 image_batch.index_select(3, idx_pad_right)),
                                3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top),
                                 image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)),
                                2)
        return image_batch

    @staticmethod
    def _paste_target(fg_image, fg_box, fg_mask, bg_image, paste_loc):
        fg_mask = fg_mask.view(fg_mask.shape[0], fg_mask.shape[1], 1)
        fg_box = fg_box.long().tolist()

        x1 = int(paste_loc[0] - 0.5 * fg_box[2])
        x2 = x1 + fg_box[2]

        y1 = int(paste_loc[1] - 0.5 * fg_box[3])
        y2 = y1 + fg_box[3]

        x1_pad = max(-x1, 0)
        y1_pad = max(-y1, 0)

        x2_pad = max(x2 - bg_image.shape[1], 0)
        y2_pad = max(y2 - bg_image.shape[0], 0)

        bg_mask = torch.zeros((bg_image.shape[0], bg_image.shape[1], 1), dtype=fg_mask.dtype,
                              device=fg_mask.device)

        if x1_pad >= fg_mask.shape[1] or x2_pad >= fg_mask.shape[1] or y1_pad >= fg_mask.shape[0] or y2_pad >= \
                fg_mask.shape[0]:
            return bg_image, bg_mask.squeeze(-1),

        fg_mask_patch = fg_mask[fg_box[1] + y1_pad:fg_box[1] + fg_box[3] - y2_pad,
                                fg_box[0] + x1_pad:fg_box[0] + fg_box[2] - x2_pad, :]

        fg_image_patch = fg_image[fg_box[1] + y1_pad:fg_box[1] + fg_box[3] - y2_pad,
                                  fg_box[0] + x1_pad:fg_box[0] + fg_box[2] - x2_pad, :]

        bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = \
            bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] * (1 - fg_mask_patch.numpy()) \
            + fg_mask_patch.numpy() * fg_image_patch

        bg_mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = fg_mask_patch

        return bg_image, bg_mask.squeeze(-1)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:
            if get_flow:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                flow_map: corresponding ground-truth flow field, shape 2xHxW
                correspondence_mask: mask of valid flow, shape HxW
            else:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                correspondence_map: correspondence_map, normalized to [-1,1], shape HxWx2,
                                    should correspond to correspondence_map_pyro[-1]
                correspondence_map_pyro: pixel correspondence map for each feature pyramid level
                mask_x: X component of the mask (valid/invalid correspondences)
                mask_y: Y component of the mask (valid/invalid correspondences)
                correspondence_mask: mask of valid flow, shape HxW, equal to mask_x and mask_y

        """
        data = self.df.iloc[idx]
        # get the transformation type flag
        transform_type = data['aff/tps/homo'].astype('uint8')
        source_img_name = osp.join(self.img_path, data.fname)
        if not os.path.exists(source_img_name):
            raise ValueError("The path to one of the original image {} does not exist, check your image path "
                             "and your csv file !".format(source_img_name))


        ############# video #############
        if "CityScape" in source_img_name:
            source_img = cv2.imread(source_img_name)
        else:
            source_img = cv2.imread(source_img_name)
            # source_img = cv2.cvtColor(cv2.imread(source_img_name),
            #                 cv2.COLOR_BGR2RGB)
        # cv2.imwrite("original_img.jpg", source_img)

        img_source = cv2.resize(source_img, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR) 
        img_src_orig = img_source.copy()
        # cv2.imwrite("original_img.jpg", img_src_orig)

        top_left_x, top_left_y = random.randint(150, 250), random.randint(150, 200)
        source_img, x, y = self.random_crop(img_src_orig, (1280, 720), (top_left_x, top_left_y))
        # cv2.imwrite("source_img.jpg", source_img)

        homo_sampling_module = RandomHomography(p_flip=0.0, max_rotation=3.0, max_shear=0.0,
                                                    max_scale=0.3, min_scale=-0.3, max_ar_factor=0.0,
                                                    min_perspective=0.0, max_perspective=0.0,
                                                    max_translation=100, pad_amount=0)
        do_flip, rot, shear_values, scales, perpective_factor, tx, ty = homo_sampling_module.roll()

        save_parameters = np.array([rot, scales[0], scales[1], tx, ty, top_left_x, top_left_y])
        np.save(os.path.join(self.save_dir, str(idx) + "_stable.npy"), save_parameters)

        ####### add moving object #########
        frames_ = 30
        
        object_dict = dict()
        if np.random.rand() < self.object_proba:
            number_of_objects = random.randint(1, self.number_of_objects)
            for ob_id in range(0, number_of_objects): 
                seq_id = random.randint(0, self.get_num_sequences() - 1)
                anno = self.get_sequence_info(seq_id)
                

                image_fg, fg_anno, fg_object_meta = self.foreground_image_dataset.get_image(seq_id, anno=anno)

                # get segmentation mask and bounding box of the foreground object to past
                mask_fg = fg_anno['mask'][0]  # float32
                bbox_fg = fg_anno['bbox'][0]

                # if the object is too big, reduce it:
                number_of_pixels = img_src_orig.shape[0]*img_src_orig.shape[1]
                if mask_fg.sum() > 0.5 * number_of_pixels:
                    scale = random.uniform(0.1, 0.4) * number_of_pixels / mask_fg.sum()
                    image_fg, bbox_fg, mask_fg, _ = self.foreground_transform.transform_with_specific_values(
                        image_fg, bbox_fg, mask_fg, do_flip=False, theta=0, shear_values=(0, 0),
                        scale_factors=(scale, scale), tx=0, ty=0)
                
                if mask_fg.sum() < 10000:
                    scale = random.uniform(0.7, 1.1) * 10000.0 / max(mask_fg.sum(), 2500)
                    image_fg, bbox_fg, mask_fg, _ = self.foreground_transform.transform_with_specific_values(
                        image_fg, bbox_fg, mask_fg, do_flip=False, theta=0, shear_values=(0, 0),
                        scale_factors=(scale, scale), tx=0, ty=0)
                


                # for the target image, put the object at random location on the target background
                if len(image_fg.shape) == 2:
                    print(idx)
                    continue
                loc_y_target = random.randint(150, img_src_orig.shape[0] - 150)
                loc_x_target = random.randint(200, img_src_orig.shape[1] - 200)
                target_image, target_mask_fg = self._paste_target(image_fg, masks_to_bboxes(mask_fg, fmt='t'),
                                                                mask_fg, img_src_orig,  # original target image
                                                                (loc_x_target, loc_y_target))
                object_dict[seq_id] = {}
                object_dict[seq_id]['image_fg'] = image_fg
                object_dict[seq_id]['bbox_fg'] = bbox_fg
                object_dict[seq_id]['mask_fg'] = mask_fg
                object_dict[seq_id]['loc_x_target'] = loc_x_target
                object_dict[seq_id]['loc_y_target'] = loc_y_target

                theta_factor = random.uniform(-14.0, 14.0)
                scale_factor = random.uniform(0.8, 1.2)

                if loc_x_target > img_src_orig.shape[1] // 2 + 300:
                    translation_x = random.uniform(-400, -200)
                elif loc_x_target < img_src_orig.shape[1] // 2 - 300:
                    translation_x = random.uniform(200, 400)
                else:
                    translation_x = np.sign(random.uniform(-1.0, 1.0)) * random.uniform(200, 400)
                
                if loc_y_target > img_src_orig.shape[0] // 2 + 300:
                    translation_y = random.uniform(-300, -150)
                elif loc_y_target < img_src_orig.shape[0] // 2 - 300:
                    translation_y = random.uniform(150, 300)
                else:
                    translation_y = np.sign(random.uniform(-1.0, 1.0)) * random.uniform(150, 300)

                object_dict[seq_id]['theta_factor'] = theta_factor
                object_dict[seq_id]['scale_factor'] = scale_factor
                object_dict[seq_id]['translation_x'] = translation_x
                object_dict[seq_id]['translation_y'] = translation_y

                pad = np.random.choice(range(3,10), frames_)
                pad = pad / np.sum(pad)
                
                temp_pad = 0.0
                for k in range(pad.shape[0]):
                    temp_pad += pad[k]
                    pad[k] = temp_pad

                object_dict[seq_id]['pad'] = pad
            
            # cv2.imwrite("target_image.jpg", target_image)
                
        ####################################

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        unstable_video_writer = cv2.VideoWriter(os.path.join(self.save_dir, str(idx) + "_unstable.mp4"), fourcc, 25, (1280, 720))
        unstable_video_writer.write(target_image)

        stable_video_writer = cv2.VideoWriter(os.path.join(self.save_dir, str(idx) + "_stable.mp4"), fourcc, 25, (1280, 720))
        stable_video_writer.write(target_image)

        unstable_parameters = []

        for frame_id in range(frames_):
            img_src_orig = img_source.copy()

            rot_pad = rot / frames_ 
            scale_pad = (scales[0] - 1.0) / frames_ 
            tx_pad = tx / frames_
            ty_pad = ty / frames_

            # smoothing tansformation
            H_stable = homo_sampling_module._construct_t_mat((self.H_HOMO, self.W_HOMO), do_flip, frame_id *rot_pad,
                                                    (0.0, 0.0), (1.0 + frame_id * scale_pad, 1.0 + frame_id * scale_pad), tx=frame_id * tx_pad, ty=frame_id * ty_pad,
                                                    perspective_factor=(0.0, 0.0))

            # adding camera flutter and perspective transformation
            do_unstable = random.random() < 0.7
            if do_unstable:
                d_r = random.uniform(-3.0, 3.0)
                d_tx = random.uniform(-12.0, 12.0)
                d_ty = random.uniform(-12.0, 12.0)
            else:
                d_r = random.uniform(-1.0, 1.0)
                d_tx = random.uniform(-3.0, 3.0)
                d_ty = random.uniform(-3.0, 3.0)

            do_perspective = random.random() < 0.5
            if do_perspective:
                perspective_x = random.uniform(0.00001, 0.00005)
                perspective_y = random.uniform(0.00001, 0.00005)
                perpective_factor = (perspective_x, perspective_y)

    
            rot_frame = frame_id * rot_pad + d_r
            scale_frame = (1.0 + frame_id * scale_pad, 1.0 + frame_id * scale_pad)
            tx_frame = frame_id * tx_pad + d_tx
            ty_frame = frame_id * ty_pad + d_ty
            unstable_parameters.append(np.array([rot_frame, scale_frame[0], tx_frame, ty_frame, perpective_factor[0], perpective_factor[1]]))
            H = homo_sampling_module._construct_t_mat((self.H_HOMO, self.W_HOMO), do_flip, rot_frame,
                                                    shear_values, scale_frame, tx=tx_frame, ty=ty_frame,
                                                    perspective_factor=perpective_factor)
        
            # Obtaining the full and crop grids out of H
            grid_fullstable, grid_crop = self.get_grid(H_stable, ccrop=(x, y))
            grid_full, grid_crop = self.get_grid(H, ccrop=(x, y))


            # warp the fullsize original source image        
            ######## add the moving object #######
            for obj_id in object_dict.keys():
                do_shear = random.random() < 0.3
                do_unstable = random.random() < 0.5
                if do_shear:
                    shear_x = random.uniform(0.01, 0.05)
                    shear_y = random.uniform(0.01, 0.05)
                    shear_factor = (shear_x, shear_y)
                    scale_obj = 1.0 + (object_dict[obj_id]['scale_factor'] - 1.0) * object_dict[obj_id]['pad'][frame_id]
                    scale_obj_factor = (scale_obj, scale_obj)
                    image_fg, bbox_fg, mask_fg, _ = self.foreground_transform.transform_with_specific_values(
                        object_dict[obj_id]["image_fg"].copy(),  object_dict[obj_id]['bbox_fg'],  object_dict[obj_id]["mask_fg"], 
                        do_flip=False, theta=object_dict[obj_id]['theta_factor'] * object_dict[obj_id]['pad'][frame_id], shear_values=shear_factor,
                        scale_factors=scale_obj_factor, tx=0, ty=0)
                    if do_unstable:
                        d_tx_obj = random.uniform(-10.0, 10.0)
                        d_ty_obj = random.uniform(-10.0, 10.0)
                    else:
                        d_tx_obj = 0.0
                        d_ty_obj = 0.0
                    target_image, target_mask_fg = self._paste_target(image_fg, masks_to_bboxes(mask_fg, fmt='t'),
                                                               mask_fg, img_src_orig,  # original target image
                                                                (object_dict[obj_id]["loc_x_target"] + object_dict[obj_id]["translation_x"] * object_dict[obj_id]['pad'][frame_id] + d_tx_obj, 
                                                                object_dict[obj_id]["loc_y_target"] + object_dict[obj_id]["translation_y"] * object_dict[obj_id]['pad'][frame_id] + d_ty_obj))
                else:
                    scale_obj = 1.0 + (object_dict[obj_id]['scale_factor'] - 1.0) * object_dict[obj_id]['pad'][frame_id]
                    scale_obj_factor = (scale_obj, scale_obj)
                    image_fg, bbox_fg, mask_fg, _ = self.foreground_transform.transform_with_specific_values(
                        object_dict[obj_id]["image_fg"].copy(),  object_dict[obj_id]['bbox_fg'],  object_dict[obj_id]["mask_fg"], 
                        do_flip=False, theta=object_dict[obj_id]['theta_factor'] * object_dict[obj_id]['pad'][frame_id], shear_values=(0, 0),
                        scale_factors=scale_obj_factor, tx=0, ty=0)
                    if do_unstable:
                        d_tx_obj = random.uniform(-10.0, 10.0)
                        d_ty_obj = random.uniform(-10.0, 10.0)
                    else:
                        d_tx_obj = 0.0
                        d_ty_obj = 0.0
                    target_image, target_mask_fg = self._paste_target(image_fg, masks_to_bboxes(mask_fg, fmt='t'),
                                                                mask_fg, img_src_orig,  # original target image
                                                                (object_dict[obj_id]["loc_x_target"] + object_dict[obj_id]["translation_x"] * object_dict[obj_id]['pad'][frame_id] + d_tx_obj, 
                                                                object_dict[obj_id]["loc_y_target"] + object_dict[obj_id]["translation_y"] * object_dict[obj_id]['pad'][frame_id] + d_ty_obj))
            
                # cv2.imwrite("target_image.jpg", target_image)

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

            # get the central crop of the target image
            img_target_crop_stable, _, _ = self.random_crop(img_orig_target_vrbl_stable, (1280, 720), (top_left_x, top_left_y))
            img_target_crop_stable = img_target_crop_stable.astype(np.uint8)

            stable_video_writer.write(img_target_crop_stable)
            


            target_image = torch.Tensor(target_image.astype(np.float32))
            target_image = target_image.permute(2, 0, 1)
            if float(torch.__version__[:3]) >= 1.3:
                img_orig_target_vrbl = F.grid_sample(target_image.unsqueeze(0),
                                                        grid_full, align_corners=True)
            else:
                img_orig_target_vrbl = F.grid_sample(target_image.unsqueeze(0),
                                                        grid_full)

            img_orig_target_vrbl = \
                img_orig_target_vrbl.squeeze().permute(1, 2, 0)
            img_orig_target_vrbl = img_orig_target_vrbl.numpy()
            # cv2.imwrite("img_orig_target_vrbl.jpg", img_orig_target_vrbl)

            # get the central crop of the target image
            img_target_crop, _, _ = self.random_crop(img_orig_target_vrbl, (1280, 720), (top_left_x, top_left_y))
            img_target_crop = img_target_crop.astype(np.uint8)
            # cv2.imwrite("img_target_crop.jpg", img_target_crop)

            unstable_video_writer.write(img_target_crop)
        
        unstable_video_writer.release()
        stable_video_writer.release()

        save_unstable_parameters = np.stack(unstable_parameters)
        np.save(os.path.join(self.save_dir, str(idx) + "_unstable.npy"), save_unstable_parameters)
        #################################
        
        return True


class TpsGridGen(nn.Module):

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=False):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda #default:False

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: load_size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # load_size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # load_size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: load_size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: load_size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: load_size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)
