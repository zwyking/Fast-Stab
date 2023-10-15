import os
from cv2 import FastFeatureDetector
import numpy as np
import argparse
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
env_path = os.path.join(os.path.dirname(__file__), '../')
if env_path not in sys.path:
    sys.path.append(env_path)
from utils_data.io import boolean_string
from datasets.geometric_matching_datasets.training_dataset import HomoAffTpsDataset
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from datasets.object_augmented_dataset.synthetic_object_augmentation_for_pairs_multiple_ob import RandomAffine
from datasets.object_augmented_dataset import MSCOCO

from utils_data.image_transforms import ArrayToTensor
from utils_data.io import writeFlow

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DGC-Net train script')
    parser.add_argument('--image_data_path', type=str, default='/data3/zhaoweiyue/data/dense_matching/DenseMatching/training_datasets', help='path to directory containing the original images.')
    parser.add_argument('--csv_path', type=str, default='assets/homo_aff_tps_test_DPED_CityScape_video.csv',
                        help='path to the CSV files')
    parser.add_argument('--save_dir', type=str, default='/data3/kennys/data/stable_video_dataset/warp_video_datasets', help='path directory to save the image pairs and corresponding ground-truth flows')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot as examples the first 4 pairs ? default is False')
    parser.add_argument('--seed', type=int, default=1981,
                        help='Pseudo-RNG seed')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    plot = args.plot
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_dir=os.path.join(save_dir, 'videos_new')

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # datasets
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    pyramid_param = [520]

    coco_path = "/data3/kennys/data/dense_matching/DenseMatching/coco_root" # you need add the coco path by yourself
    min_target_area = 1300
    coco_dataset_train = MSCOCO(root=coco_path, split='train', version='2017',
                                min_area=min_target_area)
    fg_tform = RandomAffine(p_flip=0.0, max_rotation=30.0,
                        max_shear=0, max_ar_factor=0.,
                        max_scale=0.3, pad_amount=0)

    # prepare dataset
    train_dataset = HomoAffTpsDataset(image_path=args.image_data_path,
                                      csv_file=args.csv_path,
                                      transforms=source_img_transforms,
                                      transforms_target=target_img_transforms,
                                      pyramid_param=pyramid_param,
                                      get_flow=True,
                                      output_image_size=(520, 520),
                                      foreground_image_dataset=coco_dataset_train,
                                      foreground_transform=fg_tform,
                                      number_of_objects=3, object_proba=1.0,
                                      save_dir=video_dir)
