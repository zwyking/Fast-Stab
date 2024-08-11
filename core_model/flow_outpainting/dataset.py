import random
from unicodedata import name
import numpy as np
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import glob
import skvideo.io

class FusionDataset(data.Dataset):
    def __init__(self, data_path, video_frames) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_frames = video_frames
        self.var_name_list = ["devia", "mask", "stable_crop", "stable_large"]
        self.data = self.load_data()
    
    def load_data(self):
        print("Loading data")
        data_items = sorted(glob.glob(os.path.join(self.data_path, 'stable_large_*.mp4')))
        return data_items

    def __getitem__(self, index):
        data_path = self.data[index]
        
        data_id = data_path.split('/')[-1].split('.')[0].split('_')[-1]
        out_data = {}
        for var in self.var_name_list:
            if var == "flow" or var == "mask" or var == "devia":
                r_data = np.load(os.path.join(self.data_path, var + '_' + data_id + '.npy'))
                out_data[var] = r_data
            else:
                v_path = os.path.join(self.data_path, var + '_' +  data_id + '.mp4')
                vc = skvideo.io.vread(v_path)
                # vc = np.flip(vc, axis=3)
                out_data[var] = vc.astype(np.float32)
        out_data["mask"] = out_data["mask"].astype(np.float32)

        start_frames = random.randint(0, out_data["stable_crop"].shape[0] - self.video_frames - 1)
        for var in self.var_name_list:
            out_data[var] = out_data[var][start_frames:start_frames+self.video_frames, ...]
        
        do_flip = True

        if do_flip:
            for var in self.var_name_list:
                out_data[var] = np.flip(out_data[var], 0).copy()
        
        return out_data

    def __len__(self):
        _len = len(self.data)
        return _len - 1


if __name__ == '__main__':
    train_dataset = FusionDataset("/data3/zhaoweiyue/data/stable_video_dataset/warp_video_datasets/stable_data", 8)

    for i in range(train_dataset.__len__()):
        temp_data = train_dataset.__getitem__(i)
        temp_data
        
        