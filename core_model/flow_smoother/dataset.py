import random
import numpy as np
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import glob
import skvideo.io

class VideoDataset(data.Dataset):
    def __init__(self, data_path, video_frames) -> None:
        super().__init__()
        self.data_path = data_path
        self.video_frames = video_frames
        self.var_name_list = ["devia", "mask", "stable_crop", "stable_crop_unstable"]
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
                out_data[var] = vc.astype(np.float32)
        out_data["mask"] = out_data["mask"].astype(np.float32)

        start_frames = random.randint(0, out_data["stable_crop"].shape[0] - self.video_frames - 1)
        for var in self.var_name_list:
            out_data[var] = out_data[var][start_frames:start_frames+self.video_frames, ...]
        
        "保证第一帧和最后一帧是稳定的"
        out_data["stable_crop_unstable"][0, ...] = out_data["stable_crop"][0, ...]
        out_data["stable_crop_unstable"][-1, ...] = out_data["stable_crop"][-1, ...]

        return out_data

    def __len__(self):
        _len = len(self.data)
        return _len - 1