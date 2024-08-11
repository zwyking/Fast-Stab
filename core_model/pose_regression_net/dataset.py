import numpy as np
import os
import pickle
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

class VideoDataset(data.Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.var_name_list = ["flow_map", "conf_map", "gt_trace", "gt_unstable_trace", "gt_unstable"]
        self.data = self.load_data()
        self.keys = list(self.data)
    
    def load_data(self):
        print("Loading data")
        infile_name = os.path.join(self.data_path, "video_data.pkl")
        with open(infile_name, "rb") as ifp:
            video_data_1 = pickle.load(ifp)

        return video_data_1

    def __getitem__(self, index):
        key = self.keys[index]
        data = self.data[key]
        
        out_data = {}
        for var in self.var_name_list:
            if var == "gt_trace" or var == "gt_unstable_trace" or var == "gt_unstable":
                gt_trace = data[var]
                out_data[var] = np.stack(gt_trace)
            else:
                out_data[var] = data[var][0]

        return out_data

    def __len__(self):
        _len = len(self.keys)
        return _len - 1