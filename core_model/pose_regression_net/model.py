import torch
import math
from torch import nn
from torch.nn import functional as F, parameter
import numpy as np
import cv2

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, input):
        if self.training:
            return super().forward(input)
        else:
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum
            scaling = self.weight / (torch.sqrt(self.running_var + self.eps))
            output = input * scaling.view(-1,1,1) + (self.bias - self.running_mean * scaling).view(-1,1,1)
            return output

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            BatchNorm(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


class RegressMotivation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = conv(in_channel, 8, kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.conv2 = conv(8, 16, kernel_size=3, stride=2, padding=(0, 1), batch_norm=False)
        self.conv3 = conv(16, 32, kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.conv4 = conv(32, out_channel, kernel_size=3, stride=1, padding=1, batch_norm=False)

        self.multiple = nn.Sequential(
                                    nn.Conv1d(out_channel, 1, kernel_size=1, bias=True),
                                    nn.ReLU())

        self.regress = nn.Sequential(
                                    nn.Conv1d(out_channel, 32, kernel_size=1, bias=True),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv1d(32, 16, kernel_size=1, bias=True),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv1d(16, 4, kernel_size=1, bias=True))

    def forward(self, input):
        out_1 = self.conv4(self.conv3(self.conv2(self.conv1(input))))
        out_1 = out_1.reshape(out_1.shape[0], out_1.shape[1], -1)
        out_2 = self.multiple(out_1)
        out_T = out_1.permute(0, 2, 1)

        out_3 = torch.bmm(out_2, out_T)
        out_3 = out_3.permute(0, 2, 1)

        out = self.regress(out_3)

        return out 
        

class regress_coarse2fine(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()

        self._ksize = kernel_size
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(in_channels, 8, kernel_size=self._ksize, stride=1, padding=1, bias=True),
                                    nn.LeakyReLU(0.1),
                                    # BatchNorm(8),
                                    nn.Conv2d(8, 32, kernel_size=self._ksize, stride=1, padding=1, bias=True),
                                    nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=True),
                                    nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=True),
                                    nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(
                                    nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=True),
                                    nn.LeakyReLU(0.1))

        self.multiple = nn.Sequential(
                                    nn.Conv1d(32, 1, kernel_size=1, bias=True),
                                    nn.ReLU())
        self.regress = nn.Sequential(
                                    nn.Conv1d(32, 16, kernel_size=1, bias=True),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv1d(16, 4, kernel_size=1, bias=True))
    
    def forward(self, flow_r, conf_map_ori):
        x_0 = flow_r * conf_map_ori
        x_1 = self.conv1(x_0)
        x_2 = self.conv2(x_1)

        x_2_pool = F.max_pool2d(x_2, 4)
        conf_map_ori_up = conf_map_ori.repeat(1, x_2_pool.shape[1]//conf_map_ori.shape[1], 1, 1)
        conf_2_pool = F.max_pool2d(conf_map_ori_up, 4)
        x_3 = x_2_pool * conf_2_pool
        x_4 = self.conv3(x_3)

        x_4_pool = F.max_pool2d(x_4, 4)
        conf_4_pool = F.max_pool2d(conf_2_pool, 4)
        x_5 = x_4_pool * conf_4_pool
        x_6 = self.conv3(x_5)

        x_6_T = x_6.reshape(x_6.shape[0], x_6.shape[1], -1)
        x_7 = self.multiple(x_6_T)
        x_7_T = x_7.permute(0, 2, 1)
        out = torch.bmm(x_6_T, x_7_T)

        params = self.regress(out)

        return params



class StableNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.frames = config.frames
        self.Width = config.Width
        self.Height = config.Height
        self.original_Width = config.original_Width
        self.original_Height = config.original_Height

        self.regress_coarse2fine_1 = regress_coarse2fine(2, 3)
        self.regress_coarse2fine_2 = regress_coarse2fine(2, 3)
        self.regress_coarse2fine_3 = regress_coarse2fine(2, 3)
    
    def forward(self, data):
        flow = data["flow_map"]
        conf_map = data["conf_map"]

        flow_r = flow.reshape(flow.shape[0] * flow.shape[1], flow.shape[2], flow.shape[3], flow.shape[4])
        conf_map_ori = conf_map.reshape(-1, conf_map.shape[2], conf_map.shape[3])

        #### prepare data
        flow_r[:, 0, :, :] = flow_r[:, 0, :, :] / self.original_Width * self.Width
        flow_r[:, 1, :, :] = flow_r[:, 1, :, :] / self.original_Height * self.Height
        conf_map_ori[conf_map_ori < 0.5] = torch.tensor(0.0, device=flow_r.device)
        conf_map_ori[conf_map_ori >= 0.5] = torch.tensor(1.0, device=flow_r.device)
        conf_map_ori = conf_map_ori.unsqueeze(1).repeat(1,2,1,1)

        predict_regressions = self.regress_coarse2fine_1(flow_r, conf_map_ori)
        predict_regressions[:, 1, :] = 3.0 * torch.sigmoid(predict_regressions[:, 1, :])

        flow_refine_2, flow_2, conf_map_2 = self.refine_predict(predict_regressions, flow_r, conf_map_ori, self.Width, self.Height)
        predict_regressions_2 = self.regress_coarse2fine_2(flow_refine_2.permute(0,3,1,2), conf_map_ori)
        predict_regressions_2[:, 0, :] = predict_regressions[:, 0, :] + predict_regressions_2[:, 0, :]
        predict_regressions_2[:, 1, :] = predict_regressions[:, 1, :] * 3.0 * torch.sigmoid(predict_regressions_2[:, 1, :])
        predict_regressions_2[:, 2, :] = predict_regressions[:, 2, :] + predict_regressions_2[:, 2, :]
        predict_regressions_2[:, 3, :] = predict_regressions[:, 3, :] + predict_regressions_2[:, 3, :]

        flow_refine_3, flow_3, conf_map_3 = self.refine_predict(predict_regressions_2, flow_r, conf_map_ori, self.Width, self.Height)
        predict_regressions_3 = self.regress_coarse2fine_3(flow_refine_3.permute(0,3,1,2), conf_map_ori)
        predict_regressions_3[:, 0, :] = predict_regressions_2[:, 0, :] + predict_regressions_3[:, 0, :]
        predict_regressions_3[:, 1, :] = predict_regressions_2[:, 1, :] * 3.0 * torch.sigmoid(predict_regressions_3[:, 1, :])
        predict_regressions_3[:, 2, :] = predict_regressions_2[:, 2, :] + predict_regressions_3[:, 2, :]
        predict_regressions_3[:, 3, :] = predict_regressions_2[:, 3, :] + predict_regressions_3[:, 3, :]

        return {"predict_regressions":[predict_regressions, predict_regressions_2, predict_regressions_3],
                "flow_refine": [flow_refine_2, flow_refine_3, conf_map_ori]}
    
    def refine_predict(self, predict_regressions, flow_r, conf_map, W, H):
        B = flow_r.shape[0]
        theta = predict_regressions[:, 0, 0]
        scale = predict_regressions[:, 1, 0]
        dx = predict_regressions[:, 2, 0]
        dy = predict_regressions[:, 3, 0]

        #### generate grid coordinate ####
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()
        grid = grid.to(flow_r.device)
        
        grid_coord = torch.cat([grid, torch.ones(B,1,H,W).to(flow_r.device)], dim=1)
        grid_coord = grid_coord.reshape(B,3,-1)


        #### compute homography matrix ####
        angle = theta / 180.0 * math.pi
        t_rot = torch.eye(3, 3).unsqueeze(0).to(flow_r.device)
        t_rot = t_rot.repeat(B,1,1)
        t_rot[:, 0, 0] = torch.cos(angle)
        t_rot[:, 0, 1] = torch.sin(angle)
        t_rot[:, 0, 2] = (1-torch.cos(angle)) * W/2 - torch.sin(angle) * H/2
        t_rot[:, 1, 0] = -torch.sin(angle)
        t_rot[:, 1, 1] = torch.cos(angle)
        t_rot[:, 1, 2] = torch.sin(angle) * W/2 + (1 - torch.cos(angle)) * H/2

        t_scale = torch.eye(3, 3).unsqueeze(0).to(flow_r.device)
        t_scale = t_scale.repeat(B,1,1)
        t_scale[:, 0, 0] = scale
        t_scale[:, 1, 1] = scale
        t_scale[:, 0, 2] = (1.0 - scale) * 0.25 * W
        t_scale[:, 1, 2] = (1.0 - scale) * 0.25 * H
        t_mat = t_scale @ t_rot
        t_mat[:, 0, 2] = t_mat[:, 0, 2] + dx
        t_mat[:, 1, 2] = t_mat[:, 1, 2] + dy



        #### warp grid coordinates ####
        warp_coord = torch.bmm(t_mat, grid_coord)
        warp_coord = warp_coord.reshape(B, 3, H, W)
        flow_predict = warp_coord[:, :2, ...] - grid
        flow_refine = flow_r - flow_predict

        flow_refine[conf_map < 0.5] = torch.tensor(0.0, device=flow_r.device)

        warp_coord_copy = warp_coord
        warp_coord_copy = warp_coord_copy.long()
        warp_coord_copy = warp_coord_copy[:, :2, ...]
        warp_coord_copy[:, 0, ...][warp_coord_copy[:, 0, ...] > W - 1 ] = 0
        warp_coord_copy[:, 0, ...][warp_coord_copy[:, 0, ...] < 0 ] = 0
        warp_coord_copy[:, 1, ...][warp_coord_copy[:, 1, ...] > H - 1 ] = 0
        warp_coord_copy[:, 1, ...][warp_coord_copy[:, 1, ...] < 0 ] = 0

        flow_new = torch.zeros_like(flow_r).to(flow_r.device)
        flow_new = flow_new.permute(0,2,3,1)
        warp_coord_copy = warp_coord_copy.permute(0,2,3,1)
        flow_refine = flow_refine.permute(0,2,3,1)

        return flow_refine, flow_new, None
