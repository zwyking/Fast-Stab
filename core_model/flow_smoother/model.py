import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt

from core_stable.flow_inpaint_model.refine import refine_net
from core_stable.flow_inpaint_model.flow_refine import flow_refine
from third_party.GOCor.GOCor.optimizer_selection_functions import define_optimizer_global_corr
from utils_flow.flow_vis import flow_to_image


# def conv1_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv1_2(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv2_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv2_2(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv3_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv3_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv3_3(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv4_1(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     dilation=dilate,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv4_2(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     dilation=dilate,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv4_3(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
#     return nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     dilation=dilate,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv5_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
#     return  nn.Sequential(nn.ConvTranspose2d(
#         in_channels,
#         out_channels,
#         kernel_size=kernel,
#         stride=stride,
#         padding=padding),
#     nn.ReLU())

# def conv5_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return  nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv5_3(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return  nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv6_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
#     return  nn.Sequential(nn.ConvTranspose2d(
#         in_channels,
#         out_channels,
#         kernel_size=kernel,
#         stride=stride,
#         padding=padding),
#     nn.ReLU())

# def conv6_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return  nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1),
#     nn.ReLU())

# def conv7_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
#     return  nn.Sequential(nn.ConvTranspose2d(
#         in_channels,
#         out_channels,
#         kernel_size=kernel,
#         stride=stride,
#         padding=padding))

# def conv7_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
#     return  nn.Sequential(nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1))

# def conv7_3(in_channels,out_channels,kernel=1,stride=1,dilate=1,padding=0):
#     return  nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel,
#     stride=stride,
#     padding=padding,
#     bias=True,
#     groups=1)


def conv1_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv1_2(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv2_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv2_2(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv3_1(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv3_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv3_3(in_channels,out_channels,kernel=4,stride=2,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv4_1(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilate,
    bias=True,
    groups=1)

def conv4_2(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilate,
    bias=True,
    groups=1)

def conv4_3(in_channels,out_channels,kernel=3,stride=1,dilate=2,padding=2):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilate,
    bias=True,
    groups=1)

def conv5_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding)

def conv5_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv5_3(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv6_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding)

def conv6_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv7_1(in_channels,out_channels,kernel=4,stride=2,padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding)

def conv7_2(in_channels,out_channels,kernel=3,stride=1,dilate=1,padding=1):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)

def conv7_3(in_channels,out_channels,kernel=1,stride=1,dilate=1,padding=0):
    return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    bias=True,
    groups=1)


class mpinet(nn.Module):
    def __init__(self, in_channels=60, out_channels=38,
                 start_filts=64, flag=True):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.flag = flag
        self.conv11=conv1_1(in_channels,start_filts*2)
        self.conv12=conv1_2(start_filts*2,start_filts*4)
        self.conv21=conv2_1(start_filts*4,start_filts*4)
        self.conv22=conv2_2(start_filts*4,start_filts*8)
        self.conv31=conv3_1(start_filts*8,start_filts*8)
        self.conv32=conv3_2(start_filts*8,start_filts*8)
        self.conv33=conv3_3(start_filts*8,start_filts*16)
        self.conv41=conv4_1(start_filts*16,start_filts*16)
        self.conv42=conv4_2(start_filts*16,start_filts*16)
        self.conv43=conv4_3(start_filts*16,start_filts*16)

        self.conv51=conv5_1(start_filts*32,start_filts*8)
        self.conv52=conv5_2(start_filts*8,start_filts*8)
        self.conv53=conv5_3(start_filts*8,start_filts*8)
        self.conv61=conv6_1(start_filts*16,start_filts*4)
        self.conv62=conv6_2(start_filts*4,start_filts*4)
        self.conv71=conv7_1(start_filts*8,start_filts*2)
        self.conv72=conv7_2(start_filts*2,start_filts*2)
        self.conv73=conv7_3(start_filts*2,out_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x1):

        out1_1=self.conv11(x1)
        out1_2=self.conv12(out1_1)
        out2_1=self.conv21(out1_2)
        out2_2=self.conv22(out2_1)
        out3_1=self.conv31(out2_2)
        out3_2=self.conv32(out3_1)
        out3_3=self.conv33(out3_2)
        out4_1=self.conv41(out3_3)
        out4_2=self.conv42(out4_1)
        out4_3=self.conv43(out4_2)
        
        out5_1=self.conv51(torch.cat((out4_3,out3_3),1))
        out5_2=self.conv52(out5_1)
        out5_3=self.conv53(out5_2)
        out6_1=self.conv61(torch.cat((out5_3,out2_2),1))
        out6_2=self.conv62(out6_1)
        out7_1=self.conv71(torch.cat((out6_2,out1_2),1))
        out7_2=self.conv72(out7_1)
        out7_3=self.conv73(out7_2)
        
        return out7_3, out7_2


class vnet(nn.Module):
    def __init__(self, in_channels=60, out_channels=38,
                 start_filts=64, flag=True):
        super().__init__()
        self.flag = flag
        self.in_channels = in_channels
        self.mpinet_1 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)
        # self.mpinet_2 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)
        # self.mpinet_3 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)
        # self.refine_net_1 = refine_net(self.mpinet_2)
        # self.refine_net_2 = refine_net(self.mpinet_3)
        # self.refine_net_3 = refine_net()

        # self.flow_refine_1 = flow_refine(164, batch_norm=True)
        # self.flow_refine_2 = flow_refine(164, batch_norm=True)

        self.refine_net_1 = refine_net()



    
    def forward(self, input):
        if self.flag:
            B, _, H, W = input["mask_unstable"].shape
            flow_map = input["flow_map"]
            mask_unstable = input["mask_unstable"]
            pr = input["pr_list"]
            # mask = torch.logical_and(mask_unstable>0.5, pr>0.5)
            mask_flag = mask_unstable > 0.5
            mask = torch.zeros_like(mask_unstable, device=mask_unstable.device)
            mask[mask_flag] = 1.0
            input["mask_final"] = mask
            mask = mask.unsqueeze(2)

            "plot image"
            # fig, axis = plt.subplots(6, 5, figsize=(20, 20), squeeze=False)
            # xv, yv = np.meshgrid(np.linspace(-1, 1, 720), np.linspace(-1, 1, 480))
            # xv = np.expand_dims(xv, axis=2)
            # yv = np.expand_dims(yv, axis=2)
            # grid = np.expand_dims(np.concatenate((xv, yv), axis=2), axis=0)
            # grid_large = torch.from_numpy(np.repeat(grid, 1, axis=0)).float()
            # for i in range(6):
            #     image_s = input["stable_crop_unstable"][0, i + 1, ...].permute(1,2,0).detach().cpu().numpy().astype(np.int32)
            #     axis[i][0].imshow(image_s)
            #     image_t = input["stable_crop_unstable"][0, i, ...].permute(1,2,0).detach().cpu().numpy().astype(np.int32)
            #     axis[i][1].imshow(image_t)
            #     m = mask[0, i, 0, ...].detach().cpu().numpy().astype(np.int32)
            #     axis[i][3].imshow(m)
            #     flow_show = flow_map[0,i,...].permute(1,2,0).cpu().numpy()
            #     flow_show = flow_to_image(flow_show)
            #     axis[i][4].imshow(flow_show)
            #     image_s_t = torch.from_numpy(image_s).permute(2,0,1).unsqueeze(0).float()
            #     temp_img = F.grid_sample(image_s_t, grid_large + flow_map[0,i,...].cpu().permute(1,2,0).unsqueeze(0))
            #     temp_img = temp_img.data.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
            #     axis[i][2].imshow(temp_img)
            # fig.tight_layout()
            # fig.savefig("cut.jpg")
            # plt.close(fig)

            data_cat = torch.cat([flow_map, mask], dim=2).reshape(B,-1,H,W)
            x1 = data_cat[:, :self.in_channels, ...]
        else:
            x1 = input
        
        out1, _ = self.mpinet_1(x1)

        
        # out2, l_1, x2 = self.refine_net_1(x1, out1, input)
        # out3 = out1 + out2

        # out4, l_2, x4 = self.refine_net_2(x1, out3, input)
        # out5 = out3 + out4

        # l_3 = self.refine_net_3(x1, out5, input)
        
        # return [out1, out3, out5], [l_1, l_2, l_3]

        l_1, l_2 = self.refine_net_1(x1, out1, input)
        
        return [out1], [l_1, l_2]


