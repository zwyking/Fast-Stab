import os
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


def get_bn_layer(output_sz):
    return torch.nn.BatchNorm2d(num_features=output_sz)

class GatedConv2d(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_c, out_c, k, p, s):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_c, out_c, k, s, p)
        self.mask_conv2d = torch.nn.Conv2d(in_c, out_c, k, s, p)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_c)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.gated(mask)
        return x


class GatedConv2d_ResNet_Block(torch.nn.Module):
    def __init__(self, in_c, in_o, downsample=None):
        super().__init__()
        bn_noise1 = get_bn_layer(output_sz=in_c)
        bn_noise2 = get_bn_layer(output_sz=in_o)

        conv_layer = GatedConv2d

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = torch.nn.Identity()

        self.ch_a = torch.nn.Sequential(
            bn_noise1,
            torch.nn.ReLU(),
            conv_aa,
            bn_noise2,
            torch.nn.ReLU(),
            conv_ab,
            norm_downsample,
        )

        if downsample or (in_c != in_o):
            self.ch_b = torch.nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = torch.nn.Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b


class GatedConv2d_Block(torch.nn.Module):
    def __init__(self, in_c, in_o, downsample=None):
        super().__init__()
        bn_noise1 = get_bn_layer(output_sz=in_o)

        conv_layer = GatedConv2d

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)

        if downsample == "Down":
            # norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            norm_downsample = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = torch.nn.Identity()

        # self.ch_a = torch.nn.Sequential(
        #     conv_aa,
        #     bn_noise1,
        #     torch.nn.ReLU(),
        #     norm_downsample
        # )
        self.ch_a = torch.nn.Sequential(
            conv_aa,
            torch.nn.ReLU(),
            norm_downsample
        )

    def forward(self, x):
        x_a = self.ch_a(x)

        return x_a



class mpinet(nn.Module):
    def __init__(self, in_channels=60, out_channels=38,
                 start_filts=64, flag=True):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.flag = flag
        
        # self.layer0 = GatedConv2d_ResNet_Block(self.in_channels, 16)
        # self.layer1 = GatedConv2d_ResNet_Block(16, 64, 'Down')
        # self.layer2 = GatedConv2d_ResNet_Block(64, 64, 'Down')
        # self.layer3 = GatedConv2d_ResNet_Block(64, 32)
        # self.layer4 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        # self.layer5 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        # self.layer6 = GatedConv2d_ResNet_Block(32, 32)
        # self.layer7 = GatedConv2d_ResNet_Block(32, 2)

        self.layer0 = GatedConv2d_Block(self.in_channels, 16)
        self.layer1 = GatedConv2d_Block(16, 64, 'Down')
        self.layer2 = GatedConv2d_Block(64, 64, 'Down')
        self.layer3 = GatedConv2d_Block(64, 32)
        self.layer4 = GatedConv2d_Block(32, 32, 'Up')
        self.layer5 = GatedConv2d_Block(32, 32, 'Up')
        self.layer6 = GatedConv2d_Block(32, 16)
        self.layer7 = torch.nn.Conv2d(16, 2, 3, 1, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x1):
        x_0 = self.layer0(x1)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)
        
        return x_7


class vnet(nn.Module):
    def __init__(self, in_channels=60, out_channels=38, batch_size=1, height=480, width=720,
                 start_filts=64, flag=True, save_img=None):
        super().__init__()
        self.plot = 0
        self.flag = flag
        self.in_channels = in_channels
        self.batch = batch_size
        self.H = height
        self.W = width
        self.save_img = save_img
        self.grid = self.generate_grid(self.batch, self.H, self.W)
        
        # self.mpinet_0 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)
        self.mpinet_1 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)

        # self.mpinet_2 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)
        # self.mpinet_3 = mpinet(in_channels=in_channels, out_channels=out_channels, flag=flag)
        # self.refine_net_1 = refine_net(self.mpinet_2)
        # self.refine_net_2 = refine_net(self.mpinet_3)
        # self.refine_net_3 = refine_net()

        # self.flow_refine_1 = flow_refine(164, batch_norm=True)
        # self.flow_refine_2 = flow_refine(164, batch_norm=True)

        self.refine_net_1 = refine_net()


    def generate_grid(self, B, H, W):
        xv, yv = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        xv = np.expand_dims(xv, axis=2)
        yv = np.expand_dims(yv, axis=2)
        grid = np.expand_dims(np.concatenate((xv, yv), axis=2), axis=0)
        grid_large = torch.from_numpy(np.repeat(grid, B, axis=0)).float().cuda()

        return grid_large

    
    def forward(self, input):
        if self.flag:
            B, _, H, W = input["flow_crop"].shape
            flow_crop = input["flow_crop"]
            flow_large = input["flow_large"]

            "mask_crop"
            image_s = input["stable_crop"][:, 0, ...]
            image_s_remap = F.grid_sample(image_s, self.grid + flow_crop.permute(0,2,3,1))
            mask_crop = (torch.sum(image_s_remap, dim=1) > 3.0).float()
            mask = F.max_pool2d(1-mask_crop.unsqueeze(1), kernel_size=(15, 15), padding=(7, 7), stride=(1,1)).squeeze(1)
            mask = ((1 - mask) > 0).float()

            "mask_large"
            image_s = input["stable_crop"][:, 0, ...]
            image_s_remap_large = F.grid_sample(image_s, self.grid + flow_large.permute(0,2,3,1))
            mask_large = (torch.sum(image_s_remap_large, dim=1) > 3.0).float()
            mask_large = F.max_pool2d(mask_large.unsqueeze(1), kernel_size=(11, 11), padding=(5, 5), stride=(1,1)).squeeze(1)
            mask_devia = ((mask_large - mask) > 0.0).float()
            mask_devia = F.max_pool2d(mask_devia.unsqueeze(1), kernel_size=(5, 5), padding=(2, 2), stride=(1,1)).squeeze(1)
            mask_out = 1 - mask
            
            # mask = input["mask"][:,-1,:,:]
            # pr = input["pr_crop"]
            # # mask = torch.logical_and(mask_unstable>0.5, pr>0.5)
            # mask_flag = mask > 0.5
            # mask = torch.zeros_like(mask, device=mask.device)
            # mask[mask_flag] = 1.0
            # mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=(31, 31), padding=(15, 15), stride=(1,1)).squeeze(1)

            data_cat = torch.cat([flow_crop * mask.unsqueeze(1), mask.unsqueeze(1)], dim=1).reshape(B,-1,H,W)
            x1 = data_cat
        else:
            x1 = input
        
        x1_low = F.interpolate(input=x1, size=(32, 64),mode='bilinear', align_corners=False)
        out0 = self.mpinet_1(x1_low)
        out1 = F.interpolate(input=out0, size=(480, 720),mode='bilinear', align_corners=False)

        # x2 = flow_crop * mask.unsqueeze(1) + out0_up * mask_out.unsqueeze(1)
        # x2 = torch.cat([flow_crop, mask.unsqueeze(1)], dim=1)
        # out1 = self.mpinet_1(x2)

        if self.plot > 10:
            self.plot = 0

            "plot image"
            fig, axis = plt.subplots(B, 11, figsize=(20, 15), squeeze=False)
            xv, yv = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
            xv = np.expand_dims(xv, axis=2)
            yv = np.expand_dims(yv, axis=2)
            grid = np.expand_dims(np.concatenate((xv, yv), axis=2), axis=0)
            grid_large = torch.from_numpy(np.repeat(grid, 1, axis=0)).float()
            for i in range(B):
                image_s = input["stable_crop"][i, 0, ...].permute(1,2,0).detach().cpu().numpy().astype(np.int32)
                axis[i][0].imshow(image_s)
                axis[i][0].set_title("source_crop")
                image_t = input["stable_crop"][i, -1, ...].permute(1,2,0).detach().cpu().numpy().astype(np.int32)
                axis[i][1].imshow(image_t)
                axis[i][1].set_title("target_crop")
                m = mask[i, ...].detach().cpu().numpy().astype(np.int32)
                axis[i][3].imshow(m)
                axis[i][3].set_title("mask")
                image_s_t = torch.from_numpy(image_s).permute(2,0,1).unsqueeze(0).float()
                temp_img = F.grid_sample(image_s_t, grid_large + flow_crop[i,...].cpu().permute(1,2,0).unsqueeze(0))
                temp_img = temp_img.data.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
                axis[i][2].imshow(temp_img)
                axis[i][2].set_title("remap")
                image_t = input["stable_large"][i, -1, ...].permute(1,2,0).detach().cpu().numpy().astype(np.int32)
                axis[i][4].imshow(image_t)
                axis[i][4].set_title("target_large")
                image_s_t = torch.from_numpy(image_s).permute(2,0,1).unsqueeze(0).float()
                temp_img = F.grid_sample(image_s_t, grid_large + flow_large[i,...].cpu().permute(1,2,0).unsqueeze(0))
                temp_img = temp_img.data.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
                axis[i][5].imshow(temp_img)
                axis[i][5].set_title("remap")

                image_s_t = torch.from_numpy(image_s).permute(2,0,1).unsqueeze(0).float()
                temp_img = F.grid_sample(image_s_t, grid_large + out1[i,...].cpu().permute(1,2,0).unsqueeze(0))
                temp_img = temp_img.data.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
                axis[i][6].imshow(temp_img)
                axis[i][6].set_title("remap")

                m = mask_devia[i, ...].detach().cpu().numpy().astype(np.int32)
                axis[i][7].imshow(m)
                axis[i][7].set_title("mask_devia")

                f_c = flow_crop[i, ...].permute(1,2,0).detach().cpu().numpy()
                f_c = flow_to_image(f_c)
                axis[i][8].imshow(f_c)
                axis[i][8].set_title("flow_crop")

                f_l = flow_large[i, ...].permute(1,2,0).detach().cpu().numpy()
                f_l = flow_to_image(f_l)
                axis[i][9].imshow(f_l)
                axis[i][9].set_title("flow_large")

                pre = out1[i, ...].permute(1,2,0).detach().cpu().numpy()
                pre = flow_to_image(pre)
                axis[i][10].imshow(pre)
                axis[i][10].set_title("flow_pre")

            fig.tight_layout()
            fig.savefig(os.path.join(self.save_img, "cut.jpg"))
            plt.close(fig)
        else:
            self.plot = self.plot + 1
        
        return out1, [mask, mask_out, mask_devia]
