import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import torch.fft
import numpy as np
from core_stable.flow_inpaint_model.gaussian import generate_gaussianmap

def compute_MM(grid, Height, Width):
    WW=torch.zeros((grid-1,grid-1,4))
    for i in range(0,(grid-1)):
        for k in range(0,(grid-1)):
            w1=((((grid-1)-k-0.5)*((grid-1)-i-0.5)))/float((grid)*(grid))
            w2=((((grid-1)-k-0.5)*(i+0.5)))/float((grid)*(grid))
            w3=(((k+0.5)*((grid-1)-i-0.5)))/float((grid)*(grid))
            w4=(((k+0.5)*(i+0.5)))/float((grid)*(grid))
            #print(w1,w2,w3,w4)
            WW[i,k,0]=w1
            WW[i,k,1]=w2
            WW[i,k,2]=w3
            WW[i,k,3]=w4
    grid=grid-1;
    W=torch.zeros((Height)*(Width),4)
    for i in range(0,(Height)):
        for k in range(0,(Width)):
            W[i*(Width)+k,0]=WW[i%grid,k%grid,0];
            W[i*(Width)+k,1]=WW[i%grid,k%grid,1];
            W[i*(Width)+k,2]=WW[i%grid,k%grid,2];
            W[i*(Width)+k,3]=WW[i%grid,k%grid,3];
    
    P1=torch.mm(W,torch.inverse(torch.mm(torch.transpose(W,0,1),W)))
    P2=torch.transpose(W,0,1)

    return P1.cuda(), P2.cuda()

class Loss:
    def __init__(self, config, output_channels):
        self.input_Width = config.input_Width
        self.input_Height = config.input_Height
        self.output_channels = output_channels

        sobel_kernel = np.array([[-0.125, -0.125, -0.125], [-0.125, 1, -0.125], [-0.125, -0.125, -0.125]], dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 2, axis=1)
        self.weight = Variable(torch.from_numpy(sobel_kernel).cpu().cuda(), requires_grad=False)

    def functional_conv2d(self, input):
        edge_detect = F.conv2d(input, self.weight, padding=1)
        return edge_detect


    def run(self, pred_in, data, mask_in, epoch):
        B, _, H, W = data["flow_crop"].shape

        flow_large = data["flow_large"]
        flow_large_low = F.interpolate(input=flow_large, size=(H//4, W//4),mode='bilinear', align_corners=False)

        if type(mask_in) == list:
            mask = mask_in[0]
            mask_out = mask_in[1]
            mask_deiva = mask_in[2]
        else:
            mask = mask_in
        
        if type(pred_in) == list:
            pred0 = pred_in[0]
            pred1 = pred_in[1]
        else:
            pred0 = pred_in

        ### compute the gradiant of predict flow ###

        if type(pred_in) == list:
            loss_edge_1 = self.functional_conv2d(pred1)
            loss_edge_mask_1 = torch.sum(torch.abs(loss_edge_1 * mask_deiva)) / (torch.sum(mask_deiva) + 1)

        ### flow loss ###
        l2_loss_0 = torch.mean(torch.sqrt(torch.sum((flow_large_low - pred0) ** 2, dim=1)))
        if type(pred_in) == list:
            l2_loss_1 = torch.sqrt(torch.sum((flow_large - pred1) ** 2, dim=1))

        ### flow+mask loss ###
        if type(pred_in) == list:
            l2_loss_mask_1 = torch.mean(l2_loss_1)

        ### out loss ###
        if type(pred_in) == list:
            out_loss_mask_1 = torch.sum(l2_loss_1 * mask_out) / (torch.sum(mask_out) + 1)


        if epoch < 50:
            loss_all = 2 * l2_loss_0 + l2_loss_mask_1 + out_loss_mask_1
        else:
            loss_all = l2_loss_0 + l2_loss_mask_1 + out_loss_mask_1
        

        return loss_all, {"l2_loss_0":l2_loss_0.item(), "l2_loss_mask_1":l2_loss_mask_1.item(), "out_loss_mask_1":out_loss_mask_1.item()}

