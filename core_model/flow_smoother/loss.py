import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import torch.fft
from core_stable.flow_inpaint_model.gaussian import generate_gaussianmap

def compute_MM(grid, Height, Width):
    WW=torch.zeros((grid-1,grid-1,4))
    for i in range(0,(grid-1)):
        for k in range(0,(grid-1)):
            w1=((((grid-1)-k-0.5)*((grid-1)-i-0.5)))/float((grid)*(grid))
            w2=((((grid-1)-k-0.5)*(i+0.5)))/float((grid)*(grid))
            w3=(((k+0.5)*((grid-1)-i-0.5)))/float((grid)*(grid))
            w4=(((k+0.5)*(i+0.5)))/float((grid)*(grid))
            WW[i,k,0]=w1
            WW[i,k,1]=w2
            WW[i,k,2]=w3
            WW[i,k,3]=w4
    grid=grid-1
    W=torch.zeros((Height)*(Width),4)
    for i in range(0,(Height)):
        for k in range(0,(Width)):
            W[i*(Width)+k,0]=WW[i%grid,k%grid,0]
            W[i*(Width)+k,1]=WW[i%grid,k%grid,1]
            W[i*(Width)+k,2]=WW[i%grid,k%grid,2]
            W[i*(Width)+k,3]=WW[i%grid,k%grid,3]
    
    P1=torch.mm(W,torch.inverse(torch.mm(torch.transpose(W,0,1),W)))
    P2=torch.transpose(W,0,1)

    return P1.cuda(), P2.cuda()

class Loss:
    def __init__(self, config, output_channels):
        self.input_Width = config.input_Width
        self.input_Height = config.input_Height
        self.output_channels = output_channels
        weight = generate_gaussianmap(config.input_Height, config.input_Width, 3.0)
        self.weight = torch.from_numpy(weight).unsqueeze(0).cuda().repeat(self.output_channels,1,1).detach()

    def run(self, input, data, l, epoch):
        B, _, H, W = data["mask_unstable"].shape
        devia_gt = data["devia"]
        devia_gt[:, :, 0, :, :] = devia_gt[:, :, 0, :, :] / W
        devia_gt[:, :, 1, :, :] = devia_gt[:, :, 1, :, :] / H
        devia_gt = devia_gt.permute(0,1,4,2,3).reshape(B,-1,H,W)
        devia_gt = devia_gt[:, 2:(2+self.output_channels), ...]
        mask_unstable = data["mask_unstable"].unsqueeze(0).repeat(1,1,2,1,1)
        mask_unstable = mask_unstable.reshape(B,-1,H,W)[:, 2:(2+self.output_channels), ...]

        ### stage_1 loss ###
        loss_gaussian = 0
        loss_regulization = 0
        mask_ = self.weight.unsqueeze(0).repeat(B,1,1,1)

        for k in range(len(input)):
            input_ = input[k]
            loss_regulization += torch.sum(torch.abs(input_))
            ri = torch.fft.fftn(input_, dim=(-2,-1), norm="forward")
            ri = torch.abs(ri)
            ri_max = torch.max(torch.max(ri, dim=-1)[0], dim=-1)[0] + 1e-6
            ri = ri / ri_max.unsqueeze(-1).unsqueeze(-1)
            ri = torch.log(ri + 1e-6)
            
            op1 = torch.cat([ri[:,:,H//2:,:], ri[:,:,:H//2,:]],dim=-2)
            op2 = torch.cat([op1[:,:,:,W//2:], op1[:,:,:,:W//2]],dim=-1)
            loss_gaussian += torch.mean(mask_ * op2)


            # "plot"
            # fig, axis = plt.subplots(10, 2, figsize=(20, 30), squeeze=False)
            # for i in range(10):
            #     # axis[i][0].imshow(ri_real[0,i*2,:,:].detach().cpu().numpy())
            #     # axis[i][1].imshow(ri_real[0,i*2 + 1,:,:].detach().cpu().numpy())
            #     # axis[i][2].imshow(ri_imag[0,i*2,:,:].detach().cpu().numpy())
            #     # axis[i][3].imshow(ri_imag[0,i*2 + 1,:,:].detach().cpu().numpy())
            #     axis[i][0].imshow(op2[0,i*2,:,:].detach().cpu().numpy())
            #     axis[i][1].imshow(op2[0,i*2 + 1,:,:].detach().cpu().numpy())
            # fig.tight_layout()
            # fig.savefig("temp.jpg")
            # plt.close(fig)
        

        ### loss linear ###
        pen4 = 0
        for k in range(len(input)):
            linear_data = input[k]
            linear_data_low = F.interpolate(input=linear_data, size=(10,10), mode='bilinear', align_corners=False)
            linear_data_ori = F.interpolate(input=linear_data_low, size=(H,W), mode='bilinear', align_corners=False)
            pen4 = torch.mean(torch.abs(input_ - linear_data_ori))


        ### stage_2 loss ###
        l1 = l[0].reshape(B,-1,2,H,W)
        mask_loss = data["mask_final"][:,:-1,:,:]
        loss_stable_1 = torch.mean(torch.norm(l1, dim=2))
        

        l2 = l[1].reshape(B,-1,2,H,W)
        loss_stable_2 = torch.mean(torch.norm(l2, dim=2))

        flow_ori = data["flow_map"][:,:-1,...]
        loss_flow = torch.sum(torch.abs(l1) - torch.abs(flow_ori))

        


        if epoch < 0:
            loss_all = 3 * loss_gaussian +  loss_stable_1
        elif epoch < 20:
            loss_all = pen4 + 2.0 * loss_stable_1
        else:
            loss_all = 1.5 * pen4 + loss_stable_1 + loss_stable_2
        

        return loss_all, {"gaussian":loss_gaussian.item(), "l_1":loss_stable_1.item(), "l_2":loss_stable_2.item(), "l_r":loss_regulization.item(), "l_flow":loss_flow.item(), "pen4":pen4.item()}

