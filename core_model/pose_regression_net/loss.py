import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class Loss:
    def __init__(self, config):
        self.Width = config.Width
        self.Height = config.Height

    def run(self, predict_input, input_data):
        gt_unstable_trace = input_data["gt_unstable"]
        theta = gt_unstable_trace[..., 0].reshape(-1)
        scale = gt_unstable_trace[..., 1].reshape(-1)
        d_x = (gt_unstable_trace[..., 2] / 1280 * 512).reshape(-1)
        d_y = (gt_unstable_trace[..., 3] / 720 * 256).reshape(-1)

        loss_flow_all = []

        for i in range(len(predict_input["flow_refine"]) - 1):
            conf_map = predict_input["flow_refine"][-1].permute(0,2,3,1)
            flow_devia = predict_input["flow_refine"][i]
            loss_flow = torch.mean(torch.abs(flow_devia[conf_map > 0.5]))
            loss_flow_all.append(loss_flow)
        loss_flow_o = torch.sum(torch.stack(loss_flow_all))    

        loss_theta_all = []  
        loss_scale_all = [] 
        loss_d_x_gt_all = [] 
        loss_d_y_gt_all = []  
        loss_transform_all = [] 

        for i in range(len(predict_input["predict_regressions"])):
            predict_params = predict_input["predict_regressions"][i].squeeze(-1)
            theta_pred = predict_params[..., 0]
            scale_pred = predict_params[..., 1]
            d_x_pred = predict_params[..., 2]
            d_y_pred = predict_params[..., 3]

            # compute the transformation loss
            W = 16
            H = 16
            B = theta.shape[0]
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy),1).float()
            
            vgrid0 = grid.clone()
            vgrid1 = grid.clone()
            vgrid0[:, 0, :, :] = 2.0 * vgrid0[:, 0, :, :].clone() / max(W-1, 1) - 1.0
            vgrid0[:, 1, :, :] = 2.0 * vgrid0[:, 1, :, :].clone() / max(H-1, 1) - 1.0
            vgrid1[:, 0, :, :] = 2.0 * vgrid1[:, 0, :, :].clone() / max(W-1, 1) - 1.0
            vgrid1[:, 1, :, :] = 2.0 * vgrid1[:, 1, :, :].clone() / max(H-1, 1) - 1.0
            vgrid0 = vgrid0.view(B, 2, -1).to(theta.device)
            vgrid1 = vgrid1.view(B, 2, -1).to(theta_pred.device)

            theta_pi = theta / 180.0 * math.pi
            transform_0 = torch.zeros((B, 2, 2))
            transform_0[:, 0, 0] = torch.cos(theta_pi)
            transform_0[:, 0, 1] = torch.sin(theta_pi)
            transform_0[:, 1, 0] = -torch.sin(theta_pi)
            transform_0[:, 1, 1] = torch.cos(theta_pi)
            transform_0 = transform_0.to(theta_pi.device)
            transform_0 = transform_0 * scale.view(-1,1,1)
            vgrid0 = torch.bmm(transform_0.float(), vgrid0)

            theta_pred_pi = theta_pred / 180.0 * math.pi
            transform_1 = torch.zeros((B, 2, 2))
            transform_1[:, 0, 0] = torch.cos(theta_pred_pi)
            transform_1[:, 0, 1] = torch.sin(theta_pred_pi)
            transform_1[:, 1, 0] = -torch.sin(theta_pred_pi)
            transform_1[:, 1, 1] = torch.cos(theta_pred_pi)
            transform_1 = transform_1.to(theta_pred_pi.device)
            transform_1 = transform_1 * scale_pred.view(-1,1,1)
            vgrid1 = torch.bmm(transform_1, vgrid1)

            loss_transform = torch.mean(torch.sum(torch.abs(vgrid0 - vgrid1), dim=1))
            loss_theta = torch.mean(torch.abs(theta_pred - theta))
            loss_scale = torch.mean(torch.abs((scale_pred - scale) / 0.1))
            loss_d_x_gt = torch.mean(torch.abs(d_x_pred - d_x))
            loss_d_y_gt = torch.mean(torch.abs(d_y_pred - d_y)) 

            loss_theta_all.append(loss_theta)
            loss_scale_all.append(loss_scale)
            loss_d_x_gt_all.append(loss_d_x_gt)
            loss_d_y_gt_all.append(loss_d_y_gt)
            loss_transform_all.append(loss_transform)
            
        loss_theta_o = torch.sum(torch.stack(loss_theta_all))
        loss_scale_o = torch.sum(torch.stack(loss_scale_all))
        loss_d_x_gt_o = torch.sum(torch.stack(loss_d_x_gt_all))
        loss_d_y_gt_o = torch.sum(torch.stack(loss_d_y_gt_all))
        loss_transform_o = torch.sum(torch.stack(loss_transform_all))

        loss_all = 2.0 * loss_theta_o + 10.0 * loss_scale_o + 0.8 * loss_d_x_gt_o + 0.8 * loss_d_y_gt_o + 2.0 * loss_transform_o

        if torch.isnan(predict_params.detach()).sum().ge(1):
            raise ValueError("predict_params has nan!")
        if torch.isinf(loss_scale.detach()).sum().ge(1):
            print(scale_pred)
            raise ValueError("loss_scale has inf!")



        return loss_all, {"theta":loss_theta_all[-1].item(), "scale":loss_scale_all[-1].item(), "d_x":loss_d_x_gt_all[-1].item(), "d_y":loss_d_y_gt_all[-1].item(),
                "trans": loss_transform_all[-1].item(), "flow_0":loss_flow_all[0].item(), "flow_1":loss_flow_all[1].item()}

