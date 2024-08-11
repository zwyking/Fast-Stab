import torch
import torch.nn as nn

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, batch_norm=False, relu=True):
    if batch_norm:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1, inplace=True))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes))
    else:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias),
                                nn.LeakyReLU(0.1))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias))


def predict_flow(in_planes, nbr_out_channels=36):
    return nn.Conv2d(in_planes, nbr_out_channels, kernel_size=3, stride=1, padding=1, bias=True)

class flow_refine(nn.Module):
    def __init__(self, input_to_refinement, batch_norm):
        super().__init__()
        self.dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                             batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(64)
    
    def forward(self, x):
        x = self.dc_conv6(self.dc_conv5(self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))))
        res = self.dc_conv7(x)
        return res