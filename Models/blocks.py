"""
MIT License

Copyright (c) 2020 linjc16

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_block(nc_in, nc_out, k, s, norm='bn', act='lrelu', dilation=1):

    blocks = [
        nn.Conv2d(nc_in, nc_out, k, s, dilation if dilation > 1 else k // 2, dilation=dilation)
    ]
    '''
    if norm is not None:
        norm = norm.lower()
        if norm == 'bn':
            blocks.append(nn.BatchNorm2d(nc_out))
        elif norm == 'in':
            blocks.append(nn.InstanceNorm2d(nc_out))
        else:
            raise RuntimeError
    '''

    if act is not None:
        act = act.lower()
        if act == 'relu':
            blocks.append(nn.ReLU(True))
        elif act == 'lrelu':
            blocks.append(nn.LeakyReLU(0.2, True))
        else:
            raise RuntimeError
    
    return nn.Sequential(*blocks)


def conv3d_block(in_planes, out_planes, kernel_size, stride, norm='bn', act='lrelu'):

    blocks = [
        nn.Conv3d(in_planes, out_planes, kernel_size, stride, kernel_size // 2)
    ]
    '''
    if norm is not None:
        norm = norm.lower()
        if norm == 'bn':
            blocks.append(nn.BatchNorm3d(out_planes))
        elif norm == 'in':
            blocks.append(nn.InstanceNorm3d(out_planes))
        else:
            raise RuntimeError
    '''
    
    if act is not None:
        act = act.lower()
        if act == 'lrelu':
            blocks.append(nn.LeakyReLU(0.2, True))
        elif act == 'relu':
            blocks.append(nn.ReLU(True))
        else:
            raise RuntimeError
    
    return nn.Sequential(*blocks)


class ResBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, dilation=1):
        super(ResBlock, self).__init__()
        self.conv = conv_block(in_planes, out_planes, kernel_size, stride, norm='bn', act='lrelu', dilation=dilation)
        
    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return out



class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        
        self.disp = torch.from_numpy(
            np.reshape(np.array(range(maxdisp)), 
            [1, maxdisp, 1, 1]
            )).cuda().float().requires_grad_(False)
    
    def forward(self, x):
        
        y = x.mul(self.disp).sum(dim=1, keepdim=True)
       
        return y

