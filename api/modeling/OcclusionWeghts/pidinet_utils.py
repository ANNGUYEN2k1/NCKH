import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def convert_pdc(op, weight):
    if op == 'cv':
        return weight
    elif op == 'cd':
        shape = weight.shape
        if shape[2] == shape[3] == 3:
            weight_c = (weight.sum(dim=[2, 3]))
            weight = weight.view(shape[0], shape[1], -1)
            with torch.no_grad(): 
                weight[:,:,4]= weight[:, :, 4] - weight_c
            return weight.view(shape)
        raise ValueError(" kernel size with op = 'cd' must be 3x3 ")
    elif op == 'ad':     
        shape = weight.shape
        if shape[2] == shape[3] == 3:
            weight = weight.view(shape[0], shape[1], -1)
            weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
            return weight_conv
        raise ValueError(" kernel size with op = 'ad' must be 3x3 ")
    elif op == 'rd':
        shape = weight.shape
        if shape[2] == shape[3] == 3:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weight = weight.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, [0,1,2,3,5,6,7,8]]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, [0,1,2,3,5,6,7,8]]
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            return buffer
        raise ValueError(" kernel size with op = 'rd' must be 3x3 ")
    raise ValueError("wrong op {}".format(str(op)))

def convert_pidinet(model, pdcs):
    
    new_dict = {}

    for pname, p in model.items():
        if 'init_block.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[0], p)
        elif 'block1_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[1], p)
        elif 'block1_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[2], p)
        elif 'block1_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[3], p)
        elif 'block2_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[4], p)
        elif 'block2_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[5], p)
        elif 'block2_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[6], p)
        elif 'block2_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[7], p)
        elif 'block3_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[8], p)
        elif 'block3_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[9], p)
        elif 'block3_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[10], p)
        elif 'block3_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[11], p)
        elif 'block4_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[12], p)
        elif 'block4_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[13], p)
        elif 'block4_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[14], p)
        elif 'block4_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[15], p)
        else:
            new_dict[pname] = p

    return new_dict

"""
Function factory for pixel difference convolutional operations with vanilla conv components
please see line 49, the theta parameter was also used in "Yu et al, Searching central difference convolutional networks for face anti-spoofing, CVPR 2020"
Author: Zhuo Su
Date: Dec 29, 2021
"""
class Conv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


## cd, ad, rd convolutions
## theta could be used to control the vanilla conv components
## theta = 0 reduces the function to vanilla conv, theta = 1 reduces the fucntion to pure pdc (used in the paper)
def createConvFunc(op_type, theta):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    assert theta > 0 and theta <= 1.0, 'theta should be within (0, 1]'

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True) * theta
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - theta * weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, [0,1,2,3,5,6,7,8]]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, [0,1,2,3,5,6,7,8]] * theta
            buffer[:, :, 12] = weights[:, :, 4] * (1 - theta)
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None

class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y

class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride=stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """
    def __init__(self, pdc, inplane, ouplane, stride=1, mode ='train'):
        super(PDCBlock_converted, self).__init__()
        self.stride=stride
        if mode == 'train':
            self.tranning = True
        else:
            self.tranning =False

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

