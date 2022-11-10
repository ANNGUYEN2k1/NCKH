'''Contains utility functions used by train/eval code.
'''
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from glob import glob
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot  as plt
import random
import os

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(path_img, path_label= None, split=0.8, seed=1912537):
    """ Tải ảnh và mặt nạ """
    images = sorted(glob(f"{path_img}/*"))
    if path_label:
        masks = sorted(glob(f"{path_label}/*"))

    """ Chia dữ liệu """
    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(
        images, train_size=split_size, random_state=seed)
    if path_label:
        train_y, valid_y = train_test_split(
            masks, train_size=split_size, random_state=seed)
    else:
        train_y = valid_y = None

    return train_x, train_y, valid_x, valid_y

def display_image_grid(dataset, model=None, samples=5, device=None, fig = '/content/drive/MyDrive/NCKH/cleargrasp/data/log' ):
    dataset = copy.deepcopy(dataset)
    cols = 3 if model else 2
    n = np.arange(0, len(dataset))
    figure, ax = plt.subplots(nrows=samples, ncols=cols, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[np.random.choice(n)]

        ax[i, 0].imshow(np.transpose(image, (1, 2, 0)))
        ax[i, 1].imshow(np.squeeze(mask, axis=0),
                        interpolation="nearest", cmap='gray')
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if model:
            with torch.no_grad():
                image = np.array(image).astype(np.float32)
                image = np.expand_dims(image, axis=0)
                image = torch.from_numpy(image)
                image = image.to(device)
                y_pred = model(image)
                y_pred = torch.max(y_pred, 1)[1]
                y_pred = y_pred[0].cpu().numpy()
                # y_pred = np.squeeze(y_pred, axis=0)
               
                # y_pred = y_pred > 0.5
                # y_pred = y_pred.astype(np.int32)
                y_pred = y_pred * 255/2
                y_pred = np.array(y_pred, dtype=np.uint8)
                ax[i, 2].imshow(
                    y_pred, interpolation="nearest", cmap='gray')
                ax[i, 2].set_title("Predicted mask")
                ax[i, 2].set_axis_off()
    figure.savefig(fig)
    plt.tight_layout()
    plt.show()

def label_to_rgb(label):
    '''Output RGB visualizations of the outlines' labels

    The labels of outlines have 3 classes: Background, Depth Outlines, Surface Normal Outlines which are mapped to
    Red, Green and Blue respectively.

    Args:
        label (torch.Tensor): Shape: (batchSize, 1, height, width). Each pixel contains an int with value of class.

    Returns:
        torch.Tensor: Shape (no. of images, 3, height, width): RGB representation of the labels
    '''
    if len(label.shape) == 4:
        # Shape: (batchSize, 1, height, width)
        rgbArray = torch.zeros((label.shape[0], 3, label.shape[2], label.shape[3]), dtype=torch.float)
        rgbArray[:, 0, :, :][label[:, 0, :, :] == 0] = 1
        rgbArray[:, 1, :, :][label[:, 0, :, :] == 1] = 1
        rgbArray[:, 2, :, :][label[:, 0, :, :] == 2] = 1
    if len(label.shape) == 3:
        # Shape: (1, height, width)
        rgbArray = torch.zeros((3, label.shape[1], label.shape[2]), dtype=torch.float)
        rgbArray[0, :, :][label[0, :, :] == 0] = 1
        rgbArray[1, :, :][label[0, :, :] == 1] = 1
        rgbArray[2, :, :][label[0, :, :] == 2] = 1

    return rgbArray


def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]
    output_tensor = torch.unsqueeze(torch.max(outputs[:max_num_images_to_save], 1)[1].float(), 1)
    output_tensor_rgb = label_to_rgb(output_tensor)
    label_tensor = labels[:max_num_images_to_save]
    label_tensor_rgb = label_to_rgb(label_tensor)

    images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb), dim=3)
    grid_image = make_grid(images, 1, normalize=True, scale_each=True)

    return grid_image


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    """
    The loss is

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

        `(minibatch, C, d_1, d_2, ..., d_K)`

    Args:
        logit (Tensor): Output of network
        target (Tensor): Ground Truth
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss
        weight (List, optional): Defaults to None. Weight assigned to each class
        batch_average (bool, optional): Defaults to True. Whether to consider the loss of each element in the batch.

    Returns:
        Float: The value of loss.
    """

    n, c, h, w = logit.shape
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')

    loss = criterion(logit, target.long())

    if batch_average:
        loss /= n

    return loss


def FocalLoss(logit, target, weight, gamma=2, alpha=0.5, ignore_index=255, size_average=True, batch_average=True):
    
    n, c, h, w = logit.shape
    target = target.squeeze(1)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                    reduction='sum')

    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    if batch_average:
        loss /= n

    return loss

def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    per_class_iou = [0] * n_classes
    num_images_per_class = [0] * n_classes
    for i in range(1,len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * (n_classes-1)
        union = [0] * (n_classes-1)
        iou_per_class = [0] * (n_classes-1)
        for j in range(1,n_classes):
            match = (pred_tmp == j).long() + (gt_tmp == j).long()

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j-1] += it
            union[j-1] += un

            if union[j-1] == 0:
                iou_per_class[j-1] = -1

            else:
                iou_per_class[j-1] = intersect[j-1] / union[j-1]
                # print('IoU for class %d is %f'%(j, iou_per_class[j]))

        iou = []
        for k in range(n_classes-1):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        for k in range(n_classes-1):
            if iou_per_class[k] == -1:
                continue
            else:
                per_class_iou[k] += iou_per_class[k]
                num_images_per_class[k] += 1

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou
    # print('class iou:', per_class_iou)
    # print('images per class:', num_images_per_class)
    return total_iou, per_class_iou, num_images_per_class

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

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