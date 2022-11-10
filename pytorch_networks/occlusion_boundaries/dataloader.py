#!/usr/bin/env python3

from __future__ import print_function, division
import os
import glob
from PIL import Image
import Imath
import numpy as np
import imageio

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia

from utils import utils


class OutlinesDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format). If no labels exists, pass empty string.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(self,
                 input_dir='data/datasets/train/milk-bottles-train/resized-files/preprocessed-rgb-imgs',
                 label_dir='',
                 transform=None,
                 input_only=None,
                 ):

        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []  # Variable containing list of all ground truth filenames in dataset
        self._extension_input = ['-rgb.jpg', '-transparent-rgb-img.jpg', 'input-img.jpg']  # The file extension of input images
        self._extension_label = '-outlineSegmentation.png'  # The file extension of labels
        self._create_lists_filenames(self.images_dir, self.labels_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label
        '''

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        # Open labels
        if self.labels_dir:
            label_path = self._datalist_label[index]
            _label = Image.open(label_path).convert('L')
            _label = np.array(_label)[..., np.newaxis]

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
            if self.labels_dir:
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)
        if self.labels_dir:
            _label_tensor = transforms.ToTensor()(_label.astype(float))
        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor

    def _create_lists_filenames(self, images_dir, labels_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        for ext in self._extension_input:
            imageSearchStr = os.path.join(images_dir, '*' + ext)
            imagepaths = sorted(glob.glob(imageSearchStr))
            self._datalist_input = self._datalist_input + imagepaths
        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError('No images found in given directory {}. Searched for {}'.format(images_dir, self._extension_input))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"'
                                               % (labels_dir))
            for ext in self._extension_label:
                labelSearchStr = os.path.join(labels_dir, '*' + self._extension_label)
                labelpaths = sorted(glob.glob(labelSearchStr))
                self._datalist_label = self._datalist_label + labelpaths
            numLabels = len(labelpaths)
            if numLabels == 0:
                raise ValueError('No labels found in given directory {}. Searched for {}'.format(labels_dir, self._extension_label))
            if numImages != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,\
                                found {} images and {} labels' .format(numImages, numLabels))

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default
    
class OutlinesDataset_(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format). If no labels exists, pass empty string.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(self,
                 _datalist_input,
                 _datalist_label='',
                 transform=None,
                 ):

        super().__init__()

        self._datalist_input = _datalist_input
        self._datalist_label = _datalist_label
        self.transform = transform



    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label
        '''

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        # Open labels
        if self._datalist_label:
            label_path = self._datalist_label[index]
            _label = Image.open(label_path).convert('L')
            _label = np.array(_label)[..., np.newaxis]

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform[0].to_deterministic()
            _img = det_tf.augment_image(_img)
            # _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
            if self._datalist_label:
                det_tf = self.transform[1].to_deterministic()
                _label = det_tf.augment_image(_label)
        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)
        if self._datalist_label:
            _label_tensor = transforms.ToTensor()(_label.astype(float))
        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor


import copy
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    imsize = 512
    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize), 0), # Resize image
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Rot90((0, 4)),
        # Blur and Noise
        #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
        #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
        iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
        # Color, Contrast, etc.
        #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
        iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
        iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
        #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    ])
    augs_test = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize), 0),
    ])

    augs = augs_train #, augs_test, None
    input_only = ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = OutlinesDataset(
        input_dir='/home/an/Desktop/cleargrasp/data/datasets/cup-with-waves-train/rgb-imgs',
        label_dir='/home/an/Desktop/cleargrasp/data/datasets/cup-with-waves-train/outlines',
        transform=augs,
        input_only=input_only
    )

    batch_size = 1
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Show 1 Shuffled Batch of Images
    dataset = copy.deepcopy(db_test)
    n = np.arange(0, len(dataset))
    figure, ax = plt.subplots(nrows=5, ncols= 2, figsize=(10, 24))
    for i in range(5):
        image, mask = dataset[np.random.choice(n)]

        ax[i, 0].imshow(np.transpose(image, (1, 2, 0)))
        ax[i, 1].imshow(np.squeeze(mask, axis=0),
                        interpolation="nearest", cmap='gray')
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

    plt.tight_layout()
    plt.show()
