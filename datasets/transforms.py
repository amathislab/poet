# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentations for images and keypoints/bboxes.
"""
import random
import numpy as np

import PIL
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
#from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "keypoints" in target:
        keypoints = target["keypoints"]

        # transform relative keypoints to absolute keypoints
        keypoints[:,3::3] = keypoints[:,3::3] + keypoints[:,0].unsqueeze(-1)
        keypoints[:,4::3] = keypoints[:,4::3] + keypoints[:,1].unsqueeze(-1)

        # check which keypoints are visible in cropped patch
        valid_x = (keypoints[:, 0::3] > j) & (keypoints[:, 0::3] < w + j)
        valid_y = (keypoints[:, 1::3] > i) & (keypoints[:, 1::3] < h + i)
        valid = torch.logical_and(valid_x, valid_y)
        keypoints[:, 2::3][~valid] = 0

        # shift coordinates to new coordinates of cropped patch
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        keypoints[:, 0::3] = keypoints[:, 0::3] - torch.tensor(j, dtype=torch.float32)
        keypoints[:, 1::3] = keypoints[:, 1::3] - torch.tensor(i, dtype=torch.float32)
        keypoints[:, 0::3] = torch.min(keypoints[:, 0::3], max_size[0]).clamp(min=0)
        keypoints[:, 1::3] = torch.min(keypoints[:, 1::3], max_size[1]).clamp(min=0)
        
        # compute a new center of mass
        #assert kpts_center == "center_of_mass"
        x_ctr = torch.Tensor(np.ma.average(keypoints[:, 3::3].clone(), axis=1,
                            weights=keypoints[:, 5::3]).filled(0))
        y_ctr = torch.Tensor(np.ma.average(keypoints[:, 4::3].clone(), axis=1,
                            weights=keypoints[:, 5::3]).filled(0))
        center = torch.stack((x_ctr, y_ctr, torch.ones(keypoints[:, 2].shape)), dim=1)
        for i in range(len(center)): 
            if (center[i, 0] and center[i, 1]) == 0:  # set center visibility of empty humans to 0
                center[i, 2] = torch.tensor(0)
            # set center to new value
            keypoints[i, 0:3] = center[i].clone()

        # transform absolute keypoints back to relative keypoints
        keypoints[:,3::3] = keypoints[:,3::3] - keypoints[:,0].unsqueeze(-1)
        keypoints[:,4::3] = keypoints[:,4::3] - keypoints[:,1].unsqueeze(-1)

        target["keypoints"] = keypoints
        fields.append("keypoints")

        # remove elements for which all keypoints are zero
        keypoints = target['keypoints']
        keep = torch.any(keypoints[:, 2::3].bool(), dim=1)
        for field in fields:
            target[field] = target[field][keep]

    # remove elements for which the boxes have zero area
    if "boxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        cropped_boxes = target['boxes'].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes


    if "keypoints" in target:
        keypoints = target["keypoints"]
        keypoints[:, 0::3] = keypoints[:, 0::3] * -1
        if any(keypoints[:,0:3].clone().flatten()): # centers are not learned
            keypoints[:, 0, None] += w

        # mirroring also keypoints
        for i in range(6, len(keypoints[0,:]), 6):
            kpts = keypoints[:, i:i+6]
            keypoints[:, i:i+6] = kpts[:, (3,4,5, 0,1,2)]

        target["keypoints"] = keypoints

    return flipped_image, target


def rotate(image, target, degree):

    image = np.asarray(image)
    angle = torch.FloatTensor(1).uniform_(-degree, degree).item()
    rot = iaa.Affine(rotate=angle)

    target = target.copy()

    #if "boxes" in target:
        # TO DO (also for 'area'; area should stay the same when rotating bbox)
        # TO DO (but not needed for now)

    if "keypoints" in target:
        keypoints = target["keypoints"]

        # transform relative keypoints to absolute keypoints
        keypoints[:,3::3] = keypoints[:,3::3] + keypoints[:,0].unsqueeze(-1)
        keypoints[:,4::3] = keypoints[:,4::3] + keypoints[:,1].unsqueeze(-1)

        # rotate coordinates to new coordinates of rotated image
        for i in range(len(keypoints)):
            kpts = KeypointsOnImage(    
                [Keypoint(x=kx, y=ky) for kx,ky in zip(keypoints[i,0::3],keypoints[i,1::3])],
                shape=image.shape)
            rotated_image, kpts_rot = rot(image=image, keypoints=kpts)
            keypoints[i, 0::3] = torch.tensor([k.x for k in kpts_rot])
            keypoints[i, 1::3] = torch.tensor([k.y for k in kpts_rot])

        # check which keypoints are still visible in rotated image
        valid_x = (keypoints[:, 0::3] > 0) & (keypoints[:, 0::3] < image.shape[1])
        valid_y = (keypoints[:, 1::3] > 0) & (keypoints[:, 1::3] < image.shape[0])
        valid = torch.logical_and(valid_x, valid_y)
        keypoints[:, 2::3][~valid] = 0
        
        # compute a new center of mass
        #assert kpts_center == "center_of_mass"
        x_ctr = torch.Tensor(np.ma.average(keypoints[:, 3::3].clone(), axis=1,
                            weights=keypoints[:, 5::3]).filled(0))
        y_ctr = torch.Tensor(np.ma.average(keypoints[:, 4::3].clone(), axis=1,
                            weights=keypoints[:, 5::3]).filled(0))
        center = torch.stack((x_ctr, y_ctr, torch.ones(keypoints[:, 2].shape)), dim=1)
        for i in range(len(center)): 
            if (center[i, 0] and center[i, 1]) == 0:  # set center visibility of empty humans to 0
                center[i, 2] = torch.tensor(0)
            # set center to new value
            keypoints[i, 0:3] = center[i].clone()

        # transform absolute keypoints back to relative keypoints
        keypoints[:,3::3] = keypoints[:,3::3] - keypoints[:,0].unsqueeze(-1)
        keypoints[:,4::3] = keypoints[:,4::3] - keypoints[:,1].unsqueeze(-1)

        target["keypoints"] = keypoints

    rotated_image = Image.fromarray(rotated_image)

    return rotated_image, target


def coarsedrop(image, target, drop_rate, size_percent):

    image = np.asarray(image)
    drop_rate = torch.FloatTensor(1).uniform_(drop_rate[0], drop_rate[1]).item()
    size_percent = torch.FloatTensor(1).uniform_(size_percent[0], size_percent[1]).item()
    drop = iaa.CoarseDropout(p=drop_rate, size_percent=size_percent, per_channel=0.5)

    target = target.copy()

    if "keypoints" in target:
        keypoints = target["keypoints"]

        # transform relative keypoints to absolute keypoints
        keypoints[:,3::3] = keypoints[:,3::3] + keypoints[:,0].unsqueeze(-1)
        keypoints[:,4::3] = keypoints[:,4::3] + keypoints[:,1].unsqueeze(-1)

        for i in range(len(keypoints)):
            kpts = KeypointsOnImage(    
                [Keypoint(x=kx, y=ky) for kx,ky in zip(keypoints[i,0::3],keypoints[i,1::3])],
                shape=image.shape)
            dropped_image, kpts_drop = drop(image=image, keypoints=kpts)
            keypoints[i, 0::3] = torch.tensor([k.x for k in kpts_drop])
            keypoints[i, 1::3] = torch.tensor([k.y for k in kpts_drop])

        # transform absolute keypoints back to relative keypoints
        keypoints[:,3::3] = keypoints[:,3::3] - keypoints[:,0].unsqueeze(-1)
        keypoints[:,4::3] = keypoints[:,4::3] - keypoints[:,1].unsqueeze(-1)

        target["keypoints"] = keypoints

    dropped_image = Image.fromarray(dropped_image)

    return dropped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    
    if "keypoints" in target:
        keypoints = target["keypoints"]
        keypoints[:, 0::3] = keypoints[:, 0::3] * torch.as_tensor(ratio_width)
        keypoints[:, 1::3] = keypoints[:, 1::3] * torch.as_tensor(ratio_height)
        target["keypoints"] = keypoints

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomRotation(object):
    #def __init__(self, degrees, p=0.5):
    def __init__(self, degree, p=0.5):
        #assert isinstance(degrees, (list, tuple))
        #self.degrees = degrees
        self.degree = degree
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            #degree = random.choice(self.degrees)
            #return rotate(img, target, degree)
            return rotate(img, target, self.degree)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomCoarseDropout(object):
    def __init__(self, drop_rate, size_percent, p=0.5):
        assert isinstance(drop_rate, tuple)
        assert isinstance(size_percent, tuple)
        self.drop_rate = drop_rate
        self.size_percent = size_percent
        self.p = p
    
    def __call__(self, img, target=None):
        if random.random() < self.p:
            return coarsedrop(img, target, self.drop_rate, self.size_percent)
        return img, target


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        
        if "keypoints" in target:
            keypoints = target["keypoints"]  # normalize relative to image size
            keypoints[:, 0::3] = keypoints[:, 0::3] / torch.tensor(w, dtype=torch.float32)
            keypoints[:, 1::3] = keypoints[:, 1::3] / torch.tensor(h, dtype=torch.float32)

            # normalize offsets to be positive
            keypoints[:, 3::3] = (keypoints[:, 3::3] * 0.5) + 0.5
            keypoints[:, 4::3] = (keypoints[:, 4::3] * 0.5) + 0.5

            target["keypoints"] = keypoints
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
