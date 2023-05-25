# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import numpy as np

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.kpts_ops import build_HSPR, build_SPR, COCO_CLASSES


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, #hierarchical,
                 kpts_center, return_boxes=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.ids = [
            img_id
            for img_id in self.ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(#hierarchical,
                                              kpts_center, return_boxes)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        """
        while not target:
            img, target = super(CocoDetection, self).__getitem__(idx)
        """
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, #hierarchical,
                 kpts_center, return_boxes=False):

        #self.hierarchical = hierarchical
        self.kpts_center = kpts_center
        self.return_boxes = return_boxes

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        if self.return_boxes:
            boxes = [obj["bbox"] for obj in anno]
            # guard against no boxes via resizing
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        keypoints = [obj["keypoints"] for obj in anno]
        #keypoints = [obj["keypoints"] for obj in anno if max(obj['keypoints'])!=0]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        keypoints[:, 0::3].clamp_(min=0, max=w)
        keypoints[:, 1::3].clamp_(min=0, max=h)
        keypoints[:, 2::3] = torch.where(keypoints[:, 2::3] == 0, keypoints[:, 2::3],
                                         keypoints[:, 2::3] - 1)  # COCO: o in (0,1,2). This makes 0,1 -> 0 and 2 -> 1

        # filtering out 'empty humans' (humans without annotated keypoints)
        #empty_kpts_ind = (keypoints[:,0:2].sum(dim=1) != 0)
        #keypoints = keypoints[empty_kpts_ind]

        #center = keypoints[:, 3:6].clone() # left eye
        if self.kpts_center == "center_of_mass":
            x_ctr = torch.Tensor(np.ma.average(keypoints[:, 0::3].clone(), axis=1,
                                            weights=keypoints[:, 2::3]).filled(0))
            y_ctr = torch.Tensor(np.ma.average(keypoints[:, 1::3].clone(), axis=1,
                                            weights=keypoints[:, 2::3]).filled(0))
            center = torch.stack((x_ctr, y_ctr, torch.ones(keypoints[:,2].shape)), dim=1)
            for i in range(len(center)): # set center visibility of empty humans to 0
                if (center[i,0] and center[i,1]) == 0:
                    center[i, 2] =  torch.tensor(0)

        elif self.kpts_center in COCO_CLASSES and not self.hierarchical:
            center_idx = COCO_CLASSES.index(self.kpts_center)
            center = keypoints[:, center_idx*3:center_idx*3+3].clone()
        else: # take point between two shoulders as center
            lshou_idx = COCO_CLASSES.index("left_shoulder")
            rshou_idx = COCO_CLASSES.index("right_shoulder")
            sho_ctr = torch.stack((keypoints[:, lshou_idx*3:lshou_idx*3+3].clone(),
                                   keypoints[:, rshou_idx*3:rshou_idx*3+3].clone()), dim=1)
            center = torch.mean(sho_ctr, dim=1)
            # if at least one of the shoulders is not visible we set the center visibility to 0
            center[:, 2] = center[:, 2].int()

        # if self.hierarchical: # Hierarchical Structured Pose Representation (HSPR)
        #     keypoints = build_HSPR(keypoints, center) 
        # else: # Structured Pose Representation (SPR)
        #     keypoints = build_SPR(keypoints, center)
        keypoints = build_SPR(keypoints, center)

        relative_keypoints = torch.cat((center, keypoints), dim=1)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        #classes = classes[empty_kpts_ind]
        empty_kpts_ind = (keypoints[:,0:2].sum(dim=1) == 0)
        classes[empty_kpts_ind] = torch.tensor(0, dtype=torch.int64)

        # visible_mask = (keypoints[:, 2::3] > 1)
        # kpts_classes = torch.as_tensor(np.tile(np.arange(1, 18), (keypoints.shape[0], 1)))
        # kpts_classes = 2 * kpts_classes - 2 + visible_mask  # 0: (nose,invisible), 1: (nose,visible), 2: (l_eye,..

        # remove annotations without a left eye? change the non-visible labels?
        target = {"labels": classes}

        """
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        """
        if self.return_boxes:
            target["boxes"] = boxes

        target["image_id"] = image_id
        """
        if keypoints is not None:
            target["keypoints"] = keypoints
        """

        target["keypoints"] = relative_keypoints
        # target["keypoints_classes"] = kpts_classes

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        """
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        """
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, downsample, max_size=1333):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if downsample:
        normalize = T.Compose([T.RandomResize([200], max_size=max_size), normalize])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            #T.RandomRotation([25, 345]),
            T.RandomRotation(25),
            T.RandomCoarseDropout((0.05,0.2), size_percent=(0.015,0.03)),
            T.RandomSelect(
                #T.RandomResize(scales, max_size=1333),
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    #T.RandomResize(scales, max_size=1333),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            #T.RandomResize([800], max_size=1333),
            T.RandomResize([800], max_size=max_size),
            #T.RandomResize([1000], max_size=max_size), # better performance!?
            #T.RandomResize([(512,512)]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'person_keypoints'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017{args.coco_filter}.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017{args.coco_filter}.json'),
    }
    # for macaque pose dataset
    if args.macaquepose:
        PATHS = {
            "train": (root / "images" / "train", root / "annotations" / f"macaque_train{args.coco_filter}.json"),
            "val": (root / "images"/ "val", root / "annotations" / "macaque_val.json"),
        }

    img_folder, ann_file = PATHS[image_set]
    
    image_set_ = 'val' if args.overfit else image_set

    dataset = CocoDetection(img_folder, ann_file,
                            transforms=make_coco_transforms(image_set_, downsample=args.downsampling, max_size=args.max_size),
                            #hierarchical=args.hierarchical,
                            kpts_center=args.kpts_center)
    return dataset
