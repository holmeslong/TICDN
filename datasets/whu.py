# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

import os
import os.path as osp
import random
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import albumentations as A

from .base import _BaseDataset


class WHUDataset(Dataset):

    def __init__(self, img_dir, gt_dir, is_train=True) -> None:
        super().__init__()

        img_A_dir = os.path.join(img_dir, 'A')
        img_B_dir = os.path.join(img_dir, 'B')
        self.img_A = [os.path.join(img_A_dir, img_name) for img_name in os.listdir(img_A_dir)]
        self.img_B = [os.path.join(img_B_dir, img_name) for img_name in os.listdir(img_B_dir)]
        self.gt_mask = [os.path.join(gt_dir, img_name) for img_name in os.listdir(gt_dir)]

        self.mean_value = [0.485, 0.456, 0.406]
        self.std_value = [0.229, 0.224, 0.225]

        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(45, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, p=0.3),
            ])
        else:
            self.transform = None

        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean_value, self.std_value)
        ])

    def __getitem__(self, index: int):
        # load image
        img_a_path = self.img_A[index]
        img_b_path = self.img_B[index]
        img_a = cv2.imread(img_a_path)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.imread(img_b_path)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        
        # load mask
        gt_mask = cv2.imread(self.gt_mask[index], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            imgs = np.concatenate([img_a, img_b], 2)

            # transform
            transformed = self.transform(image=imgs, mask=gt_mask)
            imgs = transformed['image']
            img_a, img_b = imgs[:, :, :3], imgs[:, :, 3:]
            gt_mask = transformed['mask']

        img_a, img_b = self.as_tensor(img_a), self.as_tensor(img_b)
        gt_mask = (gt_mask / 255.0).astype(np.float32)
        gt_mask = torch.from_numpy(gt_mask)

        return img_a, img_b, gt_mask[None]

    def __len__(self) -> int:
        return len(self.gt_mask)


class WHUAugBox(_BaseDataset):
    """
    WHU building dataset

    Data path format:
    root:
        ImageSets:
            train.txt
            test.txt
            val.txt
        Images:
            A:*.png
            B:*.png
        Label_bbox: *.png
        Label_grabcut: *.png
        Label: *.png
    """

    def __init__(self, pickle_file=None, **kwargs):
        super(WHUAugBox, self).__init__(**kwargs)

        self.pseudo_label = None
        if isinstance(pickle_file, str):
            self.pseudo_label = pickle.load(open(pickle_file, 'rb'))

        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[104.008, 116.669, 122.675],
                        std=[0.229, 0.224, 0.225])])

    def _set_files(self):
        if self.split in ["train", "val", "test"]:
            file_list = osp.join(
                self.root, "ImageSets", self.split + ".txt"
            )
            self.files = [i.strip() for i in open(file_list) if not i.strip() == ' ']
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_a_path = osp.join(self.root, 'Images', "A", image_id)
        image_b_path = osp.join(self.root, 'Images', "B", image_id)

        # Load image
        image_a = cv2.imread(image_a_path, cv2.IMREAD_COLOR).astype(np.float32)
        image_b = cv2.imread(image_b_path, cv2.IMREAD_COLOR).astype(np.float32)
        images = [image_a, image_b]
        if self.split == 'val' or self.split == 'test':
            label_path = osp.join(self.root, 'Label', image_id)  # GT label for val.
            label = np.asarray(Image.open(label_path).convert('L'), dtype=np.int32)
            label[label == 255] = 1
            unsure_mask = np.zeros(shape=label.shape)
        else:
            if not self.pseudo_label:
                label_path = osp.join(self.root, 'Label_grabcut', image_id)  # grab cut pseudo label for training.
                label = np.asarray(Image.open(label_path).convert('L'), dtype=np.int32)
                label[label == 255] = 1
            else:
                label = self.pseudo_label[image_id]
                label = np.array(label, dtype=np.uint8)
            unsure_mask_path = osp.join(self.root, "unsure_mask", image_id)
            unsure_mask = np.asarray(Image.open(unsure_mask_path).convert('L'), dtype=np.int32)
            unsure_mask[unsure_mask == 255] = 1

        bbox_path = osp.join(self.root, "Label_bbox", image_id)
        bbox = np.asarray(Image.open(bbox_path).convert('L'), dtype=np.int32)
        bbox[bbox == 255] = 1

        return image_id, images, label, bbox, unsure_mask

    def _augmentation(self, images, label, bbox, unsure_mask):
        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        images = [cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR) for image in images]
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)
        bbox = Image.fromarray(bbox).resize((w, h), resample=Image.NEAREST)
        bbox = np.asarray(bbox, dtype=np.int64)
        unsure_mask = Image.fromarray(unsure_mask).resize((w, h), resample=Image.NEAREST)
        unsure_mask = np.asarray(unsure_mask, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            images = [cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs) for image in images]
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)
            bbox = cv2.copyMakeBorder(bbox, value=self.ignore_label, **pad_kwargs)
            unsure_mask = cv2.copyMakeBorder(unsure_mask, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        images = [image[start_h:end_h, start_w:end_w] for image in images]
        label = label[start_h:end_h, start_w:end_w]
        bbox = bbox[start_h:end_h, start_w:end_w]
        unsure_mask = unsure_mask[start_h:end_h, start_w:end_w]

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                images = [np.fliplr(image).copy() for image in images]  # HWC
                label = np.fliplr(label).copy()  # HW
                bbox = np.fliplr(bbox).copy()  # HW
                unsure_mask = np.fliplr(unsure_mask).copy()  # HW

        return images, label, bbox, unsure_mask

    def __getitem__(self, index):
        image_id, images, label, bbox, unsure_mask = self._load_data(index)
        if self.augment:
            images, label, bbox, unsure_mask = self._augmentation(images, label, bbox, unsure_mask)

        # # Mean subtraction
        # images = [image - self.mean_bgr for image in images]

        # # HWC -> CHW
        # images = [image.transpose(2, 0, 1).astype(np.float32) for image in images]
        # images = [torch.from_numpy(image) for image in images]

        images = [T.ToTensor()(img) for img in images]
        images = [torch.sub(torch.multiply(torch.tensor(2.), 
                                           torch.div(img, torch.tensor(255.))), torch.tensor(1.)) for img in images]
        # images = [self.as_tensor(img) for img in images]
        label = torch.from_numpy(label.astype(np.int64))
        bbox = torch.from_numpy(bbox.astype(np.int64))
        sure_mask = np.ones(unsure_mask.shape, dtype=np.bool_)
        sure_mask[unsure_mask == 1] = False
        sure_mask = torch.from_numpy(sure_mask)
        return image_id, images[0], images[1], label[None], bbox[None], sure_mask[None]


if __name__ == '__main__':
    transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.Transpose(p=1),
            A.Rotate(45, p=1),
            A.ShiftScaleRotate(shift_limit=0.1, p=1),
            A.GaussNoise(mean=0, p=1)
        ])
    as_tensor = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=[0.37772245912313807, 0.4425350597897193, 0.4464795300397427],
        #             std=[0.1762166286060892, 0.1917139949806914, 0.20443966020731438])
    ])
    img_a = cv2.imread("G:/RemoteDataset/WHU_Two/seg_coco_format/after/img_6.png")
    img_b = cv2.imread("G:/RemoteDataset/WHU_Two/seg_coco_format/before/img_6.png")
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    imgs = np.concatenate([img_a, img_b], axis=2)
    mask = cv2.imread("G:/RemoteDataset/WHU_Two/seg_coco_format/gt/img_6.png")
    res = transform(image=imgs, mask=mask)
    transformed_img = res['image']
    img_a = as_tensor(transformed_img[:, :, :3])
    print(torch.max(img_a))
