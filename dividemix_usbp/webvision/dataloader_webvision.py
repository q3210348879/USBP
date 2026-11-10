"""WebVision dataloader with USBP modifications:
- labeled mode returns (img1, img2_aug, target, sampled_w, original_index)
- img2 uses RandAugment with magnitude m_i = (1 - H_tilde_i) * max_magnitude
- eval_train returns (img, target, index) and supports H_values pass-through
- run('train', pred, sampled_w, H_values) wires USBP signals into both branches
"""
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as _mp
import os

# Avoid /dev/shm bus errors when workers pass tensors back to main process.
try:
    _mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass


class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir + 'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root + str(c))
            for img in imgs:
                self.val_data.append([c, os.path.join(self.root, str(c), img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class webvision_dataset(Dataset):
    def __init__(self, root_dir, mode, num_class,
                 transform=None,
                 # USBP: split the train transform into composable pieces so we can
                 # apply RandAugment dynamically only to img2 (the strong view).
                 transform_resize_crop=None, transform_flip=None,
                 transform_to_tensor=None, normalize=None,
                 max_magnitude=20,
                 pred=[], sampled_clean_prob=[], H_values=None,
                 log=''):
        self.root = root_dir
        self.mode = mode

        if self.mode == 'test':
            self.transform = transform
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:
            with open(self.root + 'info/train_filelist_google.txt') as f:
                lines = f.readlines()
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target

            if self.mode == 'all':
                # Used by warmup; the full set with simple train transform.
                self.train_imgs = train_imgs
                self.transform = transform
            elif self.mode == 'eval_train':
                # Same as 'all' but the loader returns index for USBP.
                self.train_imgs = train_imgs
                self.transform = transform
            elif self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
                self.train_imgs = [train_imgs[i] for i in pred_idx]
                self.indices = pred_idx
                self.sampled_w = [sampled_clean_prob[i] for i in pred_idx]
                if H_values is not None:
                    self.H_vals = [H_values[i] for i in pred_idx]
                else:
                    self.H_vals = [0.0] * len(pred_idx)
                # Keep composable transforms for dynamic-magnitude RandAugment.
                self.transform_resize_crop = transform_resize_crop
                self.transform_flip = transform_flip
                self.transform_to_tensor = transform_to_tensor
                self.normalize = normalize
                self.max_magnitude = max_magnitude
                print('%s data has a size of %d' % (self.mode, len(self.train_imgs)))
                log.write('Number of labeled samples:%d \n' % (pred.sum()))
                log.flush()
            elif self.mode == 'unlabeled':
                pred_idx = (1 - pred).nonzero()[0]
                self.train_imgs = [train_imgs[i] for i in pred_idx]
                if H_values is not None:
                    self.H_vals = [H_values[i] for i in pred_idx]
                else:
                    self.H_vals = [0.0] * len(pred_idx)
                self.transform_resize_crop = transform_resize_crop
                self.transform_flip = transform_flip
                self.transform_to_tensor = transform_to_tensor
                self.normalize = normalize
                self.max_magnitude = max_magnitude
                print('%s data has a size of %d' % (self.mode, len(self.train_imgs)))

    def _two_views(self, image, h_val):
        """Return (img1: weak, img2: dynamic-RandAugment) per USBP UIAA."""
        h_val = float(np.clip(h_val, 0.0, 1.0))
        # img1 (weak)
        img1 = self.transform_resize_crop(image)
        img1 = self.transform_flip(img1)
        img1 = self.normalize(self.transform_to_tensor(img1))
        # img2 (UIAA: magnitude inverse to H_tilde)
        img2 = self.transform_resize_crop(image)
        img2 = self.transform_flip(img2)
        dynamic_magnitude = int((1.0 - h_val) * self.max_magnitude)
        if dynamic_magnitude > 0:
            img2 = transforms.RandAugment(num_ops=3, magnitude=dynamic_magnitude)(img2)
        img2 = self.normalize(self.transform_to_tensor(img2))
        return img1, img2

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            sampled_w = self.sampled_w[index]
            h_val = self.H_vals[index]
            original_index = self.indices[index]
            image = Image.open(self.root + img_path).convert('RGB')
            img1, img2 = self._two_views(image, h_val)
            return img1, img2, target, sampled_w, int(original_index)

        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            h_val = self.H_vals[index]
            image = Image.open(self.root + img_path).convert('RGB')
            img1, img2 = self._two_views(image, h_val)
            return img1, img2

        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

        elif self.mode == 'eval_train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + img_path).convert('RGB')
            img = self.transform(image)
            # WebVision has no clean ground-truth; return target as both noisy and "true".
            return img, target, index, target

        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root + 'val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class webvision_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, log):
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        # ---- training-time transforms (composable for UIAA) ----
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform_resize_crop = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
        ])
        self.transform_flip = transforms.RandomHorizontalFlip()
        self.transform_to_tensor = transforms.ToTensor()
        self.max_magnitude = 20  # UIAA upper bound

        # legacy fixed train transform (used by warmup + USBP-disabled path)
        self.transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            self.normalize,
        ])

    def run(self, mode, pred=[], prob=[], sampled_w=None, H_values=None):
        # Throughput-tuned: shm is large (~10GB), so bump workers + prefetch
        # to keep RTX 4090 fed. pin_memory still off (file_system strategy +
        # large batches make it unnecessary and bus-error-safe).
        if mode == 'warmup':
            all_dataset = webvision_dataset(
                root_dir=self.root_dir, mode='all',
                num_class=self.num_class, transform=self.transform_train)
            return DataLoader(
                dataset=all_dataset, batch_size=self.batch_size * 2,
                shuffle=True, num_workers=self.num_workers, pin_memory=True,
                persistent_workers=True, prefetch_factor=4)

        elif mode == 'train':
            sw = sampled_w if sampled_w is not None else prob
            labeled_dataset = webvision_dataset(
                root_dir=self.root_dir, mode='labeled',
                num_class=self.num_class,
                transform_resize_crop=self.transform_resize_crop,
                transform_flip=self.transform_flip,
                transform_to_tensor=self.transform_to_tensor,
                normalize=self.normalize, max_magnitude=self.max_magnitude,
                pred=pred, sampled_clean_prob=sw, H_values=H_values,
                log=self.log)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers, pin_memory=True,
                persistent_workers=False, prefetch_factor=4)
            unlabeled_dataset = webvision_dataset(
                root_dir=self.root_dir, mode='unlabeled',
                num_class=self.num_class,
                transform_resize_crop=self.transform_resize_crop,
                transform_flip=self.transform_flip,
                transform_to_tensor=self.transform_to_tensor,
                normalize=self.normalize, max_magnitude=self.max_magnitude,
                pred=pred, H_values=H_values)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers, pin_memory=True,
                persistent_workers=False, prefetch_factor=4)
            return labeled_loader, unlabeled_loader

        elif mode == 'test':
            test_dataset = webvision_dataset(
                root_dir=self.root_dir, mode='test',
                num_class=self.num_class, transform=self.transform_test)
            return DataLoader(
                dataset=test_dataset, batch_size=self.batch_size * 8,
                shuffle=False, num_workers=self.num_workers, pin_memory=True,
                prefetch_factor=4)

        elif mode == 'eval_train':
            eval_dataset = webvision_dataset(
                root_dir=self.root_dir, mode='eval_train',
                num_class=self.num_class, transform=self.transform_test)
            return DataLoader(
                dataset=eval_dataset, batch_size=self.batch_size * 8,
                shuffle=False, num_workers=self.num_workers, pin_memory=True,
                prefetch_factor=4)

        elif mode == 'imagenet':
            imagenet_val = imagenet_dataset(
                root_dir=self.root_dir, transform=self.transform_imagenet,
                num_class=self.num_class)
            return DataLoader(
                dataset=imagenet_val, batch_size=self.batch_size * 8,
                shuffle=False, num_workers=self.num_workers, pin_memory=True,
                prefetch_factor=4)
