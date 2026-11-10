from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os

from Asymmetric_Noise import noisify_cifar10_asymmetric, noisify_cifar100_asymmetric


def unpickle(file):
    import _pickle as cPickle
    with open(file, "rb") as fo:
        data = cPickle.load(fo, encoding="latin1")
    return data


class cifar_dataset(Dataset):
    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        root_dir,
        mode,
        noise_label,
        train_data,
        train_label,
        transform=None,
        transform_crop=None,
        transform_flip=None,
        transform_to_tensor=None,
        normalize=None,
        max_magnitude=20,
        pred=None,
        sampled_clean_prob=None,
        log=None,
        H_values=None,
    ):
        self.r = r
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

        if self.mode == "test":
            if dataset == "cifar10":
                test_dic = unpickle("%s/test_batch" % root_dir)
                self.test_data = test_dic["data"].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
                self.test_label = test_dic["labels"]
            elif dataset == "cifar100":
                test_dic = unpickle("%s/test" % root_dir)
                self.test_data = test_dic["data"].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
                self.test_label = test_dic["fine_labels"]
            self.transform = transform
            return

        if self.mode in ["labeled", "unlabeled"]:
            pred_arr = np.asarray(pred if pred is not None else [], dtype=bool)
            if pred_arr.size == 0:
                pred_arr = np.zeros(len(train_data), dtype=bool)

            if self.mode == "labeled":
                pred_idx = np.where(pred_arr)[0]
                if log is not None:
                    log.write("Number of labeled samples:%d\n" % pred_arr.sum())
                    log.flush()
            else:
                pred_idx = np.where(~pred_arr)[0]

            sampled_arr = np.asarray(sampled_clean_prob if sampled_clean_prob is not None else [], dtype=np.float32)
            if sampled_arr.size == 0:
                sampled_arr = np.ones(len(train_data), dtype=np.float32)

            h_arr = np.asarray(H_values if H_values is not None else [], dtype=np.float32)
            if h_arr.size == 0:
                h_arr = np.zeros(len(train_data), dtype=np.float32)

            self.train_data = train_data[pred_idx]
            self.noise_label = np.asarray(noise_label)[pred_idx].tolist()
            self.indices = pred_idx

            self.sampled_w = sampled_arr[pred_idx].tolist()
            self.H_vals = h_arr[pred_idx].tolist()

            self.transform_crop = transform_crop
            self.transform_flip = transform_flip
            self.transform_to_tensor = transform_to_tensor
            self.normalize = normalize
            self.max_magnitude = max_magnitude
            return

        if self.mode in ["all", "eval_train"]:
            self.train_data = train_data
            self.noise_label = noise_label
            self.true_label = train_label
            self.transform = transform
            return

        raise ValueError("Unsupported mode: %s" % self.mode)

    def _build_views(self, image, h_val):
        h_val = float(np.clip(h_val, 0.0, 1.0))

        weak = self.transform_crop(image)
        weak = self.transform_flip(weak)
        weak = self.normalize(self.transform_to_tensor(weak))

        strong = self.transform_crop(image)
        strong = self.transform_flip(strong)
        magnitude = int(np.clip((1.0 - h_val) * self.max_magnitude, 0, self.max_magnitude))
        if magnitude > 0:
            strong = transforms.RandAugment(num_ops=3, magnitude=magnitude)(strong)
        strong = self.normalize(self.transform_to_tensor(strong))

        return weak, strong

    def __getitem__(self, index):
        if self.mode == "labeled":
            img = Image.fromarray(self.train_data[index])
            target = int(self.noise_label[index])
            sampled_w = float(self.sampled_w[index])
            original_index = int(self.indices[index])
            weak, strong = self._build_views(img, self.H_vals[index])
            return weak, strong, target, sampled_w, original_index

        if self.mode == "unlabeled":
            img = Image.fromarray(self.train_data[index])
            weak, strong = self._build_views(img, self.H_vals[index])
            return weak, strong

        if self.mode == "all":
            img = Image.fromarray(self.train_data[index])
            target = int(self.noise_label[index])
            img = self.transform(img)
            return img, target, index

        if self.mode == "eval_train":
            img = Image.fromarray(self.train_data[index])
            target = int(self.noise_label[index])
            true_target = int(self.true_label[index])
            img = self.transform(img)
            return img, target, index, true_target

        if self.mode == "test":
            img = Image.fromarray(self.test_data[index])
            target = int(self.test_label[index])
            img = self.transform(img)
            return img, target

        raise ValueError("Unsupported mode: %s" % self.mode)

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        return len(self.test_data)


class cifar_dataloader:
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file="", max_magnitude=20):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.max_magnitude = max_magnitude

        if self.dataset == "cifar10":
            self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        elif self.dataset == "cifar100":
            self.normalize = transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            raise ValueError("Unsupported dataset: %s" % self.dataset)

        self.transform_crop = transforms.RandomCrop(32, padding=4)
        self.transform_flip = transforms.RandomHorizontalFlip()
        self.transform_to_tensor = transforms.ToTensor()
        self.transform_warmup = transforms.Compose(
            [self.transform_crop, self.transform_flip, self.transform_to_tensor, self.normalize]
        )

        self.train_data = []
        self.true_labels = []
        if dataset == "cifar10":
            for n in range(1, 6):
                dpath = "%s/data_batch_%d" % (root_dir, n)
                data_dic = unpickle(dpath)
                self.train_data.append(data_dic["data"])
                self.true_labels += data_dic["labels"]
            self.train_data = np.concatenate(self.train_data)
        elif dataset == "cifar100":
            train_dic = unpickle("%s/train" % root_dir)
            self.train_data = train_dic["data"]
            self.true_labels = train_dic["fine_labels"]

        self.train_data = self.train_data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

        if os.path.exists(noise_file):
            with open(noise_file, "r") as f:
                self.noise_label = json.load(f)
        else:
            self.noise_label = self._inject_noise()
            noise_dir = os.path.dirname(noise_file)
            if noise_dir:
                os.makedirs(noise_dir, exist_ok=True)
            with open(noise_file, "w") as f:
                json.dump(self.noise_label, f)
            print("Saving noisy labels to %s ..." % noise_file)

    def _inject_noise(self):
        if self.noise_mode == "asym":
            if self.dataset == "cifar10":
                noisy_labels, _ = noisify_cifar10_asymmetric(np.array(self.true_labels), self.r)
            else:
                noisy_labels, _ = noisify_cifar100_asymmetric(np.array(self.true_labels), self.r)
            return np.asarray(noisy_labels).astype(int).tolist()

        noise_label = []
        idx = list(range(50000))
        random.shuffle(idx)
        num_noise = int(self.r * 50000)
        noise_idx = set(idx[:num_noise])
        for i in range(50000):
            if i in noise_idx:
                if self.dataset == "cifar10":
                    noiselabel = random.randint(0, 9)
                else:
                    noiselabel = random.randint(0, 99)
                noise_label.append(noiselabel)
            else:
                noise_label.append(int(self.true_labels[i]))
        return noise_label

    def run(self, mode, pred=None, sampled_w=None, H_values=None):
        if mode == "warmup":
            all_dataset = cifar_dataset(
                dataset=self.dataset,
                r=self.r,
                noise_mode=self.noise_mode,
                root_dir=self.root_dir,
                transform=self.transform_warmup,
                mode="all",
                noise_label=self.noise_label,
                train_data=self.train_data,
                train_label=self.true_labels,
            )
            trainloader = DataLoader(
                dataset=all_dataset, batch_size=self.batch_size * 2, shuffle=True, num_workers=self.num_workers
            )
            return trainloader

        if mode == "train":
            labeled_dataset = cifar_dataset(
                dataset=self.dataset,
                r=self.r,
                noise_mode=self.noise_mode,
                root_dir=self.root_dir,
                transform_crop=self.transform_crop,
                transform_flip=self.transform_flip,
                transform_to_tensor=self.transform_to_tensor,
                normalize=self.normalize,
                max_magnitude=self.max_magnitude,
                mode="labeled",
                noise_label=self.noise_label,
                train_data=self.train_data,
                train_label=self.true_labels,
                pred=pred,
                sampled_clean_prob=sampled_w,
                log=self.log,
                H_values=H_values,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

            unlabeled_dataset = cifar_dataset(
                dataset=self.dataset,
                r=self.r,
                noise_mode=self.noise_mode,
                root_dir=self.root_dir,
                transform_crop=self.transform_crop,
                transform_flip=self.transform_flip,
                transform_to_tensor=self.transform_to_tensor,
                normalize=self.normalize,
                max_magnitude=self.max_magnitude,
                mode="unlabeled",
                noise_label=self.noise_label,
                train_data=self.train_data,
                train_label=self.true_labels,
                pred=pred,
                sampled_clean_prob=sampled_w,
                H_values=H_values,
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
            return labeled_trainloader, unlabeled_trainloader

        if mode == "test":
            test_dataset = cifar_dataset(
                dataset=self.dataset,
                r=self.r,
                noise_mode=self.noise_mode,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="test",
                noise_label=self.noise_label,
                train_data=self.train_data,
                train_label=self.true_labels,
            )
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            return test_loader

        if mode == "eval_train":
            eval_dataset = cifar_dataset(
                dataset=self.dataset,
                r=self.r,
                noise_mode=self.noise_mode,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="eval_train",
                noise_label=self.noise_label,
                train_data=self.train_data,
                train_label=self.true_labels,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            return eval_loader

        raise ValueError("Unsupported mode: %s" % mode)

