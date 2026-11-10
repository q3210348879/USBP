from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, mode, noise_label, train_data, train_label, 
                 transform=None, 
                 transform_crop=None, transform_flip=None, transform_to_tensor=None, normalize=None, max_magnitude=20,
                 noise_file='', pred=[], sampled_clean_prob=[], log='', H_values=None): 
        
        self.r = r
        self.mode = mode
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 
     
        if self.mode == 'test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']
            self.transform = transform 
                            
        elif self.mode in ['labeled', 'unlabeled']:
            if self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
                log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                log.flush()
            elif self.mode == 'unlabeled':
                pred_idx = (1-pred).nonzero()[0]
                
            self.train_data = train_data[pred_idx]
            self.noise_label = [noise_label[i] for i in pred_idx]
            self.indices = pred_idx
            self.transform_crop = transform_crop
            self.transform_flip = transform_flip
            self.transform_to_tensor = transform_to_tensor
            self.normalize = normalize
            self.max_magnitude = max_magnitude 
            self.sampled_w = [sampled_clean_prob[i] for i in pred_idx]
            
            if H_values is not None:
                self.H_vals = [H_values[i] for i in pred_idx]
            else:
                self.H_vals = [0] * len(pred_idx) 
                                       
        elif self.mode in ['all', 'eval_train']:
            self.train_data = train_data
            self.noise_label = noise_label
            self.true_label = train_label
            self.transform = transform 

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, sampled_w = self.train_data[index], self.noise_label[index], self.sampled_w[index]
            h_val = self.H_vals[index] 
            # Clip for numerical safety
            h_val = np.clip(h_val, 0.0, 1.0)
            original_index = self.indices[index]
            
            img = Image.fromarray(img)
            
            # img1 (weak)
            img1_aug = self.transform_crop(img)
            img1_aug = self.transform_flip(img1_aug)
            img1 = self.normalize(self.transform_to_tensor(img1_aug))
            
            # img2 (dynamic strong)
            img2_aug = self.transform_crop(img)
            img2_aug = self.transform_flip(img2_aug)
            
            # Uncertainty-inverse augmentation: the more confident the sample, the stronger the augmentation
            dynamic_magnitude = int((1 - h_val) * self.max_magnitude) 
            
            if dynamic_magnitude > 0:
                img2_aug = transforms.RandAugment(num_ops=3, magnitude=dynamic_magnitude)(img2_aug)
            
            img2 = self.normalize(self.transform_to_tensor(img2_aug))
                
            return img1, img2, target, sampled_w, original_index
            
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            h_val = self.H_vals[index]
            # Clip for numerical safety
            h_val = np.clip(h_val, 0.0, 1.0)
            
            img = Image.fromarray(img)
            
            # img1 (weak)
            img1_aug = self.transform_crop(img)
            img1_aug = self.transform_flip(img1_aug)
            img1 = self.normalize(self.transform_to_tensor(img1_aug))
            
            # img2 (dynamic strong)
            img2_aug = self.transform_crop(img)
            img2_aug = self.transform_flip(img2_aug)
            
            dynamic_magnitude = int((1 - h_val) * self.max_magnitude) 
            
            if dynamic_magnitude > 0:
                img2_aug = transforms.RandAugment(num_ops=3, magnitude=dynamic_magnitude)(img2_aug)
            
            img2 = self.normalize(self.transform_to_tensor(img2_aug))
                
            return img1, img2

        elif self.mode == 'all':
             img, target, index_val = self.train_data[index], self.noise_label[index], index
             img = Image.fromarray(img)
             img = self.transform(img) 
             return img, target, index_val

        elif self.mode == 'eval_train':
             img, target, true_target, index_val = self.train_data[index], self.noise_label[index], self.true_label[index], index
             img = Image.fromarray(img)
             img = self.transform(img) 
             return img, target, index_val, true_target

        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img) 
            return img, target
           
    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        
        if self.dataset=='cifar10':
            self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize,
                ])    
        elif self.dataset=='cifar100':    
            self.normalize = transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize,
                ])
                
        self.transform_crop = transforms.RandomCrop(32, padding=4)
        self.transform_flip = transforms.RandomHorizontalFlip()
        self.transform_to_tensor = transforms.ToTensor()
        
        self.transform_warmup = transforms.Compose([
            self.transform_crop,
            self.transform_flip,
            self.transform_to_tensor,
            self.normalize
        ])
        
        self.max_magnitude = 20 
                
        self.train_data = []
        self.true_labels = []
        if dataset=='cifar10': 
            for n in range(1,6):
                dpath = '%s/data_batch_%d'%(root_dir,n)
                data_dic = unpickle(dpath)
                self.train_data.append(data_dic['data'])
                self.true_labels += data_dic['labels']
            self.train_data = np.concatenate(self.train_data)
        elif dataset=='cifar100':    
            train_dic = unpickle('%s/train'%root_dir)
            self.train_data = train_dic['data']
            self.true_labels = train_dic['fine_labels']
        self.train_data = self.train_data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

        if noise_mode == 'cifarn':
            # ---- CIFAR-N: load real human-annotated noisy labels ----
            cifarn_dir = os.path.join(os.path.dirname(root_dir), 'cifar-n-labels')
            if dataset == 'cifar10':
                cifarn_path = os.path.join(cifarn_dir, 'CIFAR-10_human.pt')
                cifarn_data = torch.load(cifarn_path, map_location='cpu')
                # noise_file doubles as key selector: 'worse_label', 'aggre_label', 'random_label1/2/3'
                key = noise_file if noise_file in cifarn_data else 'worse_label'
                self.noise_label = cifarn_data[key].tolist()
            elif dataset == 'cifar100':
                cifarn_path = os.path.join(cifarn_dir, 'CIFAR-100_human.pt')
                cifarn_data = torch.load(cifarn_path, map_location='cpu')
                self.noise_label = cifarn_data['noisy_label'].tolist()
            # compute actual noise rate
            n_noisy = sum(1 for a, b in zip(self.noise_label, self.true_labels) if a != b)
            actual_r = n_noisy / len(self.true_labels)
            print(f"[CIFAR-N] Loaded human noise labels from {cifarn_path}")
            print(f"[CIFAR-N] Actual noise rate: {actual_r:.2%} ({n_noisy}/{len(self.true_labels)})")
        elif os.path.exists(noise_file):
            self.noise_label = json.load(open(noise_file,"r"))
        else:
            self.noise_label = []
            idx = list(range(50000))
            random.shuffle(idx)
            num_noise = int(self.r*50000)
            noise_idx = idx[:num_noise]
            for i in range(50000):
                if i in noise_idx:
                    if noise_mode=='sym':
                        if dataset=='cifar10':
                            noiselabel = random.randint(0,9)
                        elif dataset=='cifar100':
                            noiselabel = random.randint(0,99)
                        self.noise_label.append(noiselabel)
                    elif noise_mode=='asym':
                        noiselabel = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}[self.true_labels[i]]
                        self.noise_label.append(noiselabel)
                else:
                    self.noise_label.append(self.true_labels[i])
            print("Saving noisy labels to %s ..."%noise_file)
            json.dump(self.noise_label,open(noise_file,"w"))

    def run(self, mode, pred=[], sampled_w=[], H_values=None):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, r=self.r, noise_mode=self.noise_mode, root_dir=self.root_dir, 
                                        transform=self.transform_warmup, mode="all", 
                                        noise_label=self.noise_label, train_data=self.train_data, train_label=self.true_labels)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, r=self.r, noise_mode=self.noise_mode, root_dir=self.root_dir, 
                                            transform_crop=self.transform_crop,
                                            transform_flip=self.transform_flip,
                                            transform_to_tensor=self.transform_to_tensor,
                                            normalize=self.normalize,
                                            max_magnitude=self.max_magnitude,
                                            mode="labeled", noise_label=self.noise_label, train_data=self.train_data, 
                                            train_label=self.true_labels, pred=pred, sampled_clean_prob=sampled_w, 
                                            log=self.log, H_values=H_values)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, r=self.r, noise_mode=self.noise_mode, root_dir=self.root_dir, 
                                              transform_crop=self.transform_crop,
                                              transform_flip=self.transform_flip,
                                              transform_to_tensor=self.transform_to_tensor,
                                              normalize=self.normalize,
                                              max_magnitude=self.max_magnitude,
                                              mode="unlabeled", noise_label=self.noise_label, train_data=self.train_data, 
                                              train_label=self.true_labels, pred=pred,
                                              sampled_clean_prob=sampled_w, 
                                              H_values=H_values)   
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, r=self.r, noise_mode=self.noise_mode, root_dir=self.root_dir, 
                                         transform=self.transform_test, mode='test', 
                                         noise_label=self.noise_label, train_data=self.train_data, train_label=self.true_labels)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, r=self.r, noise_mode=self.noise_mode, root_dir=self.root_dir, 
                                         transform=self.transform_test, mode='eval_train', 
                                         noise_label=self.noise_label, train_data=self.train_data, train_label=self.true_labels)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader