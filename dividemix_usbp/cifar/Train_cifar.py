"""DivideMix + USBP on CIFAR-10/100 (synthetic noise and CIFAR-N).

USBP adds three modules on top of DivideMix, all driven by the MC-Dropout
predictive entropy H (min-max normalised over the training set):
  (a) stochastic Beta partitioning: w ~ Beta(K*p, K*(1-p)), K = K_max / (1 + H)
  (b) confidence-gated cross-network rectification with weight 1 - H
  (c) uncertainty-inverse adaptive RandAugment on the second view (dataloader)
"""
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from PreResNet import ResNet18, ResNet34

# --- Arguments ---
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--model_name', default='resnet18', type=str, help='local model name to use')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=360, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dropout_p', default=0.6, type=float, help='dropout probability for MC-Dropout')
parser.add_argument('--K_mcdropout', default=10, type=int, help='number of forward passes for MC-Dropout')
parser.add_argument('--K_max', default=5.0, type=float, help='max concentration parameter for Beta distribution')
parser.add_argument('--relabel_threshold', default=0.9, type=float, help='Confidence threshold to relabel a noisy sample')

parser.add_argument('--stats_dir', default='./stats', type=str, help='Directory to save epoch statistics')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# --- Training ---
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, relabel_map={}):
    net.train() 
    net2.eval() 

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x_sampled, indices_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)

        # Apply relabeling
        for i, idx in enumerate(indices_x):
            if idx.item() in relabel_map:
                new_label = relabel_map[idx.item()]
                labels_x[i] = new_label 
                # The sample weight is left unchanged: it already holds 1 - H,
                # which the main loop assigns to rectified samples.

        labels_x_onehot = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        w_x_sampled = w_x_sampled.view(-1,1).type(torch.FloatTensor) 
        
        inputs_x, inputs_x2, labels_x_onehot, w_x_sampled = inputs_x.cuda(), inputs_x2.cuda(), labels_x_onehot.cuda(), w_x_sampled.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # Co-Guessing
            outputs_u11, outputs_u12 = net(inputs_u), net(inputs_u2)
            outputs_u21, outputs_u22 = net2(inputs_u, use_dropout=False), net2(inputs_u2, use_dropout=False) 
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            # Label Refinement
            outputs_x, outputs_x2 = net(inputs_x), net(inputs_x2)
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x_sampled * labels_x_onehot + (1 - w_x_sampled) * px
            ptx = px**(1/args.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        loss = Lx + lamb * Lu  + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()


def warmup(epoch,net,optimizer,dataloader):
    net.train() 
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs, use_dropout=False)
        loss = CEloss(outputs, labels)
        if args.noise_mode=='asym':
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode=='sym':
            L = loss
        L.backward()
        optimizer.step()
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs, use_dropout=False) 
            outputs2 = net2(inputs, use_dropout=False) 
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()

def get_mcdropout_uncertainty(model, inputs):
    model.eval() 
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train() 
            
    all_probs = []
    with torch.no_grad():
        for _ in range(args.K_mcdropout):
            outputs = model(inputs, use_dropout=True)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
    mean_probs = torch.stack(all_probs).mean(dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    return entropy

def eval_train_and_get_uncertainty(model, all_loss):
    model.eval() 
    
    num_samples = 50000
    losses = torch.zeros(num_samples)
    uncertainties = torch.zeros(num_samples)
    all_logits = torch.zeros(num_samples, args.num_class)
    all_true_labels = torch.zeros(num_samples, dtype=torch.long)
    all_noisy_labels = torch.zeros(num_samples, dtype=torch.long)

    with torch.no_grad():
        for batch_idx, (inputs, noisy_targets, index, true_targets) in enumerate(eval_loader):
            inputs, noisy_targets_cuda = inputs.cuda(), noisy_targets.cuda()
            outputs = model(inputs, use_dropout=False)
            loss = CE(outputs, noisy_targets_cuda)
            uncertainty = get_mcdropout_uncertainty(model, inputs)

            for b in range(inputs.size(0)):
                idx = index[b].item()
                losses[idx] = loss[b]
                all_logits[idx] = outputs[b]
                all_true_labels[idx] = true_targets[b]
                all_noisy_labels[idx] = noisy_targets[b]
                uncertainties[idx] = uncertainty[b]

    losses_normalized = (losses-losses.min())/(losses.max()-losses.min() + 1e-10)
    uncertainties_normalized = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-10)
    all_loss.append(losses_normalized)

    if args.r==0.9:
        history = torch.stack(all_loss)
        input_loss = history[-min(5, len(history)):].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses_normalized.reshape(-1,1)

    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss.cpu().numpy())
    prob = gmm.predict_proba(input_loss.cpu().numpy())
    prob = prob[:,gmm.means_.argmin()]

    return prob, uncertainties_normalized.numpy(), all_logits.cpu(), all_loss, losses.numpy(), all_true_labels.numpy(), all_noisy_labels.numpy()


# --- Helpers ---
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    if args.model_name == 'resnet18':
        local_model = ResNet18(num_classes=args.num_class, dropout_p=args.dropout_p)
    elif args.model_name == 'resnet34':
        local_model = ResNet34(num_classes=args.num_class, dropout_p=args.dropout_p)
    else:
        raise ValueError(f"Model '{args.model_name}' is not defined in PreResNet.py.")
    print(f"| Building local {args.model_name} model with dropout p={args.dropout_p}.")
    return local_model.cuda()


# --- Main script starts here ---
stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w')
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')

stats_dir = os.path.join(args.stats_dir, f'{args.dataset}_{args.r}_{args.noise_mode}_{args.id}')
os.makedirs(stats_dir, exist_ok=True)
print(f"| Saving epoch stats to: {stats_dir}")

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30 

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()
all_loss = [[],[]]

for epoch in range(args.num_epochs+1):
    lr=args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader)

    else:
        # --- 1. GMM clean probability, MC-Dropout uncertainty and logits ---
        prob1, H1, logits1, all_loss[0], losses1, true_labels, noisy_labels = eval_train_and_get_uncertainty(net1, all_loss[0])
        prob2, H2, logits2, all_loss[1], losses2, _, _ = eval_train_and_get_uncertainty(net2, all_loss[1])

        # --- 2. Beta concentration K = K_max / (1 + H) ---
        K1 = args.K_max / (1 + H1)
        K2 = args.K_max / (1 + H2)
        
        # Beta parameters (logged in the epoch stats); 1e-10 for numerical stability
        alpha1 = K1 * prob1 + 1e-10
        beta1 = K1 * (1 - prob1) + 1e-10
        alpha2 = K2 * prob2 + 1e-10
        beta2 = K2 * (1 - prob2) + 1e-10

        if epoch == warm_up:
            print(f"\n>> Warmup ended. Activating STOCHASTIC Beta Sampling for partitioning.")

        # Stochastic Beta sampling of the partition weight
        prob1_t = torch.from_numpy(prob1).float().cuda()
        K1_t = torch.from_numpy(K1).float().cuda()
        prob2_t = torch.from_numpy(prob2).float().cuda()
        K2_t = torch.from_numpy(K2).float().cuda()

        beta_dist1 = torch.distributions.beta.Beta(K1_t * prob1_t + 1e-10, K1_t * (1 - prob1_t) + 1e-10)
        w_sampled_1 = beta_dist1.sample().cpu().numpy()
        
        beta_dist2 = torch.distributions.beta.Beta(K2_t * prob2_t + 1e-10, K2_t * (1 - prob2_t) + 1e-10)
        w_sampled_2 = beta_dist2.sample().cpu().numpy()


        # --- 3. Confidence-gated cross-network rectification ---
        print(f"\n>> Activating Relabeling Mechanism (Epoch {epoch})")
        
        pred1_hard = (w_sampled_1 > args.p_threshold)
        pred2_hard = (w_sampled_2 > args.p_threshold)

        # Net2 rectifies the samples Net1 will treat as noisy
        noisy_indices_for_net1 = np.where(pred2_hard == False)[0]
        relabel_map2 = {}
        relabel_acc_net1 = 0.0 
        
        if len(noisy_indices_for_net1) > 0:
            noisy_logits_for_net1 = logits2[noisy_indices_for_net1]
            max_probs, pred_labels = torch.max(torch.softmax(noisy_logits_for_net1, dim=1), dim=1)
            original_noisy_labels_subset = torch.tensor(noisy_labels)[noisy_indices_for_net1]
            relabel_candidates_mask = (max_probs > args.relabel_threshold) & (pred_labels != original_noisy_labels_subset)
            
            relabel_indices_net2 = noisy_indices_for_net1[relabel_candidates_mask.numpy()]
            relabel_new_labels_net2 = pred_labels[relabel_candidates_mask].numpy()
            relabel_map2 = dict(zip(relabel_indices_net2, relabel_new_labels_net2))
            
            if len(relabel_indices_net2) > 0:
                true_labels_for_net1 = true_labels[relabel_indices_net2]
                correct_relabels = (true_labels_for_net1 == relabel_new_labels_net2).sum()
                relabel_acc_net1 = (correct_relabels / len(relabel_indices_net2)) * 100 
                print(f"Net1 will train with {len(relabel_map2)} relabeled samples from Net2 (Acc: {relabel_acc_net1:.2f}%)")
            else:
                print(f"Net1 will train with 0 relabeled samples from Net2.")
        else:
            print(f"Net1 will train with 0 relabeled samples from Net2.")

        # Net1 rectifies the samples Net2 will treat as noisy
        noisy_indices_for_net2 = np.where(pred1_hard == False)[0]
        relabel_map1 = {}
        relabel_acc_net2 = 0.0 
        
        if len(noisy_indices_for_net2) > 0:
            noisy_logits_for_net2 = logits1[noisy_indices_for_net2]
            max_probs, pred_labels = torch.max(torch.softmax(noisy_logits_for_net2, dim=1), dim=1)
            original_noisy_labels_subset = torch.tensor(noisy_labels)[noisy_indices_for_net2]
            relabel_candidates_mask = (max_probs > args.relabel_threshold) & (pred_labels != original_noisy_labels_subset)
            
            relabel_indices_net1 = noisy_indices_for_net2[relabel_candidates_mask.numpy()]
            relabel_new_labels_net1 = pred_labels[relabel_candidates_mask].numpy()
            relabel_map1 = dict(zip(relabel_indices_net1, relabel_new_labels_net1))
            
            if len(relabel_indices_net1) > 0:
                true_labels_for_net2 = true_labels[relabel_indices_net1]
                correct_relabels = (true_labels_for_net2 == relabel_new_labels_net1).sum()
                relabel_acc_net2 = (correct_relabels / len(relabel_indices_net1)) * 100 
                print(f"Net2 will train with {len(relabel_map1)} relabeled samples from Net1 (Acc: {relabel_acc_net2:.2f}%)")
            else:
                print(f"Net2 will train with 0 relabeled samples from Net1.")
        
        
        # --- 4. Final partition; rectified samples join the labeled set ---
        pred1_final = (w_sampled_1 > args.p_threshold).astype(bool)
        pred2_final = (w_sampled_2 > args.p_threshold).astype(bool)

        if len(relabel_map1) > 0:
            relabel_idxs_1 = list(relabel_map1.keys())
            pred1_final[relabel_idxs_1] = True 
            # Uncertainty-aware weight w = 1 - H: the more uncertain a rectified
            # sample is, the less it counts as labeled data in MixMatch.
            w_sampled_1[relabel_idxs_1] = 1.0 - H1[relabel_idxs_1]

        if len(relabel_map2) > 0:
            relabel_idxs_2 = list(relabel_map2.keys())
            pred2_final[relabel_idxs_2] = True 
            # Uncertainty-aware weight w = 1 - H
            w_sampled_2[relabel_idxs_2] = 1.0 - H2[relabel_idxs_2]
            
        
        # --- 5. Co-training on the partitions produced by the peer network ---
        
        print(f'\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2_final, 
                                                                sampled_w=w_sampled_2, 
                                                                H_values=H2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, relabel_map=relabel_map2)

        print(f'\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1_final, 
                                                                sampled_w=w_sampled_1, 
                                                                H_values=H1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, relabel_map=relabel_map1)

        # --- 6. Save per-epoch statistics ---
        stats_dict = {
            'epoch': epoch,
            'true_labels': true_labels,
            'noisy_labels': noisy_labels,
            'net1_gmm_prob': prob1,
            'net1_H': H1,
            'net1_K': K1,
            'net1_beta_alpha': alpha1,
            'net1_beta_beta': beta1,
            'net1_w_sampled': w_sampled_1,
            'net1_relabel_count': len(relabel_map1), 
            'net1_relabel_acc': relabel_acc_net2,
            'net2_gmm_prob': prob2,
            'net2_H': H2,
            'net2_K': K2,
            'net2_beta_alpha': alpha2,
            'net2_beta_beta': beta2,
            'net2_w_sampled': w_sampled_2,
            'net2_relabel_count': len(relabel_map2),
            'net2_relabel_acc': relabel_acc_net1,
        }
        
        save_path = os.path.join(stats_dir, f'epoch_{epoch:03d}.npz')
        np.savez(save_path, **stats_dict) 
        print(f"\n| Saved epoch {epoch} stats to {save_path}")

    test(epoch,net1,net2)