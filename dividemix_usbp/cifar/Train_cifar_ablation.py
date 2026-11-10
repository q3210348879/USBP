"""
USBP Ablation Study Script for CIFAR-100 sym 0.9
=================================================
Controls each module via command-line flags:
  --enable_beta     : Enable stochastic Beta partitioning (USBP core)
  --enable_rect     : Enable cross-network rectification
  --enable_uiaa     : Enable uncertainty-inverse adaptive augmentation

Ablation configurations:
  ② USBP Only:        --enable_beta
  ③ USBP + Rect:      --enable_beta --enable_rect
  ④ USBP + UIAA:      --enable_beta --enable_uiaa
  ⑤ UIAA + Rect:      --enable_rect --enable_uiaa
  ⑥ Full:             --enable_beta --enable_rect --enable_uiaa

Example:
  # USBP Only on CIFAR-100 sym 0.9
  python Train_cifar_ablation.py --dataset cifar100 --num_class 100 --r 0.9 \
      --noise_mode sym --data_path ./cifar-100-python --enable_beta

  # UIAA + Rect (no Beta) on CIFAR-100 sym 0.9
  python Train_cifar_ablation.py --dataset cifar100 --num_class 100 --r 0.9 \
      --noise_mode sym --data_path ./cifar-100-python --enable_rect --enable_uiaa
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
parser = argparse.ArgumentParser(description='USBP Ablation Study')
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float)
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta in MixMatch')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=360, type=int)
parser.add_argument('--r', default=0.9, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='./cifar-100-python', type=str)
parser.add_argument('--dataset', default='cifar100', type=str)

# USBP hyperparameters
parser.add_argument('--dropout_p', default=0.6, type=float)
parser.add_argument('--K_mcdropout', default=10, type=int)
parser.add_argument('--K_max', default=5.0, type=float)
parser.add_argument('--relabel_threshold', default=0.9, type=float)

# === Ablation flags ===
parser.add_argument('--enable_beta', action='store_true', help='Enable stochastic Beta partitioning')
parser.add_argument('--enable_rect', action='store_true', help='Enable cross-network rectification')
parser.add_argument('--enable_uiaa', action='store_true', help='Enable uncertainty-inverse adaptive augmentation')

args = parser.parse_args()

# Derive ablation tag for logging
modules = []
if args.enable_beta: modules.append('Beta')
if args.enable_rect: modules.append('Rect')
if args.enable_uiaa: modules.append('UIAA')
ablation_tag = '+'.join(modules) if modules else 'Baseline'
print(f"\n{'='*60}")
print(f"  ABLATION CONFIG: {ablation_tag}")
print(f"  Beta={args.enable_beta}  Rect={args.enable_rect}  UIAA={args.enable_uiaa}")
print(f"  Dataset={args.dataset}  Noise={args.noise_mode} r={args.r}")
print(f"{'='*60}\n")

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# --- train function ---
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, relabel_map={}):
    net.train()
    net2.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x_sampled, indices_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)

        # Apply relabeling (only if rectification is enabled)
        if args.enable_rect:
            for i, idx in enumerate(indices_x):
                if idx.item() in relabel_map:
                    new_label = relabel_map[idx.item()]
                    labels_x[i] = new_label

        labels_x_onehot = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x_sampled = w_x_sampled.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x_onehot, w_x_sampled = inputs_x.cuda(), inputs_x2.cuda(), labels_x_onehot.cuda(), w_x_sampled.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs_u11, outputs_u12 = net(inputs_u), net(inputs_u2)
            outputs_u21, outputs_u22 = net2(inputs_u, use_dropout=False), net2(inputs_u2, use_dropout=False)
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            outputs_x, outputs_x2 = net(inputs_x), net(inputs_x2)
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x_sampled * labels_x_onehot + (1 - w_x_sampled) * px
            ptx = px ** (1 / args.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        loss = Lx + lamb * Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s [%s] | Epoch [%3d/%3d] Iter[%3d/%3d]\t Lx: %.2f  Lu: %.2f'
                         % (args.dataset, args.r, args.noise_mode, ablation_tag,
                            epoch, args.num_epochs, batch_idx + 1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs, use_dropout=False)
        loss = CEloss(outputs, labels)
        if args.noise_mode == 'asym':
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode == 'sym':
            L = loss
        L.backward()
        optimizer.step()
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs,
                            batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs, use_dropout=False)
            outputs2 = net2(inputs, use_dropout=False)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()
    return acc


def get_mcdropout_uncertainty(model, inputs):
    """MC-Dropout uncertainty estimation. Returns per-sample entropy."""
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

            # Only compute MC-Dropout uncertainty if Beta or UIAA or Rect needs it
            if args.enable_beta or args.enable_uiaa or args.enable_rect:
                uncertainty = get_mcdropout_uncertainty(model, inputs)
            else:
                uncertainty = torch.zeros(inputs.size(0))

            for b in range(inputs.size(0)):
                idx = index[b].item()
                losses[idx] = loss[b]
                all_logits[idx] = outputs[b]
                all_true_labels[idx] = true_targets[b]
                all_noisy_labels[idx] = noisy_targets[b]
                uncertainties[idx] = uncertainty[b]

    losses_normalized = (losses - losses.min()) / (losses.max() - losses.min() + 1e-10)
    uncertainties_normalized = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-10)
    all_loss.append(losses_normalized)

    if args.r == 0.9:
        history = torch.stack(all_loss)
        input_loss = history[-min(5, len(history)):].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses_normalized.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss.cpu().numpy())
    prob = gmm.predict_proba(input_loss.cpu().numpy())
    prob = prob[:, gmm.means_.argmin()]

    return prob, uncertainties_normalized.numpy(), all_logits.cpu(), all_loss, losses.numpy(), all_true_labels.numpy(), all_noisy_labels.numpy()


# --- Utility classes ---
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    if args.model_name == 'resnet18':
        local_model = ResNet18(num_classes=args.num_class, dropout_p=args.dropout_p)
    elif args.model_name == 'resnet34':
        local_model = ResNet34(num_classes=args.num_class, dropout_p=args.dropout_p)
    else:
        raise ValueError(f"Model '{args.model_name}' is not defined.")
    print(f"| Building {args.model_name} with dropout p={args.dropout_p}")
    return local_model.cuda()


# --- Main ---
log_name = f'{args.dataset}_{args.r}_{args.noise_mode}_ablation_{ablation_tag}'
stats_log = open(f'./checkpoint/{log_name}_stats.txt', 'w')
test_log = open(f'./checkpoint/{log_name}_acc.txt', 'w')

# Write config header
test_log.write(f'# Ablation: {ablation_tag}\n')
test_log.write(f'# Beta={args.enable_beta} Rect={args.enable_rect} UIAA={args.enable_uiaa}\n')
test_log.flush()

if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(
    args.dataset, r=args.r, noise_mode=args.noise_mode,
    batch_size=args.batch_size, num_workers=5,
    root_dir=args.data_path, log=stats_log,
    noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()
all_loss = [[], []]

best_acc = 0.0

for epoch in range(args.num_epochs + 1):
    lr = args.lr
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
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)
    else:
        # --- 1. Evaluate and get uncertainty ---
        prob1, H1, logits1, all_loss[0], losses1, true_labels, noisy_labels = eval_train_and_get_uncertainty(net1, all_loss[0])
        prob2, H2, logits2, all_loss[1], losses2, _, _ = eval_train_and_get_uncertainty(net2, all_loss[1])

        # --- 2. Partitioning: Beta vs Deterministic ---
        if args.enable_beta:
            # Stochastic Beta partitioning (USBP core)
            K1 = args.K_max / (1 + H1)
            K2 = args.K_max / (1 + H2)

            prob1_t = torch.from_numpy(prob1).float().cuda()
            K1_t = torch.from_numpy(K1).float().cuda()
            prob2_t = torch.from_numpy(prob2).float().cuda()
            K2_t = torch.from_numpy(K2).float().cuda()

            beta_dist1 = torch.distributions.beta.Beta(K1_t * prob1_t + 1e-10, K1_t * (1 - prob1_t) + 1e-10)
            w_sampled_1 = beta_dist1.sample().cpu().numpy()

            beta_dist2 = torch.distributions.beta.Beta(K2_t * prob2_t + 1e-10, K2_t * (1 - prob2_t) + 1e-10)
            w_sampled_2 = beta_dist2.sample().cpu().numpy()

            if epoch == warm_up:
                print(f"\n>> [Beta] Stochastic Beta Sampling ENABLED")
        else:
            # Deterministic: use GMM posterior directly (DivideMix default)
            w_sampled_1 = prob1.copy()
            w_sampled_2 = prob2.copy()

            if epoch == warm_up:
                print(f"\n>> [Beta] DISABLED — using deterministic GMM posterior")

        pred1_hard = (w_sampled_1 > args.p_threshold)
        pred2_hard = (w_sampled_2 > args.p_threshold)

        # --- 3. Rectification ---
        relabel_map1 = {}
        relabel_map2 = {}

        if args.enable_rect:
            # Net2 prepares relabel_map for Net1
            noisy_indices_for_net1 = np.where(pred2_hard == False)[0]
            if len(noisy_indices_for_net1) > 0:
                noisy_logits = logits2[noisy_indices_for_net1]
                max_probs, pred_labels = torch.max(torch.softmax(noisy_logits, dim=1), dim=1)
                original_noisy = torch.tensor(noisy_labels)[noisy_indices_for_net1]
                candidates = (max_probs > args.relabel_threshold) & (pred_labels != original_noisy)
                relabel_idx = noisy_indices_for_net1[candidates.numpy()]
                relabel_labels = pred_labels[candidates].numpy()
                relabel_map2 = dict(zip(relabel_idx, relabel_labels))

                if len(relabel_idx) > 0:
                    correct = (true_labels[relabel_idx] == relabel_labels).sum()
                    acc = correct / len(relabel_idx) * 100
                    print(f"  [Rect] Net1 gets {len(relabel_map2)} relabeled samples (Acc: {acc:.1f}%)")

            # Net1 prepares relabel_map for Net2
            noisy_indices_for_net2 = np.where(pred1_hard == False)[0]
            if len(noisy_indices_for_net2) > 0:
                noisy_logits = logits1[noisy_indices_for_net2]
                max_probs, pred_labels = torch.max(torch.softmax(noisy_logits, dim=1), dim=1)
                original_noisy = torch.tensor(noisy_labels)[noisy_indices_for_net2]
                candidates = (max_probs > args.relabel_threshold) & (pred_labels != original_noisy)
                relabel_idx = noisy_indices_for_net2[candidates.numpy()]
                relabel_labels = pred_labels[candidates].numpy()
                relabel_map1 = dict(zip(relabel_idx, relabel_labels))

                if len(relabel_idx) > 0:
                    correct = (true_labels[relabel_idx] == relabel_labels).sum()
                    acc = correct / len(relabel_idx) * 100
                    print(f"  [Rect] Net2 gets {len(relabel_map1)} relabeled samples (Acc: {acc:.1f}%)")
        else:
            if epoch == warm_up:
                print(f"  [Rect] DISABLED")

        # --- 4. Final partition with relabel integration ---
        pred1_final = pred1_hard.copy()
        pred2_final = pred2_hard.copy()

        if args.enable_rect and len(relabel_map1) > 0:
            relabel_idxs_1 = list(relabel_map1.keys())
            pred1_final[relabel_idxs_1] = True
            # Uncertainty-aware weight for rectified samples
            w_sampled_1[relabel_idxs_1] = 1.0 - H1[relabel_idxs_1]

        if args.enable_rect and len(relabel_map2) > 0:
            relabel_idxs_2 = list(relabel_map2.keys())
            pred2_final[relabel_idxs_2] = True
            w_sampled_2[relabel_idxs_2] = 1.0 - H2[relabel_idxs_2]

        # --- 5. UIAA: pass H_values only if enabled ---
        H_for_loader1 = H1 if args.enable_uiaa else None
        H_for_loader2 = H2 if args.enable_uiaa else None

        if epoch == warm_up and not args.enable_uiaa:
            print(f"  [UIAA] DISABLED — using fixed augmentation")
        elif epoch == warm_up and args.enable_uiaa:
            print(f"  [UIAA] ENABLED — adaptive augmentation by uncertainty")

        # --- 6. Train ---
        print(f'\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run(
            'train', pred2_final, sampled_w=w_sampled_2, H_values=H_for_loader2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader,
              relabel_map=relabel_map2)

        print(f'\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(
            'train', pred1_final, sampled_w=w_sampled_1, H_values=H_for_loader1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader,
              relabel_map=relabel_map1)

    acc = test(epoch, net1, net2)
    if acc > best_acc:
        best_acc = acc

print(f"\n{'='*60}")
print(f"  ABLATION: {ablation_tag}")
print(f"  BEST TEST ACCURACY: {best_acc:.2f}%")
print(f"{'='*60}")

# Save best accuracy to a summary file
with open('./checkpoint/ablation_results.txt', 'a') as f:
    f.write(f'{ablation_tag}\t{args.dataset}\t{args.r}\t{args.noise_mode}\t{best_acc:.2f}\n')
