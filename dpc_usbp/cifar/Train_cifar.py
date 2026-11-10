"""DPC + USBP on CIFAR-10/100.

Adds the three USBP modules (stochastic Beta partitioning, confidence-gated
cross-network rectification with weight 1 - H, uncertainty-inverse adaptive
augmentation) to DPC (Dirichlet-based prediction calibration, AAAI 2024).
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
from loss import EDL_Loss


parser = argparse.ArgumentParser(description="PyTorch CIFAR Training (DPC + USBP)")
parser.add_argument("--model_name", default="resnet18", type=str, help="model backbone to use")
parser.add_argument("--batch_size", default=64, type=int, help="train batchsize")
parser.add_argument("--lr", "--learning_rate", default=0.02, type=float, help="initial learning rate")
parser.add_argument("--noise_mode", default="sym")
parser.add_argument("--alpha", default=4, type=float, help="parameter for MixMatch Beta")
parser.add_argument("--lambda_u", default=0, type=float, help="weight for unsupervised loss")
parser.add_argument("--p_threshold", default=0.5, type=float, help="clean probability threshold")
parser.add_argument("--T", default=0.5, type=float, help="sharpening temperature")
parser.add_argument("--num_epochs", default=300, type=int)
parser.add_argument("--r", default=0.8, type=float, help="noise ratio")
parser.add_argument("--id", default="dpc_usbp")
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--gpuid", default=0, type=int)
parser.add_argument("--num_class", default=10, type=int)
parser.add_argument("--data_path", default="./dataset/cifar-10", type=str, help="path to dataset")
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--dropout_p", default=0.6, type=float, help="dropout probability for MC-Dropout")
parser.add_argument("--K_mcdropout", default=10, type=int, help="number of forward passes for MC-Dropout")
parser.add_argument("--K_max", default=5.0, type=float, help="max concentration parameter for Beta distribution")
parser.add_argument("--relabel_threshold", default=0.9, type=float, help="confidence threshold for relabeling")
parser.add_argument("--max_magnitude", default=20, type=int, help="max RandAugment magnitude for UIAA")
args = parser.parse_args()


torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def conv_p(logits):
    alpha_t = torch.exp(logits) + 10.0 / args.num_class
    total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
    return alpha_t / (total_alpha_t + 1e-12)


def normalize_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def get_mcdropout_uncertainty(model, inputs):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_probs = []
    with torch.no_grad():
        for _ in range(args.K_mcdropout):
            logits_main, _ = model(inputs, use_dropout=True)
            all_probs.append(conv_p(logits_main))

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    return entropy


def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, relabel_map=None):
    net.train()
    net2.eval()

    if relabel_map is None:
        relabel_map = {}

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x_sampled, indices_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        for i, idx in enumerate(indices_x):
            idx_val = int(idx.item())
            if idx_val in relabel_map:
                labels_x[i] = int(relabel_map[idx_val])

        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x_sampled = w_x_sampled.view(-1, 1).float()

        inputs_x = inputs_x.cuda()
        inputs_x2 = inputs_x2.cuda()
        labels_x = labels_x.cuda()
        w_x_sampled = w_x_sampled.cuda()
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            outputs_u11, _ = net(inputs_u, use_dropout=False)
            outputs_u12, _ = net(inputs_u2, use_dropout=False)
            outputs_u21, _ = net2(inputs_u, use_dropout=False)
            outputs_u22, _ = net2(inputs_u2, use_dropout=False)

            pu = (conv_p(outputs_u11) + conv_p(outputs_u12) + conv_p(outputs_u21) + conv_p(outputs_u22)) / 4
            ptu = pu ** (1.0 / args.T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            outputs_x, _ = net(inputs_x, use_dropout=False)
            outputs_x2, _ = net(inputs_x2, use_dropout=False)
            px = (conv_p(outputs_x) + conv_p(outputs_x2)) / 2
            px = w_x_sampled * labels_x + (1.0 - w_x_sampled) * px
            ptx = px ** (1.0 / args.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        mix_l = np.random.beta(args.alpha, args.alpha)
        mix_l = max(mix_l, 1 - mix_l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = mix_l * input_a + (1 - mix_l) * input_b
        mixed_target = mix_l * target_a + (1 - mix_l) * target_b

        logits_main, logits_aux = net(mixed_input, use_dropout=False)
        logits_x = logits_main[: batch_size * 2]
        logits_u = logits_aux[batch_size * 2 :]

        Lx, Lu, lamb = criterion(
            logits_x,
            mix_l,
            target_a[: batch_size * 2].argmax(1),
            target_b[: batch_size * 2].argmax(1),
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            warm_up,
        )

        prior = torch.ones(args.num_class).cuda() / args.num_class
        pred_mean = conv_p(logits_main).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write("\r")
        sys.stdout.write(
            "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f"
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                Lx.item(),
                Lu.item(),
            )
        )
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader_):
    net.train()
    num_iter = (len(dataloader_.dataset) // dataloader_.batch_size) + 1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader_):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs, _ = net(inputs, use_dropout=False)
        loss1, loss2 = EDL_Loss(args.num_class)(outputs, labels)
        loss = loss1.mean() + loss2.mean()

        if args.noise_mode == "asym":
            loss = loss + conf_penalty(outputs)

        loss.backward()
        optimizer.step()

        sys.stdout.write("\r")
        sys.stdout.write(
            "%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f"
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                loss.item(),
            )
        )
        sys.stdout.flush()


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs, use_dropout=False)
            outputs2, _ = net2(inputs, use_dropout=False)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.0 * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write("Epoch:%d   Accuracy:%.2f\n" % (epoch, acc))
    test_log.flush()


def eval_train_and_get_uncertainty(model):
    model.eval()
    num_samples = len(eval_loader.dataset)

    margins = torch.zeros(num_samples)
    uncertainties = torch.zeros(num_samples)
    all_logits = torch.zeros(num_samples, args.num_class)
    all_noisy_labels = torch.zeros(num_samples, dtype=torch.long)

    with torch.no_grad():
        for _, (inputs, noisy_targets, index, _) in enumerate(eval_loader):
            inputs = inputs.cuda()
            noisy_targets_cuda = noisy_targets.cuda()

            outputs, _ = model(inputs, use_dropout=False)
            uncertainty = get_mcdropout_uncertainty(model, inputs)

            evidence_pos = outputs.gather(1, noisy_targets_cuda.view(-1, 1)).squeeze(1)
            copy_outputs = outputs.clone()
            copy_outputs.scatter_(1, noisy_targets_cuda.view(-1, 1), -1e5)
            evidence_neg = copy_outputs.max(dim=1).values
            margin_batch = evidence_pos - evidence_neg

            for b in range(inputs.size(0)):
                idx = int(index[b].item())
                margins[idx] = margin_batch[b].detach().cpu()
                uncertainties[idx] = uncertainty[b].detach().cpu()
                all_logits[idx] = outputs[b].detach().cpu()
                all_noisy_labels[idx] = int(noisy_targets[b].item())

    margins_norm = normalize_01(margins)
    uncertainties_norm = normalize_01(uncertainties)

    input_margin = margins_norm.numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_margin)
    prob = gmm.predict_proba(input_margin)
    prob_clean = prob[:, gmm.means_.argmax()]

    return (
        prob_clean,
        uncertainties_norm.numpy(),
        all_logits,
        all_noisy_labels.numpy(),
    )


def linear_rampup(current, warm_up_epoch, rampup_length=16):
    current = np.clip((current - warm_up_epoch) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, mix_l, targets_x1, targets_x2, outputs_u, targets_u, epoch, warm_up_epoch):
        probs_u = conv_p(outputs_u)

        Lx11, Lx12 = EDL_Loss(args.num_class)(outputs_x, targets_x1)
        Lx21, Lx22 = EDL_Loss(args.num_class)(outputs_x, targets_x2)
        Lx = (mix_l * Lx11 + (1 - mix_l) * Lx21).mean() + (mix_l * Lx12 + (1 - mix_l) * Lx22).mean()

        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up_epoch)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = conv_p(outputs)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    if args.model_name == "resnet18":
        model = ResNet18(num_classes=args.num_class, dropout_p=args.dropout_p)
    elif args.model_name == "resnet34":
        model = ResNet34(num_classes=args.num_class, dropout_p=args.dropout_p)
    else:
        raise ValueError("Unsupported model_name: %s" % args.model_name)
    return model.cuda()


os.makedirs("./checkpoint", exist_ok=True)
stats_log = open("./checkpoint/%s_%.1f_%s_usbp_stats.txt" % (args.dataset, args.r, args.noise_mode), "w")
test_log = open("./checkpoint/%s_%.1f_%s_usbp_acc.txt" % (args.dataset, args.r, args.noise_mode), "w")

if args.dataset == "cifar10":
    warm_up = 10
elif args.dataset == "cifar100":
    warm_up = 30
else:
    raise ValueError("Unsupported dataset: %s" % args.dataset)

loader = dataloader.cifar_dataloader(
    args.dataset,
    r=args.r,
    noise_mode=args.noise_mode,
    batch_size=args.batch_size,
    num_workers=5,
    root_dir=args.data_path,
    log=stats_log,
    noise_file="%s/%.1f_%s.json" % (args.data_path, args.r, args.noise_mode),
    max_magnitude=args.max_magnitude,
)

print("| Building net")
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.noise_mode == "asym":
    conf_penalty = NegEntropy()

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 10

    for param_group in optimizer1.param_groups:
        param_group["lr"] = lr
    for param_group in optimizer2.param_groups:
        param_group["lr"] = lr

    test_loader = loader.run("test")
    eval_loader = loader.run("eval_train")

    if epoch < warm_up:
        warmup_trainloader = loader.run("warmup")
        print("Warmup Net1")
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print("\nWarmup Net2")
        warmup(epoch, net2, optimizer2, warmup_trainloader)
    else:
        prob1, H1, logits1, noisy_labels = eval_train_and_get_uncertainty(net1)
        prob2, H2, logits2, _ = eval_train_and_get_uncertainty(net2)

        K1 = args.K_max / (1.0 + H1)
        K2 = args.K_max / (1.0 + H2)
        alpha1 = K1 * prob1 + 1e-10
        beta1 = K1 * (1.0 - prob1) + 1e-10
        alpha2 = K2 * prob2 + 1e-10
        beta2 = K2 * (1.0 - prob2) + 1e-10

        prob1_t = torch.from_numpy(prob1).float().cuda()
        K1_t = torch.from_numpy(K1).float().cuda()
        prob2_t = torch.from_numpy(prob2).float().cuda()
        K2_t = torch.from_numpy(K2).float().cuda()

        beta_dist1 = torch.distributions.beta.Beta(K1_t * prob1_t + 1e-10, K1_t * (1.0 - prob1_t) + 1e-10)
        beta_dist2 = torch.distributions.beta.Beta(K2_t * prob2_t + 1e-10, K2_t * (1.0 - prob2_t) + 1e-10)
        w_sampled_1 = beta_dist1.sample().cpu().numpy()
        w_sampled_2 = beta_dist2.sample().cpu().numpy()

        pred1_hard = w_sampled_1 > args.p_threshold
        pred2_hard = w_sampled_2 > args.p_threshold

        relabel_map2 = {}
        noisy_indices_for_net1 = np.where(~pred2_hard)[0]
        if len(noisy_indices_for_net1) > 0:
            noisy_logits = logits2[noisy_indices_for_net1]
            probs = conv_p(noisy_logits)
            max_probs, pred_labels = torch.max(probs, dim=1)
            original_noisy = torch.from_numpy(noisy_labels[noisy_indices_for_net1]).long()
            relabel_mask = (max_probs > args.relabel_threshold) & (pred_labels != original_noisy)

            relabel_indices = noisy_indices_for_net1[relabel_mask.numpy()]
            relabel_labels = pred_labels[relabel_mask].cpu().numpy()
            relabel_map2 = {int(idx): int(lbl) for idx, lbl in zip(relabel_indices, relabel_labels)}

        relabel_map1 = {}
        noisy_indices_for_net2 = np.where(~pred1_hard)[0]
        if len(noisy_indices_for_net2) > 0:
            noisy_logits = logits1[noisy_indices_for_net2]
            probs = conv_p(noisy_logits)
            max_probs, pred_labels = torch.max(probs, dim=1)
            original_noisy = torch.from_numpy(noisy_labels[noisy_indices_for_net2]).long()
            relabel_mask = (max_probs > args.relabel_threshold) & (pred_labels != original_noisy)

            relabel_indices = noisy_indices_for_net2[relabel_mask.numpy()]
            relabel_labels = pred_labels[relabel_mask].cpu().numpy()
            relabel_map1 = {int(idx): int(lbl) for idx, lbl in zip(relabel_indices, relabel_labels)}

        pred1_final = pred1_hard.copy()
        pred2_final = pred2_hard.copy()

        if len(relabel_map1) > 0:
            relabel_idxs = np.array(list(relabel_map1.keys()), dtype=int)
            pred1_final[relabel_idxs] = True
            w_sampled_1[relabel_idxs] = 1.0 - H1[relabel_idxs]

        if len(relabel_map2) > 0:
            relabel_idxs = np.array(list(relabel_map2.keys()), dtype=int)
            pred2_final[relabel_idxs] = True
            w_sampled_2[relabel_idxs] = 1.0 - H2[relabel_idxs]

        print("Train Net1")
        labeled_trainloader, unlabeled_trainloader = loader.run(
            "train", pred=pred2_final, sampled_w=w_sampled_2, H_values=H2
        )
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, relabel_map=relabel_map2)

        print("\nTrain Net2")
        labeled_trainloader, unlabeled_trainloader = loader.run(
            "train", pred=pred1_final, sampled_w=w_sampled_1, H_values=H1
        )
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, relabel_map=relabel_map1)

    test(epoch, net1, net2)
