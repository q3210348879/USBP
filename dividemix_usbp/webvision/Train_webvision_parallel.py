"""DivideMix + USBP for WebVision (mini-WebVision, 50 classes), dual-GPU.

USBP modifications over DivideMix-parallel:
  (i)  stochastic Beta partitioning replaces hard threshold
  (ii) confidence-gated cross-network rectification with weight w = 1 - H_tilde
  (iii) UIAA: dynamic-magnitude RandAugment on the second view (in dataloader)

Each network runs on its own GPU (cuda1 / cuda2). Eval phase computes both
GMM-on-loss probability and MC-Dropout uncertainty H_tilde (K_mcdropout passes).
Logits are passed back through the queue so that the rectification step can
use the peer network's predictions on the rejected samples.
"""
from __future__ import print_function
import sys
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
# Avoid /dev/shm bus errors in containers with small shared memory
mp.set_sharing_strategy('file_system')

from sklearn.mixture import GaussianMixture
import torchnet

from InceptionResNetV2 import InceptionResNetV2
import dataloader_webvision as dataloader


# ----------------------------- args -----------------------------
parser = argparse.ArgumentParser(description='DivideMix+USBP WebVision (dual-GPU)')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float)
parser.add_argument('--alpha', default=0.5, type=float, help='MixMatch Beta param')
parser.add_argument('--lambda_u', default=0, type=float)
parser.add_argument('--p_threshold', default=0.5, type=float)
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='', type=str)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid1', default=0, type=int)
parser.add_argument('--gpuid2', default=1, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./webvision/', type=str)
# USBP-specific
parser.add_argument('--dropout_p', default=0.3, type=float, help='MC-Dropout rate')
parser.add_argument('--K_mcdropout', default=10, type=int, help='MC-Dropout forward passes')
parser.add_argument('--K_max', default=5.0, type=float, help='Beta concentration cap')
parser.add_argument('--relabel_threshold', default=0.9, type=float,
                    help='peer-net confidence gate for rectification')
parser.add_argument('--stats_dir', default='./stats_webvision', type=str)
parser.add_argument('--num_workers', default=4, type=int,
                    help='DataLoader workers per loader; lower if /dev/shm is small')
parser.add_argument('--warm_up', default=3, type=int,
                    help='warmup epochs (DivideMix used 1; USBP needs 3-5 for reliable H_tilde)')
parser.add_argument('--usbp_start', default=5, type=int,
                    help='epoch to fully turn on USBP (rectification + soft Beta)')
parser.add_argument('--pin_memory', action='store_true',
                    help='enable pin_memory in DataLoader (disable if shm is tight)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '%s,%s' % (args.gpuid1, args.gpuid2)
random.seed(args.seed)
cuda1 = torch.device('cuda:0')
cuda2 = torch.device('cuda:1')


# ----------------------------- losses ---------------------------
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


# ----------------------------- workers --------------------------
def train(epoch, net, net2, optimizer, labeled_loader, unlabeled_loader,
          device, whichnet, relabel_map=None):
    """USBP-aware training. relabel_map: {global_index: corrected_label} from peer."""
    if relabel_map is None:
        relabel_map = {}
    criterion = SemiLoss()
    net.train()
    net2.eval()

    unlabeled_iter = iter(unlabeled_loader)
    num_iter = (len(labeled_loader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x_sampled, indices_x) in enumerate(labeled_loader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            inputs_u, inputs_u2 = next(unlabeled_iter)
        batch_size = inputs_x.size(0)

        # Apply rectification: replace label for rectified samples
        for i, idx in enumerate(indices_x):
            idx_int = int(idx.item())
            if idx_int in relabel_map:
                labels_x[i] = int(relabel_map[idx_int])

        labels_x_onehot = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1)
        w_x = w_x_sampled.view(-1, 1).type(torch.FloatTensor)

        inputs_x = inputs_x.to(device, non_blocking=True)
        inputs_x2 = inputs_x2.to(device, non_blocking=True)
        labels_x_onehot = labels_x_onehot.to(device, non_blocking=True)
        w_x = w_x.to(device, non_blocking=True)
        inputs_u = inputs_u.to(device, non_blocking=True)
        inputs_u2 = inputs_u2.to(device, non_blocking=True)

        with torch.no_grad():
            # co-guessing on unlabeled
            outputs_u11 = net(inputs_u, use_dropout=False)
            outputs_u12 = net(inputs_u2, use_dropout=False)
            outputs_u21 = net2(inputs_u, use_dropout=False)
            outputs_u22 = net2(inputs_u2, use_dropout=False)
            pu = (torch.softmax(outputs_u11, 1) + torch.softmax(outputs_u12, 1) +
                  torch.softmax(outputs_u21, 1) + torch.softmax(outputs_u22, 1)) / 4
            ptu = pu ** (1 / args.T)
            targets_u = (ptu / ptu.sum(dim=1, keepdim=True)).detach()

            # label refinement on labeled
            outputs_x = net(inputs_x, use_dropout=False)
            outputs_x2 = net(inputs_x2, use_dropout=False)
            px = (torch.softmax(outputs_x, 1) + torch.softmax(outputs_x2, 1)) / 2
            px = w_x * labels_x_onehot + (1 - w_x) * px
            ptx = px ** (1 / args.T)
            targets_x = (ptx / ptx.sum(dim=1, keepdim=True)).detach()

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
        mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]

        logits = net(mixed_input)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # uniform-prior penalty
        prior = (torch.ones(args.num_class) / args.num_class).to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            sys.stdout.write('\n%s |%s Epoch [%3d/%3d] Iter[%4d/%4d]\t Lx: %.3f' %
                             (args.id, whichnet, epoch, args.num_epochs,
                              batch_idx + 1, num_iter, Lx.item()))
            sys.stdout.flush()


def warmup(epoch, net, optimizer, loader, device, whichnet):
    CEloss = nn.CrossEntropyLoss()
    net.train()
    num_iter = (len(loader.dataset) // loader.batch_size) + 1
    for batch_idx, (inputs, labels, _idx) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs, use_dropout=False)
        loss = CEloss(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            sys.stdout.write('\n%s |%s Warmup [%3d/%3d] Iter[%4d/%4d]\t CE: %.3f' %
                             (args.id, whichnet, epoch, args.num_epochs,
                              batch_idx + 1, num_iter, loss.item()))
            sys.stdout.flush()


def test(epoch, net1, net2, test_loader, device, queue):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    acc_meter.reset()
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device, non_blocking=True)
            outputs1 = net1(inputs, use_dropout=False)
            outputs2 = net2(inputs, use_dropout=False)
            outputs = outputs1 + outputs2
            acc_meter.add(outputs, targets)
    queue.put(acc_meter.value())


def get_mcdropout_uncertainty(model, inputs, K_mc):
    """K_mc stochastic forward passes; return per-sample predictive entropy.
    Accumulates mean in-place to avoid stacking K_mc tensors (shm/RAM friendly)."""
    model.eval()
    # turn on dropout layers
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    mean_probs = None
    with torch.no_grad():
        for _ in range(K_mc):
            outputs = model(inputs, use_dropout=True)
            p = torch.softmax(outputs, dim=1)
            mean_probs = p if mean_probs is None else mean_probs + p
        mean_probs = mean_probs / K_mc
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    return entropy


def eval_train_usbp(eval_loader, model, device, whichnet, queue, K_mc):
    """USBP eval: returns (prob_GMM, H_tilde_normalised, logits, noisy_labels)."""
    CE = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    N = len(eval_loader.dataset)
    losses = torch.zeros(N)
    uncertainties = torch.zeros(N)
    all_logits = torch.zeros(N, args.num_class)
    all_noisy_labels = torch.zeros(N, dtype=torch.long)

    num_iter = (N // eval_loader.batch_size) + 1
    with torch.no_grad():
        for batch_idx, (inputs, targets, index, _true) in enumerate(eval_loader):
            inputs = inputs.to(device)
            targets_cuda = targets.to(device, non_blocking=True)
            outputs = model(inputs, use_dropout=False)
            loss = CE(outputs, targets_cuda)
            uncertainty = get_mcdropout_uncertainty(model, inputs, K_mc)
            for b in range(inputs.size(0)):
                idx = int(index[b].item())
                losses[idx] = loss[b].cpu()
                all_logits[idx] = outputs[b].cpu()
                all_noisy_labels[idx] = int(targets[b].item())
                uncertainties[idx] = uncertainty[b].cpu()
            if batch_idx % 20 == 0:
                sys.stdout.write('\n|%s Eval+UncEst [%3d/%3d]' %
                                 (whichnet, batch_idx, num_iter))
                sys.stdout.flush()

    losses_n = (losses - losses.min()) / (losses.max() - losses.min() + 1e-10)
    H_n = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-10)

    # GMM on normalised loss
    input_loss = losses_n.numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=1e-3)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    queue.put({
        'prob': prob.astype(np.float32),
        'H': H_n.numpy().astype(np.float32),
        'logits': all_logits.numpy().astype(np.float32),
        'noisy_labels': all_noisy_labels.numpy().astype(np.int64),
    })


# ----------------------------- factory --------------------------
def create_model(device):
    model = InceptionResNetV2(num_classes=args.num_class, dropout_p=args.dropout_p)
    return model.to(device)


# ----------------------------- USBP helpers ---------------------
def usbp_sample_w(prob, H, K_max=None, eps=1e-10):
    """Beta sampling per-sample: w_i ~ Beta(K M_i, K(1-M_i)), K = K_max/(1+H).
    K_max can be overridden to ramp up softness from sharp (large K_max) to
    target across early epochs."""
    if K_max is None:
        K_max = args.K_max
    prob_t = torch.from_numpy(prob).float()
    H_t = torch.from_numpy(H).float()
    K = K_max / (1.0 + H_t)
    alpha = K * prob_t + eps
    beta = K * (1.0 - prob_t) + eps
    beta_dist = torch.distributions.beta.Beta(alpha, beta)
    return beta_dist.sample().numpy()


def build_relabel_map(noisy_indices, peer_logits, noisy_labels, tau_rect):
    """Confidence-gated rectification using peer logits on rejected samples."""
    relabel_map = {}
    if len(noisy_indices) == 0:
        return relabel_map
    sub_logits = torch.from_numpy(peer_logits[noisy_indices])
    probs = torch.softmax(sub_logits, dim=1)
    max_probs, pred_labels = torch.max(probs, dim=1)
    sub_noisy = torch.from_numpy(noisy_labels[noisy_indices])
    mask = (max_probs > tau_rect) & (pred_labels != sub_noisy)
    sel_idx = noisy_indices[mask.numpy()]
    sel_lab = pred_labels[mask].numpy()
    return dict(zip(sel_idx.tolist(), sel_lab.tolist()))


# ----------------------------- main -----------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)
    stats_log = open('./checkpoint/%s_stats.txt' % args.id, 'w')
    test_log = open('./checkpoint/%s_acc.txt' % args.id, 'w')

    warm_up = args.warm_up

    loader = dataloader.webvision_dataloader(
        batch_size=args.batch_size, num_class=args.num_class,
        num_workers=args.num_workers, root_dir=args.data_path, log=stats_log)

    print('| Building net (InceptionResNetV2 + dropout p=%.2f)' % args.dropout_p)
    net1 = create_model(cuda1)
    net2 = create_model(cuda2)
    net1_clone = create_model(cuda2)  # peer of net2, lives on cuda2
    net2_clone = create_model(cuda1)  # peer of net1, lives on cuda1

    cudnn.benchmark = True
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    web_valloader = loader.run('test')
    imagenet_valloader = loader.run('imagenet')

    # last-epoch eval results (used as next-epoch's partition signal)
    eval1, eval2 = None, None  # dicts {prob, H, logits, noisy_labels}

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 50:
            lr /= 10
        for pg in optimizer1.param_groups:
            pg['lr'] = lr
        for pg in optimizer2.param_groups:
            pg['lr'] = lr

        # ---------- warm-up or USBP-trained ----------
        if epoch < warm_up:
            wm1 = loader.run('warmup')
            wm2 = loader.run('warmup')
            p1 = mp.Process(target=warmup, args=(epoch, net1, optimizer1, wm1, cuda1, 'net1'))
            p2 = mp.Process(target=warmup, args=(epoch, net2, optimizer2, wm2, cuda2, 'net2'))
            p1.start(); p2.start()
        else:
            # USBP partitioning: net1 trains with peer=net2's signals; net2 trains with peer=net1's
            prob1, H1, logits1, noisy_labels = eval1['prob'], eval1['H'], eval1['logits'], eval1['noisy_labels']
            prob2, H2, logits2, _            = eval2['prob'], eval2['H'], eval2['logits'], eval2['noisy_labels']

            # Ramp-up: early epochs use a sharp Beta (K_max large -> near-hard
            # decision, behaviour close to vanilla DivideMix) and skip
            # rectification, then anneal toward the target K_max.
            if epoch < args.usbp_start:
                ramp = (epoch - warm_up) / max(1, args.usbp_start - warm_up)
                ramp = float(np.clip(ramp, 0.0, 1.0))
                # log-interp K_max from K_sharp -> args.K_max
                K_sharp = 50.0
                K_now = np.exp((1 - ramp) * np.log(K_sharp) + ramp * np.log(args.K_max))
                use_rect = False
            else:
                K_now = args.K_max
                use_rect = True

            # Stochastic Beta sampling per net (sharp early, soft late)
            w1 = usbp_sample_w(prob1, H1, K_max=K_now)
            w2 = usbp_sample_w(prob2, H2, K_max=K_now)
            pred1 = (w1 > args.p_threshold)
            pred2 = (w2 > args.p_threshold)

            # Cross-network rectification: only after USBP signals matured
            if use_rect:
                noisy_idx_for_net1 = np.where(pred2 == False)[0]
                noisy_idx_for_net2 = np.where(pred1 == False)[0]
                relabel_map_for_net1 = build_relabel_map(noisy_idx_for_net1, logits2, noisy_labels, args.relabel_threshold)
                relabel_map_for_net2 = build_relabel_map(noisy_idx_for_net2, logits1, noisy_labels, args.relabel_threshold)
            else:
                relabel_map_for_net1, relabel_map_for_net2 = {}, {}

            # Promote rectified samples into labeled set with soft weight (1 - H)
            pred1_final = (w1 > args.p_threshold).astype(bool)
            pred2_final = (w2 > args.p_threshold).astype(bool)
            if len(relabel_map_for_net2) > 0:
                idxs = np.array(list(relabel_map_for_net2.keys()))
                pred1_final[idxs] = True
                w1[idxs] = 1.0 - H1[idxs]
            if len(relabel_map_for_net1) > 0:
                idxs = np.array(list(relabel_map_for_net1.keys()))
                pred2_final[idxs] = True
                w2[idxs] = 1.0 - H2[idxs]

            print('\n>> Epoch %d  K_now=%.2f  rect: net1<-%d, net2<-%d' %
                  (epoch, K_now, len(relabel_map_for_net1), len(relabel_map_for_net2)))

            # Build co-divide loaders
            lt1, ut1 = loader.run('train', pred=pred2_final, sampled_w=w2, H_values=H2)
            lt2, ut2 = loader.run('train', pred=pred1_final, sampled_w=w1, H_values=H1)

            p1 = mp.Process(target=train,
                            args=(epoch, net1, net2_clone, optimizer1, lt1, ut1, cuda1, 'net1'),
                            kwargs={'relabel_map': relabel_map_for_net1})
            p2 = mp.Process(target=train,
                            args=(epoch, net2, net1_clone, optimizer2, lt2, ut2, cuda2, 'net2'),
                            kwargs={'relabel_map': relabel_map_for_net2})
            p1.start(); p2.start()

        p1.join(); p2.join()

        # sync clones
        net1_clone.load_state_dict(net1.state_dict())
        net2_clone.load_state_dict(net2.state_dict())

        # ---------- test (dual GPU) ----------
        q1, q2 = mp.Queue(), mp.Queue()
        pt1 = mp.Process(target=test, args=(epoch, net1, net2_clone, web_valloader, cuda1, q1))
        pt2 = mp.Process(target=test, args=(epoch, net1_clone, net2, imagenet_valloader, cuda2, q2))
        pt1.start(); pt2.start()
        web_acc = q1.get(); imagenet_acc = q2.get()
        pt1.join(); pt2.join()

        print('\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)' %
              (epoch, web_acc[0], web_acc[1], imagenet_acc[0], imagenet_acc[1]))
        test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n' %
                       (epoch, web_acc[0], web_acc[1], imagenet_acc[0], imagenet_acc[1]))
        test_log.flush()

        # ---------- USBP eval (dual GPU): GMM prob + H_tilde + logits ----------
        el1 = loader.run('eval_train')
        el2 = loader.run('eval_train')
        q1, q2 = mp.Queue(), mp.Queue()
        pe1 = mp.Process(target=eval_train_usbp,
                         args=(el1, net1, cuda1, 'net1', q1, args.K_mcdropout))
        pe2 = mp.Process(target=eval_train_usbp,
                         args=(el2, net2, cuda2, 'net2', q2, args.K_mcdropout))
        pe1.start(); pe2.start()
        eval1 = q1.get(); eval2 = q2.get()
        pe1.join(); pe2.join()

        # snapshot per-epoch stats (small subset)
        np.savez(os.path.join(args.stats_dir, 'epoch_%03d.npz' % epoch),
                 prob1=eval1['prob'], H1=eval1['H'],
                 prob2=eval2['prob'], H2=eval2['H'],
                 web_acc_top1=web_acc[0], web_acc_top5=web_acc[1],
                 imagenet_acc_top1=imagenet_acc[0], imagenet_acc_top5=imagenet_acc[1])
