"""DivideMix + USBP on Clothing1M (ImageNet-pretrained ResNet-50 with MC-Dropout)."""
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)

# --- USBP arguments ---
parser.add_argument('--dropout_p', default=0.2, type=float, help='dropout probability for MC-Dropout')
parser.add_argument('--K_mcdropout', default=5, type=int, help='number of forward passes for MC-Dropout')
parser.add_argument('--K_max', default=30.0, type=float, help='max concentration parameter for Beta distribution')
parser.add_argument('--relabel_threshold', default=2.0, type=float, help='Confidence threshold to relabel a noisy sample')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# --- Model Wrapper for MC-Dropout ---
class ResNet50_MCDropout(nn.Module):
    def __init__(self, num_classes, dropout_p=0.1):
        super(ResNet50_MCDropout, self).__init__()
        self.base = models.resnet50(pretrained=True)
        self.base.fc = nn.Linear(2048, num_classes)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x, use_dropout=False):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        if use_dropout and self.dropout_p > 0:
            x = self.dropout(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        
        if use_dropout and self.dropout_p > 0:
            x = self.dropout(x)
            
        x = self.base.fc(x)
        return x

def create_model():
    model = ResNet50_MCDropout(num_classes=args.num_class, dropout_p=args.dropout_p)
    model = model.cuda()
    return model     

# --- Helper Functions ---
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

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() 
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)             
        batch_size = inputs_x.size(0)
        
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) 
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) 
            targets_u = targets_u.detach()       
            
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)         
            targets_x = targets_x.detach()       
        
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()
    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = './checkpoint/%s_net%d.pth.tar'%(args.id,k)
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)       
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc    

# Modified eval_train
def eval_train_and_uncertainty(epoch, model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    uncertainties = torch.zeros(num_samples)
    all_logits = [] 
    paths = []
    
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets_cuda = inputs.cuda(), targets.cuda() 
            
            outputs = model(inputs) 
            loss = CE(outputs, targets_cuda)
            uncertainty = get_mcdropout_uncertainty(model, inputs)
            
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                uncertainties[n] = uncertainty[b]
                paths.append(path[b])
                n+=1
            
            all_logits.append(outputs.cpu())

            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss & uncertainty Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses_norm = losses.reshape(-1,1)
    
    uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-10)
    
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses_norm)
    prob = gmm.predict_proba(losses_norm) 
    prob = prob[:,gmm.means_.argmin()]       
    
    all_logits = torch.cat(all_logits, dim=0)
    
    return prob, paths, uncertainties.numpy(), all_logits
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
log=open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
    
    # --- Warm-up: 5 epochs ---
    if epoch < 5:      
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        print('\n==== Net 1 Evaluate ====') 
        eval_loader = loader.run('eval_train')
        prob1, paths1, H1, logits1 = eval_train_and_uncertainty(epoch, net1)
        print(f" Net1 Mean Uncertainty: {H1.mean():.4f}")
        
        print('\n==== Net 2 Evaluate ====') 
        eval_loader = loader.run('eval_train')
        prob2, paths2, H2, logits2 = eval_train_and_uncertainty(epoch, net2)
        print(f" Net2 Mean Uncertainty: {H2.mean():.4f}")

        # --- Stochastic Beta Sampling ---
        K1 = args.K_max / (1 + H1 + 1e-10)
        K2 = args.K_max / (1 + H2 + 1e-10)
        
        prob1_t = torch.from_numpy(prob1).float().cuda()
        prob2_t = torch.from_numpy(prob2).float().cuda()
        K1_t = torch.from_numpy(K1).float().cuda()
        K2_t = torch.from_numpy(K2).float().cuda()

        beta_dist1 = torch.distributions.beta.Beta(K1_t * prob1_t + 1e-10, K1_t * (1 - prob1_t) + 1e-10)
        w_sampled_1 = beta_dist1.sample().cpu().numpy()
        
        beta_dist2 = torch.distributions.beta.Beta(K2_t * prob2_t + 1e-10, K2_t * (1 - prob2_t) + 1e-10)
        w_sampled_2 = beta_dist2.sample().cpu().numpy()

        pred1_hard = (w_sampled_1 > args.p_threshold)
        pred2_hard = (w_sampled_2 > args.p_threshold)
        
        relabel_map1 = {} 
        relabel_map2 = {} 
        
        if args.relabel_threshold <= 1.0:
             # Not implemented for Clothing1M
             pass
        else:
             if epoch == 5:
                 print(">> Relabeling DISABLED.")

        # --- Training ---
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2_hard, prob2, paths=paths2, 
                                                                sampled_w=w_sampled_2, H_values=H2,
                                                                relabel_map=relabel_map2) 
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1_hard, prob1, paths=paths1, 
                                                                sampled_w=w_sampled_1, H_values=H1,
                                                                relabel_map=relabel_map1) 
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)

    val_loader = loader.run('val') 
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 

test_loader = loader.run('test')
net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar'%args.id))
net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar'%args.id))
acc = test(net1,net2,test_loader)      

log.write('Test Accuracy:%.2f\n'%(acc))
log.flush()