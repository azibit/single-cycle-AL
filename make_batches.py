import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import *
from loader import Loader, RotationLoader
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def compute_KL_divergence_score(tensor):
    # Compute the KL divergence from a uniform distribution
    kl_divergence = F.kl_div(F.log_softmax(tensor, dim=0), torch.ones_like(tensor) / tensor.size(0))

    return kl_divergence.item()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Change batch_size to 100 in testloader
testset = RotationLoader(is_train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

net = ResNet18()
net.linear = nn.Linear(512, 4)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/rotation.pth')
net.load_state_dict(checkpoint['net'])

# Modify criterion to compute individual losses for each sample
criterion = nn.CrossEntropyLoss(reduction='none')

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            outputs3 = net(inputs3)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1 + loss2 + loss3 + loss4) / 4.
            test_loss += loss.mean().item()  # Take the mean over the batch dimension
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kl_score = (compute_KL_divergence_score(outputs) + compute_KL_divergence_score(outputs1) 
            + compute_KL_divergence_score(outputs2) + compute_KL_divergence_score(outputs3))/4.
            # print(f"Outputs: {outputs}, Outputs1: {outputs1}, Outputs2: {outputs2}, Outputs3: {outputs3}")

            # loss = loss.mean().item()
            # correct_preds = predicted.eq(targets).item() + predicted1.eq(targets1).sum().item() + predicted2.eq(targets2).sum().item() + predicted3.eq(targets3).sum().item()
            # s = str(float(loss)) + '_' + str(correct_preds) + "_" + str(kl_score) + "_" + str(path[0]) + "\n"

            # with open('./rotation_loss.txt', 'a') as f:
            #     f.write(s)
            for i in range(len(path)):
                individual_losses = sum([loss1[i].item(), loss2[i].item(), loss3[i].item(), loss4[i].item()])/4.
                individual_kl_scores = sum([
                    compute_KL_divergence_score(outputs[i]),
                    compute_KL_divergence_score(outputs1[i]),
                    compute_KL_divergence_score(outputs2[i]),
                    compute_KL_divergence_score(outputs3[i])
                ])/4.0
                individual_correct_preds = predicted[i].eq(targets[i]).item() + predicted1[i].eq(targets1[i]).item() + predicted2[i].eq(targets2[i]).item() + predicted3[i].eq(targets3[i]).item()
                s = f"{float(individual_losses)}_{individual_correct_preds}_{individual_kl_scores}_{path[i]}\n"

                with open('./rotation_loss.txt', 'a') as f:
                    f.write(s)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    test(1)
    with open('./rotation_loss.txt', 'r') as f:
        losses = f.readlines()

    loss_1 = []
    name_2 = []

    for j in losses:
        loss_value = float(j[:-1].split('_')[0])
        loss_1.append(loss_value)
        name = j[:-1].split('_')[3]
        name_2.append(name)

    s = np.array(loss_1)
    sort_index = np.argsort(s)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x) # convert to high loss first

    items_per_class = 100

    if not os.path.isdir('loss'):
        os.mkdir('loss')
    for i in range(10):
        # sample minibatch from unlabeled pool 
        sample5000 = sort_index[i*items_per_class:(i+1)*items_per_class]
        # sample1000 = sample5000[[j*5 for j in range(1000)]]
        b = np.zeros(10)
        for jj in sample5000:
            # print(name_2[jj].split('/'))
            b[int(name_2[jj].split('/')[-2])] +=1
        print(f'{i} Class Distribution: {b}')
        s = './loss/batch_' + str(i) + '.txt'
        for k in sample5000:
            with open(s, 'a') as f:
                f.write(name_2[k]+'\n')
