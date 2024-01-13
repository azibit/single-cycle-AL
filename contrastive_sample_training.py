'''Train CIFAR10 with PyTorch.'''
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

from models import *
from loader import Loader, RotationLoader
from utils import progress_bar
import numpy as np
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
import glob as glob
from sklearn.metrics import confusion_matrix
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('--n-views', default=10, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('-data', metavar='DIR', default='./DATA-cifar_sub_100',
                    help='path to dataset')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

dataset = ContrastiveLearningDataset(args.data)
trainset = dataset.get_dataset(args.dataset_name, args.n_views)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=False)

# Model
print('==> Building model..')
net = ResNet18()
net.linear = nn.Linear(512, len(glob.glob(args.data + "/train/*/*")))
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])


def info_nce_loss(labels, features):

        # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

        logits = logits / args.temperature
        return logits, labels

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global best_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    preds = torch.Tensor([]).to(device)
    truths = torch.Tensor([]).to(device)

    for batch_idx, (images, targets) in enumerate(trainloader):

        images = torch.cat(images, dim=0)
        targets = torch.cat(targets, dim=0)

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = net(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

        preds = torch.concat((preds, predicted), dim=0)
        truths = torch.concat((truths, targets), dim=0)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # Save checkpoint.
    acc = 100.*correct/total
    with open('./best_contrastive_learning.txt','a') as f:
        f.write(str(acc)+':'+str(epoch)+'\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # save rotation weights
        torch.save(state, './checkpoint/contrastive_learning.pth')
        best_acc = acc
        
    return preds, truths

total_preds = torch.Tensor([]).to(device)
total_truths = torch.Tensor([]).to(device)
for epoch in range(start_epoch, args.epochs):
    preds, truths = train(epoch)

    print("Length of preds: ", len(preds))
    print("Length of truths: ", len(truths))

    total_preds = torch.concat((total_preds, preds), dim=0)
    total_truths = torch.concat((total_truths, truths), dim=0)

    scheduler.step()

confusion = confusion_matrix(total_truths.cpu().numpy().tolist(), total_preds.cpu().numpy().tolist())
print(confusion)

with open('confusion_matrix_pt4al_test.pkl', 'wb') as file:
    pickle.dump(confusion, file)

# Save the list as a text file
np.savetxt("total_preds_cifar10_100_per_class", total_preds.cpu().numpy().tolist())
np.savetxt("total_truths_cifar10_100_per_class", total_truths.cpu().numpy().tolist())