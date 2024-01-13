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
from utils import progress_bar
import numpy as np
from data_aug.contrastive_learning_dataset import ContrastiveLearningUniversalRepresentationDataset
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
parser.add_argument('--epochs', default=500, type=int, metavar='N',
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
parser.add_argument('-feature-size', default=512, type=int,
                    help='Size of feature before classification (default: 512)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

dataset = ContrastiveLearningUniversalRepresentationDataset(args.data)
trainset = dataset.get_dataset(args.dataset_name, args.n_views)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=False)


class RotationClassifier(nn.Module):
    def __init__(self, num_rotations):
        super(RotationClassifier, self).__init__()
        self.rotation_fc = nn.Linear(args.feature_size, num_rotations)

    def forward(self, x):
        rotation_output = self.rotation_fc(x)
        return rotation_output
    
class ClassClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ClassClassifier, self).__init__()
        self.classification_fc = nn.Linear(args.feature_size, num_classes)

    def forward(self, x):
        class_output = self.classification_fc(x)
        return class_output
    
class CombinedModel(nn.Module):
    def __init__(self, num_rotations, num_classes):
        super(CombinedModel, self).__init__()
        self.backbone = ResNet18()
        self.backbone.linear = nn.Linear(args.feature_size, args.feature_size)  # Add an intermediate fully connected layer if needed
        self.class_branch = ClassClassifier(num_classes)
        self.rotation_branch = RotationClassifier(num_rotations)

    def forward(self, x):
        features = self.backbone(x)
        features =  F.relu(self.backbone.linear(features))  # Optionally pass through an intermediate layer
        class_output = self.class_branch(features)
        rotation_output = self.rotation_branch(features)
        return features, class_output, rotation_output
    
# Model
print('==> Building model..')
net = CombinedModel(num_rotations = 4, num_classes =  len(glob.glob(args.data + "/train/*/*")))
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

def info_nce_loss(labels, features):

        print("Labels: ", labels)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        print("Labels after unsqueezing(): ", labels)
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
    class_correct = 0
    class_total = 0

    rot_correct = 0
    rot_total = 0

    rot_preds, class_preds = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    rot_truths, class_truths = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

    for batch_idx, (images, class_targets, rotation_labels) in enumerate(trainloader):

        images = torch.cat(images, dim=0)
        class_targets = torch.cat(class_targets, dim=0)
        rotation_labels = torch.cat(rotation_labels, dim=0)

        images = images.to(device)
        class_targets = class_targets.to(device)
        rotation_labels = rotation_labels.to(device)

        optimizer.zero_grad()

        features, class_outputs, rotation_outputs = net(images)

        logits, labels = info_nce_loss(class_targets, features)
        
        self_supervised_loss = criterion(logits, labels)
        rotation_loss = criterion(rotation_outputs, rotation_labels) 
        classification_loss = criterion(class_outputs, class_targets)

        loss = (classification_loss + rotation_loss + self_supervised_loss)/3.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, class_predicted = class_outputs.max(1)
        _, rot_predicted = rotation_outputs.max(1)
        class_total += class_targets.size(0)
        rot_total += class_targets.size(0)

        class_correct += class_predicted.eq(class_targets).sum().item()
        rot_correct += rot_predicted.eq(rotation_labels).sum().item()

        class_preds = torch.concat((class_preds, class_predicted), dim=0)
        class_truths = torch.concat((class_truths, class_targets), dim=0)

        rot_preds = torch.concat((rot_preds, rot_predicted), dim=0)
        rot_truths = torch.concat((rot_truths, rotation_labels), dim=0)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Class Acc: %.3f%% (%d/%d) | Rot Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*class_correct/class_total, class_correct, class_total, 100.*rot_correct/rot_total, rot_correct, rot_total))
        
    # Save checkpoint.
    acc = 100.*class_correct/class_total
    with open('./best_contrastive_learning_test.txt','a') as f:
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
        torch.save(state, './checkpoint/contrastive_learning_test.pth')
        best_acc = acc
        
    return class_preds, class_truths

total_preds = torch.Tensor([]).to(device)
total_truths = torch.Tensor([]).to(device)
for epoch in range(start_epoch, args.epochs):
    preds, truths = train(epoch)

    total_preds = torch.concat((total_preds, preds), dim=0)
    total_truths = torch.concat((total_truths, truths), dim=0)

    scheduler.step()

confusion = confusion_matrix(total_truths.cpu().numpy().tolist(), total_preds.cpu().numpy().tolist())
print(confusion)

with open('confusion_matrix_pt4al_universal_representation_test.pkl', 'wb') as file:
    pickle.dump(confusion, file)

# Save the list as a text file
np.savetxt("total_preds_cifar10_100_per_class_universal_representation_test", total_preds.cpu().numpy().tolist())
np.savetxt("total_truths_cifar10_100_per_class_universal_representation_test", total_truths.cpu().numpy().tolist())