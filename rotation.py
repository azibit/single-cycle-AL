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
import optuna

from models import *
from loader import Loader, RotationLoader
from utils import progress_bar
import numpy as np

from auto_augment import RandAugmentPolicy, SplitAugmentPolicy
from torch.optim.swa_utils import AveragedModel, SWALR


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def get_train_and_test_data():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugmentPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = RotationLoader(is_train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = RotationLoader(is_train=False,  transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

def build_model(device):
    net = ResNet18()
    net.linear = nn.Linear(512, 4)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net

def train_model(net, optimizer, criterion, device, train_epochs, swa_epochs):

    train_acc, test_acc, best_acc = 0, 0, 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    trainloader, testloader = get_train_and_test_data()

    for epoch in range(train_epochs):
        train_acc = train_epoch(net, trainloader, criterion, optimizer, epoch, device)
        test_acc = test_epoch(net, testloader, criterion, epoch, device, best_acc) 

        if test_acc > best_acc:
            best_acc = test_acc

        scheduler.step()

    swa_model = AveragedModel(net)
    swa_start = 5
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    for epoch in range(swa_epochs):
        train_acc = train_epoch(net, trainloader, criterion, optimizer, epoch, device)
        # test_acc = test_epoch(net, testloader, criterion, epoch, device, best_acc) 

        # if test_acc > best_acc:
        #     best_acc = test_acc

        if epoch > swa_start:
            swa_model.update_parameters(net)
            swa_scheduler.step()
        else:
            scheduler.step()

    # Update bn statistics for the swa_model at the end
    torch.optim.swa_utils.update_bn(trainloader, swa_model)

    test_acc = test_epoch(swa_model, testloader, criterion, epoch, device, 0) 

    return train_acc, test_acc

# Training
def train_epoch(net, trainloader, criterion, optimizer, epoch, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(trainloader):
        inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
        inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
        optimizer.zero_grad()
        outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)

        loss1 = criterion(outputs, targets)
        loss2 = criterion(outputs1, targets1)
        loss3 = criterion(outputs2, targets2)
        loss4 = criterion(outputs3, targets3)
        loss = (loss1+loss2+loss3+loss4)/4.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        _, predicted3 = outputs3.max(1)
        total += targets.size(0)*4

        correct += predicted.eq(targets).sum().item()
        correct += predicted1.eq(targets1).sum().item()
        correct += predicted2.eq(targets2).sum().item()
        correct += predicted3.eq(targets3).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    return 100.*correct/total


def test_epoch(net, testloader, criterion, epoch, device, best_acc):
    # global best_acc
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
            loss = (loss1+loss2+loss3+loss4)/4.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0)*4

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
            correct += predicted3.eq(targets3).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    with open('./best_rotation.txt','a') as f:
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
        torch.save(state, './checkpoint/rotation.pth')
        best_acc = acc

    return best_acc


def objective(trial, device, automl_epochs, swa_epochs):

    net = build_model(device)
    criterion = nn.CrossEntropyLoss()

    lr, wd, momentum, optimizer = suggest_hyperparams(trial)

    if optimizer == "ADAM":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    _, test_acc = train_model(net, optimizer, criterion, device, automl_epochs, swa_epochs)
    return test_acc


def suggest_hyperparams(trial):

    # auto-ml hyperparams
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 1.5, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["ADAM", "SGD"])

    return lr, wd, momentum, optimizer

# AUTO_ML_ROUNDS = 5

# study = optuna.create_study(
#             sampler=optuna.samplers.TPESampler(),
#             direction="maximize",
#             pruner=optuna.pruners.SuccessiveHalvingPruner(),
#         )


# criterion = nn.CrossEntropyLoss()
# automl_epochs = 2
# swa_epochs = 13

# study.optimize(
#         lambda trial: objective(trial, device, automl_epochs, swa_epochs),
#         n_trials=5,
#         n_jobs=1,
#     )
    
# complete_trials = [
#         t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
#         ]
# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial

# print("  Value: {}".format(trial.value))

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

#1. Run the SWA on all the trials.
# train_model(net, optimizer, criterion, device, epochs)

