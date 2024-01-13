import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse

from models import *
from utils import progress_bar
from loader import RotationLoader
import pandas as pd

from helper_methods import create_empty_file, compute_KL_divergence_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--DATASET_NAME', required=True, help='Name of the dataset')
parser.add_argument('--IMAGE_SIZE', required=True, type=int, help='Size of the images in the dataset')
parser.add_argument('--BATCH_SIZE', type=int, default=256, help='Size of the images in the dataset')

args = parser.parse_args()

dataset_name = args.DATASET_NAME
image_size = args.IMAGE_SIZE
batch_size = args.BATCH_SIZE

saved_model_name = f"{dataset_name}_rotation_pretext_model.pth"
output_filename_list = f"{dataset_name}_result_list.csv"

if not os.path.exists(f'./DATA/{dataset_name}'):
    print(f"The dataset '{dataset_name}' does not exist inside of the ./DATA folder.")
    sys.exit(1)


def test(net, testloader, location_to_save_results):
    """
    Save the rotation loss, rotation confusion score and the uncertainty metric for the test data.
    """
    create_empty_file(location_to_save_results)

    # Modify criterion to compute individual losses for each sample
    criterion = nn.CrossEntropyLoss(reduction='none')

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    metrics_list = []
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
            
            for i in range(len(path)):
                individual_losses = sum([loss1[i].item(), loss2[i].item(), loss3[i].item(), loss4[i].item()])/4.
                individual_kl_scores = sum([
                    compute_KL_divergence_score(outputs[i]),
                    compute_KL_divergence_score(outputs1[i]),
                    compute_KL_divergence_score(outputs2[i]),
                    compute_KL_divergence_score(outputs3[i])
                ])/4.0
                individual_correct_preds = predicted[i].eq(targets[i]).item() + predicted1[i].eq(targets1[i]).item() + predicted2[i].eq(targets2[i]).item() + predicted3[i].eq(targets3[i]).item()
                metrics_list.append([individual_losses, individual_correct_preds, individual_kl_scores, path[i]])

                
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(metrics_list, columns=['individual_losses', 'individual_correct_preds', 'individual_kl_scores', 'path'])

    # Save the DataFrame to a CSV file
    df.to_csv(location_to_save_results, index=False)

def get_train_and_test_data():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = RotationLoader(is_train=True, transform=transform_train, dataset_name=dataset_name)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = RotationLoader(is_train=False,  transform=transform_test, dataset_name=dataset_name)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def build_model(device):
    net = ResNet18()
    net.linear = nn.Linear(512, 4)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net

# Training
def train_epoch(net, trainloader, criterion, optimizer, device):
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
        
    state = {
        'net': net.state_dict()
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/{saved_model_name}')
        
    return 100.*correct/total

net = build_model(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
trainloader, testloader = get_train_and_test_data()
train_epoch(net, trainloader, criterion, optimizer, device)
test(net, testloader, location_to_save_results=output_filename_list)