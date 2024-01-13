from annoy import AnnoyIndex
import random, pickle
from collections import defaultdict

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
import numpy as np

from models import *
from loader import Loader, RotationLoader, SampleDataLoader
from utils import progress_bar
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset


import glob as glob
from annoy import AnnoyIndex

import os
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('--n-views', default=1, type=int, metavar='N',
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

checkpoint = torch.load('./checkpoint/contrastive_learning.pth')
net.load_state_dict(checkpoint['net'])

# Define a hook function to store the intermediate output
def hook_fn(module, input, output):
    hook_fn.intermediate_input = input

# Function to get intermediate output from a specific layer
def get_intermediate_input(model, layer_name, input_data):
    target_layer = getattr(model.module, layer_name)
    hook_handle = target_layer.register_forward_hook(hook_fn)
    output = model(input_data)
    intermediate_input = hook_fn.intermediate_input
    hook_handle.remove()
    return intermediate_input, output

def get_feature_representation_by_index(index_value):
    data, _ = trainset.__getitem__(index_value)
    intermediate_input, output = get_intermediate_input(net, 'linear', data[0].unsqueeze(0))

    return intermediate_input[0][0]

paths = glob.glob(args.data + '/train/*/*')
index_value = random.randint(0, len(paths) - 1)

# Load the confusion matrix from the saved file
with open('confusion_matrix_pt4al_test_sample.pkl', 'rb') as file:
    confusion = pickle.load(file)

print("Index to check: ", index_value)
print("File Path: ", )
print("Description: ", paths[index_value], "of class:", paths[index_value].split("/")[-2])

# Access the index_value row of the confusion matrix
numbers = confusion[index_value]

print("Total Number: ", len(numbers))
print("Sum: ", sum(numbers))

f = 512  # Length of item vector that will be indexed

dist_metric = 'angular'
t = AnnoyIndex(f, dist_metric)
feat_repr = get_feature_representation_by_index(index_value)
t.add_item(index_value, feat_repr)

# Print the first row
# print("First Row of Confusion Matrix:")
# print(first_row)

# Remove a certain value (e.g., remove all occurrences of 4)
# index_value = 102
filtered_numbers = [numbers[i] for i in range(len(numbers)) if i != index_value]

# Calculate the average of the numbers
average = math.floor((sum(filtered_numbers) / len(filtered_numbers)) + 0.5)

# Find indices greater than the average
indices_greater_than_average = [i for i, x in enumerate(filtered_numbers) if x > average]
indices_less_than_average = [i for i, x in enumerate(filtered_numbers) if x < average]

# Print results
print("Average:", average)
# print("Filtered List (without value {}):\n".format(index_value), filtered_numbers)
print("Indices Greater Than Average:\n", indices_greater_than_average)

max_value = max(filtered_numbers)
print("Max Value: ", max_value)

# Find indices greater than the average
indices_at_max = [i for i, x in enumerate(filtered_numbers) if x == max_value]
print("Indices at Max Value: ", [(paths[i] + " of class: " + paths[i].split("/")[-2]) for i in indices_at_max])

for max_index in indices_at_max:
    feat_repr = get_feature_representation_by_index(max_index)
    t.add_item(max_index, feat_repr)

# Compare distance to all indices close.
for index in indices_greater_than_average:

    feat_repr = get_feature_representation_by_index(index)
    t.add_item(index, feat_repr)

# Compare distance to all indices far away.
for index in indices_less_than_average:

    feat_repr = get_feature_representation_by_index(index)
    t.add_item(index, feat_repr)

t.build(10) # 10 trees
t.save('test.ann')

u = AnnoyIndex(f, dist_metric)
u.load('test.ann') # super fast, will just mmap the file
nearest_neigh = u.get_nns_by_item(index_value, 2)
print(nearest_neigh) 
print([paths[i] for i in nearest_neigh])

def save_images(index_value, max_index, nearest_neigh):
    # Define the directory path where you want to save the image
    directory_path = "Sample" + str(index_value)  # Replace with your desired directory path

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Open an image file
    index_value_img = Image.open(paths[index_value])  # Replace "your_image.jpg" with the path to your image file
    index_value_img.save(os.path.join(directory_path, "main.png")) 

    max_index_img = Image.open(paths[max_index])  # Replace "your_image.jpg" with the path to your image file
    max_index_img.save(os.path.join(directory_path, "confusion.png"))   

    nearest_neigh_img = Image.open(paths[nearest_neigh])  # Replace "your_image.jpg" with the path to your image file
    nearest_neigh_img.save(os.path.join(directory_path, "nearest_neigh.png"))   


save_images(index_value, indices_at_max[0], nearest_neigh[1])