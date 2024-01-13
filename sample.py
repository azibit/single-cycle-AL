from annoy import AnnoyIndex
import random
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

number_of_instances = 1000
testset = SampleDataLoader(is_train=True,  transform=transform_test, count=number_of_instances)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)

net = ResNet18()
repr_output_length = 512
net.linear = nn.Linear(repr_output_length, 4)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/rotation.pth')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

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

def test(epoch):
    global best_acc
    net.eval()
    results = []
    files = []
    loss_values = []

    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs = inputs.to(device)

            #1. Get the representation of each instance
            intermediate_input, outputs = get_intermediate_input(net, 'linear', inputs)
            intermediate_input1, outputs1 = get_intermediate_input(net, 'linear', inputs1)
            intermediate_input2, outputs2 = get_intermediate_input(net, 'linear', inputs2)
            intermediate_input3, outputs3 = get_intermediate_input(net, 'linear', inputs3)

            loss = criterion(outputs.to(device), targets.to(device))
            loss1 = criterion(outputs1.to(device), targets1.to(device))
            loss2 = criterion(outputs2.to(device), targets2.to(device))
            loss3 = criterion(outputs3.to(device), targets3.to(device))

            # loss = (loss + loss1 + loss2 + loss3) / 4
            print("Loss: ", loss)
            # results.extend(intermediate_input[0])
            files.extend(path)
            loss_values.append(loss.item())

            results = intermediate_input #[intermediate_input, intermediate_input1, intermediate_input2, intermediate_input3]
            # print("Targets: ", targets)
            get_similarity_score(intermediate_input[0], 'euclidean', name_to_save = 'test.ann')
            get_similarity_score(intermediate_input[0], 'angular', name_to_save = 'test.ann')
            print(loss)

            # print("Targets1: ", targets1)
            get_similarity_score(intermediate_input1[0], 'euclidean', name_to_save = 'test.ann')
            get_similarity_score(intermediate_input1[0], 'angular', name_to_save = 'test.ann')
            
            print(loss1)

            # print("Targets2: ", targets2)
            get_similarity_score(intermediate_input2[0], 'euclidean', name_to_save = 'test.ann')
            get_similarity_score(intermediate_input2[0], 'angular', name_to_save = 'test.ann')
            print(loss2)

            # print("Targets3: ", targets3)
            get_similarity_score(intermediate_input3[0], 'euclidean', name_to_save = 'test.ann')
            get_similarity_score(intermediate_input2[0], 'angular', name_to_save = 'test.ann')
            print(loss3)

            intermediates = torch.concat((intermediate_input[0], intermediate_input1[0], intermediate_input2[0], intermediate_input3[0]), dim = 0)
            all_targets = torch.concat((targets, targets1, targets2, targets3), dim = 0)

            create_class_representation(intermediates, all_targets)

            print("*"*50)


    return results, files, loss_values

def get_similarity_score(results, metric_distance, name_to_save = 'test.ann'):
    print("Metric Distance: ", metric_distance)

    results = torch.split(results, 1, dim=0)  # Split into individual tensors of shape (1, 512)

    # If you want to convert each tensor to a shape of (512,), you can squeeze them
    results = [t.squeeze() for t in results]

    number_of_instances = len(results)

    #3. Vector output length
    f = results[0].shape[0] # Length of item vector that will be indexed

    #4. Distance metric to use
    t = AnnoyIndex(f, metric_distance)
    for i in range(number_of_instances):
        #5. Build representation
        t.add_item(i, results[i])

    t.build(10) # 10 trees
    t.save(name_to_save)

    u = AnnoyIndex(f, metric_distance)
    u.load(name_to_save) # super fast, will just mmap the file

    #6. For each instance, find the top 100 closest representations and find their average.
    for instance in range(number_of_instances):

        #7. Find the closest n neighbors to this node
        n = number_of_instances
        closest_neighbors = u.get_nns_by_item(1, n)

        #8. Find its distance to its neighbors. 
        distance_score = []
        for item in closest_neighbors:
            distance_score.append(t.get_distance(instance, item))

    print("The distance between the samples is: ", sum(distance_score) / len(distance_score))

def create_class_representation(A, B):

    results = []

    # Create a dictionary to store grouped tensors
    grouped_tensors = defaultdict(list)

    # Group items in A based on class labels in B
    for i, class_label in enumerate(B):
        grouped_tensors[class_label.item()].append(A[i])

    # Convert the dictionary to a regular dictionary
    grouped_tensors = dict(grouped_tensors)

    # Print the grouped tensors
    for class_label, tensors in grouped_tensors.items():
        # results.append(torch.stack(tensors))  # Convert the list of tensors to a single tensor
        res = torch.stack(tensors)
        print("For class, ", class_label)
        get_similarity_score(res, 'euclidean')
        get_similarity_score(res, 'angular')

    return results

test(2)