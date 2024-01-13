import torch
import torchvision
from PIL import Image
import os


class save_dataset(torch.utils.data.Dataset):

  def __init__(self, dataset, split='train'):
    self.dataset = dataset
    self.split = split

  def __getitem__(self, i):
      x, y = self.dataset[i]
      path = './DATA_CIFAR100/'+self.split+'/'+str(y)+'/'+str(i)+'.png'

      if not os.path.isdir('./DATA_CIFAR100/'+self.split+'/'+str(y)):
          os.mkdir('./DATA_CIFAR100/'+self.split+'/'+str(y))

      x.save(path)

  def __len__(self):
    return len(self.dataset)

trainset = torchvision.datasets.CIFAR100(root='./DATA_CIFAR100', train=True, download=True, transform=None)

testset = torchvision.datasets.CIFAR100(root='./DATA_CIFAR100', train=False, download=True, transform=None)

train_dataset = save_dataset(trainset, split='train')
test_dataset = save_dataset(testset, split='test')

if not os.path.isdir('./DATA_CIFAR100'):
    os.mkdir('./DATA_CIFAR100')

if not os.path.isdir('./DATA_CIFAR100/train'):
    os.mkdir('./DATA_CIFAR100/train')

if not os.path.isdir('./DATA_CIFAR100/test'):
    os.mkdir('./DATA_CIFAR100/test')

for idx, i in enumerate(train_dataset):
    train_dataset[idx]
    print(idx)

for idx, i in enumerate(test_dataset):
    test_dataset[idx]
    print(idx)
