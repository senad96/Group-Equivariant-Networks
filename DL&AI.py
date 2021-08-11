import torch
import numpy as np
import sklearn
import scipy
import pandas as pd
import torch
import matplotlib.pyplot as plt

import torchvision
from typing import *
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import TensorDataset
from torch.nn import functional as F
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

from torchvision import datasets, models, transforms
import os

import nets


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}') 



# basic transformation
image_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)



# build the dataloader for trainingset
train_loader = torch.utils.data.DataLoader( datasets.CIFAR10("../prova/data", train=True, 
                            download=True, transform=image_transforms
                            ),
                            batch_size=64,
                            shuffle=True,
                            )


# build the dataloader for testset
test_loader = torch.utils.data.DataLoader( datasets.CIFAR10("../prova/data", train=False,
                            download=True, transform=image_transforms
                            ),
                            batch_size=1000,
                            shuffle=True,
                            )



# build the dataloader for trainingset
train_loader2 = torch.utils.data.DataLoader( datasets.FashionMNIST("../prova/data", train=True, 
                            download=True, transform=image_transforms
                            ),
                            batch_size=64,
                            shuffle=True,
                            )



# build the dataloader for trainingset
test_loader2 = torch.utils.data.DataLoader( datasets.FashionMNIST("../prova/data", train=False, 
                            download=True, transform=image_transforms
                            ),
                            batch_size=1000,
                            shuffle=True,
                            )


# Retrieve the image size and the number of color channels for both the dataset
x, _ = next(iter(train_loader))

batch_size = x.shape[0]
n_channels = x.shape[1]
input_size_w = x.shape[2]
input_size_h = x.shape[3]
input_size = input_size_w * input_size_h


dataiter = iter(train_loader)
images, labels = dataiter.next()


print(images.shape)
print(labels.shape)
print(len(train_loader))




log_freq = len(train_loader)//batch_size * 5


# SOME UTILITY FUNCTION IN ORDER TO MAKE MORE ORGANIZED THE CODE
def train(epoch, loader , net, optimizer, loss_func, log_freq=log_freq):
    
    running_loss = 0.0
    correct = 0
    total = 0
    losses = []
    
    for i, data in enumerate(loader, start=1):
        
        # get the inputs and load into GPU if available
        x, labels = data
        x, labels = x.to(device), labels.to(device)


        # zero the parameter gradients
        optimizer.zero_grad()
        

        # forward + backward + optimize
        outputs = net(x)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        #get predications
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

        # print statistics every log_freq mini batch
        running_loss += loss.item()
        if (i) % log_freq == 0:    # print every log_freq mini batches
            print('[Epoch : %d, Iter: %5d] loss: %.3f' %
                  (epoch + 1, i, running_loss / log_freq))
            losses.append( running_loss / log_freq)
            running_loss = 0.0
            print('Top1 Accuracy of the network on the trainset images:', correct / total)
            correct=0
            total=0   
            
    return losses




def test(net, loader, train_data=False):
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        net.eval()
        
        for batch_idx, data in enumerate(loader):
            
            if batch_idx == len(loader):
                break
            
            #load into GPU if available
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # predict
            output = net(inputs)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze().sum().item()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        
        print('Accuracy of the network on the testset images:', correct / total)




epochs = 20 
learning_rate = 0.001
momentum = 0.9
loss_func = nn.CrossEntropyLoss()



#construct the models + optimizer
model1 = nets.GroupCNN()
optimizer = optim.Adam(model1.parameters(), lr=learning_rate)
loss_value = []


# training
for i in range(epochs):
    model1.train()
    loss_avg = train(epoch=i, loader=train_loader, net=model1, optimizer=optimizer, loss_func=loss_func)
    loss_value.append(loss_avg)

    
y_ax = np.reshape(loss_value, (-1))
x = range(len(y_ax))
    

plt.plot(x,  y_ax)

#testing
test(model, test_loader)

torch.save(model1.state_dict(), "../prova/model1.pt" )







