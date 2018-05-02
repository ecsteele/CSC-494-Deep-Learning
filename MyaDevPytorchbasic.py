# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:49:45 2018

@author: matev
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:02:05 2018

@author: matev
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

import NkuMyaDevMaker as nmd


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=(5,5), padding=0)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 16 * 7, 11)
        self.fc2 = nn.Linear(11, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2,2), stride=2)
        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
class MyaDevDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        item = (self.X[idx], self.Y[idx])
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
def accuracy(predictions, labels):
    pred = predictions[:,0]
    lab = labels[:,0]
    correct = 0
    for i in range(len(pred)):
        if abs(pred[i].data[0] - lab[i]) < 0.33:
            correct += 1
    return correct
    
def main():
    n = 36
    trset_size = 30000
    print('Generating training set...')
    X,Y = nmd.makeDataSet(n,trset_size, training=True)
    # For convolutional neural nets, we want 2d single plane images.
    Xtrain = np.array(X).reshape([-1,n,n,1])
    # Make it a single output, not 2 output with 1-hot!
    Ytrain= np.array([ [ y] for y in Y ], dtype=np.float32 )
    ds = []
    for i in range(trset_size):
        ds.append((Xtrain[i],Ytrain[i]))
    
    train_set = MyaDevDataset(Xtrain,Ytrain)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    
    tsset_size = 10000
    print('Generating test set...')
    X,Y = nmd.makeDataSet(n,tsset_size, training=False)
    # For convolutional neural nets, we want 2d single plane images.
    Xtest = np.array(X).reshape([-1,n,n,1])
    # HW4 expects a single output, not 2 output with 1-hot!
    Ytest= np.array([ [ y] for y in Y ], dtype=np.float32 )
    ds = []
    for i in range(tsset_size):
        ds.append((Xtest[i],Ytest[i]))
    
    test_set = MyaDevDataset(Xtest,Ytest)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)
    
    cnn = Net()

    # Hyper Parameters
    num_epochs = 4
    learning_rate = 0.01

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(cnn.parameters(), lr=learning_rate)
    
    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data):
            images = images.transpose(1,3)
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 300 == 0:
                print ('Epoch [%d/%d]: Loss: %.4f' %(epoch+1, num_epochs, loss.data[0]))

    # Test the Model
    print("Test")
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_data:
        images = images.transpose(1,3)
        images = Variable(images)
        outputs = cnn(images)
        total += labels.size(0)
        correct += accuracy(outputs,labels)

    print('Test Accuracy of the model on the 100 test images: %d %%' % (100 * correct / total))
    
main()
