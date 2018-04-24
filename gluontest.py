#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:59:31 2018

@author: Mimetic
"""

import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np
import matplotlib.pyplot as plt

import NkuMyaDevMaker as nmd

def render_as_image(a):
    img = a.asnumpy() # convert to numpy array
    #img = img.astype(np.uint8)  # use uint8 (0-255)
    plt.imshow(img[:,:,0])
    plt.show()
    
class MyaDevDataset(mx.gluon.data.Dataset):
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

def main():
    #ds = mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data, label))
    # Print np.arrays nicely (3 places after decimal, no scientific notation).
    np.set_printoptions( precision=3, suppress=True)
        
    # Generate the training set, with nxn images.
    n= 36
    trset_size= 3000
    print('Generating training set...')
    X,Y = nmd.makeDataSet(n,trset_size, training=True)
    # For convolutional neural nets, we want 2d single plane images.
    Xtrain= np.array(X).reshape([-1,n,n,1])
    # HW4 expects a single output, not 2 output with 1-hot!
    Ytrain= np.array([ [ y] for y in Y ], dtype=np.float32 )
    """
    ds = []
    for i in range(trset_size):
        ds.append((Xtrain[i],Ytrain[i]))
    ds = np.array(ds)
    """
    ds = MyaDevDataset(Xtrain,Ytrain)
    train_data = mx.gluon.data.DataLoader(ds, batch_size=100, shuffle=True)
    
    # Generate test set.
    teset_size= 1000
    print('Generating test set...')
    X,Y = nmd.makeDataSet(n,teset_size,training=False)
    Xtest= np.array(X).reshape([-1,n,n,1])
    Ytest= np.array([ [y] for y in Y ], dtype=np.float32 )
    """
    for i in range(teset_size):
        ds.append((Xtest[i],Ytest[i]))
    ds = np.array(ds)
    """
    ds = MyaDevDataset(Xtest,Ytest)
    test_data = mx.gluon.data.DataLoader(ds, batch_size=1, shuffle=False)
    
    # Initialize the model
    net = gluon.nn.Sequential()
    
    ps = 2
    nf = 10
    k = 5
    nh = 20
    
    # Define the model architecture
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=nf, kernel_size=(5,5), use_bias=True, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=ps, strides=ps))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(nh, activation="relu", use_bias=True))
        net.add(gluon.nn.Dense(1, activation="relu", use_bias=True)) # Output layer
    
    # We start with random values for all of the model’s parameters from a 
    # normal distribution with a standard deviation of 0.05
    #net.collect_params().initialize(mx.init.Normal(sigma=0.05))
    net.collect_params().initialize(mx.init.Uniform())
    mean_squared_error = gluon.loss.L2Loss()
    
    # We opt to use the stochastic gradient descent (sgd) training algorithm 
    # and set the learning rate hyperparameter to .1
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
    
    # Loop through several epochs and watch the model improve
    epochs = 10
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(mx.cpu()).swapaxes(3,1)
            label = label.as_in_context(mx.cpu())
            with autograd.record(): # Start recording the derivatives
                output = net(data) # the forward iteration
                loss = mean_squared_error(output, label)
                loss.backward()
            trainer.step(data.shape[0])
            # Provide stats on the improvement of the model over each epoch
            curr_loss = ndarray.mean(loss).asscalar()
        print("Epoch {}. Current Loss: {}.".format(e, curr_loss))
    
    acc = mx.metric.Accuracy()
    # Run against testing set
    for i, (data, label) in enumerate(test_data):
        data = data.as_in_context(mx.cpu()).swapaxes(3,1)
        label = label.as_in_context(mx.cpu())
        output = net(data)
        acc.update(labels=label, preds=output)
        if i < 10:
            img = data.swapaxes(3,1)
            render_as_image(img[0])
            print("expected: " + str(label) + "| actual: " + str(output))
            
    print("Test accuracy: " + str(acc.get()[1]))
    

main()