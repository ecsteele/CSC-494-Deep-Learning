# Project_Steele_Virag_Tiemon_Gluon_Redesign.py - Eric Steele, Mate Virag, Liam Tiemon
# NKU CSC/DSC 494/594 Deep Learning Spring 2018 K. Kirby
# ---------------------------------------------------------------------
# The Project_Steele_Virag_Tiemon_Gluon_Redesign.py file creates and trains 
# a neural network using the Gluon machine learning library to solve the 
# Myanmar/Devanagari classification problem. This file requires 
# NkuMyaDevMaker.py to generate the images.


import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np
import matplotlib.pyplot as plt
import NkuMyaDevMaker as nmd

"""
Function to display a mxnet image using matplotlib. Converts the mxnet NDArray
into a numpy NDArray, chooses the first channel (not working with RGB images),
and displays the image.
"""
def display_image(a):
    img = a.asnumpy()
    plt.imshow(img[:,:,0])
    plt.show()
    
"""
Dataset class for use with Gluon's DataLoader. Takes in lists of images and
labels from the makeDataSet function in NkuMyaDevMaker.py in the contstructor.
__len__ and __getitem__ functions implemented as required.
"""
class MyaDevDataset(mx.gluon.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X                  # NkuMyaDevMaker images
        self.Y = Y                  # NkuMyaDevMaker labels
        self.transform = transform  # Transformation function (optional)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        item = (self.X[idx], self.Y[idx])
        
        if self.transform:
            item = self.transform(item)
        
        return item

"""
Accuracy function for a two-class classifier. Receieves floats where one class
is associated with 0.0 and the other with 1.0. A prediction within 0.33 of the
label is considered a correct result. The function returns the number of
correct classifications across a batch of predictions and labels.
"""
def accuracy(predictions, labels):
    # Convert mxnet NDArrays to numpy NDArrays
    pred = predictions.asnumpy()[:,0]
    lab = labels.asnumpy()[:,0]
    correct = 0
    for i in range(len(pred)):
        if abs(pred[i] - lab[i]) < 0.33:
            correct += 1
    return correct

def cnn():
    # Format options for numpy
    np.set_printoptions( precision=3, suppress=True)
        
    # Generate the training set, with nxn images
    n = 36
    trset_size = 30000
    
    print('Generating training set...')
    
    # Use NkuMyaDevMaker to generate images, then format
    X,Y = nmd.makeDataSet(n,trset_size, training=True)
    # For convolutional neural nets, we want 2d single plane images
    Xtrain= np.array(X).reshape([-1,n,n,1])
    # Make it a single output, not 2 output with 1-hot
    Ytrain= np.array([ [ y] for y in Y ], dtype=np.float32 )
    
    # Use generated images for Dataset, use Dataset to create DataLoader for training
    # Gluon does mini-batching by defining a parameter in DataLoader
    ds = MyaDevDataset(Xtrain,Ytrain)
    train_data = mx.gluon.data.DataLoader(ds, batch_size=100, shuffle=True)
    
    # Generate the test set, with nxn images
    teset_size= 10000
    print('Generating test set...')
    
    # Use NkuMyaDevMaker to generate images, then format
    X,Y = nmd.makeDataSet(n,teset_size,training=False)
    # For convolutional neural nets, we want 2d single plane images
    Xtest= np.array(X).reshape([-1,n,n,1])
    # Make it a single output, not 2 output with 1-hot
    Ytest= np.array([ [y] for y in Y ], dtype=np.float32 )
    
    # Use generated images for Dataset, use Dataset to create DataLoader for testing 
    ds = MyaDevDataset(Xtest,Ytest)
    test_data = mx.gluon.data.DataLoader(ds, batch_size=1, shuffle=False)
    
    # Initialize the network
    net = gluon.nn.Sequential()
    
    # Declare hyperparameters
    convo1_kernels = 20
    convo1_kernel_size = (5,5)
    convo2_kernels = 40
    convo2_kernel_size = (5,5)
    pooling = 2
    
    hidden1_neurons = 20
    dropout_rate = 0.3
    hidden2_neurons = 15
    
    # Define our network
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=convo1_kernels, kernel_size=convo1_kernel_size, use_bias=True, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=pooling, strides=pooling))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=convo2_kernels, kernel_size=convo2_kernel_size, use_bias=True, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=pooling, strides=pooling))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(hidden1_neurons, activation="relu", use_bias=True))
        net.add(gluon.nn.Dropout(dropout_rate))
        net.add(gluon.nn.Dense(hidden2_neurons, activation="relu", use_bias=True))
        net.add(gluon.nn.Dense(1, activation="sigmoid", use_bias=True)) # Output layer

    # Initialize parameters using normal distribution
    net.collect_params().initialize(mx.init.Normal(sigma=0.05))
    # Use Mean Squared Error for our loss function
    mean_squared_error = gluon.loss.L2Loss()
    
    # Declare our training algorithm.
    trainer = gluon.Trainer(net.collect_params(), 'ADAM', {'learning_rate': .01})
    
    # Begin training
    print('Training...')
    max_epochs = 5
    for e in range(max_epochs):
        correct = 0 # Count of correct results across epoch, for calculating accuracy
        
        # Get a tuple containing the images/labels for an entire batch
        for i, (data, label) in enumerate(train_data):
            # Specify that we are running this on our cpu. gpu is another option
            data = data.as_in_context(mx.cpu()).swapaxes(3,1)
            label = label.as_in_context(mx.cpu())
            with autograd.record(): # Start recording the derivatives
                output = net(data) # The forward iteration
                loss = mean_squared_error(output, label)
                correct += accuracy(output, label) # Just to print for our benefit, doesn't affect learning
                loss.backward() # Backprop
            trainer.step(data.shape[0])
            curr_loss = ndarray.mean(loss).asscalar() # Also to print
        acc = correct / trset_size
        print("Epoch {}. Current Accuracy: {}. Current Loss: {}.".format(e, acc, curr_loss))
    
    # Begin testing
    print('Testing...')
    # Count of correct results across entire test
    count = 0 
    for i, (data, label) in enumerate(test_data):
        # Specify running on cpu
        data = data.as_in_context(mx.cpu()).swapaxes(3,1)
        label = label.as_in_context(mx.cpu())
        # Push forward through network
        output = net(data) 
        # Count correct results
        count += accuracy(output, label) 
        
        # Print out 10 example images
        if i < 10:
            img = data.swapaxes(3,1)
            display_image(img[0])
            print("expected: " + str(label) + "| actual: " + str(output))
    acc = count / teset_size
    print("Test accuracy: {}%".format(acc*100))

cnn()