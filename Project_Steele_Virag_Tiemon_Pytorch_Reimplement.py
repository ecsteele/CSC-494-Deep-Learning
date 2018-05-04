# Project_Steele_Virag_Tiemon_Pytorch_Reimplement.py - Eric Steele, Mate Virag, Liam Tiemon
# NKU CSC/DSC 494/594 Deep Learning Spring 2018 K. Kirby
# ---------------------------------------------------------------------
# The Project_Steele_Virag_Tiemon_Pytorch_Reimplement.py file creates and 
# trains a neural network using the PyTorch machine learning library to 
# solve the Myanmar/Devanagari classification problem. The network used in
# this implementation is a replica of the original TensorFlow network used
# in the TfCnn-MyaDev_For_HW4.py file. This file requires 
# NkuMyaDevMaker.py to generate the images.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt

import NkuMyaDevMaker as nmd

"""
Function to display a PyTorch image using matplotlib. Converts the PyTorch FloatTensor
into a numpy array, chooses the first channel (not working with RGB images),
and displays the image.
"""
def render_as_image(a):
    img = a.numpy()    # Convert to numpy array
    plt.imshow(img[:,:,0])
    plt.show()

"""Class that defines the neural network."""
class Net(nn.Module):
    """Defines the layers in the neural network."""
    def __init__(self, depth, nk, kernel_size, padding, hidden_neurons, nc):
        super(Net, self).__init__()
        # out_channels defines the number of kernels
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=nk, kernel_size=kernel_size, padding=padding)
        # nc is the image size after convolution and pooling
        self.fc1 = nn.Linear(nc * nc * nk, hidden_neurons)
        # Single value output
        self.fc2 = nn.Linear(hidden_neurons, 1)

    """
    Defines the connections and the activation functions between layers, pushes the 
    input patterns through the network and returns the network's output.
    """
    def forward(self, x, pooling):
        # Max pooling over a square window with stride of pool size to avoid overlaps
        # Activatin functions are specified in this function even for the convolutional layer
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=pooling, stride=pooling)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    """
    Calculates the size of the flat array for the input of the first dense layer 
    after the last convolutional layer.
    """
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
"""
Dataset class for use with Gluon's DataLoader. Takes in lists of images and
labels from the makeDataSet function in NkuMyaDevMaker.py in the contstructor.
__len__ and __getitem__ functions implemented as required.
"""
class MyaDevDataset(Dataset):
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
    # predictions is a list of PyTorch Variable objects
    pred = predictions[:,0]
    lab = labels[:,0]
    correct = 0
    for i in range(len(pred)):
        # Get the output value of a Variable object and compare it to the expected output
        if abs(pred[i].data[0] - lab[i]) < 0.33:
            correct += 1
    return correct
    
"""Defines the network, generates the training and test sets, and trains and tests the network."""
def main():
    # Format options for numpy
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    
    # Generate the training set, with nxn images
    n = 36
    trset_size = 30000
    
    print('Generating training set...')
    
    # Use NkuMyaDevMaker to generate images, then format
    X,Y = nmd.makeDataSet(n,trset_size, training=True)
    # For convolutional neural nets, we want 2d single plane images
    Xtrain = np.array(X).reshape([-1,n,n,1])
    # Make it a single output, not 2 output with 1-hot
    Ytrain = np.array([ [ y] for y in Y ], dtype=np.float32 )

    # Declare hyperparameters
    convo_kernels = 7
    convo_kernel_size = 5
    convo_padding = 0
    pooling = 2
    hidden_neurons = 11
    batch_size = 100
    num_epochs = 4
    learning_rate = 0.01
    
    # Training set size should be divisible by batch size
    assert trset_size % batch_size == 0
    
    # Result of convolving nxn image (stride 1, with no padding) will be n_conv x n_conv
    n_conv = n-(convo_kernel_size-1)  
    # Pools should evenly divide images being pooled
    assert n_conv % pooling == 0     
    
    # Use generated images for Dataset, use Dataset to create DataLoader for training
    # PyTorch does mini-batching by defining a parameter in DataLoader
    train_set = MyaDevDataset(Xtrain,Ytrain)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Initialize the network
    cnn = Net(1,convo_kernels,convo_kernel_size,convo_padding,hidden_neurons,n_conv//pooling)

    # Loss and Optimizer
    # Using mean squared error for our loss and RMSProp for our optimimzer
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(cnn.parameters(), lr=learning_rate)
    
    # Train the Model
    print('Training...')
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data):
            images = images.transpose(1,3)
            
            # PyTorch uses its Variable class to wrap tensors
            # and record the operations applied to tensors
            images = Variable(images)           
            labels = Variable(labels)           
        
            optimizer.zero_grad()                       # Clears gradients of all optimized Variables
            outputs = cnn.forward(images,pooling)       # Pushes the input pattern through the network
            loss = criterion(outputs, labels)           # Evaluates the loss function
            loss.backward()                             # Calculate gradients based on loss
            optimizer.step()                            # Updates parameters based on gradients
        
            # Print out the loss at the end of every epoch
            if (i+1) % (trset_size // batch_size) == 0:
                print ('Epoch [%d/%d]: Loss: %.4f' %(epoch+1, num_epochs, loss.data[0]))
            
    # Generate the test set, with nxn images
    tsset_size = 10000
    print('Generating test set...')
    
    # Use NkuMyaDevMaker to generate images, then format
    X,Y = nmd.makeDataSet(n,tsset_size, training=False)
    # For convolutional neural nets, we want 2d single plane images
    Xtest = np.array(X).reshape([-1,n,n,1])
    # Make it a single output, not 2 output with 1-hot
    Ytest = np.array([ [ y] for y in Y ], dtype=np.float32 )
    
    # Use generated images for Dataset, use Dataset to create DataLoader for testing
    # PyTorch does mini-batching by defining a parameter in DataLoader
    test_set = MyaDevDataset(Xtest,Ytest)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)

    # Test the Model
    print("Testing...")
    
    # Changes model to 'eval' mode (BN uses moving mean/var)
    cnn.eval()
    # Count of correct results across entire test
    correct = 0
    # Index to limit the number of printed examples
    i = 0
    
    for images, labels in test_data:
        # For printing the images, we don't want to make them PyTorch Variables
        displayImages = images
        images = images.transpose(1,3)
        images = Variable(images)
        # Push forward through network
        outputs = cnn(images,pooling)
        # Count correct results
        correct += accuracy(outputs,labels)
        
        # Print out 5 example images
        if i < 5:
            render_as_image(displayImages[i])
            lab = labels[:,0]
            pred = outputs[:,0]
            print("expected: " + str(lab[i]) + " | actual: %.4f" %(pred[i].data[0]))
            i += 1

    print("Test accuracy: {}%".format(100 * correct / tsset_size))
    
main()
