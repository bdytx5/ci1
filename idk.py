# note: by this point, you should know how to look the below packages up if you don't understand what it is!!!
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import struct
from skimage.transform import resize
import numpy as np
import random
import math


###########################################
###########################################
# declare the MLP
###########################################
###########################################

# lets make a simple MLP in PyTorch 
# the following code DECLARES a particular MLP
#  note: it is not an "INSTANCE" of a MLP yet, thats later in the code below, this just says "what is in a MLP"
# in the following class, we define the MLP in the 'init' and computation/forward pass is in the 'forward'
#  note: XORMlp is the name of the class that we are making (I made up that name!)
#  note: nn.Module is what we are inheriting from in PyTorch to work with making neurons/networks
#  note: since we keep this all in PyTorch, it will do the back prop for us (read up on autograd)
class XORMlp(nn.Module):
    def __init__(self, D_in, H, D_out): # D_in = number of inputs to the MLP
                                        # H = number of hidden nodes (we are assuming a single layer here)
                                        # D_out = number of output neurons
        super(XORMlp, self).__init__()  # this says call the nn.Module init's function and do whatever it needs
        self.linear1 = nn.Linear(D_in, H) # input to hidden layer (note: nn.Linear is for a dot product)
        self.linear2 = nn.Linear(H, D_out) # hidden layer to output (note: nn.Linear is for a dot product)
    def forward(self, x):                # this is the function for doing "forward propagation"/firing of the MLP
        h_pred = torch.sigmoid(self.linear1(x)) # h = dot(input,w1) 
                                         #  and nonlinearity (relu), you could have used a sigmoid or something
        y_pred = self.linear2(h_pred) # network_output = dot(h,w2) (note, no nonlinearity here, add if you like)
        return y_pred 

###########################################
###########################################
# create an instance of a MLP
###########################################
###########################################

# here is a network with 2 inputs connected to 4 hidden neurons, which are connected to one output neuron    
D_in, H, D_out = 196, 100, 10
net = XORMlp(D_in, H, D_out)

###########################################
###########################################
# what error function?
###########################################
###########################################

# lets define a custom criteria/loss function 
#  note: does this on each output value, so this is our sum os squared error function
def criterion(out,label): # we are going to pass out for our network output and label is what we wanted to get! 
    return torch.mean((label - out)**2) # the ** 2 means squared

###########################################
###########################################
# what optimization algorithm to use?
###########################################
###########################################

# lets pick an optimizer, SGD = stochastic gradient descent, with learning rate and momentum
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)




with open('train-labels-idx1-ubyte', 'rb') as f:
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:6008]

w, h = 10,10;
labs = [[0 for x in range(w)] for y in range(h)] 

for i in range(10):
    for j in range(10):
        if i == j:
            labs[i][j] = 1


L = torch.randn(2000, 10)
y = []
yindexes = []
yc = np.zeros((10))
index = 0
for i in range(6000):
    if(yc[data[i]] < 200):
        for j in range(10):
            L[index][j] = labs[data[i]][j]
        np.append(y, labs[data[i]]) 
        yc[data[i]] = yc[data[i]] + 1
        yindexes.append(i)
        index = index + 1
yindexes = np.array(yindexes)
y = np.array(y)




with open('train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

import matplotlib.pyplot as plt

torchdata = torch.randn(2000,196)


x = []
index = 0
for dex in yindexes:
    im = data[dex,:,:]
    res = resize(im,(14, 14))
    fres = res.flatten()
    nfres = [g / 255 for g in fres]
    for j in range(10):
        torchdata[index][j] = nfres[j]
    index = index + 1
    




###########################################
###########################################
# Make our data set
###########################################
###########################################

# xor data set


# the labels




# lets serial train (versus batch or mini-batch)
for epoch in range(100):
    for i in range(2000):
        X = Variable(torchdata[i,:])
        Y = Variable(L[i])
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

# display the results



with open('t10k-labels-idx1-ubyte', 'rb') as f:
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:6008]

w, h = 10,10;
labs = [[0 for x in range(w)] for y in range(h)] 

for i in range(10):
    for j in range(10):
        if i == j:
            labs[i][j] = 1


L = torch.randn(2000, 10)
y = []
yindexes = []
yc = np.zeros((10))
index = 0
for i in range(6000):
    if(yc[data[i]] < 200):
        for j in range(10):
            L[index][j] = labs[data[i]][j]
        np.append(y, labs[data[i]]) 
        yc[data[i]] = yc[data[i]] + 1
        yindexes.append(i)
        index = index + 1
yindexes = np.array(yindexes)
y = np.array(y)




with open('t10k-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

import matplotlib.pyplot as plt

torchdata = torch.randn(2000,196)


x = []
index = 0
for dex in yindexes:
    im = data[dex,:,:]
    res = resize(im,(14, 14))
    fres = res.flatten()
    nfres = [g / 255 for g in fres]
    for j in range(10):
        torchdata[index][j] = nfres[j]
    index = index + 1
    




s = 0
conf = np.array(np.zeros((10,10)))
for i in range(2000):
        # layer 1
    X = Variable(torchdata[i,:])
    Y = Variable(L[i])
    optimizer.zero_grad()
    outputs = net(X)
    values, pred = outputs.max(0)
    values, act = Y.max(0)    
    if act == pred:
        s = s + 1
    conf[act][pred] = conf[act][pred] + 1

print(s/2000)
print(conf)
# plt.ylabel('error')
# plt.xlabel('epochs')
# plt.show()
        
