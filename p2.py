import numpy as np
import struct
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

labs = np.zeros(shape=(10,10))
for i in range(10):
    for j in range(10):
        if i == j:
            labs[i][j] = 1

with open('/Users/macbookpro/Desktop/ci1/train-labels-idx1-ubyte', 'rb') as f:
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:6008]

y = []
yindexes = []
yc = np.zeros((10))
for i in range(6000):
    if(yc[data[i]] < 200):
        y.append(labs[data[i]])
        np.append(y, labs[data[i]]) 
        yc[data[i]] = yc[data[i]] + 1
        yindexes.append(i)
yindexes = np.array(yindexes)
y = np.array(y)

with open('/Users/macbookpro/Desktop/ci1/train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

x = []
for dex in yindexes:
    im = data[dex,:,:]
    res = cv2.resize(im, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
    x.append(np.append(res.flatten(),1))
x = np.array(x)/255





# alt 

# with open('train-labels-idx1-ubyte', 'rb') as f:
#     data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:6008]

# labs = np.zeros(shape=(10,10))

# for i in range(10):
#     for j in range(10):
#         if i == j:
#             labs[i][j] = 1

# y = []
# for i in range(6000):
#     y.append(labs[data[i]])
# y = np.array(y)

# with open('train-images-idx3-ubyte','rb') as f:
#     magic, size = struct.unpack(">II", f.read(8))
#     nrows, ncols = struct.unpack(">II", f.read(8))
#     data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
#     data = data.reshape((size, nrows, ncols))

# import matplotlib.pyplot as plt

# x = []
# for dex in range(6000):
#     im = data[dex,:,:]
#     res = cv2.resize(im, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
#     x.append(np.append(res.flatten(),1))
# x = np.array(x)/255





def tanh2(x, derive=False): # x is the input, derive is do derivative or not
    if derive:
        return (1.0 - x**2)
                           # depends on how you call the function
    return ((eee(x)-eee(-x))/(eee(x)+eee(-x)))



def eee(val):
    return np.exp(val)

def tanh(x, derive=False): 
    if derive: 
        return x * (1.0 - x) 
    return ( 1.0 / (1.0 + np.exp(-x)))

epochs = 1000000
eta = 0.6 # learning rate
B = 0.4
bs = 10

w1 = np.random.normal(0,1,(100, 197))
w2 = np.random.normal(0,1,(10, 101))
bw1 = np.array(np.zeros((2001,100,197)))
bw2 = np.array(np.zeros((2001,10,101)))

mbw1 = np.array(np.zeros((100,197)))
mbw2 = np.array(np.zeros((10,101)))
bc = 0
actualEpochs = 0
ee = np.zeros(epochs)
for e in range(epochs):
    actualEpochs = e
    for i in range(2000):
        
        if bc == bs:
            bc = 0
            w1 = w1 - mbw1
            w2 = w2 - mbw2
            mbw1 = np.array(np.zeros((100,197)))
            mbw2 = np.array(np.zeros((10,101)))
        bc = bc + 1
            
        # layer 1
        v1 = np.dot(x[i, :], np.transpose(w1))
        y1 = tanh(v1)
        # layer 2
        v2 = np.dot(np.append(y1,1), np.transpose(w2))
        y2 = tanh(v2)
        #backprop 
        err = -np.array(y[i, :]-y2)
        errphiprimev2 = err*tanh(y2,derive=True)
        dEdW2 = np.dot(np.transpose(np.array([errphiprimev2])), np.array([np.append(y1,1)])) # e/dw2
        errphiprimev2w2 = np.array(np.dot(np.array(errphiprimev2), w2))[0:(w2.shape[1] - 1)] # exclude bias since its not part of de/dy2
        errphiprimev2w2phiprimev1 = errphiprimev2w2 * tanh(y1, derive=True)
        dEdW1 = np.dot(np.transpose(np.array([errphiprimev2w2phiprimev1])), np.array([x[i, :]]))
        ee[e] = ee[e] + ((1.0/2.0) * ((y[i, :] - y2)**2).mean(axis=0))
        # adjustments
        mbw2 = mbw2 + (bw2[i] + eta*dEdW2)
        mbw1 = mbw1 + (bw1[i]+ eta*dEdW1)
        bw1[i+1] = B*(bw1[i]+ eta*dEdW1)
        bw2[i+1] = B*(bw2[i] + eta*dEdW2)
    print(ee[e])
    if(ee[e] < 3):
        print('total epochs ', e)
        break
 
print('w1----',w1)
print('w2----',w2)



with open('/Users/macbookpro/Desktop/ci1/t10k-labels-idx1-ubyte', 'rb') as f:
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:2008]

labs = np.zeros(shape=(10,10))

for i in range(10):
    for j in range(10):
        if i == j:
            labs[i][j] = 1

y = []
for i in range(2000):
    y.append(labs[data[i]])
y = np.array(y)

with open('/Users/macbookpro/Desktop/ci1/t10k-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

import matplotlib.pyplot as plt

x = []
for dex in range(2000):
    im = data[dex,:,:]
    res = cv2.resize(im, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
    x.append(np.append(res.flatten(),1))
x = np.array(x)/255

s = 0
conf = np.array(np.zeros((10,10)))
for i in range(2000):
        # layer 1
    v1 = np.dot(x[i, :], np.transpose(w1))
    y1 = tanh(v1)
        # layer 2
    v2 = np.dot(np.append(y1,1), np.transpose(w2))
    y2 = tanh(v2)
    act = y[i].argmax()
    pred = y2.argmax()
    if act == pred:
        s = s + 1
    conf[act][pred] = conf[act][pred] + 1

print(s/2000)
print(conf)
for i in range(10):
    print(i,conf[i])
print(actualEpochs)

plt.plot(ee[0:actualEpochs])
plt.ylabel('error')
plt.xlabel('epochs')
plt.show()
        