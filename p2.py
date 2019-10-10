import numpy as np
import struct
import cv2
import numpy as np
import random
import math

with open('train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

import matplotlib.pyplot as plt
x = np.ones(shape=(100,197))
for dex in range(100):

    im = data[dex,:,:]
    res = cv2.resize(im, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
    x[dex][0:196] = np.array(res).flatten()


with open('train-labels-idx1-ubyte', 'rb') as f:
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:108]

labs = np.zeros(shape=(10,10))

for i in range(10):
    for j in range(10):
        if i == j:
            labs[i][j] = 1
y = np.zeros(shape=(60000,10))

for i in range(100):
    y[i] = labs[data[i]]



def eee(val):
    return np.exp(val)


def tanh2(x, derive=False): # x is the input, derive is do derivative or not
    if derive:
        return (1.0 - x**2)
                           # depends on how you call the function
    return ((eee(x)-eee(-x))/(eee(x)+eee(-x)))

def tanh(x, derive=False): # x is the input, derive is do derivative or not
    if derive: # ok, says calc the deriv?
        return x * (1.0 - x) # note, you might be thinking ( sigmoid(x) * (1 - sigmoid(x)) )
                           # depends on how you call the function
    return ( 1.0 / (1.0 + eee(-x)) )


epochs = 2500
eta = 0.01# learning rate


w1 = np.random.normal(0,2,(100, 197))
w2 = np.random.normal(0,2,(10, 101))





for e in range(epochs):
    ee = 0 # error
    for i in range(50):
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

        errphiprimev2 = np.array(np.dot(errphiprimev2, w2))[0:(w2.shape[1] - 1)] # exclude bias since its not part of de/dy2
        errphiprimev2w2phiprimev1 = errphiprimev2 * tanh(y1, True)
        dEdW1 = np.dot(np.transpose(np.array([errphiprimev2w2phiprimev1])), np.array([x[i, :]]))
 

        ee = ee + ((1.0/2.0) * np.power((y[i, :] - y2), 2.0))

        # adjustments

        w2 = w2 - eta*dEdW2
        w1 = w1 - eta*dEdW1
        # print(ee)
 
print('w1----',w1)
print('w2----',w2)
print(ee)


