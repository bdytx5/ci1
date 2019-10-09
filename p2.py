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
x = np.zeros(shape=(100,196))
for dex in range(100):

    im = data[dex,:,:]
    # plt.imshow(im, cmap='gray')
    # plt.show()
    res = cv2.resize(im, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
    x[dex] = np.array(res).flatten()


with open('train-labels-idx1-ubyte', 'rb') as f:
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))[8:108]

labs = np.zeros(shape=(10,10))

for i in range(10):
    for j in range(10):
        if i == j:
            labs[i][j] = 1
Y = np.zeros(shape=(60000,10))

for i in range(100):
    Y[i] = labs[data[i]]



def eee(val):
    return np.exp(val)


def tanh(x, derive=False): # x is the input, derive is do derivative or not
    if derive:
        return (1.0 - x**2)
                           # depends on how you call the function
    return ((eee(x)-eee(-x))/(eee(x)+eee(-x)))



epochs = 1000
eta = 0.1# learning rate

x = np.array([
    [0, 0, 0, 1],  # data point (x,y,z, bias) 
    [0, 0, 1, 1],  
    [0, 1, 0, 1],  
    [0, 1, 1, 1],
    [1, 0, 0, 1],   
    [1, 0, 1, 1],  
    [1, 1, 0, 1],  
    [1, 1, 1, 1],
]) 

# labels
y = np.array([[1,0], #outputs 
              [0,1], 
              [0,1],
              [1,0],
              [0,1], 
              [1,0],
              [1,0],
              [1,0]
             ])

w1 = np.random.normal(0,2,(100, 97))
w2 = np.random.normal(0,2,(10, 101))





# for e in range(epochs):
#     ee = 0 # errorx
#     for i in range(8):
#         # layer 1
#         v1 = np.dot(x[i, :], np.transpose(w1))
#         y1 = tanh(v1)
#         # layer 2
#         v2 = np.dot(np.append(y1,1), np.transpose(w2))
#         y2 = tanh(v2)
#         #layer 3
#         y3 = np.dot(np.append(y2,1), np.transpose(w3))
#         #backprop 
#         err = -np.array(y[i, :]-y3)
#         dEdW3 = np.dot(np.transpose(np.array([err])),np.array([np.append(y2,1)])) # e/dw3
#         errw3 = np.array(np.dot(err, w3))[0:(w3.shape[1] - 1)] # exclude bias since its not part of de/dy2
#         tanhv2errw3 = errw3 * tanh(y2, True)
#         dEdW2 = np.dot(np.transpose(np.array([tanhv2errw3])),np.array([np.append(y1,1)]))# e/dw2
#         tanhv2errw3w2 = np.dot(tanhv2errw3, w2)[0:(w2.shape[1] - 1)]
#         tanhv2errw3w2tanhv1 = tanhv2errw3w2 * tanh(y1, True)
#         dEdW1 = np.dot(np.transpose(np.array([tanhv2errw3w2tanhv1])), np.array([x[i, :]]))# e/dw1
#         ee = ee + ((1.0/2.0) * np.power((y[i, :] - y3), 2.0))
#         # adjustments
#         w3 = w3 - eta*dEdW3
#         w2 = w2 - eta*dEdW2
#         w1 = w1 - eta*dEdW1
#         print(ee)

# print('w1----',w1)
# print('w2----',w2)
# print('w3----', w3)


