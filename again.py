import numpy as np
import random
import math


def eee(val):
    return np.exp(np.clip(val, -709, 709))


def tanh(x, derive=False): # x is the input, derive is do derivative or not
    if derive:
        return (1.0 - x**2)
                           # depends on how you call the function
    return ((eee(x)-eee(-x))/(eee(x)+eee(-x)))



epochs = 1
eta = 0.01# learning rate

x = np.array([
    [0, 0, 0],  # data point (x,y,z, bias) 
    [0, 0, 1],  
    [0, 1, 0],  
    [0, 1, 1],
    [1, 0, 0],   
    [1, 0, 1],  
    [1, 1, 0],  
    [1, 1, 1],
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
  
# weights in format - w12 = layer 1, neuron 2
w1 = np.array([[0.1,0.2,0.3], 
               [0.1,0.1,0.1],
                [0.3,0.3,0.3]])
b1 = np.array([0.2, 0.1, 0.9])


w2 = np.array([[0.0,0.0,0.0],
                 [0.1,0.1,0.1], 
                 [0.1,0.1,0.1],
                 [0.2,0.2,0.2]])
b2 = np.array([0.0, 0.2, 0.0, -0.1])



w3 = np.array([[1.5,1.2,1.0,0.0], [0.0,0.8,0.1,0.0]])
b3 = np.array([[-0.2, -0.1]])


for e in range(epochs):
    ee = 0 # errorx
    for i in range(2):
        # layer 1
        v1 = np.dot(x[i, :], np.transpose(w1)) + b1
        y11D = tanh(v1)

        # layer 2
        v2 = np.dot(y11D, np.transpose(w2)) + b2
        y2 = tanh(v2)
                #layer 3

        y3 = np.dot(y2, np.transpose(w3)) + b3
        err = np.array(y[i, :]-y3)
        y2 = np.array([y2])
        # print(err)
        dEdW3 = np.dot(np.transpose(err),y2)
        # print(dEdW3)


        y1 = np.array([y11D])

        errw3 = np.array(np.dot(err, w3))
        t2 = np.array([tanh(v2, True)])
        tanhv2errw3 = np.dot(t2, np.transpose(errw3))
        
        
        dEdW2 = np.dot(tanhv2errw3, y1)


        print(y1, '------',tanhv2errw3w2)


        dEdW2 = np.dot(w2, np.transpose(tanhv2errw3w2))
        







        
        # dy_dw3 = np.array([y2])
        # dEdw2 = np.dot(np.transpose(np.array([tanh(y2, True)])), np.dot(err,w3))

        
        # # dE_dw = np.dot(np.transpose(dE_dY), dy_dw)
        # # w3 = w3 - (eta * dE_dw)
        # #dE/dw2 
        # dE_w2 = np.dot(np.transpose(w3), np.transpose(dE_dY)) 
        # print(dE_w2)

        # dE_w2 = np.dot(tanh(y2, True), dE_w2)
        # # dE_w2 = np.dot(np.array([y1]), dE_w2)
        # print(dE_w2)
        # # dE_w1 = np.dot(np.transpose(w2), np.transpose(dE_dY)) 
        # # dE_w1 = np.dot(np.array([y1]), dE_w1)








       
