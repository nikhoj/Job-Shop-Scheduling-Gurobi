import numpy as np
import matplotlib.pyplot as plt

with open('t10k-images-idx3-ubyte.idx3-ubyte', 'rb') as test_images:
    testX = bytearray(test_images.read())[16:]
    testX = np.array(testX).reshape((10000,784))

with open('t10k-labels-idx1-ubyte.idx1-ubyte' , 'rb') as test_label:
    testY = bytearray(test_label.read())[8:]
    testY = np.array(testY).reshape((10000,1))


def init_cheat(testX, testY):
    mu = np.zeros((10,784))  
    
    for i in range(10):
        indx = np.arange(0,10000).reshape((10000,1))
        indx = indx[testY == i]
        dset = testX[indx]
        n = np.random.randint(0, dset.shape[0] -1)
        mu[i] = dset[n]
    
    return mu



