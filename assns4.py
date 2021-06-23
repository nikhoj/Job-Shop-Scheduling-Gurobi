import numpy as np


        #Data import
data = np.loadtxt('crash.txt')
vdata = data.copy()
np.random.shuffle(vdata)
vdata = (vdata - np.min(vdata , axis = 0)) / (np.max(vdata , axis = 0) - np.min(vdata, axis = 0))
x = data[:,0].reshape((len(data),1))
x_N = (x - np.min(x)) / (np.max(x) - np.min(x))
y = data[:,1].reshape((len(data),1))
y_N = (y - np.min(y)) / (np.max(y) - np.min(y))

#s = np.std(x)
sigma = np.std(y)
sigma_N = (sigma - np.min(y)) / (np.max(y) - np.min(y))
beta = 1 / sigma_N ** 2


def Kernel_1(xi, xj, sigma):
        
    return np.exp(-(xi - xj)**2./(2. * sigma ** 2.))

k = Kernel_1(x, x.T, .1)