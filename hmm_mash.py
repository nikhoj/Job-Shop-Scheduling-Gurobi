import numpy as np
import matplotlib.pyplot as plt


T = 1000
s = 2
die = ""
rolls = ""
l = 0
a = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
for t in range(T):
    if not l:
        die += "F"
        rolls += str(np.random.randint(1,7))
        if np.random.rand() > .95:
            l = 1
    else:
        die += "L"
        rolls += str(a[np.random.randint(0, 10)])
        if np.random.rand() > .9:
            l = 0
print('Rolls:', rolls)
print('Die:  ',die)

A = np.array([[.95, .10],[.05, .90]])
xz = np.vstack((np.array([1./ 6., 1./ 6., 1./ 6., 1./ 6., 1./ 6., 1./ 6.]),
               np.array([.1, .1, .1, .1, .1, .5])))

xt = np.zeros(T)
for i in range(T):
    if die[i] == 'L':
        xt[i] = 1.

xf = np.zeros(shape=(s, T))
xf[:, 0] = 1
for t in range(1, T):
    xf[:, t] = (xf[0, t-1] * A[0, :] + xf[1, t-1] * A[1, :]) * xz[:, int(rolls[t])-1]
    xf[:, t] *= (1./(xf[0, t] + xf[1, t]))

plt.figure(figsize = (14,4))
plt.plot(range(T), xt, c = '.85')
plt.plot(range(T), xf[1, :], 'r-')
plt.xlabel("Time")
plt.ylabel("P(z)")
plt.title("Forward pass")
plt.show()

xb = np.zeros(shape=(s, T))
xb[:, T-1] = 1.
for t in range(T-2, -1, -1):
    xb[:, t] = (xb[0, t+1] * A[:, 0] + xb[1, t+1] * A[:, 1]) * xz[:, int(rolls[t])-1]
    xb[:, t] *= (1. / (xb[0, t] + xb[1, t]))

x = np.zeros(shape=(s, T))
for t in range(T):
    x[:, t] = xf[:, t] * xb[:, t]
    x[:, t] *= (1./np.sum(x[:, t]))

plt.figure(figsize = (14,4))
plt.plot(range(T), xt, c = '.85')
plt.plot(range(T), x[1, :], 'r-')
plt.xlabel("Time")
plt.ylabel("P(z)")
plt.title("Complete inference")
plt.show()