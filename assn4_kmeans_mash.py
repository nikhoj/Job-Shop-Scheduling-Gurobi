#!/usr/bin/python

########################################################
# K-means clustering #
# Sk. Mashfiqur Rahman (CWID: A20102717) #
# collect data from: http://yann.lecun.com/exdb/mnist/#
#######################################################

import numpy as np
import matplotlib.pyplot as plt


def k_means(k, data, _mean):
    N = data.shape[0]
    min_d = np.zeros((N,1),dtype="double")
    min_k = np.zeros((N,1),dtype=int)
    z = np.zeros(shape=(N, k),dtype=int)
    mean_init = np.zeros(np.shape(_mean))
    count = 0
    allowance = 1e-7
    change = np.linalg.norm(_mean - mean_init)
    while change > allowance:
        mean_init = _mean.copy()
        mean_update = np.zeros((k,data.shape[1]),dtype=np.float)
        for i in range(N):
            dist = np.zeros((k,1),dtype="double")
            for j in range(k):
                dist[j] = np.linalg.norm(data[i] - _mean[j])
            min_k[i] = np.argmin(dist) # return the indices of minimum value
            min_d[i] = np.amin(dist)

        for i,j in enumerate(min_k):
                z[i][j] = 1.

        for l in range(k):
            clusters = []
            for i in range(len(min_k)):
                if min_k[i] == l:
                    clusters.append(i)
            center = np.mean(data[clusters],axis=0)
            mean_update[l,:] = center

        _mean = mean_update
        count = count + 1
        change = np.linalg.norm(_mean - mean_init)
        cost = np.sum(min_d)
        print('iteration:',count,'Change:',change,'Cost:',cost)

    return z,_mean

# Test images
test_images_file = open('t10k-images-idx3-ubyte.idx3-ubyte','rb')
test_images = test_images_file.read()
test_images = bytearray(test_images)
test_images = test_images[16:]
test_images_file.close()

test_images = np.array(test_images,"float64")
d = 28 * 28
n = test_images.shape[0] / d
test_images = test_images.reshape(int(n), d)

# Test labels
test_labels_file = open('t10k-labels-idx1-ubyte.idx1-ubyte','rb')
test_labels = test_labels_file.read()
test_labels = bytearray(test_labels)
test_labels = test_labels[8:]
test_labels_file.close()

test_labels = np.array(test_labels)
test_labels_1 = test_labels.reshape(test_labels.shape[0], 1)

N = test_images.shape[0]
flag = 2  # 0: random data points, 1: k-means++ assignment, 2: data point drawn from each label class, 3:final

if flag == 0:
    k = 10
    mean = test_images[np.random.randint(low=0, high=N, size=k)]
    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mean[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid for random datapoints")
    plt.savefig("kmeans1.png", format='png')
    plt.show()

    Z, m = k_means(k, test_images, mean)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid for random datapoints")
    plt.savefig("kmeans2.png", format='png')
    plt.show()

if flag == 1:
    k = 10
    mean = np.zeros(shape=(k, test_images.shape[1]))
    mean[0] = test_images[np.random.randint(N)]
    for j in range(1,k):
        dist = np.array([np.min([np.linalg.norm(x-c)**2 for c in mean]) for x in test_images])
        p = np.true_divide(dist,np.sum(dist))
        cp = p.cumsum()
        r = np.random.random()
        ix = np.where(cp >= r)[0][0]
        mean[j] = np.copy(test_images[ix])

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mean[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid by K-means++")
    plt.savefig("kmeans3.png", format='png')
    plt.show()

    Z, m = k_means(k, test_images, mean)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid for K-means++")
    plt.savefig("kmeans4.png", format='png')
    plt.show()

if flag == 2:
    k = 10
    mean = np.zeros(shape=(k, test_images.shape[1]))

    for i in range(k):
        label_images = np.copy(test_images[test_labels == i])
        np.random.shuffle(label_images)
        mean[i] = label_images[0]

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mean[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid by known label")
    plt.savefig("kmeans5.png", format='png')
    plt.show()

    Z, m = k_means(k, test_images, mean)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid for known label")
    plt.savefig("kmeans6.png", format='png')
    plt.show()

if flag == 3:
    k = 3
    mean = np.zeros(shape=(k, test_images.shape[1]))
    mean[0] = test_images[np.random.randint(N)]
    for j in range(1,k):
        dist = np.array([np.min([np.linalg.norm(x-c)**2 for c in mean]) for x in test_images])
        p = np.true_divide(dist,np.sum(dist))
        cp = p.cumsum()
        r = np.random.random()
        ix = np.where(cp >= r)[0][0]
        mean[j] = np.copy(test_images[ix])

    figure, axes = plt.subplots(nrows=1, ncols=3)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mean[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid for K-means++ using k=3")
    plt.savefig("kmeans7.png", format='png')
    plt.show()

    Z, m = k_means(k, test_images, mean)

    figure, axes = plt.subplots(nrows=1, ncols=3)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid for K-means++ using k=3")
    plt.savefig("kmeans8.png", format='png')
    plt.show()