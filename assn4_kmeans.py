import numpy as np
import matplotlib.pyplot as plt

def problem2():

    def fileop():
        with open('t10k-images-idx3-ubyte.idx3-ubyte', 'rb') as test_images:
            testX = bytearray(test_images.read())[16:]
            testX = np.array(testX).reshape((10000,784))
            #testX[testX > 0] = 1
            
        with open('t10k-labels-idx1-ubyte.idx1-ubyte' , 'rb') as test_label:
            testY = bytearray(test_label.read())[8:]
            testY = np.array(testY).reshape((10000,1))
        
        return testX, testY
    
    testX, testY = fileop()
    rawX, rawY = fileop()
    
    #initializing by taking randomly 10 data points
    def init_rand_kmeans(testX= testX):
        np.random.shuffle(testX)
        mu = testX[np.random.randint(0,10000, 10)] #random initial mu
        
        return mu
    
    def init_kmeans_pp(testX = testX, k = 10):
        N = testX.shape[0]
        
        np.random.shuffle(testX)
        mu = testX[np.random.randint(0,N-1, 1)]
        dist = np.array([(np.linalg.norm(x-mu)) for x in testX])
        dist_sort = np.argsort(dist)
        n = np.linspace(0,N-1,k).astype(int)
        mu = np.zeros((k,784))
        for i in range(k):
            indx = dist_sort[n]
            mu = testX[indx]
            
        return mu
    
    def init_cheat(testX = testX, testY = testY):
        mu = np.zeros((10,784))  
        
        for i in range(10):
            indx = np.arange(0,10000).reshape((10000,1))
            indx = indx[testY == i]
            dset = testX[indx]
            np.random.shuffle(dset)
            n = np.random.randint(0, dset.shape[0] -1)
            mu[i] = dset[n]
        
        return mu
    
    
    def plot_mu(mu, k = 10):
        if k == 3:
            for i in range(3):
                axisn = str(131 + i)
                plt.subplot(axisn)        
                plt.imshow(np.reshape(mu[i], (28,28)), cmap = 'bone')
            plt.show()
        else:
            for i in range(10):
                axisn = str(251+i)
                plt.subplot(axisn)        
                plt.imshow(np.reshape(mu[i], (28,28)), cmap = 'bone')
            plt.show()
    
    
    
    def k_means(testX, mu , k = 10):
        distance = np.array([([np.linalg.norm(x-c) for c in mu]) for x in testX])
        cluster = np.argmin(distance, axis = 1)
    #newmean
    
        mu_N = np.array([np.mean(testX[cluster == i], axis = 0) for i in range(k)])
        
        return np.reshape(cluster, (10000,1)), mu_N
    
    def J(Xn, mu, cluster, k = 10):
        penalty = np.zeros((10000,1))
        for n in range(10000):
            for i in range(k):
                if cluster[n] == i:
                    a = np.reshape(Xn[n,:], (784,1))
                    b = np.reshape(mu[i,:], (784,1))
                    
                    squared_dist = np.sum((a - b) ** 2)
                    penalty[n] += squared_dist
        
        J = np.sum(penalty)
        
        return J
    
    
    def cost_calculation(testX, mu, k = 10):
        
        min_cost = 100000000000
        total_J = 0
    
        for i in range(10000):
            #cluster, mu = random_based_iteration(testX, mu)
            cluster, mu = k_means(testX, mu, k)
            cost = J(testX, mu, cluster, k)
            #print(cost)
            
            if cost < min_cost:
                min_cost = cost
                total_J = min_cost
            else:
                iteration = i
                break
            
            print("(iteration number = " + str(i) + ") : J = " + str(cost))
            
        print(str(iteration) + " iterations, J = " +str(total_J))
        
        return cluster, mu    
    
    
    #kmeans random start from here
    mu = init_rand_kmeans()
    #plot_mu(mu)
    print("Random initialization")
    cluster, mu = cost_calculation(testX, mu)
    plot_mu(mu)
    
    #kmeans ++ start from here
    
    mu = init_kmeans_pp()
    #plot_mu(mu)
    print("K-means ++ initialization")
    cluster, mu = cost_calculation(testX, mu)
    plot_mu(mu)
    
    #cheating initialization start from here
    #testX, testY = fileop()
    mu = init_cheat(rawX, rawY)
    #plot_mu(mu)
    print("Cheating initialization")
    cluster, mu = cost_calculation(rawX, mu)
    plot_mu(mu)
    
    # k = 3 k means ++
    rX, rY = fileop()
    mu = init_kmeans_pp(rX, k = 3)
    #plot_mu(mu, 3)
    print("For k = 3")
    cluster, mu = cost_calculation(testX, mu, 3)
    plot_mu(mu, 3)
    
    
    #printing some cluster with label
    
    def sample(testX = testX, cluster = cluster, k = 3):    
        for k in range(3):
            for i in range(2):
                indx = np.arange(0,10000).reshape((10000,1))
                indx = indx[cluster == k]
                dset = testX[indx]
                np.random.shuffle(dset)
                n = np.random.randint(0, dset.shape[0] -1)
                item = dset[n]
                
                axisn = str(121+i )
                plt.subplot(axisn)
                plt.title("Cluster " + str(k))
                plt.imshow(np.reshape(item, (28,28)), cmap = 'bone')
            plt.show()
    sample()
    
problem2()
    
    

        