"""
Submitted by: Md Mahabub Uz Zaman
Assignment 4
"""

import numpy as np
import matplotlib.pyplot as plt

def problem1():
    def GP(kernel = 1):
        
        n = kernel
        #Data import
        data = np.loadtxt('crash.txt')

        vdata = data.copy()
        np.random.shuffle(vdata)
        vdata = (vdata - np.min(vdata , axis = 0)) / (np.max(vdata , axis = 0) - np.min(vdata, axis = 0))
        
        
        x = data[:,0].reshape((len(data),1))
        x = (x - np.min(x)) / (np.max(x) - np.min(x))#x / np.max(x, axis = 0).reshape((1,1))
        
        y = data[:,1].reshape((len(data),1))
        
        sd = np.std(y)
        sd_n = (sd - np.min(y)) / (np.max(y) - np.min(y))
        b = 1 / sd_n ** 2
        
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        

        #Kernel Definition
        def Kernel_1(xi, xj, sigma):
        
            return np.exp(-(xi - xj)**2./(2. * sigma ** 2.))
        
        
        def Kernel_2(xi, xj, sigma):
        
            return np.exp(-np.abs(xi - xj) / sigma)
        
        def MSE(y, yhat):
                
                e =  ((y - yhat) **2)
                e = 0.5 * np.sum(e, axis = 0)
                e = (2 * e) / len(y)
                amse = e / len(y)
                
                return amse
        
        def best_sigma_finder(vx ,vy ,vx_star, vy_star , kernel = n, beta = b):
            EM_min = 1000

            for s in sigma:    

                if kernel == 1:
                    k = Kernel_1(vx, vx_star.T, s)
                    K = Kernel_1(vx , vx.T, s)
                
                else:
                    k = Kernel_2(vx, vx_star.T, s)
                    K = Kernel_2(vx , vx.T, s)
                C = K + (1 / beta) * np.identity(72)
                vy_star_hat = k.T.dot(np.linalg.inv(C).dot(vy))
                
                EM = MSE(vy_star, vy_star_hat)
                if EM < EM_min:
                    EM_min = EM
                    best_sigma = s
                    
            return best_sigma
        
        N = x.shape[0]
        sigma = np.linspace(0.05,.5,100).reshape((100,1))   #taking different sigma
        fold_num = 5
        fold_size = int(N / fold_num)
        vdata_x = vdata[:,0].reshape((len(data),1))
        vdata_y = vdata[:,1].reshape((len(data),1))
        fd_x = vdata_x[0:fold_size*5]
        fd_y = vdata_y[0:fold_size*5]
        
        
         #validation start here
        vx1 = fd_x[18:]
        vy1 = fd_y[18:]
        vx1_star = fd_x[:18]
        vy1_star = fd_y[:18]   
        
        vx2 = np.vstack((fd_x[0:18], fd_x[36:]))
        vy2 = np.vstack((fd_y[0:18], fd_y[36:]))
        vx2_star = fd_x[18:36]
        vy2_star = fd_y[18:36] 
        
        vx3 = np.vstack((fd_x[0:36], fd_x[54:]))
        vy3 = np.vstack((fd_y[0:36], fd_y[54:]))
        vx3_star = fd_x[36:54]
        vy3_star = fd_y[36:54] 
        
        vx4 = np.vstack((fd_x[0:54], fd_x[72:]))
        vy4 = np.vstack((fd_y[0:54], fd_y[72:]))
        vx4_star = fd_x[54:72]
        vy4_star = fd_y[54:72] 
        
        vx5 = fd_x[0:72]
        vy5 = fd_y[0:72]
        vx5_star = fd_x[72:]
        vy5_star = fd_y[72:]
        
        bs1 = best_sigma_finder(vx1,vy1,vx1_star, vy1_star)
        bs2 = best_sigma_finder(vx2,vy2,vx2_star, vy2_star)
        bs3 = best_sigma_finder(vx3,vy3,vx3_star, vy3_star)
        bs4 = best_sigma_finder(vx4,vy4,vx4_star, vy4_star)
        bs5 = best_sigma_finder(vx5,vy5,vx5_star, vy5_star)
        
        best_sigma = (bs1 + bs2 + bs3 + bs4 + bs5) / 5
        
        #applying best sigma in test set
        sigma = best_sigma
        beta = 1 / sigma **2
        
        
        
        x_star = np.linspace(x.min(), x.max(), N).reshape((N,1))
        y_star = np.zeros(N).reshape((N,1))
        
        if n == 1:
            k = Kernel_1(x, x_star.T, sigma)
            K =  Kernel_1(x, x.T, sigma)
        else:
            k = Kernel_2(x, x_star.T, sigma)
            K =  Kernel_2(x, x.T, sigma)
            
        C = K + (1 / beta) * np.identity(N)
        
        y_star = k.T.dot(np.linalg.inv(C)).dot(y)
        
        print("Best sigma = " + str(best_sigma))
        plt.plot(x, y, c='g', label="Real data")
        plt.plot(x_star, y_star, c='b', label="GP")
        plt.legend(loc = 'upper right')
        plt.show()
        

    print("Squared Exponential")
    GP(1)
    print("Exponential")
    GP(2)
    
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


def problem3():


    time = np.arange(0,1000)
    
    a = np.array([1./6, 1./6, 1./6, 1./6, 1./6, 1./6])
    b = np.array([.1,.1,.1,.1,.1,.5])
    E = np.vstack((a,b)).T
    A = np.array([[.95, .10],[.05, .90]])
    
    f_rolls = [1,2,3,4,5,6]
    l_rolls = [1,2,3,4,5,6,6,6,6,6]
    
    
    rolls = np.zeros((1000,1))
    state = np.zeros((1000,1))
    
    start = f_rolls[np.random.randint(0,6)]
    state[0] = 0
    rolls[0] = start
    
    
    for i in range(1,1000):
        if state[i-1] == 0:
            rolls[i] = f_rolls[np.random.randint(0,6)]
            state[i] = 1 if np.random.rand() > .95 else 0
        
            
        elif state[i-1] == 1:
            rolls[i] = l_rolls[np.random.randint(0,10)]
            state[i] = 0 if np.random.rand() > .90 else 1
    
    
    #prediction start here
    alpha = np.zeros((1000,2))
    alpha[0,0] = E[start - 1,0] / E[start - 1,: ].sum()
    alpha[0,1] = E[start - 1,1] / E[start - 1,: ].sum()
    
    for i in range(1, 1000):
        alpha[i] = E[int(rolls[i])-1, :] * (A.dot(alpha[i-1] * E[int(rolls[i-1])-1]))
        alpha[i] /= alpha[i].sum()
    
    beta = np.zeros((1000,2))
    beta[999,0] = 1
    beta[999,1] = 1
    
    for i in range(998, -1, -1):
        beta[i] = E[int(rolls[i])-1, :] * (A.dot(alpha[i+1] * E[int(rolls[i+1])-1]))
        beta[i] /= beta[i].sum()
    
    post = np.zeros((1000,2))
    post = (alpha * beta)
    post /= np.sum(alpha * beta, axis = 1).reshape(1000,1)
         
       
    
    plt.figure(figsize = (14,4))
    plt.plot(time, state, c = '.75' , label = "Actual Loaded dice" )
    plt.plot(time,alpha[:,1], c = 'b', label = "probability of Loaded dice")
    plt.xlabel("Time steps")
    plt.ylabel ( "p(loaded)")
    plt.title("Forward step")
    plt.legend(loc = 'upper right')
    plt.show()
    
    plt.figure(figsize = (14,4))
    plt.plot(time, state, c = '.75', label = "Actual Loaded dice")
    plt.plot(time,post[:,1], c = 'g' , label = "probability of Loaded dice")
    plt.ylabel ( "p(loaded)")
    plt.xlabel("Time steps")
    plt.title("posterior")
    plt.legend(loc = 'upper right')
    plt.show()

problem1()
problem2()
problem3()
