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
    
        
problem1()
