"""
Submitted by: Md Mahabub Uz Zaman
Assignment 4
"""

import numpy as np
import matplotlib.pyplot as plt


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


problem3()
