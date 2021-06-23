import numpy as np
import matplotlib.pyplot as plt


    
def plot(x, post, title):
    plt.figure(figsize=(14, 6))
    plt.plot(x, c='grey', label="Actual Loaded Dice")
    plt.plot(post[:,1], label="Probability of Loaded Dice")
    plt.title(title)
    plt.ylabel("P(Loaded)")
    plt.xlabel("Time steps")
    plt.legend(loc="upper left")
    plt.show()
    
face = np.arange(1,7)
Z = np.array([[1/6 for i in range(6)], [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]]).T
A = np.array([[.95, .05], [.1, .9]])
    
rolls = np.zeros(1000, dtype=int)
x = np.zeros(1000)
state = 0
    
for i in range(1000):
    side = int(np.random.choice(face, p=Z.T[state]))
    rolls[i] = side
    x[i] = state
    
    r = np.random.random()
    if r > A[state,state]:
        state = np.random.randint(0,2)
    
alpha_t  = np.ones((1000, 2))
beta_t = np.ones((1000, 2))
post =  np.ones((1000, 2))
beta_t[999] = 1
alpha_t[0,0] = Z[rolls[0]-1, 0]/ Z[rolls[0]-1,:].sum()
alpha_t[0,1] = Z[rolls[0]-1, 1]/ Z[rolls[0]-1,:].sum()
    
for t in range(1, 999):
    alpha_t[t] = Z[rolls[t]-1, :] * (A.dot(alpha_t[t-1] * Z[rolls[t-1]-1]))
    alpha_t[t] /= alpha_t[t].sum()
    
for t in range(1000):
    post[t] = A.dot(alpha_t[t])
    
plot(x, post,"Forward Step")
for t in range(998, -1, -1):
    beta_t[t] = A.dot(Z[rolls[t+1]-1, :] * beta_t[t+1]* Z[rolls[t+1]-1])
    beta_t[t] /= beta_t[t].sum()
    
for t in range(1000):
    post[t] = alpha_t[t]*beta_t[t]
    post[t] /= post[t].sum()
plot(x, post,"Posterior")
    
