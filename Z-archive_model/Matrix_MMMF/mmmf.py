#!/usr/bin/env python
# coding: utf-8

# In[10]:


from cvxpy import *
import numpy as np
import pandas as pd
import random
random.seed(58)
from dataset import extract_matrix_users_movies_ratings


# In[17]:


data = pd.read_csv('./data/data_train.csv')
number_of_users, number_of_movies = (10000, 1000)
matrix, _ = extract_matrix_users_movies_ratings(data)


# In[22]:


def learnMMMF(y, c):
    n,m = y.shape
    n_obs = np.count_nonzero(y)
    obs = np.nonzero(y)

    A = Variable((n,n))
    B = Variable((m,m))
    X = Variable((n,m))

    t = Variable()
    e = Variable(n_obs)

    objective = Minimize(t + c * sum(e))

    constraints = []
    constraints.append(bmat([[A,X],[X.T,B]]) >> 0)
    constraints.append(diag(A) <= t)
    constraints.append(diag(B) <= t)
    constraints.append(e >= 0)

    for x in enumerate(zip(obs[0], obs[1])):
        ind = x[0]
        i, a = x[1][0], x[1][1]
        constraints.append(y[i,a] * X[i,a] >= 1 - e[ind])

    prob = Problem(objective, constraints)
    result = prob.solve(kktsolver=ROBUST_KKTSOLVER, verbose=True)
    return result


# In[ ]:


learnMMMF(matrix, 1.0)

