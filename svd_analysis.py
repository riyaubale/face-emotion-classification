#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Enable interactive rotation of graph
get_ipython().run_line_magic('matplotlib', 'widget')

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Load data for activity
X = np.loadtxt('sdata.csv',delimiter=',')


# In[2]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='o', alpha=0.1)
ax.scatter(0,0,0,c='b', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

plt.show()


# In[3]:


# Subtract mean
X_m = X - np.mean(X, 0)


# In[4]:


# display zero mean scatter plot
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_m[:,0], X_m[:,1], X_m[:,2], c='r', marker='o', alpha=0.1)

ax.scatter(0,0,0,c='b', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

plt.show()


# In[5]:


# Use SVD to find first principal component

U,s,VT = np.linalg.svd(X_m,full_matrices=False)

# complete the next line of code to assign the first principal component to a
a = VT[0,:]


# In[6]:


# display zero mean scatter plot and first principal component

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#scale length of line by root mean square of data for display
ss = s[0]/np.sqrt(np.shape(X_m)[0])

ax.scatter(X_m[:,0], X_m[:,1], X_m[:,2], c='r', marker='o', label='Data', alpha=0.1)

ax.plot([0,ss*a[0]],[0,ss*a[1]],[0,ss*a[2]], c='b',label='Principal Component')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')


ax.legend()
plt.show()


# # Question 2h

# In[7]:


# Use SVD to find first principal component

U,s,VT = np.linalg.svd(X_m,full_matrices=False)

a1 = VT[0,:] # first principal component
a2 = VT[1,:] # second principal component
print(a1)
print(a2)

# display zero mean scatter plot and principal components

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#scale length of line by root mean square of data for display
ss1 = s[0]/np.sqrt(np.shape(X_m)[0])
ss2 = s[1]/ np.sqrt(np.shape(X_m)[0])

ax.scatter(X_m[:,0], X_m[:,1], X_m[:,2], c='r', marker='o', label='Data', alpha=0.1)

ax.plot([0,ss1*a1[0]],[0,ss1*a1[1]],[0,ss1*a1[2]], c='b',label='1st Principal Component')
ax.plot([0,ss2*a2[0]],[0,ss2*a2[1]],[0,ss2*a2[2]], c='b',label='2nd Principal Component')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')


ax.legend()
plt.show()


# # Question 2i

# In[8]:


U,s,VT = np.linalg.svd(X_m,full_matrices=False)


U2 = U[:, :2]
s2 = np.diag(s[:2])
V2 = VT[:2, :]

X_2 = U2 @ s2 @ V2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_m[:, 0], X_m[:, 1], X_m[:, 2], c='r', marker='o', label='Original Data', alpha=0.1)
ax.scatter(X_2[:, 0], X_2[:, 1], X_2[:, 2], c='b', marker='o', label='Rank-two Approximation', alpha=0.1)


ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')


ax.legend()
plt.show()


# # Question 2k

# In[9]:


X_1 = U[:, :1] @ np.diag(s[:1]) @ VT[:1, :] 
E1 = X_m - X_1
E2 = X_m - X_2

fro_1 = np.linalg.norm(E1, 'fro') ** 2
fro_2 = np.linalg.norm(E2, 'fro') ** 2

print("Rank 1 Approximation:", fro_1)
print("Rank 2 Approximation:", fro_2)


# # Question 3a&b

# In[11]:


data = loadmat('face_emotion_data.mat')
X = data['X']
y = data['y'].flatten()


# In[12]:


import numpy as np

indices = np.arange(X.shape[0])
index_groups = np.array_split(indices, 8)
SVD_errors = []
ridge_regression_errors = []

for i in range(len(index_groups)):
    X_valid, Y_valid = X[index_groups[i]], y[index_groups[i]]
    
    for j in range(len(index_groups)):
        if i == j:
            continue
        
        X_test, Y_test = X[index_groups[j]], y[index_groups[j]]
        
        # Excludes validation and test indices
        mask = np.ones(X.shape[0], dtype=bool)
        mask[index_groups[i]] = False
        mask[index_groups[j]] = False
        X_train, Y_train= X[mask], y[mask]
        
        U, s, VT = np.linalg.svd(X_train, full_matrices=False)
        
        best_weights = None
        best_error = float('inf')
        for num_components in range(1, 10):
            weights = VT.T[:, :num_components] @ np.diag(1.0 / s[:num_components]) @ U.T[:num_components, :] @ Y_train
            validation_error = np.mean(np.sign(X_valid @ weights) != np.sign(Y_valid)) # How many are misclassified
            if validation_error < best_error:
                best_weights = weights
                best_error = validation_error
        
        # Test SVD
        test_err = np.mean(np.sign(X_test @ best_weights) != np.sign(Y_test))
        SVD_errors.append(test_err)

        # Ridge Regression
        best_weights = None
        best_error = float('inf')

        for exponent in range(-2, 5):
            regression_coefficient = 2 ** exponent
            if exponent == -2:  regression_coefficient = 0
            
            # Ridge regression formula
            weights = (VT.T[:, :num_components] @ np.diag(s[:num_components] / (s[:num_components]**2 + regression_coefficient)) @ U.T[:num_components, :] @ Y_train)
            
            # Error on validation set
            validation_error = np.mean(np.sign(X_valid @ weights) != np.sign(Y_valid))
            
            if validation_error < best_error:
                best_weights = weights
                best_error = validation_error

        ridge_regression_errors.append(np.mean(np.sign(X_test @ best_weights) != np.sign(Y_test)))


print("SVD Error:", np.mean(SVD_errors))
print("Ridge Regression Error:", np.mean(ridge_regression_errors))

