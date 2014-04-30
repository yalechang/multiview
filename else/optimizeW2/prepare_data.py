"""This script generate a synthetic dataset to be used for testing the
effectiveness of Alternative Clustering Embedding Algorithm presented in
Donglin Niu's paper <Iterative Discovery of Multiple Alternative Clustering
Views>
"""
print __doc__

import numpy as np
import matplotlib.pyplot as plt
import pickle

n_features = 4
data = np.zeros((600,n_features))

# Noise level
eps = 0.2

## Generate dimension 1,2
x = np.arange(-5,5,10./300)
y_1 = -2.+0.1*x**2+eps*np.random.randn(len(x))
y_2 = 0.25*x**2+eps*np.random.randn(len(x))

# For class 1
data[0:300,0] = x
data[0:300,1] = y_1

# For class 2
data[300:600,0] = x
data[300:600,1] = y_2 

# Generate corresponding ground truth labels
labels_view_1 = [0]*300+[1]*300

# Generate dimension 3,4,5

# For class 1
x_1 = np.arange(-10,10,20./150)
y_1 = np.sqrt(100-x_1**2)+eps*np.random.randn(len(x_1))
x_2 = np.arange(10,-10,-20./150)
y_2 = -np.sqrt(100-x_2**2)+eps*np.random.randn(len(x_2))
data[0:150,2] = x_1
data[0:150,3] = y_1
data[150:300,2] = x_2
data[150:300,3] = y_2

# For class 2
x_1 = np.arange(-5,5,10./100)
y_1 = np.sqrt(25-x_1**2)+eps*np.random.randn(len(x_1))
x_2 = np.arange(5,-5,-10./100)
y_2 = -np.sqrt(25-x_2**2)+eps*np.random.randn(len(x_2))
data[300:400,2] = x_1
data[300:400,3] = y_1
data[400:500,2] = x_2
data[400:500,3] = y_2

# For class 3
data[500:600,2:4] = np.random.rand(100,2)

# Generate corresponding ground truth labels
labels_view_2 = [0]*300+[1]*200+[2]*100

# Store the generated dataset in a pickle file to be used later
file_pkl = open("synthetic_1.pkl","wb")
pickle.dump([data,labels_view_1,labels_view_2],file_pkl)
file_pkl.close()

# Make plots of the generated data
plt.figure(0)
plt.scatter(data[0:300,0],data[0:300,1])
plt.scatter(data[300:600,0],data[300:600,1])

plt.figure(1)
plt.scatter(data[500:600,2],data[500:600,3])
plt.scatter(data[0:300,2],data[0:300,3])
plt.scatter(data[300:500,2],data[300:500,3])
plt.show()

