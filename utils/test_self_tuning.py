import numpy as np
from sklearn.preprocessing import scale
import numpy.linalg as la

basepath = '/Users/changyale/dataset/mfeat/'
filename_fou = 'mfeat-fou'
filename_fac = 'mfeat-fac'

data_fou = scale(np.loadtxt(basepath+filename_fou))
data_fac = scale(np.loadtxt(basepath+filename_fac))

dis_fou = np.zeros((2000,2000))
dis_fac = np.zeros((2000,2000))

# Compute pariwise distances between samples
for i in range(2000):
    for j in range(2000):
        dis_fou[i,j] = la.norm(data_fou[i,:]-data_fou[j,:])
        dis_fou[j,i] = dis_fou[i,j]
        dis_fac[i,j] = la.norm(data_fac[i,:]-data_fac[j,:])
        dis_fac[j,i] = dis_fac[i,j]

print "Distance Computing Finished"

# Sort distances
dis_fou_sorted = np.sort(dis_fou)
dis_fac_sorted = np.sort(dis_fac)
print "Sorting Finished"

# Number of nearest neighbors
nn = 6
sigma_fou = dis_fou_sorted[:,nn]
sigma_fac = dis_fac_sorted[:,nn]

print sigma_fou
print sigma_fac

