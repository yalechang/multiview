import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import spectral_clustering
from python.multiview.utils.find_sigma import find_sigma
from time import time
import numpy.linalg as la
from sklearn.cluster import KMeans
from cvxopt import matrix
from cvxopt.solvers import qp

basepath = "/Users/changyale/dataset/mfeat/"
filename_fou = "mfeat-fou"
filename_fac = "mfeat-fac"

data = scale(np.loadtxt(basepath+filename_fac))

K = 10
label_true = []
for i in range(K):
    for j in range(200):
        label_true.append(i)

H = np.ones((2000,10))
H_old = np.ones((2000,10))
residue = 2.
residue_old = 1.
n_iter_max = 50
n_iter = 0
eps = 1e-3
sq_val = np.zeros((2000,2000))
for i in range(2000):
    for j in range(i,2000):
        sq_val[i,j] = la.norm(data[i,:]-data[j,:])**2
        sq_val[j,i] = sq_val[i,j]
print "Computing sq_val finished"

"""
sigma = 100.
while abs(residue-residue_old)/residue_old>eps and n_iter<n_iter_max:
    H_old = H
    # Find H by eigendecomposition
    aff = rbf_kernel(data,gamma=1./sigma)
    eig_val,eig_vec = la.eig(aff)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    H = eig_vec[:,0:10]

    # Find sigma by solving an equation
    B = H.dot(H.T)
    tmp = 1./2.*np.sum(B*sq_val)
    sigma_max = 200.
    sigma_min = 0.
    iter_sigma_max = 50
    iter_sigma = 0
    eps_sigma = 1e-3
    sigma_old = np.infty
    sigma = (sigma_max+sigma_min)/2
    f_sigma = np.sum(np.exp(-sq_val/sigma)*sq_val)

    while abs(f_sigma-tmp)>eps_sigma and iter_sigma<iter_sigma_max:
        if f_sigma < tmp:
            sigma_min = sigma
            sigma = (sigma_min+sigma_max)/2
        else:
            sigma_max = sigma
            sigma = (sigma_max+sigma_min)/2

        f_sigma = np.sum(np.exp(-sq_val/sigma)*sq_val)
        iter_sigma = iter_sigma+1
        print "iter_sigma: ",iter_sigma,"sigma: ",sigma," error: ",f_sigma-tmp
    
    residue_old = residue
    residue = la.norm(aff-H.dot(H.T))

    n_iter = n_iter+1
    print "Iteration: ",n_iter,"sigma: ",sigma,"Residue: ",residue
"""
