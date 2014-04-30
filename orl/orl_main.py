import numpy as np
import pickle
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import spectral_clustering
from python.multiview.utils.find_sigma import find_sigma
import numpy.linalg as la
import copy
from sklearn.cluster import KMeans
from cvxopt import matrix
from cvxopt.solvers import qp

# Load PCA feature and Gabor feature
file_pkl = open("pca_gabor.pkl","rb")
tmp = pickle.load(file_pkl)

f_pca = scale(tmp['pca'])
f_gabor = scale(tmp['gabor'])
f_both = np.concatenate((f_pca,f_gabor),axis=1)

sigma_pca = find_sigma(f_pca)
print "sigma_pca: ",sigma_pca
sigma_gabor = find_sigma(f_gabor)
print "sigma_gabor: ",sigma_gabor

tp_aff_pca = rbf_kernel(f_pca,gamma=1./sigma_pca)
tp_aff_gabor = rbf_kernel(f_gabor,gamma=1./sigma_gabor)
aff_pca = tp_aff_pca
aff_gabor = tp_aff_gabor

n_img = 400
# Number of clusters
K = 40

# True labels
label_true = []
for i in range(40):
    for j in range(10):
        label_true.append(i)

# Spectral Clustering using Eigenfaces
#label_pred_pca = spectral_clustering(aff_pca,n_clusters=K)
#nmi_pca = nmi(label_pred_pca,label_true)
#print nmi_pca

# Spectral Clustering using Gabor features
#label_pred_gabor = spectral_clustering(aff_gabor,n_clusters=K)
#nmi_gabor = nmi(label_pred_gabor,label_true)
#print nmi_gabor

# Spectral Clustering using both features
#clf_both = SpectralClustering(n_clusters=K,gamma=2.,affinity='rbf')
#label_pred_both = clf_both.fit_predict(f_both)
#nmi_both = nmi(label_pred_both,label_true)
#print nmi_both

#print "Combined:"
#for alpha in np.arange(0.1,1.0,0.1):
#    aff = alpha*aff_pca+(1-alpha)*aff_gabor
#    label_pred_add = spectral_clustering(aff,n_clusters=K)
#    nmi_add = nmi(label_pred_add,label_true)
#    print nmi_add

# Optimization for kernel weights
# Number of sources
M = 2

# Construct Q
Q = matrix([[np.trace(aff_pca.dot(aff_pca)),np.trace(aff_pca.dot(aff_gabor))],\
            [np.trace(aff_gabor.dot(aff_pca)),np.trace(aff_gabor.dot(aff_gabor))]])
#Q = matrix([1.,0.5,0.5,1.0],(2,2))
print "Q: ",Q

# Initialization of beta and U
beta = np.array([0.5,0.5])
beta_old = np.array([0.,0.])
U = np.ones((n_img,40))
U_old = np.zeros((n_img,40))
eps = 1e-6
n_iter_max = 50
n_iter = 0
residue = 2.
residue_old = 1.

while abs(residue-residue_old)/residue_old>eps and n_iter<n_iter_max:
    U_old = U
    # Find U by eigendecomposition
    aff = beta[0]*aff_pca+beta[1]*aff_gabor
    eig_val,eig_vec = la.eig(aff)
    # Sort eigenvalues and eigenvectors
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    U = eig_vec[:,0:40]
    
    # Optimize beta
    gamma = matrix([np.trace(aff_pca.dot(U).dot(U.T)),\
            np.trace(aff_gabor.dot(U).dot(U.T))])
    G = matrix([[-1.0,0.],[0.,-1.0]])
    h = matrix([0.,0.])
    A = matrix([1.,1.],(1,2))
    b = matrix(1.0)
    res = qp(2*Q,-2*gamma,G,h,A,b)
    beta = res['x'].T
    residue_old = la.norm(aff-U_old.dot(U_old.T))
    residue = la.norm(aff-U.dot(U.T))
    n_iter = n_iter+1
    print n_iter
    print "beta: ",beta
    print "-2*gamma: ",-2*gamma
    print "Residue: ",residue

clf = KMeans(n_clusters=K,init='random')
label_u = clf.fit_predict(U)
nmi_u = nmi(label_u,label_true)
print nmi_u
