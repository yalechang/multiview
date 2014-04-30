import numpy as np
import pickle 
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import spectral_clustering
from sklearn.metrics import normalized_mutual_info_score

# Load dataset from pickle file
file_pkl = open("synthetic_1.pkl","rb")
data,labels_view_1,labels_view_2 = pickle.load(file_pkl)
file_pkl.close()

# Settings of parameters
val_lambda = 1.0
val_sigma = 1.0

# Normalization of the original dataset
data_scaled = scale(data)
n_instances,n_features = data_scaled.shape

# Compute kernel matrix from the scaled dataset
mtr_k = rbf_kernel(data_scaled,gamma=1./val_sigma**2)

"""
# Compute degree for each sample
vec_d = sum(mtr_k)

# Compute Laplacian matrix
mtr_l = np.zeros((n_instances,n_instances))
for i in range(n_instances):
    for j in range(i,n_instances):
        mtr_l[i,j] = mtr_k[i,j]/np.sqrt(vec_d[i]*vec_d[j])
        mtr_l[j,i] = mtr_l[i,j]

# Compute the eigenvalues of Laplacian matrix
eig_val,eig_vec = np.linalg.eig(mtr_l)

# Sort eigenvalues
idx = eig_val.argsort()
eig_val = eig_val[idx]
eig_vec = eig_vec[:,idx]
"""

labels_pred = spectral_clustering(mtr_k,n_clusters=3,n_init=10)
nmi_1 = normalized_mutual_info_score(labels_pred,labels_view_2)
print nmi_1

