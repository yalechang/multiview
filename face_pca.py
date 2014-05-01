""" This script extracts PCA features from CMU face dataset
"""

import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
from skimage import feature
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA


# Load faces dataset
file_pkl = open("faces4_data.pkl","rb")
img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye = pickle.load(file_pkl)
file_pkl.close()

# Normalize image
for i in range(img.shape[0]):
    img[i] = (img[i]-img[i].min())*1./(img[i].max()-img[i].min())

# PCA on image
pca = PCA(n_components=40)
feat_pca = pca.fit_transform(img.reshape(img.shape[0],\
    img.shape[1]*img.shape[2]))
print "Variance Ratio: ",sum(pca.explained_variance_ratio_)

# save PCA image
file_pkl = open("face_pca.pkl","wb")
pickle.dump(feat_pca,file_pkl)
file_pkl.close()

# compute affinity matrix
flag_sigma = 'global'

sigma_pca,aff_pca = compute_affinity(feat_pca,flag_sigma=flag_sigma,\
        sigma=100.,nn=8)

label_pred_identity = spectral_clustering(aff_pca,n_clusters=20)
nmi_identity = nmi(label_pred_identity,img_identity)

label_pred_pose = spectral_clustering(aff_pca,n_clusters=4)
nmi_pose = nmi(label_pred_pose,img_pose)

print "nmi_identity",nmi_identity,"nmi_pose",nmi_pose

