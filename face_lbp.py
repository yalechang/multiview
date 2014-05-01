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
import mahotas
from sklearn.decomposition import PCA

# Load faces dataset
file_pkl = open("faces4_data.pkl","rb")
img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye = pickle.load(file_pkl)
file_pkl.close()

# Extract local binary pattern from images
feat_lbp = []
for i in range(img.shape[0]):
    tmp = feature.local_binary_pattern(img[i],8,1,method='uniform')
    feat_lbp.append(np.double(np.histogram(tmp,bins=range(10),normed=True)[0]))
    #feat_lbp.append(mahotas.features.lbp(img[i],1,8))
feat_lbp = scale(np.array(feat_lbp))

# PCA on LBP features
#pca = PCA(n_components=20)
#feat_lbp = pca.fit_transform(feat_lbp)
#print "Variance Ratio: ",sum(pca.explained_variance_ratio_)

# Normalization of features
#feat_lbp = scale(feat_lbp)

# Save LBP features
file_pkl = open("face_lbp.pkl","wb")
pickle.dump(feat_lbp,file_pkl)
file_pkl.close()

# Compute affinity matrix
flag_sigma = 'global'
sigma_lbp, aff_lbp = compute_affinity(feat_lbp,flag_sigma=flag_sigma,\
        sigma=100.,nn=8)
print "kernel computation finished"

label_pred_identity = spectral_clustering(aff_lbp,n_clusters=20)
nmi_identity = nmi(label_pred_identity,img_identity)
print "NMI with identity: ",nmi_identity

label_pred_pose = spectral_clustering(aff_lbp,n_clusters=4)
nmi_pose = nmi(label_pred_pose,img_pose)
print "NMI with pose: ",nmi_pose

