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

# Extract HoG features for each image
feat_hog = []
for i in range(img.shape[0]):
    newarr,hog_image = feature.hog(img[i],orientations=9,\
        pixels_per_cell=(8,8),cells_per_block=(3,3),visualise=True,\
        normalise=True)
    feat_hog.append(newarr)
    #print i

feat_hog = np.array(feat_hog)
print feat_hog.shape

# PCA on HoG features
pca = PCA(n_components=40)
feat_hog = pca.fit_transform(feat_hog)
print "Variance Ratio: ",sum(pca.explained_variance_ratio_)

#feat_hog = scale(feat_hog)

# save HoG features
file_pkl = open("face_hog.pkl","wb")
pickle.dump(feat_hog,file_pkl)
file_pkl.close()

# Compute similarity matrix
flag_sigma = 'global'
sigma_hog, aff_hog = compute_affinity(feat_hog,flag_sigma=flag_sigma,\
        sigma=100.,nn=8)
print "kernel computation finished"

label_pred_identity = spectral_clustering(aff_hog,n_clusters=20)
nmi_identity = nmi(label_pred_identity,img_identity)
print "NMI with identity: ",nmi_identity

label_pred_pose = spectral_clustering(aff_hog,n_clusters=4)
nmi_pose = nmi(label_pred_pose,img_pose)
print "NMI with pose: ",nmi_pose

plt.imshow(hog_image,cmap=cm.Greys_r)
plt.show()
