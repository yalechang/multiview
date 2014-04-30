import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
from skimage.filter import gabor_kernel
from python.multiview.utils.compute_feats import compute_feats
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

# Load faces dataset
file_pkl = open("faces4_data.pkl","rb")
img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye = pickle.load(file_pkl)
file_pkl.close()

# Extract Gabor features
# Step 1: Prepare Gabor filter bank kernels
kernels = []
for theta in range(4):
    theta = theta/4.*np.pi
    for sigma in (1.,2.):
        for frequency in (0.5,1.0):
            kernel = np.real(gabor_kernel(frequency,theta=theta,sigma_x=sigma,\
                    sigma_y=sigma))
            kernels.append(kernel)

# Compute Gabor features
feat_gabor = np.zeros((img.shape[0],16*2))
for i in range(img.shape[0]):
    img[i] = (img[i]-img[i].min())/(img[i].max()-img[i].min())
    feat_gabor[i,:] = compute_feats(img[i],kernels).reshape(1,32)
    #print i

# PCA on Gabor features
#pca = PCA(n_components=4)
#feat_gabor = pca.fit_transform(feat_gabor)
#print "Variance Ratio: ",sum(pca.explained_variance_ratio_)

feat_gabor = scale(feat_gabor)
# Save Gabor Features
file_pkl = open("face_gabor.pkl","wb")
pickle.dump(feat_gabor,file_pkl)
file_pkl.close()

# Compute affinity matrix
flag_sigma = 'global'

sigma_gabor, aff_gabor = compute_affinity(feat_gabor,flag_sigma=flag_sigma,\
        sigma=100.,nn=8)

print "kernel computation finished"

label_pred_identity = spectral_clustering(aff_gabor,n_clusters=20)
nmi_identity = nmi(label_pred_identity,img_identity)
print "NMI with identity: ",nmi_identity

label_pred_pose = spectral_clustering(aff_gabor,n_clusters=4)
nmi_pose = nmi(label_pred_pose,img_pose)
print "NMI with pose: ",nmi_pose

