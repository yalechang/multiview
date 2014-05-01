import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
from sklearn.decomposition import PCA

# Load faces dataset
file_pkl = open("faces4_data.pkl","rb")
img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye = pickle.load(file_pkl)
file_pkl.close()

tmp_fft = []
# Normalization of dataset
for i in range(img.shape[0]):
    img[i] = (img[i]-img[i].min())*1./(img[i].max()-img[i].min())
    tmp_fft.append(np.fft.rfft2(img[i]))

img_fft = np.zeros((img.shape[0],tmp_fft[0].shape[0],3*2))
for i in range(img.shape[0]):
    img_fft[i,:,:] = np.concatenate((np.real(tmp_fft[i][:,0:3]),\
            np.imag(tmp_fft[i][:,0:3])),axis=1)

img_fft = img_fft.reshape(img_fft.shape[0],img_fft.shape[1]*img_fft.shape[2])

print img_fft.shape
# PCA on FFT features
pca = PCA(n_components=14)
feat_fft = pca.fit_transform(img_fft)
print "Variance Ratio: ",sum(pca.explained_variance_ratio_)

# Normalization of features
#feat_fft = scale(feat_fft)

# save FFT data
file_pkl = open("face_fft.pkl","wb")
pickle.dump(feat_fft,file_pkl)
file_pkl.close()

# Compute kernel matrix for FFT data
flag_sigma = 'global'

sigma_fft, aff_fft = compute_affinity(feat_fft,flag_sigma=flag_sigma,\
        sigma=336.,nn=8)
if flag_sigma == 'local':
    sigma_fft_init = sum(sigma_fft**2)/len(sigma_fft)
    print "sigma_fft_init: ",sigma_fft_init

label_pred_identity = spectral_clustering(aff_fft,n_clusters=20)
nmi_identity = nmi(label_pred_identity,img_identity)

label_pred_pose = spectral_clustering(aff_fft,n_clusters=4)
nmi_pose = nmi(label_pred_pose,img_pose)

print "nmi_identity",nmi_identity,"nmi_pose",nmi_pose

