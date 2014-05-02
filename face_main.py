import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from python.multiview.utils.opt_affinity_weight import opt_affinity_weight
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
import numpy.linalg as la
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score

# Load faces dataset
file_face = open("faces4_data.pkl","rb")
img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye = pickle.load(file_face)
file_face.close()

# Load FFT features
file_fft = open("face_fft.pkl","rb")
tmp = pickle.load(file_fft)
feat_fft = tmp
file_fft.close()

# Load Gabor features
file_gabor = open("face_gabor.pkl","rb")
feat_gabor = pickle.load(file_gabor)
file_gabor.close()

# Load LBP features
file_lbp = open("face_lbp.pkl","rb")
feat_lbp = pickle.load(file_lbp)
file_lbp.close()

# Load HoG features
file_hog = open("face_hog.pkl","rb")
feat_hog = pickle.load(file_hog)
file_hog.close()

# Load PCA features
file_pca = open("face_pca.pkl","rb")
feat_pca = pickle.load(file_pca)
file_pca.close()


# Compute similarity matrix for RawData, FFT, Gabor, LBP, HoG, PCA
flag_sigma = 'global'
sigma_raw, aff_raw = compute_affinity(img.reshape(img.shape[0],img.shape[1]*\
        img.shape[2]),flag_sigma=flag_sigma)
sigma_fft, aff_fft = compute_affinity(feat_fft,flag_sigma=flag_sigma)
sigma_gabor, aff_gabor = compute_affinity(feat_gabor,flag_sigma=flag_sigma)
sigma_lbp, aff_lbp = compute_affinity(feat_lbp,flag_sigma=flag_sigma)
sigma_hog, aff_hog = compute_affinity(feat_hog,flag_sigma=flag_sigma)
sigma_pca, aff_pca = compute_affinity(feat_pca,flag_sigma=flag_sigma)

# Normalization of matrix using Frobenius norm
flag_normalization = False
if flag_normalization == True:
    aff_raw = aff_raw/la.norm(aff_raw)
    aff_fft = aff_fft/la.norm(aff_fft)
    aff_gabor = aff_gabor/la.norm(aff_gabor)
    aff_lbp = aff_lbp/la.norm(aff_lbp)
    aff_hog = aff_hog/la.norm(aff_hog)
    aff_pca = aff_pca/la.norm(aff_pca)

# Centering kernel matrix
flag_centering = False
H = np.eye(img.shape[0])-1./img.shape[0]*np.ones((img.shape[0],img.shape[0]))
if flag_centering == True:
    aff_raw = H.dot(aff_raw).dot(H)
    aff_fft = H.dot(aff_fft).dot(H)
    aff_gabor = H.dot(aff_gabor).dot(H)
    aff_lbp = H.dot(aff_lbp).dot(H)
    aff_hog = H.dot(aff_hog).dot(H)
    aff_pca = H.dot(aff_pca).dot(H)

# Spectral Clustering using FFT and Gabor
n_identity = 20
#label_pred_fft = spectral_clustering(aff_fft,n_clusters=n_identity)
#label_pred_gabor = spectral_clustering(aff_gabor,n_clusters=n_identity)
#nmi_fft_identity = nmi(label_pred_fft,img_identity)
#nmi_gabor_identity = nmi(label_pred_gabor,img_identity)
#print "nmi_fft_identity: ", nmi_fft_identity
#print "nmi_gabor_identity: ",nmi_gabor_identity

#for alpha in np.arange(0.1,1.0,0.1):
#    aff_add = alpha*aff_fft+(1-alpha)*aff_gabor
#    label_pred_add = spectral_clustering(aff_add,n_clusters=n_identity)
#    nmi_add_identity = nmi(label_pred_add,img_identity)
#    print (alpha,nmi_add_identity)

# Existing solution: identity
n_instances = img.shape[0]
Y = np.zeros((n_instances,n_identity))
for i in range(n_instances):
    Y[i,img_identity[i]] = 1

################################ Parameter Settings ##########################
#affs = [aff_raw,aff_pca,aff_lbp,aff_hog,aff_gabor,aff_fft]
affs = [aff_pca,aff_lbp]
v_lambda_range = np.arange(0,10,1)
# Upper bound for 1-norm of beta
mu = 1.
dim_q = 4
tol = 1e-6
n_iter_max = 200


# Store iteration results
nmi_pose = []
nmi_identity = []
beta_vec = []
res_vec = []
mse_vec = []
# Metric from running K-Means
inertia_vec = []
score_vec = []

# Show Quadratic term matrix
n_sources = len(affs)
Q = np.zeros((n_sources,n_sources))
for i in range(n_sources):
    for j in range(i,n_sources):
        Q[i,j] = np.trace(affs[i].dot(affs[j]))
        Q[j,i] = Q[i,j]
print "Q: ",Q

# Iterations
for v_lambda_idx in range(len(v_lambda_range)):
    # Optimization
    v_lambda = v_lambda_range[v_lambda_idx]
    U,beta,res,mse, flag = opt_affinity_weight(affs,Q,Y,mu,v_lambda=v_lambda,\
            dim_q=dim_q,tol=tol,n_iter_max=n_iter_max)

    if flag == True:
        print "Optimization is successful",v_lambda_idx
        beta_vec.append(np.array(beta)[0])
        res_vec.append(res)
        mse_vec.append(mse)
    else:
        print "Optimization fails",v_lambda_idx

    n_pose = 4
    n_eye = 2

    # Repeat KMeans 50 times to reduce randomness
    label_tmp = []
    inertia_tmp = []
    for i in range(50):
        clf = KMeans(n_clusters=n_pose,init='random')
        clf.fit(U)
        label_tmp.append(list(clf.labels_))
        inertia_tmp.append(clf.inertia_)

    idx_tmp = inertia_tmp.index(min(inertia_tmp))
    label_u = label_tmp[idx_tmp]
    inertia_vec.append(min(inertia_tmp))
    score_vec.append(silhouette_score(U,np.array(label_u)))
    nmi_identity.append(nmi(label_u,img_identity))
    nmi_pose.append(nmi(label_u,img_pose))

    n_show = n_pose
    # Show mean image
    img_avg = np.zeros((n_show,img.shape[1],img.shape[2]))
    cnt_avg = np.zeros((n_show,1))
    for i in range(len(label_u)):
        img_avg[label_u[i]] += img[i]
        cnt_avg[label_u[i]] += 1.

    for i in range(n_show):
        img_avg[i] = img_avg[i]/cnt_avg[i]
        #plt.imshow(img_avg[i],cmap=cm.Greys_r)
        #plt.show()

beta_vec = np.array(beta_vec)

print nmi_pose
# Plot the result
#plt.figure(0)
#plt.plot(nmi_pose,mse_vec)
#plt.show()

#plt.figure(1)
#plt.plot(v_lambda_range,inertia_vec)
#plt.xlabel("lambda")
#plt.ylabel("Inertia Value")
#plt.figure(2)

#plt.plot(v_lambda_range,mse_vec)
#plt.xlabel("lambda")
#plt.ylabel("MSe")
