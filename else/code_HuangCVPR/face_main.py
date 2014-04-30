import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
from cvxopt import matrix
from cvxopt.solvers import qp
import numpy.linalg as la
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load faces dataset
file_face = open("faces4_data.pkl","rb")
img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye = pickle.load(file_face)
file_face.close()

# Load FFT features
file_fft = open("face_fft.pkl","rb")
tmp = pickle.load(file_fft)
feat_fft = tmp[:,0:32]
file_fft.close()

# Load Gabor features
file_gabor = open("face_gabor.pkl","rb")
feat_gabor = pickle.load(file_gabor)
file_gabor.close()

# Load LBP features
file_lbp = open("face_lbp.pkl","rb")
feat_lbp = pickle.load(file_lbp)
file_lbp.close()

# Compute similarity matrix for FFT and Gabor
flag_sigma = 'global'
sigma_fft, aff_fft = compute_affinity(feat_fft,flag_sigma=flag_sigma)
sigma_gabor, aff_gabor = compute_affinity(feat_gabor,flag_sigma=flag_sigma)
sigma_lbp, aff_lbp = compute_affinity(feat_lbp,flag_sigma=flag_sigma)
print "kernel computation finished"

# Spectral Clustering using FFT
K = 4
label_pred_fft = spectral_clustering(aff_fft,n_clusters=K)
label_pred_gabor = spectral_clustering(aff_gabor,n_clusters=K)

nmi_fft_identity = nmi(label_pred_fft,img_identity)
nmi_gabor_identity = nmi(label_pred_gabor,img_identity)
print "nmi_fft_identity: ", nmi_fft_identity
print "nmi_gabor_identity: ",nmi_gabor_identity

for alpha in np.arange(0.1,1.0,0.1):
    aff_add = alpha*aff_fft+(1-alpha)*aff_gabor
    label_pred_add = spectral_clustering(aff_add,n_clusters=K)
    nmi_add_identity = nmi(label_pred_add,img_identity)
    print (alpha,nmi_add_identity)

# Weighted summation
M = 2
Q = matrix([[np.trace(aff_fft.dot(aff_fft)),\
             np.trace(aff_fft.dot(aff_gabor))],\
            [np.trace(aff_gabor.dot(aff_fft)),\
             np.trace(aff_gabor.dot(aff_gabor))]])
print "Q: ",Q

n_instances = feat_fft.shape[0]
dim_q = K
# Initialization of beta and U
tmp = np.random.rand(1)[0]
beta = np.array([tmp,1-tmp])
U = np.ones((n_instances,dim_q))
U_old = np.zeros((n_instances,dim_q))
eps = 1e-6
n_iter_max = 50
n_iter = 0
residue = 2.
residue_old = 1.

# Existing solution
Y = np.zeros((n_instances,20))
for i in range(n_instances):
    Y[i,img_identity[i]] = 1
v_lambda = 200.
arr_tmp = v_lambda*Y.dot(Y.T)

while abs(residue-residue_old)/residue_old>eps and n_iter<n_iter_max:
    U_old = U
    # Find U by eigendecomposition
    aff = beta[0]*aff_fft + beta[1]*aff_gabor
    eig_val,eig_vec = la.eig(2*aff-arr_tmp)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    U = eig_vec[:,0:dim_q]

    # Optimize beta
    gamma = matrix([np.trace(aff_fft.dot(U).dot(U.T)),\
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
    print "Residue: ",residue

clf = KMeans(n_clusters=K,init='random')
label_u = clf.fit_predict(U)
nmi_identity = nmi(label_u,img_identity)
print "NMI identity(Optimization w.r.t weights): ",nmi_identity
nmi_pose = nmi(label_u,img_pose)
print "NMI pose",nmi_pose


img_avg = np.zeros((4,img.shape[1],img.shape[2]))
cnt_avg = np.zeros((4,1))
for i in range(len(label_u)):
    img_avg[label_u[i]] += img[i]
    cnt_avg[label_u[i]] += 1.

for i in range(4):
    img_avg[i] = img_avg[i]/cnt_avg[i]
    plt.imshow(img_avg[i],cmap=cm.Greys_r)
    plt.show()

