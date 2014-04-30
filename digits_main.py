import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import spectral_clustering
from time import time
import numpy.linalg as la
from sklearn.cluster import KMeans
from cvxopt import matrix
from cvxopt.solvers import qp
from python.multiview.utils.compute_affinity import compute_affinity
from python.multiview.utils.opt_affinity_weight import opt_affinity_weight

basepath = "/Users/changyale/dataset/mfeat/"
filename_fou = "mfeat-fou"
filename_fac = "mfeat-fac"

data_fou = scale(np.loadtxt(basepath+filename_fou))
data_fac = scale(np.loadtxt(basepath+filename_fac))

# 'global', 'local', 'manual'
flag_sigma = 'global'

# Default sigma=50.
sigma_fou, aff_fou = compute_affinity(data_fou,flag_sigma=flag_sigma,\
        sigma=147.9369,nn=8)
# Default sigma = 100.
sigma_fac, aff_fac = compute_affinity(data_fac,flag_sigma=flag_sigma,\
        sigma=422.6228,nn=8)
print "kernel computing finished"
if flag_sigma == 'local':
    sigma_fou_init = sum(sigma_fou**2)/len(sigma_fou)
    sigma_fac_init = sum(sigma_fac**2)/len(sigma_fac)

K = 10
label_true = []
for i in range(K):
    for j in range(200):
        label_true.append(i)

# Spectral Clustering: Fourier coefficient
label_fou = spectral_clustering(aff_fou,n_clusters=K)
nmi_fou = nmi(label_fou,label_true)
print "NMI(Source 1)",nmi_fou

# SC: Autocorrelation Profile
label_fac = spectral_clustering(aff_fac,n_clusters=K)
nmi_fac = nmi(label_fac,label_true)
print "NMI(Source 2)",nmi_fac

# kernel addition
for alpha in np.arange(0.1,1.0,0.1):
    aff_add = alpha*aff_fou+(1-alpha)*aff_fac
    label_add = spectral_clustering(aff_add,n_clusters=K)
    nmi_add = nmi(label_add,label_true)
    print "NMI(a*source_1+(1-a)*source_2)",(alpha,nmi_add)

# Parameter settings
affs = [aff_fac,aff_fou]
v_lambda = 0.
dim_q = 10
tol = 1e-6
n_iter_max = 200

Y = np.identity(aff_fac.shape[0])
# Optimization
U,beta,res,flag = opt_affinity_weight(affs,Y,v_lambda=v_lambda,dim_q=dim_q,\
        tol=tol,n_iter_max=n_iter_max)

if flag == True:
    print "Optimization is successful"
    print "weights: ",beta
    print "residue: ",res
else:
    print "Optimization fails"

clf = KMeans(n_clusters=K,init='random')
label_u = clf.fit_predict(U)
nmi_u = nmi(label_u,label_true)
print "NMI(Optimization w.r.t weights)",nmi_u


