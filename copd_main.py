import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from python.multiview.utils.compute_affinity import compute_affinity
from python.multiview.utils.opt_affinity_weight import opt_affinity_weight
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load case
file_case = open("copd_case.pkl","rb")
data_case = pickle.load(file_case)
file_case.close()

# Load control
file_control = open("copd_control.pkl","rb")
data_control,features_name = pickle.load(file_control)
file_control.close()

# Existing labels: case control
label_e = np.hstack([np.zeros((1,1000)),np.ones((1,1000))])[0]
Y = np.zeros((2000,2))
for i in range(0,1000):
    Y[i,0] = 1
for i in range(1000,2000):
    Y[i,1] = 1

# Stack two dataset vertically
data = np.vstack([data_case,data_control])

# Compute affinity matrix for each source
n_instances,n_features = data.shape
affs = []
flag_sigma = 'global'
for j in range(n_features):
    sigma,tmp = compute_affinity(data[:,j].reshape(n_instances,1),\
            flag_sigma=flag_sigma)
    affs.append(tmp)
    print j

# Save information
# file_pkl = open("copd_all.pkl","wb")
# pickle.dump([data,features_name,affs,label_e],file_pkl)
# file_pkl.close()

v_lambda_range = np.arange(0,10,0.5)
v_mu_range = [0]
dim_q = 4
tol = 1e-6
n_iter_max = 200

# Store iteration results
nmi_e = []
beta_vec = []
res_vec = []
inertia_vec = []
mse_vec = []
# Silhouette Score
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
    for v_mu_idx in range(len(v_mu_range)):
        # Optimization
        v_lambda = v_lambda_range[v_lambda_idx]
        v_mu = v_mu_range[v_mu_idx]
        U,beta,res,mse, flag = opt_affinity_weight(affs,Q,Y,v_lambda=v_lambda,\
                v_mu=v_mu,dim_q=dim_q,tol=tol,n_iter_max=n_iter_max)

        if flag == True:
            print "Optimization is successful",v_lambda_idx,v_mu_idx
            #print "weights: ",beta
            #print "residue: ",res
            beta_vec.append(np.array(beta)[0])
            res_vec.append(res)
            mse_vec.append(mse)
        else:
            print "Optimization fails",v_lambda_idx

        # Repeat KMeans 50 times to reduce randomness
        label_tmp = []
        inertia_tmp = []
        for i in range(50):
            clf = KMeans(n_clusters=dim_q,init='random')
            clf.fit(U)
            label_tmp.append(list(clf.labels_))
            inertia_tmp.append(clf.inertia_)
    
        idx_tmp = inertia_tmp.index(min(inertia_tmp))
        label_u = label_tmp[idx_tmp]
        inertia_vec.append(min(inertia_tmp))
        score_vec.append(silhouette_score(U,np.array(label_u)))
        nmi_e.append(nmi(label_e,label_u))

beta_vec = np.array(beta_vec)

# Plot the result
plt.figure(0)
plt.plot(v_lambda_range,nmi_e,'r',label='nmi_e')
plt.xlabel("lambda(tradeoff between clustering quality and novelty)")
plt.ylabel("NMI value")
plt.legend(loc='upper left')

plt.figure(1)
plt.plot(v_lambda_range,inertia_vec)
plt.xlabel("lambda")
plt.ylabel("Inertia Value")
plt.figure(2)

plt.plot(v_lambda_range,mse_vec)
plt.xlabel("lambda")
plt.ylabel("MSe")

plt.show()

