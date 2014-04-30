import numpy as np
import pickle
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import spectral_clustering
from sklearn.preprocessing import scale
from python.multiview.utils.opt_affinity_weight import opt_affinity_weight
from python.multiview.utils.normalized_kernel import normalized_kernel
from python.multiview.utils.compute_affinity import compute_affinity
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load webkb data
file_webkb = open("webkb.pkl","rb")
web_id,web_word,aff_link,label_univ,label_topic,univ,topic = \
        pickle.load(file_webkb)
file_webkb.close()

# Construct affinity matrix from webpage content
kernel_content = 'polynomial'

if kernel_content == 'polynomial':
    degree = 3
    coef0 = 1
    aff_word = polynomial_kernel(web_word,degree=degree,coef0=coef0)
    #aff_word = (aff_word-np.min(aff_word))/(np.max(aff_word)-np.min(aff_word))
if kernel_content == 'linear':
    aff_word = linear_kernel(web_word)
    # Normalization of obtained kernel
    aff_word = (aff_word-np.min(aff_word))/(np.max(aff_word)-np.min(aff_word))

if kernel_content == 'cosine':
    aff_tmp = web_word.dot(web_word.T)
    aff_word = np.identity(aff_tmp.shape[0])
    for i in range(aff_tmp.shape[0]):
        for j in range(aff_tmp.shape[1]):
            aff_word[i,j] = aff_tmp[i,j]/np.sqrt(aff_tmp[i,i]*aff_tmp[j,j])
    #aff_word = (aff_word-np.min(aff_word))/(np.max(aff_word)-np.min(aff_word))

if kernel_content == 'gaussian':
    sigma_word,aff_word = compute_affinity(web_word,flag_sigma='global')

# spectral clustering with content affinity matrix
label_pred_word = spectral_clustering(aff_word,n_clusters=4)
nmi_word_univ = nmi(label_univ,label_pred_word)
nmi_word_topic = nmi(label_topic,label_pred_word)
#print "nmi_word_univ: ",nmi_word_univ
#print "nmi_word_topic: ",nmi_word_topic

# spectral clustering with link matrix
label_pred_link = spectral_clustering(aff_link,n_clusters=4)
nmi_link_univ = nmi(label_univ,label_pred_link)
nmi_link_topic = nmi(label_topic,label_pred_link)
#print "nmi_link_univ: ",nmi_link_univ
#print "nmi_link_topic: ",nmi_link_topic

# linear combination of two affinit matrices
for alpha in np.arange(0.1,1.0,0.1):
    aff_add = alpha*aff_word+(1-alpha)*aff_link
    label_pred_add = spectral_clustering(aff_add,n_clusters=4)
    nmi_add_univ = nmi(label_univ,label_pred_add)
    nmi_add_topic = nmi(label_topic,label_pred_add)
    #print (alpha,nmi_add_univ,nmi_add_topic)


# Existing Solution
Y = np.zeros((len(web_id),5))
for i in range(len(web_id)):
    Y[i,label_topic[i]] = 1

# Parameter Settings
aff_word_norm = normalized_kernel(aff_word)
aff_link_norm = normalized_kernel(aff_link)
affs = [aff_word_norm,aff_link_norm]
v_lambda_range = np.arange(0,20,2)
dim_q = 4
tol = 1e-6
n_iter_max = 200

# store iteration results
nmi_univ = []
nmi_topic = []
beta_vec = []
res_vec = []
mse_vec = []
inertia_vec = []

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
    U,beta,res,mse,flag = opt_affinity_weight(affs,Q,Y,v_lambda=v_lambda,\
            dim_q=dim_q,tol=tol,n_iter_max=n_iter_max)

    if flag == True:
        print "Optimization is successful",v_lambda_idx
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
        clf.fit_predict(U)
        label_tmp.append(list(clf.labels_))
        inertia_tmp.append(clf.inertia_)
    
    idx_tmp = inertia_tmp.index(min(inertia_tmp))
    label_u = label_tmp[idx_tmp]
    inertia_vec.append(min(inertia_tmp))
    nmi_univ.append(nmi(label_univ,label_u))
    nmi_topic.append(nmi(label_topic,label_u))

beta_vec = np.array(beta_vec)

# Plot the result
plt.figure(0)
plt.plot(v_lambda_range,nmi_univ,'r',label='NMI_UNIV')
plt.plot(v_lambda_range,nmi_topic,'b',label='NMI_TOPIC')
plt.xlabel("lambda(tradeoff between clustering quality and novelty)")
plt.ylabel("NMI value")
plt.legend(loc='upper left')

plt.figure(1)
plt.plot(v_lambda_range,inertia_vec)
plt.xlabel("lambda")
plt.ylabel("Inertia Value")

plt.show()

print nmi_univ[-1],inertia_vec[-1]

