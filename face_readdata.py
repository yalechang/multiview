import numpy as np
from scipy import misc
from scipy import ndimage
import os
import fnmatch
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import spectral_clustering
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy.linalg as la
import pickle
from python.multiview.utils.compute_affinity import compute_affinity

# Directory of face image
basepath = '/Users/changyale/dataset/faces_4/'

n_img = 32*20

identity = os.walk(basepath).next()[1]

pose = ['left','right','straight','up']
expression = ['angry','happy','neutral','sad']
eye = ['open','sunglasses']

# There're altogether 20 persons
img_name = []
img_identity = []
img_pose = []
img_expression = []
img_eye = []
img = []

for i in range(len(identity)):
    current_path = basepath+identity[i]+'/'
    for j in range(len(pose)):
        for k in range(len(expression)):
            for l in range(len(eye)):
                filename = current_path+identity[i]+'_'+pose[j]+'_'+\
                        expression[k]+'_'+eye[l]+'_4.pgm'
                if os.path.isfile(filename) == True:
                    img_name.append(filename)
                    img_identity.append(i)
                    img_pose.append(j)
                    img_expression.append(k)
                    img_eye.append(l)
                    img.append(np.double(misc.imread(filename)))

img = np.array(img)

# Save img data
file_pkl = open("faces4_data.pkl","wb")
pickle.dump([img,img_name,img_identity,img_pose,img_expression,img_eye,\
        identity,pose,expression,eye],file_pkl)
file_pkl.close()

# Normalization of each image
for i in range(img.shape[0]):
    img[i] = (img[i]-img[i].min())*1./(img[i].max()-img[i].min())

img = img.reshape(img.shape[0],img.shape[1]*img.shape[2])
img = scale(img)

# 'global','local','manual'
flag_sigma = 'global'

# Compute similarity matrix
sigma,aff_img = compute_affinity(img,flag_sigma=flag_sigma,sigma=100.,nn=7)
if flag_sigma == 'local':
    sigma_init = sum(sigma**2)/len(sigma)
    print "Average Sigma(local): ",sigma_init

K = 20
# Construct existing solution Y
Y = np.zeros((img.shape[0],20))
for i in range(img.shape[0]):
    Y[i,img_identity[i]] = 1
val_lambda = 1.2
arr_tmp = val_lambda*Y.dot(Y.T)

label_pred_identity = spectral_clustering(aff_img,n_clusters=K)
nmi_identity = nmi(label_pred_identity,img_identity)
print nmi_identity
