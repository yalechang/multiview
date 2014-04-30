""" This script prepares ORL dataset
According to Descriptions in CVPR 2012 <Affinity Aggregation for Spectral
Clustering>:
    1) All images were first normalized and cropped to 88 x 88
    2) Eigenface, Gabor texture and Local binary pattern are used to obtain
    different perspectives.
"""

import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cv2
from sklearn.decomposition import PCA
from skimage.filter import gabor_kernel
from skimage.feature import local_binary_pattern
from skimage.transform import rotate
from skimage.color import label2rgb
import pickle
from sklearn.preprocessing import scale
from python.multiview.utils.compute_feats import compute_feats

# Specify the directory of ORL image dataset
basepath = '/Users/changyale/dataset/orl_faces'

# Specify the image number and size
n_img = 400
n_row = 88
n_col = 88

# Read from the image dataset
img = np.zeros((400,88,88))
for i in range(40):
    path_tmp = basepath+'/s'+str(i+1)+'/'
    for j in range(10):
        img_name = path_tmp+str(j+1)+'.pgm'
        tmp = misc.imread(img_name)
        img[i*10+j,:,:] = tmp[12:100,2:90]

# Show images
#plt.imshow(img[100,:,:],cmap=cm.Greys_r)
#plt.show()


# Normalize image
for i in range(n_img):
    img[i] = (img[i]-img[i].mean())/img[i].std()

# Feature set 1
# PCA on images
pca = PCA(n_components=150)
f_pca = pca.fit_transform(img.reshape(n_img,n_row*n_col))

print "Variance Ratio: ",sum(pca.explained_variance_ratio_)

# Feature set 2
# Gabor filters
# Prepare filter bank kernels
kernels = []
frequency = 0.05
for theta in range(8):
    theta = theta/4.*np.pi
    for sigma in range(1,6):
        kernel = np.real(gabor_kernel(frequency,theta=theta,sigma_x=sigma,\
                sigma_y=sigma))
        kernels.append(kernel)

# Compute Gabor features 
f_gabor = np.zeros((n_img,40*2))
for i in range(n_img):
    f_gabor[i,:] = compute_feats(img[i],kernels).reshape(1,80)
    print i

pickle.dump({'pca':f_pca,'gabor':f_gabor},open("pca_gabor.pkl","wb"))

"""
# Feature set 3 Local binary feature
# Settings for LBP
radius = 1
n_points = 8

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

lbp = np.zeros((400,88,88))

for i in range(400):
    lbp[i] = local_binary_pattern(img[i],n_points,radius,'uniform')
    print i
"""
