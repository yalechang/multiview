"""This function compute features from Gabor kernel
"""
import numpy as np
from scipy import ndimage

def compute_feats(image,kernels):
    feats = np.zeros((len(kernels),2),dtype=np.double)
    for k,kernel in enumerate(kernels):
        filtered = ndimage.convolve(image,kernel,mode='wrap')
        feats[k,0] = filtered.mean()
        feats[k,1] = filtered.var()
    return feats

