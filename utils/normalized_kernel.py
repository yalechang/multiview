import numpy as np

def normalized_kernel(aff):
    """ This function return a normalized kernel matrix
    aff_normed = D^(-1/2)*K^D(-1/2)
    """
    #aff = aff*(np.ones(aff.shape)-np.eye(aff.shape[0]))
    # Construct degree matrix
    aff_tmp = np.diag(np.sum(aff,1)**(-0.5))
    return aff_tmp.dot(aff).dot(aff_tmp)
    
