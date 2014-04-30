import numpy as np

def find_sigma(mtr,gamma=0.005):
    n_row,n_col = mtr.shape
    res = np.infty
    for i in range(n_row-1):
        for j in range(i+1,n_row):
            tp = min(-np.sum((mtr[i,:]-mtr[j,:])**2)/np.log(gamma),res)
            if tp !=0:
                res = tp
    return res

