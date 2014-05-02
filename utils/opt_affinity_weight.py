import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import numpy.linalg as la


def opt_affinity_weight(affs,Q,Y,mu,v_lambda=100.,dim_q=4,tol=1e-6,\
        n_iter_max=200):
    """ This function optimize the global residue w.r.t the weights of all the
    kernels and the common low dimensional embedding.

    Parameters
    ----------
    affs: list, len(M)
        list containing the kernel matrices of M sources. Each kernel matrix is
        a n_instances x n_instances p.s.d matrix
    
    Q: array, shape(n_source,n_sources)
        Quadratic matrix, avoid repeat computation

    Y: array, shape(n_instances,n_clusters_ex)
        cluster assignments of existing clustering solution. There's only one
        non-zero element in each row
    
    mu: upperbound for 1-norm of beta

    v_lambda: float
        non-negative parameter controlling the tradeoff between clustering
        quality and the novelty of the new clustering solution

    dim_q: int
        the number of low dimensions

    tol: float
        tolerance of convergence

    n_iter_max: int
        maximal number of iterations

    Returns
    -------
    U:  array, shape(n_instances,dim_q)
        low dimensional embedding for samples

    beta: array, shape(1,M)
        weights for M sources

    res: float
        residue value when achieving convergence
    """
    
    # number of sources  
    n_sources = len(affs)

    # number of samples
    assert affs[0].shape[0] == affs[0].shape[1]
    n_instances = affs[0].shape[0]

    # Construct n_sources x n_sources matrix
    #Q = np.zeros((n_sources,n_sources))
    #for i in range(n_sources):
    #    for j in range(i,n_sources):
    #        Q[i,j] = np.trace(affs[i].dot(affs[j]))
    #        Q[j,i] = Q[i,j]
    Q = matrix(Q)
    #print "Q: ",Q 
    # Compute kernel matrix for existing solution
    aff_Y = v_lambda*Y.dot(Y.T)
    #H = np.eye(n_instances)-1./n_instances*np.ones((n_instances,n_instances))
    #aff_Y = H.dot(aff_Y).dot(H)
    # Random initialization of beta and U
    U = np.random.rand(n_instances,dim_q)
    beta = np.ones((n_sources,1))*1./n_sources
    
    # Initialization of res and res_old
    res = 2.
    res_old = 1.
    
    # Count the number of loops
    n_iter = 0

    while abs(res-res_old)/res_old>tol and n_iter<n_iter_max:
        U_old = U

        # convex combination of kernels of M sources
        aff = np.zeros((n_instances,n_instances))
        for i in range(n_sources):
            aff += beta[i]*affs[i]
        
        # Find U by eigendecomposition
        eig_val,eig_vec = la.eig(2*aff-aff_Y)
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        U = eig_vec[:,0:dim_q]
        
        # OR Find U by singular value decomposition
        #tp_U,tp_S,tp_V = la.svd(2*aff-aff_Y)
        #U = tp_U[:,0:dim_q].dot(np.diag(tp_S[0:dim_q]))

        beta_old = beta
        # Optimize beta by QP
        gamma = np.zeros((n_sources,1))
        for i in range(n_sources):
            gamma[i] = np.trace(affs[i].dot(U).dot(U.T))
        gamma = matrix(gamma)
       
        #G = matrix(np.vstack((-np.identity(n_sources),np.ones((1,n_sources)))))
        #h = matrix(np.vstack((np.zeros((n_sources,1)),mu*np.ones((1,1)))))
        G = matrix(-np.identity(n_sources))
        h = matrix(np.zeros((n_sources,1)))
        A = matrix(np.ones((1,n_sources)))
        b = matrix(1.0)
        
        # Supress the display of output
        solvers.options['show_progress'] = False
        opt_res = solvers.qp(2*Q,-2*gamma,G,h,A,b)
        beta = opt_res['x'].T
        
        aff_old = np.zeros((n_instances,n_instances))
        for i in range(n_sources):
            aff_old += beta_old[i]*affs[i]
        # Compute residues
        res_old = la.norm(aff_old-U_old.dot(U_old.T))
        res = la.norm(aff-U.dot(U.T))
        
        n_iter += 1

    # Mean of Square error
    mse =  np.trace(U.T.dot(aff).dot(U))
    
    if n_iter < n_iter_max:
        flag = True
    else:
        flag = False

    return (U,beta,res,mse,flag)

