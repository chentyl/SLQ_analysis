import numpy as np
import scipy as sp

from .lanczos import exact_lanczos
from .distribution import Distribution,get_GQ_distr,get_ave_distr,d_KS,d_W,d_Wbd

def get_GQs(lam,n_samples,Ks,reorth=True,seed=0):
    """
    Generate Gaussian quadrature rules for CESM of lam

    Parameters
    ----------
    lam : ndarray of eigenvalues
    n_samples : int
    Ks: values to evaluate GQ at 
    
    Returns
    -------
    GQ: dictionary of lists of GQs
    vs: RHSs used
    """
    
    n = len(lam)
    A = sp.sparse.spdiags(lam,0,n,n)

    np.random.seed(seed)
    vs = np.random.randn(n_samples,n)
    vs /= np.linalg.norm(vs,axis=1)[:,None]

    GQs = {k:[] for k in Ks}
    for i in range(n_samples):

        Q,(a_,b_) = exact_lanczos(A,vs[i],max(Ks),reorth=reorth)
        
        for k in Ks:
            # define Gaussian quadrature
            GQs[k].append(get_GQ_distr(a_[:k],b_[:k-1]))
    
    return GQs,vs


def KS_experiment(GQs,lam,vs,lb,ub):

    v2_ave = np.mean(vs**2,axis=0)
    n_samples = vs.shape[0]
    Ks = np.array(list(GQs.keys()))
    
    GQ_ave = {}

    for k in Ks:
        # average bounds
        GQ_ave[k] = get_ave_distr(GQs[k])

    wCESM_ave = Distribution()
    wCESM_ave.from_weights(lam,v2_ave)

    t_KS = np.zeros(len(Ks))
    t_KS_bd = np.zeros(len(Ks))

    for i,k in enumerate(Ks):
        t_KS[i] = d_KS(wCESM_ave,GQ_ave[k])
        t_KS_bd[i] = np.mean([np.max(GQs[k][i].weights) for i in range(n_samples)])
        
    return n_samples,Ks,t_KS,t_KS_bd

def W_experiment(GQs,lam,vs,lb,ub):

    v2_ave = np.mean(vs**2,axis=0)
    n_samples = vs.shape[0]
    Ks = np.array(list(GQs.keys()))
    
    GQ_ave = {}

    for k in Ks:
        # average bounds
        GQ_ave[k] = get_ave_distr(GQs[k])

    wCESM_ave = Distribution()
    wCESM_ave.from_weights(lam,v2_ave)

    t_W = np.zeros(len(Ks))
    t_W_bd = np.zeros(len(Ks))

    for i,k in enumerate(Ks):
        t_W[i] = d_W(wCESM_ave,GQ_ave[k])
        t_W_bd[i] = np.mean([d_Wbd(GQs[k][i],lb,ub) for i in range(n_samples)])
        
    return n_samples,Ks,t_W,t_W_bd


def get_IQs(lam,n_samples,Ks,seed=0):
    """
    Generate interpalatory quadrature rules for CESM of lam

    Parameters
    ----------
    lam : ndarray of eigenvalues
    n_samples : int
    Ks: values to evaluate IQ at 
    
    Returns
    -------
    IQ: dictionary of lists of IQs
    vs: RHSs used
    """
    
    n = len(lam)
    A = sp.sparse.spdiags(lam,0,n,n)

    np.random.seed(seed)
    vs = np.random.randn(n_samples,n)
    vs /= np.linalg.norm(vs,axis=1)[:,None]

    IQs = {k:[] for k in Ks}
    for i in range(n_samples):      
        for k in Ks:

            T = np.polynomial.chebyshev.chebvander(lam,k-1)
            cT = np.sum(T.T*(vs[i]**2),axis=1)

            ej = np.zeros(k+1)
            ej[k] = 1

            thetaT = np.polynomial.chebyshev.chebroots(ej)
            PT = np.polynomial.chebyshev.chebvander(thetaT,k-1)
            PT = PT.T
            
            BT = np.copy(PT)
            BT[0] *= 1/2
            BT /= k/2
            
            wT = BT.T@cT

            D = Distribution()
            D.from_weights(thetaT,wT)
            
            IQs[k].append(D)
    
    return IQs,vs