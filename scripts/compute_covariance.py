import numpy as np
import healpy as hp
import sys
sys.path.append('../')
from scripts.compute_cl import compute_cl
def compute_covariance(maps, lmax, almsize, nfreqs):
    alms=np.zeros((nfreqs, almsize), dtype=complex) 
    
    for nf in range(nfreqs):
        print(f"Converting maps {nf}")
        alms[nf,:]= hp.map2alm(maps[nf,:], lmax=lmax) 
        
    cl=compute_cl(alms, lmax, nfreqs)
    
    return cl