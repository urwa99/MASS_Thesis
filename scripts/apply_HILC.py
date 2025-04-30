import numpy as np
import healpy as hp
import sys
sys.path.append('../')

from scripts.compute_decoupled_cl import compute_decoupled_cl
from scripts.compute_weights import compute_weights
from scripts.mask_maps import mask_maps
from scripts.compute_alm_ilc import compute_alm_ilc
from scripts.cosine_smoothed_mask import cosine_smoothed_mask


def apply_HILC(map_in:np.ndarray)->np.ndarray:
    
    """
    Applies the HILC algorithim to input maps
    
    Parameters:
        map_in (np.ndarray): input maps

    Returns:
        clean_maps (np.ndarray): _description_
    """
    nside=128
    npix=hp.nside2npix(nside=nside)
    freq = np.array([28.4,  44.1,  70.4,  100.0,  143.0,  217.0,  353.0])
    n_freqs = len(freq)
    
    #transform maps to spherical harmonics
    lmax=3*nside -1 #default is lmax= 3*nside+1
    alm_size=hp.Alm.getsize(lmax)
    """alm_maps=np.zeros((n_freqs, alm_size), dtype=complex) #storing coefficients in an array of dim n_freqs*almsize. dtype is complex to account for -ive values

    for nf in range(n_freqs):
        alm_maps[nf,:]= hp.map2alm(map_in[nf,:], lmax=lmax, mmax=None, iter=0, pol=False) #record alm corresponding to each f for a corresponding l and m"""
        
    # Generate and plot the mask
    mask = cosine_smoothed_mask(nside)
    mask= (mask*(-1))+1
    cl_dec=compute_decoupled_cl(map_in, mask,False, nside, lmax, n_freqs)
    _, weights = compute_weights(n_freqs, lmax, cl_dec)
    _, alm_masked= mask_maps(map_in, mask, alm_size,lmax, n_freqs, npix)
    alm_ilc=compute_alm_ilc(alm_masked, alm_size, weights, lmax)
    clean_maps = hp.alm2map(alm_ilc, nside, lmax=lmax, mmax=None)
    clean_maps=np.tile(clean_maps, (1, n_freqs))
    clean_maps = clean_maps.reshape(n_freqs, npix)
    
    return clean_maps