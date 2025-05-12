import numpy as np
import healpy as hp
import sys
sys.path.append('../')

from scripts.compute_master import compute_master
from scripts.compute_weights import compute_weights
from scripts.mask_maps import mask_maps
from scripts.compute_alm_ilc import compute_alm_ilc



def apply_HILC(map_in:np.ndarray, mask, nside, lmax, nfreqs)->np.ndarray:
    
    """
    Applies the HILC algorithim to input maps
    
    Parameters:
        map_in (np.ndarray): input maps

    Returns:
        clean_maps (np.ndarray): _description_
    """
    
    alm_size=hp.Alm.getsize(lmax)
    npix=hp.nside2npix(nside)
    
    mask= (mask*(-1))+1
    cl_dec= compute_master(map_in[:nfreqs], mask, False, nside, lmax, nfreqs)
    weights = compute_weights(nfreqs, lmax, cl_dec)

    masked_maps, alms_masked= mask_maps(map_in[:nfreqs], mask, alm_size,lmax, nfreqs, npix)
    alm_ilc=compute_alm_ilc(alms_masked, alm_size, weights, lmax)
    clean_maps = hp.alm2map(alm_ilc, nside, lmax=lmax, mmax=None)
    clean_maps=np.tile(clean_maps, (1, nfreqs))
    clean_maps = clean_maps.reshape(nfreqs, npix)
    print("Cleaned maps")
    hilc= compute_master(clean_maps[:nfreqs], mask, True, nside, lmax, nfreqs)
    
    return hilc