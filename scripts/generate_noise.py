import healpy as hp
import numpy as np


def generate_noise(nside:int, sigma:float, nfreqs:float) -> np.ndarray: 
    """
    Generate Gaussian noise map with given nside and standard deviation.
    
    Args:
        nside (int): HEALPix nside parameter
        sigma (float): standard deviation of the noise
        nfreqs (int): number of frequency channels
    Returns:
        noise_maps (np.ndarray): noise map of shape (nfreqs, npix)
    """
    npix = hp.nside2npix(nside)
    noise_maps = np.random.normal(0, sigma, size=(nfreqs,npix))
    # noise_maps = np.tile(noise_map, (nfreqs, 1))
    return noise_maps