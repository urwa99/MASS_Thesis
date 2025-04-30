import numpy as np
import healpy as hp

def sigma_noise(nside:int, f_sky:float) -> float:
    """Calculate the noise standard deviation for a given nside and f_sky.
    The noise is calculated using the formula:
    sigma_N = T_sys / sqrt(2 * t_pix * delta_v)
    where:
    - T_sys is the system temperature in Kelvin
    - t_pix is the time spent per pixel in seconds
    - delta_v is the frequency resolution in Hz
    - N_dish is the number of dishes
    - omega_pix is the solid angle of a pixel in steradians
    - omega_sur is the solid angle of the survey in steradians
    - f_sky is the fraction of the sky covered by the survey
    - omega_pix is the solid angle of a pixel in steradians
    - omega_sur is the solid angle of the survey in steradians

    Args:
        nside (int):HEALPix nside parameter
        f_sky (float): fraction of sky covered by the survey
    Returns:
        float: noise standard deviation
    """
    
    npix=hp.nside2npix(nside)
    omega_pix=4*np.pi/npix
    omega_sur= 4*np.pi*f_sky
    N_dish= 64
    t_obs=4000 * 3600  # convert hours to seconds
    T_sys=20 # system temperature in Kelvin
    delta_v= 1e6  # frequency resolution in Hz (1 MHz = 1e6 Hz)
    
    # Time spent per pixel
    t_pix = t_obs * N_dish * (omega_pix / omega_sur)

    # Noise standard deviation
    sigma_N = T_sys / np.sqrt(2 * t_pix * delta_v)
    
    return sigma_N