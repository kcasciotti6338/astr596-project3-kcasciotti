# utils.py

'''

'''
import numpy as np
from numba import njit
from scipy.integrate import simpson
from src.constants import K_BOLTZMANN as K, H_PLANCK as H, C

def integrate_band(func, band_min, band_max, *args):
    """
    Integrate function over wavelength band. Use scipy.integrate.simpson.
    
    Parameters:
    -----------
    func : callable
        Function to integrate
    band_min, band_max : float
        Band limits in cm
    *args : additional arguments for func
    
    Returns:
    --------
    result : float
        Integrated value
    """
    
    x = np.linspace(band_min, band_max, 100)
    y = func(x, *args)
    
    return simpson(y, x=x)    
    
def planck_function(wavelength, temperature):
    """
    Planck function B_lambda(T) in CGS units.
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm
    temperature : float
        Temperature in K
    
    Returns:
    --------
    B_lambda : float or array
        Planck function in erg/s/cm^2/sr/cm
    """
    
    return (2 * H * C**2) / (wavelength**5 * (np.exp(H * C / (wavelength * K * temperature)) -1 ))

def sphere_to_cartesian(r, theta, phi):
    """
    Translates spherical coordinates to cartesian coordinates
        
    Parameters
    ----------
    r : radius
    theta : 
        units = radians
    phi : 
        units = radians
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z

def cartesian_to_sphere(x, y, z):
    """
    Translates cartesian coordinates to spherical coordinates
        
    Parameters
    ----------
    x, y, z
    """
    r = np.sqrt(x*x + y*y + z*z)
    
    if r == 0.0:
        return 0.0, 0.0
    
    cos_th = z / r
    
    if cos_th > 1.0: cos_th = 1.0
    if cos_th < -1.0: cos_th = -1.0
    
    theta = np.arccos(cos_th)
    phi = np.arctan2(y, x)
    
    return theta, phi

@njit(cache=True, fastmath=True)
def random():
    """
    Separated np.random.random for jit efficiency
    
    Returns:
    --------
    np.random.random() : int
    """
    return np.random.random()
