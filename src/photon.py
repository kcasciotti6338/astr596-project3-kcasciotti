# photon.py

'''

'''
import numpy as np
from numba import njit
from utils import random

class Photon:
    def __init__(self, pos, dir, L, band, star_id, kappa):
        """
        Initialize photon packet.
        
        Required attributes:
        - x, y, z: position (cm)
        - dir_x, dir_y, dir_z: direction unit vector
        - L: luminosity carried (erg/s)
        - band: band identifier ('B', 'V', or 'K')
        - star_id: source star ID
        - kappa: opacity for this packet (cm^2/g)
        """
        
        self.pos = pos
        self.dir = dir
        self.L = L
        self.band = band
        self.star_id = star_id
        self.kappa = kappa

@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    """
    Sample random isotropic direction.
    
    (Same as original function, delete one!)
    
    Returns:
    --------
    dir_x, dir_y, dir_z : float
        Unit direction vector
    """
    
    u = random()
    v = random()
    
    theta = np.arccos(2*u - 1)
    phi = 2 * np.pi * v
    
    x_hat = np.sin(theta)*np.cos(phi)
    y_hat = np.sin(theta)*np.sin(phi)
    z_hat = np.cos(theta)
    
    return np.array([x_hat, y_hat, z_hat])

@njit(cache=True, fastmath=True)
def initialize_packet_jit(cdf, star_pos, star_R, star_kappa):
    """
    Initializes a packet
    
    Returns:
    --------
    
    """
    u = random()
    v = random()
    
    idx = cdf.size - 1
    for i in range(cdf.size):
        if u <= cdf[i]:
            idx = i
            break
    
    dir = sample_isotropic_direction()
    
    pos = star_pos[idx].copy()
    pos[0] += star_R[idx] * dir[0]
    pos[1] += star_R[idx] * dir[1]
    pos[2] += star_R[idx] * dir[2]

    # avoid log 0 errors
    while v <= 0.0: 
        v = random()
    tau_target = -np.log(v)
    
    kappa = star_kappa[idx]

    return pos, dir, tau_target, kappa    











    
