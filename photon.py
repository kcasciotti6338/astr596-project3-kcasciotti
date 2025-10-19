# photon.py

'''

'''
import numpy as np
from src.grid import Grid
from src.constants import PC

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
    
    def move(self, distance):
        """
        Move photon along its direction.
        
        Parameters:
        -----------
        distance : float
            Distance to move (cm)
        """
        
        self.pos = self.pos + self.dir * distance
        
        pass
   
def emit_packet_from_star(star, band, L_packet):
    """
    Create packet emitted from stellar surface.
    
    Parameters:
    -----------
    star : Star object
    band : Band object or string
    L_packet : float
        Luminosity carried by packet
    
    Returns:
    --------
    packet : Photon object
    """
    
    dir = sample_isotropic_direction()
    pos = star.pos + star.R * dir
    
    return Photon(pos, dir, L_packet, band, star.id, star.kappa_band[band])

def sample_isotropic_direction():
    """
    Sample random isotropic direction.
    
    Returns:
    --------
    dir_x, dir_y, dir_z : float
        Unit direction vector
    """
    
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    
    theta = np.arccos(2*u - 1)
    phi = 2 * np.pi * v
    
    x_hat = np.sin(theta)*np.cos(phi)
    y_hat = np.sin(theta)*np.sin(phi)
    z_hat = np.cos(theta)
    
    return np.array([x_hat, y_hat, z_hat])

def distance_to_next_boundary(packet, grid):
    """
    Calculate distance to next cell boundary.
    
    Parameters:
    -----------
    packet : Photon object
    grid : Grid object
    
    Returns:
    --------
    d_next : float
        Distance to next boundary (cm)
    """
    
    # find min & max corners of cell
    cell_center = ((grid.get_cell_indices(packet.pos) * grid.cell_size) - (grid.L/2) + grid.cell_size/2)
    corner_min = cell_center - grid.cell_size/2
    corner_max = cell_center + grid.cell_size/2
    
    # find distances to each face
    dmin = corner_min - packet.pos
    dmax = corner_max - packet.pos
    d_faces = np.append(dmin, dmax)
    
    # avoid divide by 0 errors
    d = packet.dir
    dir = np.where(d == 0.0, np.inf, d)

    # scale with direction
    dmin_scaled = dmin / dir 
    dmax_scaled = dmax / dir
    d_faces_scaled = np.append(dmin_scaled, dmax_scaled)
    
    # find which face will hit first
    min = np.inf
    for d in d_faces_scaled:
        if d < min and d > 0: 
            min = d

    return min + grid.cell_size[0]/1e6

def propagate_packet(packet, grid):
    """
    Propagate packet through grid until absorbed or escaped.
    
    Parameters:
    -----------
    packet : Photon object
    grid : Grid object
    
    Returns:
    --------
    outcome : str
        'absorbed' or 'escaped'
    location : tuple
        (ix, iy, iz) if absorbed, (x, y, z) if escaped
    """
    u = np.random.uniform(0, 1)
    
    tau_sample = -np.log(u)
    tau_accumulated = 0
    
    while tau_accumulated < tau_sample and grid.is_inside(packet.pos):
        rho_dust = grid.get_dust_density(pos=packet.pos)
        kappa = packet.kappa

        distance = distance_to_next_boundary(packet, grid)
        d_absorb = (tau_sample - tau_accumulated) / (kappa * rho_dust)
        
        if d_absorb <= distance:
            packet.move(d_absorb)
            tau_accumulated = tau_sample
            break
        else:
            tau_accumulated += kappa * rho_dust * distance
            packet.move(distance)
        
    if grid.is_inside(packet.pos) is False:
        return 'escaped', packet.pos
    else: 
        return 'absorbed', packet.pos
    



def main():
    """
    Testing
    """
    grid = Grid(4, 4/PC)
    print('L:', grid.L)
    print('cell size:', grid.cell_size)
    print('upper bounds:', grid.upper_bounds)
    
    dir = sample_isotropic_direction()
    pos = np.array([1.5, 1.5, 1.5])
    packet = Photon(pos, dir, 1, 'B', 'test', 1)
    print('position:', packet.pos)
    print('direction:', packet.dir)
    
    state, final_pos = propagate_packet(packet, grid)
    print('state:', state)
    print('final position:', final_pos)

if __name__ == "__main__":
    main()
    
