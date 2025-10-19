# detectors.py

'''

'''
import numpy as np
from dataclasses import dataclass, field
from src.utils import cartesian_to_sphere
from grid import Grid

@dataclass
class EscapeTracker:
    """
    Track escaping packets per band.
    
    Attributes:
    -----------
    - L_escaped: luminosity
    - n_escaped: packet counts
    - escape_fraction: escape fraction
    - grid: grid after mcrt
    
    - L_escaped_total: total escaped luminosity    
    - escape_directions: list of (theta, phi) for each packet
    
    """
    
    L_escaped: float = 0.0
    n_escaped: float = 0.0
    escape_fraction: float = 0.0
    grid: Grid = None
    
    #L_escaped_total: float
    #escape_directions_total: list

    def record_escapes(self, outcomes, L_packet, L_input):
        """
        Record all escaped packets.
        """
        
        self.n_escaped += (len(outcomes) - np.sum(outcomes))
        self.L_escaped += (self.n_escaped * L_packet)
        self.escape_fraction = self.L_escaped / L_input
        
        #self.L_escaped_total += self.n_escaped
        #theta, phi = cartesian_to_sphere(dir[0], dir[1], dir[2])
        #self.escape_directions.append((theta, phi))
        
    def record_absorbs(self, grid, outcomes, x, y, z, L_packet):
        """
        Record all absorbed packets.
        """
        absorbs = (outcomes == 1)

        ix = x[absorbs].astype(int)
        iy = y[absorbs].astype(int)
        iz = z[absorbs].astype(int)
        
        nx, ny, nz = map(int, grid.n_cells)
        flat = np.ravel_multi_index((ix, iy, iz), (nx, ny, nz))
        counts = np.bincount(flat, minlength=nx*ny*nz).reshape(nx, ny, nz)

        grid.n_absorbed += counts
        grid.L_absorbed += counts * L_packet
        
        self.grid = grid