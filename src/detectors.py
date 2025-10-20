# detectors.py

'''

'''
import numpy as np
from dataclasses import dataclass
from src.grid import Grid
from copy import deepcopy

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
    - L_input: total stars luminosity
    - kappa_bar: band-averaged opacity
    
    - L_escaped_total: total escaped luminosity    
    - escape_directions: list of (theta, phi) for each packet
    
    """
    
    L_escaped: float = 0.0
    n_escaped: float = 0.0
    escape_fraction: float = 0.0
    grid: Grid = None
    L_input: float = 0.0
    kappa_bar: float = 0.0

    def record_escapes(self, outcomes, L_packet):
        """
        Record all escaped packets.
        """
        new_escaped = (len(outcomes) - np.sum(outcomes))
        self.n_escaped += new_escaped
        self.L_escaped += (new_escaped * L_packet)
        self.escape_fraction = self.L_escaped / self.L_input
        
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
        
        self.grid = deepcopy(grid)
        
def build_results_table(results, bands = ('B','V','K'), out_csv = "outputs/results/results_table.csv"):
    """
    Builds and saves results table. 
    Per band: 'Band-averaged opacity','Input luminosity','Escaped luminosity','Escape fraction','Mean optical depth'
    """
    rows = []
    for band in bands:
        if band not in results:
            continue
        
        tau = -np.log(results[band].escape_fraction)
        
        rows.append({'Quantity': f'{band}-band',
                     'Band-averaged opacity': results[band].kappa_bar,
                     'Input luminosity': results[band].L_input,
                     'Escaped luminosity': results[band].L_escaped,
                     'Escape fraction': results[band].escape_fraction,
                     'Mean optical depth': tau })

    header = ['Quantity','Band-averaged opacity','Input luminosity','Escaped luminosity','Escape fraction','Mean optical depth']
    with open(out_csv, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(str(r[h]) for h in header) + '\n')

    return rows