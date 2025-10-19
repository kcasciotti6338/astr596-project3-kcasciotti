# grid.py

'''

'''
import numpy as np
from src.constants import PC

class Grid:
    def __init__(self, n_cells=128, L_pc=1.0, f_dust_to_gas=0.01):
        """
        Initialize 3D grid.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells per dimension (nx = ny = nz)
        L_pc : float
            Box size in parsecs
        f_dust_to_gas : float
            Dust-to-gas mass ratio (default 0.01)
        
        Required attributes:
        - nx, ny, nz: number of cells
        - L: box size in cm
        - dx, dy, dz: cell size in cm
        - x_min, x_max: boundaries (-L/2, L/2)
        - y_min, y_max: boundaries (-L/2, L/2)
        - z_min, z_max: boundaries (-L/2, L/2)
        - f_dust_to_gas: dust-to-gas ratio
        - rho_gas: 3D array of gas density
        - L_absorbed: 3D array for absorbed luminosity
        - n_absorbed: 3D array for packet count
        """
        
        self.n_cells = np.array([n_cells, n_cells, n_cells])
        self.L = L_pc * PC
        self.cell_size = self.L / self.n_cells
        self.upper_bounds = np.array([self.L/2, self.L/2, self.L/2])
        self.lower_bounds = np.array([-self.L/2, -self.L/2, -self.L/2])
        self.f_dust_to_gas = f_dust_to_gas
        self.rho_gas = np.ones(self.n_cells) * 3.84e-21
        self.L_absorbed = np.zeros(self.n_cells)
        self.n_absorbed = np.zeros(self.n_cells)
  
    def get_cell_indices(self, pos):
        """
        Get cell indices for position.
        
        Returns:
        --------
        ix, iy, iz : int
            Cell indices
        """
        
        if self.is_inside(pos):
            indices = np.floor((pos + self.L/2) / self.cell_size).astype(int)
            return tuple(indices)
        else:
            raise ValueError("Can't return cell indices; packet has escaped.")
    
    def is_inside(self, pos):
        """
        Check if position is inside grid.
        
        Returns:
        --------
        inside : bool
        """
        if np.all(pos >= self.lower_bounds) and np.all(pos <= self.upper_bounds):
            return True
        else:
            return False
    
    def get_dust_density(self, indices=None, pos=None):
        """
        Get dust density in cell.
        
        Returns:
        --------
        rho_dust : float
            Dust density (g/cm^3)
        """

        if indices is not None:
            return self.rho_gas[indices] * self.f_dust_to_gas
        elif pos is not None: 
            indices = self.get_cell_indices(pos)
            return self.rho_gas[indices] * self.f_dust_to_gas
        else: 
            raise ValueError("Must pass either indices or position.")
        
    def sample_positions(self, N):
        """
        Sample positions within the grid
        
        Returns:
        --------
        pos : (N, 3) ndarray
            x, y, z positions
        """
        low = - self.L/2 * 0.75
        high = self.L/2 * 0.75
        
        return np.random.uniform(low, high, (N,3))

def main():
    """
    Testing
    """
    grid = Grid(16, 16/PC)
    print('L:', grid.L)
    print('cell size:', grid.cell_size)
    print('upper bounds:', grid.upper_bounds)
    
    pos = np.array([-1.9, -1.9, -1.9])
    indices = grid.get_cell_indices(pos)
    print('cell indices:', indices)
    print('is inside?', grid.is_inside(pos))
    
    print('dust density:', grid.get_dust_density(indices=indices))
    print('dust density:', grid.get_dust_density(pos=pos))
    
    print('positions', grid.sample_positions(3))

if __name__ == "__main__":
    main()