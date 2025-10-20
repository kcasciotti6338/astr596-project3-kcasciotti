# grid.py

'''

'''
import numpy as np
from src.constants import PC
from numba import njit

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

    def fill_sphere(self):
        '''
        Sets rho_dust outside of sphere to 0.
        For uniform sphere test.
        '''
        r = self.L/2
        nx, ny, nz = map(int, self.n_cells)
        
        xs = self.lower_bounds[0] + (np.arange(nx) + 0.5) * self.cell_size[0]
        ys = self.lower_bounds[1] + (np.arange(ny) + 0.5) * self.cell_size[1]
        zs = self.lower_bounds[2] + (np.arange(nz) + 0.5) * self.cell_size[2]

        dx2 = xs**2
        dy2 = ys**2
        dz2 = zs**2
        r2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]

        self.rho_gas[r2 > r**2] = 0

    def reset_rho_gas(self):
        '''
        Resets rho gas after testing. 
        '''
        self.rho_gas[:] = 3.84e-21
                        
@njit(cache=True, fastmath=True)
def inside_jit(pos, lower_bounds, upper_bounds):
    """
    Check if position is inside grid. 
    
    Works better with jit
    
    Returns:
    --------
    inside : bool
    """
    return (pos[0] >= lower_bounds[0] and pos[0] <= upper_bounds[0] and
            pos[1] >= lower_bounds[1] and pos[1] <= upper_bounds[1] and
            pos[2] >= lower_bounds[2] and pos[2] <= upper_bounds[2])

@njit(cache=True, fastmath=True)
def get_cell_indices_jit(pos, lower_bounds, cell_size, n_cells):
    """
    Get cell indices for position.
    
    Works better for jit.
    
    Returns:
    --------
    ix, iy, iz : int
        Cell indices
    """
    
    ix = int(np.floor((pos[0] - lower_bounds[0]) / cell_size[0]))
    iy = int(np.floor((pos[1] - lower_bounds[1]) / cell_size[1]))
    iz = int(np.floor((pos[2] - lower_bounds[2]) / cell_size[2]))
    
    # Sets to 0 if outside lower
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if iz < 0: iz = 0
    
    # Sets to max index if outside upper
    if ix >= n_cells[0]: ix = n_cells[0]-1
    if iy >= n_cells[1]: iy = n_cells[1]-1
    if iz >= n_cells[2]: iz = n_cells[2]-1
    
    return ix, iy, iz

@njit(cache=True, fastmath=True)
def distance_to_next_boundary_jit(pos, dir, lower_bounds, cell_size, n_cells):
    """
    Calculate distance to next cell boundary.
    
    Works better for jit.
    
    Returns:
    --------
    d_next : float
        Distance to next boundary (cm)
    """
    
    ix, iy, iz = get_cell_indices_jit(pos, lower_bounds, cell_size, n_cells)
    
    # find min corners of cell
    corner_min_x = lower_bounds[0] + ix * cell_size[0]
    corner_min_y = lower_bounds[1] + iy * cell_size[1]
    corner_min_z = lower_bounds[2] + iz * cell_size[2]
    
    # find max corners of cell
    corner_max_x = corner_min_x + cell_size[0]
    corner_max_y = corner_min_y + cell_size[1]
    corner_max_z = corner_min_z + cell_size[2]

    # avoid divide by 0 errors (fastmath don't like np.inf)
    dx = dir[0] if dir[0] != 0.0 else 1e-300
    dy = dir[1] if dir[1] != 0.0 else 1e-300
    dz = dir[2] if dir[2] != 0.0 else 1e-300

    # find scaled distances to each face
    dmin_x = (corner_min_x - pos[0]) / dx
    dmax_x = (corner_max_x - pos[0]) / dx
    dmin_y = (corner_min_y - pos[1]) / dy
    dmax_y = (corner_max_y - pos[1]) / dy
    dmin_z = (corner_min_z - pos[2]) / dz
    dmax_z = (corner_max_z - pos[2]) / dz

    # find which face will hit first
    min = 1e300
    if dmin_x > 0 and dmin_x < min: min = dmin_x
    if dmax_x > 0 and dmax_x < min: min = dmax_x
    if dmin_y > 0 and dmin_y < min: min = dmin_y
    if dmax_y > 0 and dmax_y < min: min = dmax_y
    if dmin_z > 0 and dmin_z < min: min = dmin_z
    if dmax_z > 0 and dmax_z < min: min = dmax_z
    
    return min + cell_size[0]/1e6
