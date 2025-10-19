# jit_helpers.py

'''
Helper functions to run photons in parallel with jit
'''

import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def random():
    """
    Separated np.random.random for jit efficiency
    
    Returns:
    --------
    np.random.random() : int
    """
    return np.random.random()

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

@njit(cache=True, fastmath=True)
def propogate_packet_jit(pos, dir, tau_target, kappa,
                         lower_bounds, upper_bounds, cell_size, n_cells, rho_gas, f_dust_to_gas):
    """
    Propagate packet through grid until absorbed or escaped.
    
    Works better for jit.
    
    Returns:
    --------
    outcome : int
        0 : escaped
        1 : absorbed
    x, y, z : floats
        (ix, iy, iz) if absorbed, (x, y, z) if escaped
    """
    
    while True:
        # checks if escaped
        if not inside_jit(pos, lower_bounds, upper_bounds):
            return 0, pos[0], pos[1], pos[2]

        # get cell info
        ix, iy, iz = get_cell_indices_jit(pos, lower_bounds, cell_size, n_cells)
        rho_dust = rho_gas[ix, iy, iz] * f_dust_to_gas
        distance = distance_to_next_boundary_jit(pos, dir, lower_bounds, cell_size, n_cells)
        absorb = kappa * rho_dust
        
        # works for empty box
        if absorb <= 0:
            pos[0] += dir[0] * distance
            pos[1] += dir[1] * distance
            pos[2] += dir[2] * distance
            continue

        # distance left to absorb
        d_absorb = tau_target / absorb
        
        # move packet, absorb
        if d_absorb <= distance:
            pos[0] += dir[0] * d_absorb
            pos[1] += dir[1] * d_absorb
            pos[2] += dir[2] * d_absorb
            return 1, ix, iy, iz
        else: 
            tau_target -= absorb * distance
            pos[0] += dir[0] * distance
            pos[1] += dir[1] * distance
            pos[2] += dir[2] * distance

@njit(parallel=True, cache=True, fastmath=True)
def run_packets_parallel(n_packets, star_pos, star_R, star_kappa, star_cdf,
                         lower_bounds, upper_bounds, cell_size, n_cells, rho_gas, f_dust_to_gas):
    """
    Main MCRT simulation loop.
    
    Runs each packet parallel with jit. 
    
    Returns:
    --------
    outcome : list of ints
        0 : escaped
        1 : absorbed
    x, y, z : list of floats
        (ix, iy, iz) if absorbed, (x, y, z) if escaped
    """
    
    # initialize outputs
    outcome = np.zeros(n_packets)
    final_x = np.zeros(n_packets)
    final_y = np.zeros(n_packets)
    final_z = np.zeros(n_packets)

    for i in prange(n_packets):
        pos, dir, tau_target, kappa = initialize_packet_jit(star_cdf, star_pos, star_R, star_kappa)

        outcome[i], final_x[i], final_y[i], final_z[i] = propogate_packet_jit(pos, dir, tau_target, kappa,
                                            lower_bounds, upper_bounds, cell_size, n_cells, rho_gas, f_dust_to_gas)
        
    return outcome, final_x, final_y, final_z