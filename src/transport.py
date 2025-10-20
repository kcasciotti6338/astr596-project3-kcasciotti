# transport.py

'''

'''
import numpy as np
from numba import njit, prange
from src.grid import Grid, inside_jit, get_cell_indices_jit, distance_to_next_boundary_jit
from src.photon import initialize_packet_jit
from src.star import Star
from src.detectors import EscapeTracker

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

def run_band_parallel(stars, grid, band, n_packets):
    """
    Run one band in parallel
    
    Parameters:
    -----------
    stars : list of Star objects
    grid : Grid object
    band : float
        'B', 'V', or 'K'
    n_packets : int
        Number of packets PER BAND
    
    Returns:
    --------
    results : EscapeTracker dataclass
        Per-band results. Structure: results.attribute[band] = ...
            - L_escaped_total
            - L_escaped_by_band
            - n_escaped_by_band
            - escape_fractions
            - grids
    """
    n_stars = len(stars)
    star_pos   = np.zeros((n_stars,3))
    star_R     = np.zeros(n_stars)
    star_L_band = np.zeros(n_stars)
    star_kappa_band = np.zeros(n_stars)

    for i, star in enumerate(stars):
        star_pos[i,:] = star.pos
        star_R[i]     = star.R
        star_L_band[i] = star.L_band[band]
        star_kappa_band[i] = star.kappa_band[band]

    L_total = star_L_band.sum()
    L_packet = L_total / n_packets

    packet_L_band = star_L_band / L_total
    cdf = np.cumsum(packet_L_band); cdf[-1] = 1.0

    lower = grid.lower_bounds
    upper = grid.upper_bounds
    cell_size = grid.cell_size
    n_cells = grid.n_cells

    outcomes, x, y, z = run_packets_parallel(n_packets, star_pos, star_R, star_kappa_band, cdf, 
                                           lower, upper, cell_size, n_cells, grid.rho_gas, grid.f_dust_to_gas)

    results = EscapeTracker()
    results.L_input = L_total
    if L_total > 0: results.kappa_bar = np.sum(star_L_band * star_kappa_band) / L_total
    results.record_escapes(outcomes, L_packet)
    results.record_absorbs(grid, outcomes, x, y, z, L_packet)
    
    return results

def run_mcrt_jit(stars, grid, bands, n_packets):
    """
    Main MCRT simulation loop.
    
    Runs in parallel with jit. 
    
    Parameters:
    -----------
    stars : list of Star objects
    grid : Grid object
    bands : array of strings
        ['B', 'V', 'K']
    n_packets : int
        Number of packets PER BAND
    
    Returns:
    --------
    results : dict 
        L_escaped_total
        L_absorbed_total
        L_input_total
        n_packets
        EscapeTracker - per-band results. Structure: results[band].attribute = ...
            - L_escaped
            - n_escaped
            - escape_fraction
            - grid
    """
    
    results = {'L_escaped_total': 0, 'L_absorbed_total': 0, 'L_input_total': 0, 'n_packets': n_packets}
    
    for band in bands:
        
        grid.L_absorbed[:] = 0
        grid.n_absorbed[:] = 0

        results[band] = run_band_parallel(stars, grid, band, n_packets)
        results['L_escaped_total'] += results[band].L_escaped
        results['L_absorbed_total'] += np.sum(results[band].grid.L_absorbed)
        results['L_input_total'] += results[band].L_input

    return results
