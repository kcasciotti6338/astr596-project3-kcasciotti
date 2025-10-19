# transport.py

'''

'''
import numpy as np
from src.grid import Grid
from src.star import Star
from matplotlib import pyplot as plt
from src.photon import emit_packet_from_star, propagate_packet
from src.detectors import EscapeTracker
from src.mcrt_viz import plot_absorption_map
from src.jit_helpers import run_packets_parallel

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
    results.record_escapes(outcomes, L_packet, L_total)
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
        EscapeTracker - per-band results. Structure: results[band].attribute = ...
            - L_escaped
            - n_escaped
            - escape_fraction
            - grid
    """
    
    results = {'L_escaped_total': 0}
    
    for band in bands:
        
        grid.L_absorbed[:] = 0
        grid.n_absorbed[:] = 0

        results[band] = run_band_parallel(stars, grid, band, n_packets)
        results['L_escaped_total'] += results[band].L_escaped

    return results

def initialize_packet(stars, L_packet, band):
    """
    Initialize one packet with luminosity-weighted star selection.
    
    Parameters:
    -----------
    stars : list of Star objects
    L_packet : float
        Luminosity carried by this packet (pre-calculated)
    band : str or None
        If specified ('B', 'V', or 'K'), run for single band
        If None, sample from all bands (advanced option)
    
    Returns:
    --------
    packet : Photon object
        Initialized with star position, direction, L_packet, band, kappa
    """
        
    Ls = [star.L_band[band] for star in stars]
    
    cdf = np.cumsum(Ls / np.sum(Ls))
    idx = np.searchsorted(cdf, np.random.rand())
    
    return emit_packet_from_star(stars[idx], band, L_packet)

def run_mcrt(stars, grid, bands, n_packets):
    """
    Main MCRT simulation loop.
    
    Parameters:
    -----------
    stars : list of Star objects
    grid : Grid object
    bands : list of Band objects or strings ['B', 'V', 'K']
    n_packets : int
        Number of packets PER BAND
    save_every : int
        Save checkpoint every N packets
    
    Returns:
    --------
    results : dict
        Per-band results. Structure: results[band] = {...}
        Should include escape fractions, absorbed/escaped luminosities
    
    CRITICAL: L_packet = L_band_total / n_packets
    where L_band_total is the sum of all stars' luminosities 
    in the CURRENT BAND ONLY (not all bands combined).
    
    Note: Reset grid.L_absorbed between bands!
    """
    
    results = { 'B': {'escape fraction': 0, 'absorbed luminosity': 0, 'escaped luminosity': 0}, 
                'V': {'escape fraction': 0, 'absorbed luminosity': 0, 'escaped luminosity': 0}, 
                'K': {'escape fraction': 0, 'absorbed luminosity': 0, 'escaped luminosity': 0} }
    
    tracker = EscapeTracker(bands)
    
    for band in bands:
        Ls = [star.L_band[band] for star in stars]
        L_packet = np.sum(Ls) / n_packets
        
        for i in range(n_packets):
            packet = initialize_packet(stars, L_packet, band)
            result, pos = propagate_packet(packet, grid)
            if result == 'escaped':
                tracker.record_escape(packet)
            elif result == 'absorbed':
                cell = grid.get_cell_indices(pos)
                grid.n_absorbed[cell] += 1
                grid.L_absorbed[cell] += packet.L
    
        results[band]['absorbed luminosity'] = grid.L_absorbed.sum()
       
        #plot_absorption_map(grid, band)
        grid.n_absorbed[:] = 0
        grid.L_absorbed[:] = 0
    
        results[band]['escaped luminosity'] = tracker.L_escaped_by_band[band]
        results[band]['escape fraction'] = tracker.L_escaped_by_band[band] / np.sum(Ls)
    
    return results

def check_energy_conservation(results, tolerance=0.001):
    """
    Verify energy conservation.
    
    Returns:
    --------
    conserved : bool
        True if |L_in - (L_abs + L_esc)|/L_in < tolerance
    error : float
        Fractional energy error
    """
    pass

def main():
    """
    Testing
    """
    
    grid = Grid(128)
    bands = ['B', 'V', 'K']
    masses = [10]
    stars = [Star(mass) for mass in masses]
    n_packets = 10000
    '''
    results = run_mcrt(stars, grid, bands, n_packets)
    
    print('----- Regular -----')
    print('B escape fraction:', results['B']['escape fraction'])
    print('V escape fraction:', results['V']['escape fraction'])
    print('K escape fraction:', results['K']['escape fraction'])
    '''
    results_jit = run_mcrt_jit(stars, grid, bands, n_packets)
    
    print('----- Jit Speedup -----')
    print('B escape fraction:', results_jit['B'].escape_fraction)
    print('V escape fraction:', results_jit['V'].escape_fraction)
    print('K escape fraction:', results_jit['K'].escape_fraction)
    
    pass

if __name__ == "__main__":
    main()