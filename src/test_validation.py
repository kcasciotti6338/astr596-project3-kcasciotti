# test_validation.py

import numpy as np
from src.grid import Grid
from src.star import Star
from src.transport import run_mcrt_jit

def test_empty_box(grid, n_packets=1000):
    """
    Test that all packets escape in empty medium.
    
    Returns:
    --------
    passed : bool
    f_escape : float (should be 1.0)
    """
    grid.rho_gas[:] = 0
    bands = ['B', 'V', 'K']
    star = [Star(1)]
    
    results = run_mcrt_jit(star, grid, bands, n_packets)
    
    grid.reset_rho_gas()
    
    if (results['L_absorbed_total'] != 0): return False
    else: return True
    
def test_opaque_box(grid, n_packets=1000):
    """
    Test that all packets escape in empty medium.
    
    Returns:
    --------
    passed : bool
    f_escape : float (should be 1.0)
    """
    grid.rho_gas[:] = 1000
    bands = ['B', 'V', 'K']
    star = [Star(1)]
    
    results = run_mcrt_jit(star, grid, bands, n_packets)
    
    grid.reset_rho_gas()
    
    if (results['L_escaped_total'] != 0): return False
    else: return True

def test_uniform_sphere(stars, grid, n_packets=10000):
    """
    Test escape fraction against analytical solution.
    Star at center, uniform medium.
    
    Returns:
    --------
    passed : bool
    residual : float
        |f_numerical - f_analytical| / f_analytical
    """
    
    grid.fill_sphere()
    bands = ['B']

    from src.star import Star
    star_center = [Star(stars[0].mass, position=np.array([0.0,0.0,0.0]))]

    r = grid.L/2
    rho_dust = grid.f_dust_to_gas * grid.rho_gas[grid.rho_gas > 0][0]

    Lw = np.array([s.L_band['B'] for s in star_center])
    kw = np.array([s.kappa_band['B'] for s in star_center])
    kappa_eff = np.sum(Lw * kw) / np.sum(Lw)

    tau = kappa_eff * rho_dust * r
    f_analytical = np.exp(-tau)

    results = run_mcrt_jit(star_center, grid, bands, n_packets)
    f_numerical = float(results['B'].escape_fraction)

    residual = abs(f_numerical - f_analytical) / f_analytical
    grid.reset_rho_gas()
    
    if (residual < 0.05): return True, residual
    else: return False, residual

def test_convergence_scaling(results_list):
    """
    Test that error scales as 1/sqrt(N).
    
    Parameters:
    -----------
    results_list : list of (n_packets, f_escape) tuples
    
    Returns:
    --------
    passed : bool
    scaling_exponent : float (should be close to -0.5)
    """
    Ns = np.array([N for (N, _) in results_list], dtype=float)
    fractions = np.array([res['B'].escape_fraction for (_, res) in results_list], dtype=float)

    order = np.argsort(Ns)
    Ns, fractions = Ns[order], fractions[order]

    resid = np.abs(fractions - fractions[-1])
    m = resid > 0
    if not np.any(m):
        resid = resid + 1e-12
        m = np.ones_like(resid, dtype=bool)

    slope, _ = np.polyfit(np.log(Ns[m]), np.log(resid[m]), 1)
    
    if slope > -0.9 and slope < -0.3: return True, slope
    else: return False, slope
    
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
    numerator = results['L_input_total'] - (results['L_absorbed_total'] + results['L_escaped_total'])
    error = np.abs(numerator) / results['L_input_total']
    
    if error < tolerance: return True, error
    else: return False, error
