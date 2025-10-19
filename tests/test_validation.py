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

def test_uniform_sphere(star, grid, n_packets=10000):
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
    kappa = star[0].kappa_band['B']
    r = grid.L/2
    rho_dust = grid.f_dust_to_gas * grid.rho_gas[grid.rho_gas > 0][0]

    tau = kappa * rho_dust * r
    f_analytical = np.exp(-tau)

    results = run_mcrt_jit(star, grid, bands, n_packets)
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
    fractions = np.zeros(len(results_list))
    Ns = np.zeros(len(results_list))
    
    for i, result in enumerate(results_list):
        fractions[i] = result['B'].escape_fraction
        Ns[i] = result['n_packets']
                    
    resid = np.abs(fractions - fractions[-1])

    m = resid > 0
    if not np.any(m):
        resid = resid + 1e-12
        m = np.ones_like(resid, dtype=bool)
        
    slope, intercept = np.polyfit(np.log(Ns[m]), np.log(resid[m]), 1)
    
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

def main():
    """
    Testing
    """
    grid = Grid(128, 1)
    star = [Star(10)]
    
    print('Test empty box:', test_empty_box(grid))
    print('Test opaque box:', test_opaque_box(grid))
    print('Test uniform sphere:', test_uniform_sphere(star, grid2))
    
    #print('Convergence Scaling:', test_convergence_scaling())


if __name__ == "__main__":
    main()