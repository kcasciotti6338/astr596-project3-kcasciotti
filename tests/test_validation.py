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
    
    if (results['L_absorbed_total'] != 0):
        return False
    else: 
        return True
    
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
    
    if (results['L_escaped_total'] != 0):
        return False
    else: 
        return True

def test_uniform_sphere(star, grid, kappa, n_packets=10000):
    """
    Test escape fraction against analytical solution.
    Star at center, uniform medium.
    
    Returns:
    --------
    passed : bool
    residual : float
        |f_numerical - f_analytical| / f_analytical
    """
    
    bands = ['B', 'V', 'K']
    star = [Star(1)]
    
    results = run_mcrt_jit(star, grid, bands, n_packets)
    
    return results

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
    
    for result in results_list:
        fractions[i] = (results['B'].escape_fraction)
                    
    resid = np.abs(fractions - fractions[-1])

    m = resid > 0
    if not np.any(m):
        resid = resid + 1e-12
        m = np.ones_like(resid, dtype=bool)
        
    slope, intercept = np.polyfit(np.log(Ns[m]), np.log(resid[m]), 1)
    
    if slope > -0.9 and slope < -0.3:
        return True, slope
    else:
        return False, slope
    

def main():
    """
    Testing
    """
    grid = Grid(128, 1)
    star = [Star(10)]
    
    print('Test empty box:', test_empty_box(grid))
    print('Test opaque box:', test_opaque_box(grid))
    print('Convergence Scaling:', test_convergence_scaling())


if __name__ == "__main__":
    main()