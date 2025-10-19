# test_validation.py

from src.grid import Grid
from src.star import Star
from src.transport import run_mcrt

def test_empty_box(grid, n_packets=1000):
    """
    Test that all packets escape in empty medium.
    
    Returns:
    --------
    passed : bool
    f_escape : float (should be 1.0)
    """
    
    bands = ['B', 'V', 'K']
    save_every = 1
    star = [Star(1)]
    
    results = run_mcrt(star, grid, bands, n_packets, save_every)
    
    if any(results[band]['absorbed luminosity'] != 0 for band in bands):
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
    save_every = 1
    star = [Star(1)]
    
    results = run_mcrt(star, grid, bands, n_packets, save_every)
    

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
    pass

def main():
    """
    Testing
    """
    empty_box = Grid(16, 1, 0)
    print('Test empty box:', test_empty_box(empty_box))
    
    grid = Grid(16, 1, 0.01)
    star = [Star(1)]
    print('Test uniform sphere:', test_uniform_sphere(star, grid, 0.01))

if __name__ == "__main__":
    main()