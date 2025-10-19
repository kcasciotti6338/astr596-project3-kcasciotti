#!/usr/bin/env python
"""
Project 3 MCRT Analysis
Run complete Monte Carlo radiative transfer simulation.
"""

import tests.test_validation as test
from src.grid import Grid
from src.star import Star
from src.transport import run_mcrt_jit
from src.dust import read_draine_opacity

def setup_stars_and_grid():
    """
    Initialize star cluster and grid.
    Returns: stars (list), grid (Grid object)
    """
    
    grid = Grid()
    
    masses = [2, 5, 10, 20, 30]
    stars = [Star(mass, grid.sample_positions(1)) for mass in masses]
    
    return grid, stars

def run_simulation(stars, grid, bands, n_packets):
    """
    Run the main MCRT simulation - SEPARATELY for each band.
    
    Parameters:
    -----------
    stars : list of Star objects
    grid : Grid object  
    bands : list of strings ['B', 'V', 'K']
    n_packets : int
        Number of packets to run PER BAND
    
    Returns: results dictionary
    """

    results = run_mcrt_jit(stars, grid, bands, n_packets)
    
    return results

def run_tests(results, stars, grid):
    """
    Run all validation tests.
    Returns: test_results dictionary
    """
    n_packets = results['n_packets']
    
    test_results = {'Empty Box': test.test_empty_box(grid, n_packets),
                    'Opaque Box': test.test_opaque_box(grid, n_packets),
                    'Uniform Sphere': test.test_uniform_sphere(stars[0], grid, n_packets),
                    'Convergence Scaling': test.test_convergence_scaling(results),
                    'Energy Conservation': test.check_energy_conservation(results)   
                    }
    
    return test_results

def make_plots(results, draine_data, bands):
    """
    Generate all required plots.
    Saves figures to outputs/figures/
    """
    ast.make_plots(results, draine_data, bands, save=True)
    
def save_results(results, bands):
    """
    Save data table and numerical results.
    Saves to outputs/results/
    """
    pass

def main():
    """
    Main analysis workflow - calls modular functions.
    Keep this function clean by delegating work to helper functions.
    """
    n_packets = 5000
    bands = ['B', 'V', 'K']
    draine_data = read_draine_opacity()
    stars, grid = setup_stars_and_grid()
    
    results = run_simulation(stars, grid, bands, n_packets)
    test_results = run_tests(results, stars, grid)
    print(test_results)
    make_plots(results, draine_data, bands)

if __name__ == "__main__":
    main()