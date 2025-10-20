# project3_analysis.py
"""
Project 3 MCRT Analysis
Run complete Monte Carlo radiative transfer simulation.
"""
import numpy as np
import timeit
import src.test_validation as test
import src.mcrt_viz as plot
from src.grid import Grid
from src.star import Star
from src.transport import run_mcrt_jit
from src.dust import read_draine_opacity
from copy import deepcopy
from src.constants import BANDS
from src.detectors import build_results_table

def setup_stars_and_grid():
    """
    Initialize star cluster and grid.
    Returns: stars (list), grid (Grid object)
    """
    
    grid = Grid()
    
    masses = [2, 5, 10, 20, 30]
    positions = grid.sample_positions(5)
    
    stars = [Star(mass, position=positions[i]) for i, mass in enumerate(masses)]
    
    return stars, grid

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
    test_stars = deepcopy(stars)
    test_grid = deepcopy(grid)
    n_packets = results['n_packets']
    
    test_results = {'Empty Box': test.test_empty_box(test_grid, n_packets),
                    'Opaque Box': test.test_opaque_box(test_grid, n_packets),
                    'Uniform Sphere': test.test_uniform_sphere(test_stars, test_grid, n_packets),
                    'Energy Conservation': test.check_energy_conservation(results)   
                    }
    
    test_grid.reset_rho_gas()
    
    plot.summarize_tests(test_results)
    
    return test_results

def run_time_test(stars, grid, bands, max_power):
    '''
    Runs time-test for n_packets, 1e3 - 1e(max).
    '''
    exp = np.arange(2, max_power+1, 1)
    n_packets = np.logspace(2, max_power, len(exp), dtype=int)
    
    times = []   
    for n in n_packets:
        t_temp = timeit.timeit(lambda: run_simulation(stars, grid, bands, n), number=3)
        times.append(t_temp)
        
    plot.plot_time_test(n_packets, times, save=True)
    
    return

def make_plots(results, draine_data=None, bands=['B','V','K'], save=True, outdir="outputs/figures"):
    """
    Generate and save individual plots for a single run.
    
    Returns:
    --------
    figs : opactiy, sed, abs_band, rbg
    
    """
    figs = {}

    if draine_data is not None:
        kappa_bars = [results[band].kappa_bar for band in bands]
        figs['opacity'] = plot.plot_opacity_with_band_means(draine_data, kappa_bars, BANDS, save=save, outdir=outdir)

    figs['sed'] = plot.plot_band_sed_lambdaL(results, BANDS, bands=bands, save=save, outdir=outdir)

    proj_maps = {}
    for band in bands:
        if band not in results:
            continue
        figs[f'abs_{band}'] = plot.plot_absorption_map_single(results[band].grid, band, save=save, outdir=outdir)
        proj_maps[band] = np.sum(results[band].grid.L_absorbed, axis=2)
        
        extent_cm = [results[band].grid.lower_bounds[0], results[band].grid.upper_bounds[0], 
                     results[band].grid.lower_bounds[1], results[band].grid.upper_bounds[1]]

    figs['rgb'] = plot.create_rgb_composite(proj_maps['B'], proj_maps['V'], proj_maps['K'],
        extent_cm=extent_cm, save=save, outdir=outdir)
    
    return figs

def save_results(results_list, bands, stars, grid):
    """
    Save data table and numerical results.
    Saves to outputs/results/
    """
    test_results_list = []
    draine_data = read_draine_opacity()
    
    for n, results in results_list:
        make_plots(results, draine_data, bands, save=True, outdir=f"outputs/figures/{n}")
        
        test_results_list.append(run_tests(results, stars, grid))
    
    plot.plot_absorption_rgb_grid(results_list, bands=('B','V','K'), save=True)
    plot.plot_convergence(results_list, bands=('B','V','K'), save=True)
    plot.plot_convergence_error(results_list, bands=('B','V','K'), save=True)
    build_results_table(results_list[-1][1], bands=('B','V','K'))
    
    test_results_list.append(test.test_convergence_scaling(results_list))
    
    return test_results_list

def main():
    """
    Main analysis workflow - calls modular functions.
    Keep this function clean by delegating work to helper functions.
    """
    n_packets = [int(10e3), int(10e4), int(10e5), int(10e6), int(10e7)]
    bands = ['B', 'V', 'K']
    stars, grid = setup_stars_and_grid()
    results_list = []
    
    for n in n_packets:
        
        results = run_simulation(stars, grid, bands, n)
        results_list.append((n, results))
        
        print(f'Completed: {n}-packets')
    
    print('Testing & Plotting ...')    
    save_results(results_list, bands, stars, grid) 
    print('Completed: Testing & Plotting')
    
    print('Starting Time Test ...')
    run_time_test(stars, grid, bands, 7)   
    print('Completed: Time Test')
    
if __name__ == "__main__":
    main()