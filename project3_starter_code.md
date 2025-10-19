# Project 3 MCRT - Starter Code Structure

## Directory Structure

Your submission must follow this exact structure:

```bash
project3_mcrt/
├── src/
│   ├── __init__.py
│   ├── constants.py      # From Project 1, ensure CGS units
│   ├── star.py           # Enhanced from Project 1
│   ├── zams.py           # From Project 1
│   ├── utils.py          # Planck function, integration
│   ├── dust.py           # Draine opacity handling
│   ├── grid.py           # 3D grid structure
│   ├── photon.py         # Packet class and propagation
│   ├── transport.py      # Main MCRT engine
│   ├── detectors.py      # Escape tracking, analysis
│   └── mcrt_viz.py       # Visualization functions
├── data/
│   └── kext_albedo_WD_MW_5.5B_30
├── outputs/
│   ├── figures/
│   └── results/
├── tests/
│   └── test_validation.py
├── project3_analysis.py  # Main script to run everything
└── README.md
```

---

## Conceptual Approach: Band-Separated MCRT

### The Key Insight

Run the Monte Carlo simulation **independently** for each wavelength band (B, V, K) rather than mixing all wavelengths in one complex simulation. Each band gets its own complete MCRT run with its own packets.

### Why This Approach?

1. **Simpler Implementation**: No complex multi-dimensional weighting needed
2. **Easier Debugging**: Get one band working perfectly before adding others
3. **Clearer Physics**: Each packet represents "monochromatic" radiation
4. **Natural Development**: Implement V-band in Phase 1, then extend to all bands in Phase 2

### Important Considerations

- Each star emits different amounts of energy in each band (based on its temperature)
- Each star has different opacity in each band (Planck mean weighted by stellar spectrum)
- Within a single band's simulation, all packets carry equal energy
- Star selection is weighted by that star's luminosity in the current band only
- The three band simulations are completely independent - they can even run in parallel

### Development Strategy

**Phase 1**: Implement complete MCRT for V-band only
**Phase 2**: Loop over all three bands, storing results separately
**Analysis**: Compare escape fractions across bands to see differential extinction

---

## Module Descriptions and Required Functions

### `constants.py`
Copy from prior Projects. Ensure all constants are in CGS units.

### `star.py`
Enhanced version from Project 1.

```python

class Star:
    def __init__(self, mass, Z=0.02):
        """
        Initialize ZAMS star.
        
        Required attributes:
        - mass: stellar mass in M_sun
        - Z: metallicity
        - T_eff: effective temperature (K)
        - L_bol: bolometric luminosity (erg/s)
        - R: stellar radius (cm)
        - x, y, z: position (cm) NEW
        - id: unique identifier NEW
        - kappa_band: dict of band-averaged opacities {'B': value, 'V': value, 'K': value} NEW
        - L_band: dict of band luminosities {'B': value, 'V': value, 'K': value} NEW
        
        IMPORTANT: Calculate opacities and luminosities HERE during initialization!
        After computing T_eff from ZAMS:
            self.kappa_band = {}
            self.L_band = {}
            for band in ['B', 'V', 'K']:
                self.kappa_band[band] = self.calculate_planck_mean_opacity(...)
                self.L_band[band] = self.calculate_band_luminosity(...)
        """
        # Use your Project 1 ZAMS calculations
        pass
    
    def calculate_band_luminosity(self, band_min, band_max):
        """
        Calculate luminosity in specific band.
        
        Returns:
        --------
        L_band : float
            Luminosity in band (erg/s)
        """
        pass
    
    def calculate_planck_mean_opacity(self, draine_data, band_min, band_max):
        """
        Calculate Planck mean opacity for this star's temperature.
        
        Returns:
        --------
        kappa : float
            Planck mean opacity (cm^2/g)
        """
        pass
```

### `utils.py`
Utility functions for physics calculations.

```python

def planck_function(wavelength, temperature):
    """
    Planck function B_lambda(T) in CGS units.
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm
    temperature : float
        Temperature in K
    
    Returns:
    --------
    B_lambda : float or array
        Planck function in erg/s/cm^2/sr/cm
    """
    pass

def integrate_band(func, band_min, band_max, *args):
    """
    Integrate function over wavelength band. Use scipy.integrate.simpson.
    
    Parameters:
    -----------
    func : callable
        Function to integrate
    band_min, band_max : float
        Band limits in cm
    *args : additional arguments for func
    
    Returns:
    --------
    result : float
        Integrated value
    """
    pass
```

### `dust.py`
Handle Draine opacity data.

```python

def read_draine_opacity(filename):
    """
    Read Draine dust opacity file.
    
    Returns:
    --------
    data : dict
        'wavelength': array in cm
        'kappa': array in cm^2/g
        'albedo': array
        'dust_to_H': float (mass ratio)
    """
    pass

def calculate_band_averaged_opacity(draine_data, band_min, band_max, T_star):
    """
    Calculate Planck mean opacity for given stellar temperature.
    
    Parameters:
    -----------
    draine_data : dict
        From read_draine_opacity
    band_min, band_max : float
        Band limits in cm
    T_star : float
        Stellar temperature (K)
    
    Returns:
    --------
    kappa_avg : float
        Band-averaged opacity (cm^2/g)
    """
    pass

class Band:
    def __init__(self, name, lambda_min, lambda_max):
        """
        Wavelength band definition.
        
        Required attributes:
        - name: 'B', 'V', or 'K'
        - lambda_min: minimum wavelength (cm)
        - lambda_max: maximum wavelength (cm)
        - lambda_center: central wavelength (cm)
        """
        pass
```

### `grid.py`
3D Cartesian grid structure.

```python
# Students: Add your own import statements here  
# Suggested: numpy

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
        pass
    
    def get_cell_indices(self, x, y, z):
        """
        Get cell indices for position.
        
        Returns:
        --------
        ix, iy, iz : int
            Cell indices
        """
        pass
    
    def is_inside(self, x, y, z):
        """
        Check if position is inside grid.
        
        Returns:
        --------
        inside : bool
        """
        pass
    
    def get_dust_density(self, ix, iy, iz):
        """
        Get dust density in cell.
        
        Returns:
        --------
        rho_dust : float
            Dust density (g/cm^3)
        """
        pass
```

### `photon.py`

Photon packet class and propagation.

```python

class Photon:
    def __init__(self):
        """
        Initialize photon packet.
        
        Required attributes:
        - x, y, z: position (cm)
        - dir_x, dir_y, dir_z: direction unit vector
        - L: luminosity carried (erg/s)
        - band: band identifier ('B', 'V', or 'K')
        - star_id: source star ID
        - kappa: opacity for this packet (cm^2/g)
        """
        pass
    
    def move(self, distance):
        """
        Move photon along its direction.
        
        Parameters:
        -----------
        distance : float
            Distance to move (cm)
        """
        pass

def emit_packet_from_star(star, band, L_packet):
    """
    Create packet emitted from stellar surface.
    
    Parameters:
    -----------
    star : Star object
    band : Band object or string
    L_packet : float
        Luminosity carried by packet
    
    Returns:
    --------
    packet : Photon object
    """
    pass

def sample_isotropic_direction():
    """
    Sample random isotropic direction.
    
    Returns:
    --------
    dir_x, dir_y, dir_z : float
        Unit direction vector
    """
    pass

def distance_to_next_boundary(packet, grid):
    """
    Calculate distance to next cell boundary.
    
    Parameters:
    -----------
    packet : Photon object
    grid : Grid object
    
    Returns:
    --------
    d_next : float
        Distance to next boundary (cm)
    face : str
        Which boundary ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')
    """
    pass

def propagate_packet(packet, grid):
    """
    Propagate packet through grid until absorbed or escaped.
    
    Parameters:
    -----------
    packet : Photon object
    grid : Grid object
    
    Returns:
    --------
    outcome : str
        'absorbed' or 'escaped'
    location : tuple
        (ix, iy, iz) if absorbed, (x, y, z) if escaped
    """
    pass
```

### `transport.py`

Main MCRT engine.

```python

def initialize_packet(stars, L_packet, band=None):
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
    pass

def run_mcrt(stars, grid, bands, n_packets, save_every=10000):
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
    pass

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
```

### `detectors.py`

Track escaping packets and compute observables.

```python
# Students: Add your own import statements here
# Suggested: numpy

class EscapeTracker:
    def __init__(self, bands):
        """
        Track escaping packets.
        
        Required attributes:
        - L_escaped_total: total escaped luminosity
        - L_escaped_by_band: dict of escaped luminosity per band
        - n_escaped_by_band: dict of packet counts per band
        - escape_directions: list of (theta, phi) for each packet
        """
        pass
    
    def record_escape(self, packet):
        """
        Record an escaping packet.
        """
        pass
    
    def calculate_escape_fractions(self, L_input_by_band):
        """
        Calculate escape fraction for each band.
        
        Returns:
        --------
        f_escape : dict
            Escape fraction for each band
        """
        pass

def create_escape_map(escape_directions, n_theta=180, n_phi=360):
    """
    Create 2D histogram of escape directions.
    
    Returns:
    --------
    escape_map : 2D array
        Map of escaping luminosity vs direction
    """
    pass
```

### `mcrt_viz.py`
Visualization functions.

```python

def plot_opacity_validation(draine_data, band_opacities):
    """
    Plot opacity curve with band averages.
    """
    pass

def plot_convergence_analysis(n_packets_array, f_escape_array):
    """
    Plot escape fraction vs number of packets.
    Include 1/sqrt(N) reference line.
    """
    pass

def plot_absorption_maps(grid, bands):
    """
    Create 2D projections of absorption.
    Sum along z-axis.
    """
    pass

def plot_sed(wavelength, L_input_by_band, L_output_by_band):
    """
    Plot input vs output SED.
    """
    pass

def create_rgb_composite(B_map, V_map, K_map):
    """
    Create RGB image from three bands.
    """
    pass
```

### `tests/test_validation.py`

Validation tests for MCRT.

```python

def test_empty_box(grid, n_packets=1000):
    """
    Test that all packets escape in empty medium.
    
    Returns:
    --------
    passed : bool
    f_escape : float (should be 1.0)
    """
    pass

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
    pass

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
```

### `project3_analysis.py`

Main script that runs everything.

```python
#!/usr/bin/env python
"""
Project 3 MCRT Analysis
Run complete Monte Carlo radiative transfer simulation.
"""

def setup_stars_and_grid():
    """
    Initialize star cluster and grid.
    Returns: stars (list), grid (Grid object)
    """
    pass

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
    pass

def run_tests(results, stars, grid):
    """
    Run all validation tests.
    Returns: test_results dictionary
    """
    pass

def make_plots(results, draine_data, bands):
    """
    Generate all required plots.
    Saves figures to outputs/figures/
    """
    pass

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

   pass

if __name__ == "__main__":
    main()
```

---

## Required Output Format

### Data Table (save as `results/escape_fractions.csv`)

```{bash}
# Header line (do not change)
# band, opacity_cm2_g, L_input_Lsun, L_escaped_Lsun, f_escape,mean_tau
B,35500,VALUE,VALUE,VALUE,VALUE
V,30500,VALUE,VALUE,VALUE,VALUE
K,3450,VALUE,VALUE,VALUE,VALUE
```

### Figure Naming Convention

All figures must be saved in `outputs/figures/` with these exact names:
- `opacity_validation.png`
- `convergence_analysis.png`
- `absorption_map_B.png`
- `absorption_map_V.png`
- `absorption_map_K.png`
- `sed_comparison.png`
- `escape_directions.png` (optional)

### Runtime Log Format

Your code must print progress updates:

```{bash}
Starting MCRT simulation with 1000000 packets...
Progress: 10% (100000 packets) - 45.2 seconds elapsed
Progress: 20% (200000 packets) - 91.3 seconds elapsed
...
Simulation complete in 453.2 seconds
Energy conservation error: 0.00012
```

---

## Important Note on Structure

This document provides a **suggested** code structure to help you organize your project. You may deviate from this structure if you have good reasons, but significant changes should be documented in your README:

**Document these changes:**
- Using a different module organization (e.g., combining grid.py and transport.py)
- Fundamentally different class hierarchies
- Alternative parallelization approaches
- Different data storage formats

**Don't document trivial changes:**
- Variable naming conventions
- Additional helper functions
- Extra attributes for debugging
- Minor reorganization within modules

---

## Submission Checklist

Before submitting, verify:
- [ ] Code runs with: `python project3_analysis.py`
- [ ] All required output files are generated in correct directories
- [ ] Energy conservation error printed to console
- [ ] Data table saved as CSV in `results/escape_fractions.csv`
- [ ] All figures saved with correct filenames in `outputs/figures/`
- [ ] No hardcoded absolute paths (use relative paths only)
- [ ] Random seed removed for final run
- [ ] Runtime printed to console

---

## Minimal Test Case

With 1000 packets, uniform medium, single star at center:
- Should run in < 1 minute
- Energy conservation < 1%
- Results reproducible with fixed seed

---

## Grading Automation

To ensure your code can be graded automatically, it must:

1. Run with: `python project3_analysis.py`
2. Complete execution for 10^6 packets
3. Generate all required output files
4. Pass all validation tests in `test_validation.py`
5. Print the final data table to stdout

---

## Implementation Notes

- Run MCRT **separately** for each band (B, V, K) - do NOT mix bands in a single simulation
- Each band simulation is independent: packets in the B-band run know nothing about V or K
- All packets within a band carry the SAME luminosity: L_total / n_packets
- Use dust density (rho_gas × f_dust_to_gas) for optical depth calculations
- Sample $τ = -\ln(ξ)$ with negative sign
- Grid coordinates: $x, y, z ∈ [-L/2, L/2]$
- Calculate separate Planck mean opacities for each star/band combination
- Phase 1: Get V-band working perfectly first, then add B and K bands
- Use consistent random seeding approach for each band to ensure fair extinction comparison
- Track statistics, not individual packet objects (memory efficiency for 10^6 packets). Storing 10^6 Photon objects will consume ~GB of RAM and crash most systems. Use scalar accumulators (L_escaped, n_absorbed) and numpy arrays for spatial maps instead.
