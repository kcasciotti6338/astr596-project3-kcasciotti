# star.py

'''

'''

import numpy as np
from src.zams import luminosity, radius
from src.constants import RSUN, LSUN, TSUN, MSUN, G, YEAR, BANDS, SIGMA_SB
from src.dust import read_draine_opacity, calculate_band_averaged_opacity
from src.utils import planck_function, integrate_band

class Star:
    def __init__(self, mass, Z=0.02, position=np.zeros(3)):
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
        """
        if isinstance(mass, np.ndarray):
            assert np.all(mass >= 0.1) and np.all(mass <= 100), \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
        else:
            assert 0.1 <= mass <= 100, \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
            
        if Z==0.02: solar_Z_only = True 
        else: solar_Z_only = False

        self.mass = mass
        self.Z = Z 
        self.R = RSUN * radius(self.mass, self.Z, solar_Z_only=solar_Z_only)
        self.L_bol = LSUN * luminosity(self.mass, self.Z, solar_Z_only=solar_Z_only)
        self.T_eff = self.T_eff_eq()
        self.id = self.make_id()
        self.pos = position
        
        draine_data = read_draine_opacity()
        self.kappa_band = {}
        self.L_band = {}
        for band in ['B', 'V', 'K']:
            self.kappa_band[band] = self.calculate_planck_mean_opacity(draine_data, BANDS[band][0], BANDS[band][1])
            self.L_band[band] = self.calculate_band_luminosity(BANDS[band][0], BANDS[band][1])
        
    def T_eff_eq(self):
        '''
        Effective temperature in K
        
        Formula: T/T_sun = (L/L_sun)^0.25 * (R/R_sun)^-0.05
        Remember: T_sun = 5777 K (present day, not ZAMS!)
        '''
        return (self.L_bol/LSUN)**0.25 * (self.R/RSUN)**(-0.5) * TSUN
    
    def t_MS(self):
        '''
        Main sequence lifetime in Gyr
        
        Formula: t_MS ∝ M/L
        Normalize so 1 M_sun gives ~10 Gyr for present-day Sun
        But YOUR 1 M_sun star has L=0.698, so t_MS will be ~14 Gyr
        '''
        return self.mass / self.L_bol * 10
    
    def t_KH(self):
        '''
        Kelvin-Helmholtz timescale in years
        
        Formula: t_KH = GM²/(RL) in CGS units
        Convert mass, radius, luminosity to CGS before calculation
        '''
        return (G * (MSUN * self.mass)**2 / (self.R * self.L_bol)) / YEAR
    
    def lambd_max(self):
        '''
        Wein's peak wavelength in cm
        
        Formula: λ_max = b/T where b = 0.2898 cm·K
        '''
        return 0.2898 / self.T_eff
    
    def f_strings(self):
        '''
        String representation using f-strings
        
        returns type: dictionary (mass, Z, T_eff, L_bol, R, pos, id, kappa_band, L_band)
        
        Example: f"Mass: {self.mass:.2f} M_sun
        '''
        
        f_str = {
            'mass': f"Mass: {self.mass:.2f} M_sun",
            'Z': f"Metalicity: {self.Z:.2f}",
            'T_eff': f"Effective Temperature: {self.T_eff:.2f} K",
            'L_bol': f"Bolometric Luminosity: {self.L_bol/LSUN:.2f} L_sun",
            'R': f"Stellar Radius: {self.R/RSUN:.2f} R_sun",
            'pos': f"Position: {self.pos} cm",
            'id': f"Unique Identifier: {self.id}",
            'kappa_band': f"Band-Averaged Opacities: {self.kappa_band}",
            'L_band': f"Band Luminosities: {self.L_band} erg/s",
        }
        return f_str

    def calculate_band_luminosity(self, band_min, band_max):
        """
        Calculate luminosity in specific band.
        
        Returns:
        --------
        L_band : float
            Luminosity in band (erg/s)
        """
        
        numerator = integrate_band(planck_function, band_min, band_max, self.T_eff)
        denominator = SIGMA_SB * self.T_eff**4 / np.pi
        
        return self.L_bol * numerator / denominator 
    
    def calculate_planck_mean_opacity(self, draine_data, band_min, band_max):
        """
        Calculate Planck mean opacity for this star's temperature.
        
        Returns:
        --------
        kappa : float
            Planck mean opacity (cm^2/g)
        """
        return calculate_band_averaged_opacity(draine_data, band_min, band_max, self.T_eff)

    def spectral_type(self):
        """
        Find spectral type from temperature
        Uses EEM_dwarf_UBVIJHK_colors_Teff.txt
        
        Returns:
        --------
        sp_t : string
            spectral type
        """
        
        filename = '/Users/kcasc/astr596/projects/astr-596-project-03-kcasciotti6338/data/EEM_dwarf_UBVIJHK_colors_Teff.txt'
        data = np.genfromtxt(filename, skip_header=23, max_rows=118, usecols=(0, 1), dtype=[('sp_t', 'U10'), ('T_eff', 'f8')])   
        
        idx = np.argmin(np.abs(data['T_eff'] - (self.T_eff)))
       
        return data['sp_t'][idx]
    
    def make_id(self):
        """
        Makes a unique ID for a star
        
        Returns:
        --------
        id : string
            spectral type and random int
        """
        u = np.random.randint(100, 999)
        
        return f'{self.spectral_type()}_{u}'

def main():
    """
    Testing
    """

    filename = '/Users/kcasc/astr596/projects/astr-596-project-03-kcasciotti6338/data/EEM_dwarf_UBVIJHK_colors_Teff.txt'
    data = np.genfromtxt(filename, skip_header=23, max_rows=118, usecols=(0, 1), dtype=[('sp_t', 'U10'), ('T_eff', 'f8')])   
    idx = np.argmin(np.abs(data['T_eff'] - TSUN))
    print('Sun\'s spectral type: ', data['sp_t'][idx])
    
    star = Star(1)
    labels = star.f_strings()
    print(labels['mass'])
    print(labels['T_eff'])
    print(labels['L_bol'])
    print(labels['R'])
    print(labels['id'])
    print(labels['kappa_band'])
    print(labels['L_band'])

    pass

if __name__ == "__main__":
    main()