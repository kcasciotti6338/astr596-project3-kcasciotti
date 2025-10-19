# dust.py

'''

'''
import numpy as np
from src.utils import planck_function, integrate_band
from src.constants import BANDS, TSUN

def read_draine_opacity():
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

    filename = '/Users/kcasc/astr596/projects/astr-596-project-03-kcasciotti6338/data/kext_albedo_WD_MW_5.5A_30.txt'
    data = np.genfromtxt(filename, skip_header=80, names=['wavelength','albedo','kappa'], usecols=(0, 1, 4))
    
    lam_um  = np.asarray(data['wavelength'], dtype=float)
    albedo  = np.asarray(data['albedo'],     dtype=float)
    kappa   = np.asarray(data['kappa'],      dtype=float)

    order = np.argsort(lam_um)
    lam_um = lam_um[order]
    albedo = albedo[order]
    kappa  = kappa[order]

    lam_cm = lam_um * 1e-4

    data_dict = {'wavelength': lam_cm,   
                 'kappa': kappa,          
                 'albedo': albedo,
                 'dust_to_H': 2.199e-26 }

    return data_dict

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
    
    def kappa_weighted_planck(wavelength, temperature):
        lambd = draine_data['wavelength'] 
        kappa = draine_data['kappa']
        kappa_val = np.interp(wavelength, lambd, kappa)
        return kappa_val * planck_function(wavelength, temperature)

    numerator = integrate_band(kappa_weighted_planck, band_min, band_max, T_star)
    denominator = integrate_band(planck_function, band_min, band_max, T_star)
    
    return numerator / denominator

def main():
    """
    Testing
    """

    draine = read_draine_opacity()

    print('B avg opacity: ', calculate_band_averaged_opacity(draine, BANDS['B'][0], BANDS['B'][1], TSUN))

if __name__ == "__main__":
    main()