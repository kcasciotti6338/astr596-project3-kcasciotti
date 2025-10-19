# zams.py
"""
ZAMS mass-luminosity and mass-radius relations from Tout et al. (1996)
Self-contained functions with coefficients from the paper
Valid for masses 0.1 - 100 M_sun and Z = 0.0001 - 0.03

ADS Link: https://ui.adsabs.harvard.edu/abs/1996MNRAS.281..257T/abstract
"""

import numpy as np

def luminosity(M, Z=0.02, solar_Z_only=True):
    """
    Calculate ZAMS luminosity using Tout et al. (1996) Eq. 1
    
    L/L_sun = (α M^2.5 + β M^11) / (M^3 + γ + δ M^5 + ε M^7 + ζ M^8 + η M^9.5)
    
    Parameters
    ----------
    M : float or np.ndarray
        Stellar mass in solar masses
        Valid range: 0.1 - 100 M_sun
    Z : float, optional
        Metallicity (default: 0.02 for solar)
        Valid range from Tout et al.: 0.0001 - 0.03
    solar_Z_only : bool, optional
        If True (default), only Z=0.02 is implemented
        Set to False for extension with Z-dependence
    
    Returns
    -------
    L : float or np.ndarray
        Luminosity in solar luminosities
    
    Raises
    ------
    AssertionError
        If mass or metallicity outside valid range
    
    References
    ----------
    Tout et al. (1996) MNRAS 281, 257
    See equations (1) for luminosity formula
    See equations (3)-(4) for metallicity dependence
    """
    
    if isinstance(M, np.ndarray):
        # np.all() returns True only if ALL elements satisfy the condition
        assert np.all(M >= 0.1) and np.all(M <= 100), \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
    else:
        assert 0.1 <= M <= 100, \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"

    if isinstance(Z, np.ndarray):
        # np.all() returns True only if ALL elements satisfy the condition
        assert np.all(Z >= 0.0001) and np.all(Z <= 0.03), \
            "Metalicities must be between 0.0001 and 0.03 (Tout et al. 1996 validity range)"
    else:
        assert 0.0001 <= Z <= 0.03, \
            "Metalicities must be between 0.0001 and 0.03 (Tout et al. 1996 validity range)"
    
    if solar_Z_only:
        assert Z == 0.02, \
            "Solar metalicity of Z == 0.02 (Tout et al. 1996)"
    
    # Coefficients from Table 1
    coeffs = {
        'alpha': 0.39704170,
        'beta': 8.52762600,
        'gamma': 0.00025546,
        'delta': 5.43288900,
        'epsilon': 5.56357900,
        'zeta': 0.78866060,
        'eta': 0.00586685
    }

    L = ((coeffs['alpha']*M**(5.5) + coeffs['beta']*M**11)
            / (coeffs['gamma'] + M**3 + coeffs['delta']*M**5
                + coeffs['epsilon']*M**7 + coeffs['zeta']*M**8
                    + coeffs['eta']*M**(9.5)))
    
    return L

def radius(M, Z=0.02, solar_Z_only=True):
    """
    Calculate ZAMS radius using Tout et al. (1996) Eq. 2
    
    R/R_sun = (θ M^2.5 + ι M^6.5 + κ M^11+ λ M^19 + + μ M^19.5) / (ν + ξ M^2 + ο M^8.5 + M^18.5 + π M^19.5)

    Parameters
    ----------
    M : float or np.ndarray
        Stellar mass in solar masses
        Valid range: 0.1 - 100 M_sun
    Z : float, optional
        Metallicity (default: 0.02 for solar)
        Valid range from Tout et al.: 0.0001 - 0.03
    solar_Z_only : bool, optional
        If True (default), only Z=0.02 is implemented
        Set to False for extension with Z-dependence
    
    Returns
    -------
    R : float or np.ndarray
        Radius in solar radii
    
    Raises
    ------
    AssertionError
        If mass or metallicity outside valid range
    
    References
    ----------
    Tout et al. (1996) MNRAS 281, 257
    See equations (2) for radius formula
    """
    if isinstance(M, np.ndarray):
        # np.all() returns True only if ALL elements satisfy the condition
        assert np.all(M >= 0.1) and np.all(M <= 100), \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
    else:
        assert 0.1 <= M <= 100, \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
    
    if isinstance(Z, np.ndarray):
        # np.all() returns True only if ALL elements satisfy the condition
        assert np.all(Z >= 0.0001) and np.all(Z <= 0.03), \
            "Metalicities must be between 0.0001 and 0.03 (Tout et al. 1996 validity range)"
    else:
        assert 0.0001 <= Z <= 0.03, \
            "Metalicities must be between 0.0001 and 0.03 (Tout et al. 1996 validity range)"
    
    if solar_Z_only:
        assert Z == 0.02, \
            "Solar metalicity of Z == 0.02 (Tout et al. 1996)"
    
    # Coefficients from Table 2
    coeffs = {
        'theta': 1.71535900, 
        'iota': 6.59778800,
        'kappa': 10.08855000,
        'lambda': 1.01249500,
        'mu': 0.07490166,
        'nu': 0.01077422,
        'xi': 3.08223400,
        'omicron': 17.84778000,
        'pi': 0.00022582
    }
    
    R = ((coeffs['theta']*M**(2.5) + coeffs['iota']*M**6.5
          + coeffs['kappa']*M**11 + coeffs['lambda']*M**19 + coeffs['mu']*M**19.5)
            / (coeffs['nu'] + coeffs['xi']*M**2 + coeffs['omicron']*M**8.5 + M**18.5
                    + coeffs['pi']*M**(19.5)))
    
    return R