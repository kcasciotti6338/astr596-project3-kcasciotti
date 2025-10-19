# constants.py

'''
Solar: LSUN, RSUN, MSUN, TSUN
Physical: G, SIGMA_SB, WIEN_B, C, AMU, N_AVAGADRO, K_BOLTZMANN, EV, RYDBERG
Particle: E, ME, MP, MN, MH
Time: YR, GYR, MYR
Distance: AU, PC, LY
Earth: MEARTH, REARTH
BANDS: B, V, K bands
Sources: 
https://sites.astro.caltech.edu/~george/constants.html
https://aa.usno.navy.mil/downloads/publications/Constants_2021.pdf
'''

LSUN = 3.828e33 #erg s^-1
RSUN = 6.957e10 #cm
MSUN = 1.9884e33 #g
TSUN = 5772 #K

G = 6.67430e-8 #cm^3 g^−1 s^−2
SIGMA_SB = 5.670374419e-5 #erg cm^−2 s^−1 K^−4
WIEN_B = 0.289777 #cm K
AMU = 1.6605402e-24 #g
N_AVAGADRO = 6.0221367e23
K_BOLTZMANN = 1.380658e-16 #erg K^-1
EV = 1.6021772e-12 #erg
RYDBERG = 2.1798741e-11 #erg
H_PLANCK = 6.6260755e-27 #erg s

E = 4.8032068e-10 #esu
ME = 9.1093897e-28 #g
MP = 1.6726231e-24 #g
MN = 1.6749286e-24 #g
MH = 1.6733e-24 #g

CSOL = C = 2.99792458e10 #cm s^−1
YEAR = 3.15576e7 #s
GYR = 3.15576e13 #s
MYR = 3.15576e16 #s

AU = 1.49597870700e13 #cm
PC = 3.086e18 #cm
LY = 9.463e17 #cm

MEARTH = 5.9722e27 #g
REARTH = 6.378e8 #cm

BANDS = {
        'B': [390e-7, 500e-7], #cm
        'V': [500e-7, 600e-7], #cm
        'K': [1.95e-4, 2.40e-4] #cm
        }