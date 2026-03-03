"""Constants and simple conversions"""
from astropy.cosmology import Planck18

c = 299792458.0 # speed of light in vacuum, m/s
c_cgs = 2.99792458e10 # speed of light in cm/s
pc_to_cm = 3.086e18 # parsec to cm
days_to_seconds = 86400.0 # days to seconds
h = 6.62607015e-34 # J s
h_erg_s = 6.6261e-27 # cm^2 g s^{-1} (i.e. erg s)
eV = 1.602176634e-19  # J
H0 = float(Planck18.H0.value)*1e3 # m/s / Mpc Planck 2018: arxiv.org/abs/1807.06209, Table 2, TT,TE,EE+lowE+lensing+BAO
Omega_m = Planck18.Om0 # Planck 2018: arxiv.org/abs/1807.06209, Table 2, TT,TE,EE+lowE+lensing+BAO, excluding massive neutrinos
msun_cgs = 1.989e33 # solar mass in grams
sigSB = 5.6704e-5 # Stefan-Boltzmann constant in erg/cm^2/s/K^4
kb = 1.3807e-16 # Boltzmann constant in erg/K
Rsun_cgs = 6.957e10 # solar radius in cm
au_cgs = 1.496e13 # astronomical unit in cm
km_cgs = 1.0e5 # kilometer in cm
