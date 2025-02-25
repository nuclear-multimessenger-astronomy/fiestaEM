"""
Constants and simple conversions, to avoid dependence on additional non-jax packages
"""

c = 299792458.0 # speed of light in vacuum, m/s
pc_to_cm = 3.086e18 # parsec to cm
days_to_seconds = 86400.0 # days to seconds
h = 6.62607015e-34 # J s
h_erg_s = 6.6261e-27 # cm^2 g s^{-1} (i.e. erg s)
eV = 1.602176634e-19  # J
dL_at_10pc = 10 * pc_to_cm # luminosity distance at 10 pc, commmonly used in afterglowpy etc
H0 = 67400 # m/s / Mpc Planck 2018: arxiv.org/abs/1807.06209
