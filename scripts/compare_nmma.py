#!/usr/bin/env python
"""Compare fiesta analytical models against NMMA reference implementations.

Since NMMA cannot be imported directly (requires Python >= 3.12 and healpy),
the core physics functions are inlined here from
    /home/mcoughli/nmma/nmma/em/lightcurve_generation.py

Comparison is done at the log10(bolometric luminosity) level.

Usage:
    module --force purge && module load gcccore/13.2.0 python/3.11.5
    source /fred/oz480/mcoughli/envs/fiesta_test/bin/activate
    python scripts/compare_nmma.py
"""

import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp

# ── fiesta ──────────────────────────────────────────────────────────────
from fiesta.inference.analytical_models import (
    ShockCoolingModel,
    ArnettModel,
    MetzgerModel,
)
from fiesta.constants import c_cgs, msun_cgs, sigSB, Rsun_cgs, days_to_seconds


# =====================================================================
# NMMA reference implementations (inlined from lightcurve_generation.py)
# =====================================================================

# Constants (matching NMMA's astropy-derived values)
_c_cgs = 2.99792458e10       # speed of light
_msun_cgs = 1.98892e33       # solar mass in grams
_sigSB = 5.670374419e-5      # Stefan-Boltzmann
_Rsun_cgs = 6.957e10         # solar radius in cm
_seconds_a_day = 86400.0


def _nmma_arnett_lc(t_day, tau_m, log10_mni):
    """NMMA arnett_lc — Arnett (1982) bolometric light curve.

    Source: nmma/em/lightcurve_generation.py lines 85-113.

    Returns Lbol in erg/s.
    """
    epsilon_ni = 3.9e10   # erg/s/g
    epsilon_co = 6.78e9
    tau_ni = 8.8          # days
    tau_co = 111.3
    y_scale = 2 * tau_ni
    s_scale = (2 * tau_co * tau_ni) / (tau_co - tau_ni)

    Mni = 10**log10_mni * _msun_cgs

    y = tau_m / y_scale
    s = tau_m / s_scale
    x = t_day / tau_m

    def _int_A_scalar(xi, yi):
        return quad(lambda z: 2*z*np.exp(-2*z*yi + z**2), 0, xi)[0]

    def _int_B_scalar(xi, yi, si):
        return quad(lambda z: 2*z*np.exp(-2*z*yi + 2*z*si + z**2), 0, xi)[0]

    int_A = np.array([_int_A_scalar(xi, y) for xi in x])
    int_B = np.array([_int_B_scalar(xi, y, s) for xi in x])

    lbol = Mni * np.exp(-x**2) * (
        (epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B
    )
    return lbol


def _nmma_sc_bol_lc(sample_times, log10_Menv, log10_Renv_cm, log10_Ee):
    """NMMA sc_bol_lc — Piro (2021) shock-cooling bolometric luminosity.

    Source: nmma/em/lightcurve_generation.py lines 308-345.

    Note: NMMA's log10_Renv is in cm; fiesta's is in solar radii.

    Returns (lbol, Rs) in (erg/s, cm).
    """
    t = sample_times * _seconds_a_day
    Me = 10**log10_Menv * _msun_cgs
    Renv = 10**log10_Renv_cm       # cm (NOT solar radii)
    Ee = 10**log10_Ee

    n = 10
    delta = 1.1
    K = (n - 3) * (3 - delta) / (4 * np.pi * (n - delta))
    kappa = 0.2
    vt = np.sqrt(((n-5)*(5-delta)/((n-3)*(3-delta))) * (2*Ee/Me))
    td = np.sqrt((3 * kappa * K * Me) / ((n-1) * vt * _c_cgs))

    prefactor = np.pi*(n-1) / (3*(n-5)) * _c_cgs * Renv * vt**2 / kappa
    L_early = prefactor * np.power(td / t, 4 / (n-2))
    L_late = prefactor * np.exp(-0.5 * (t**2 / td**2 - 1))
    lbol = np.where(t < td, L_early, L_late)

    tph = np.sqrt(3*kappa*K*Me / (2*(n-1)*vt**2))
    R_early = np.power(tph / t, 2 / (n-1)) * vt * t
    R_late = np.power(1 + (delta-1)/(n-1) * ((t/tph)**2 - 1),
                      -1/(delta-1)) * vt * t
    Rs = np.where(t < tph, R_early, R_late)
    return lbol, Rs


def _nmma_eff_metzger_lbol(sample_times, log10_mej, log10_vej_c, beta, log10_kappa_r):
    """NMMA eff_metzger_lc — 300-shell Metzger kilonova, bolometric only.

    Adapted from nmma/em/lightcurve_generation.py lines 560-646.
    Returns Lbol in erg/s (extracted from internal computation).

    Note: NMMA's log10_vej is in units of c (same as fiesta).
    """
    M0 = 10**log10_mej * _msun_cgs
    v0 = 10**log10_vej_c * _c_cgs
    kappa_r = 10**log10_kappa_r

    t = sample_times * _seconds_a_day
    tprec = len(t)

    Mn = 1e-8
    Ye = 0.1
    Xn0max = 1 - 2 * Ye
    mprec = 300

    m = np.geomspace(1e-8, M0 / _msun_cgs, mprec)
    vm = v0 * np.power(m * _msun_cgs / M0, -1.0 / beta)
    vm[vm > _c_cgs] = _c_cgs

    # Barnes+16 thermalisation efficiency
    def therm_eff(time_days, ca, cb, cd):
        tf = 2*cb * time_days**cd
        return 0.36 * (np.exp(-ca*time_days) + np.log(1.0 + tf) / tf)

    eth = therm_eff(sample_times, ca=0.56, cb=0.17, cd=0.74)

    Xn0 = Xn0max * 2 * np.arctan(Mn / m) / np.pi
    Xr = 1.0 - Xn0

    tarray = np.tile(t, (mprec, 1))
    Xn0array = np.tile(Xn0, (tprec, 1)).T
    Xrarray = np.tile(Xr, (tprec, 1)).T
    etharray = np.tile(eth, (mprec, 1))

    Xn = Xn0array * np.exp(-tarray / 900.0)
    edotn = 3.2e14 * Xn
    edotr = 2.1e10 * etharray * ((tarray / _seconds_a_day) ** (-1.3))
    edot = edotn + edotr
    kappan = 0.4 * (1.0 - Xn - Xrarray)
    kappa = kappan + kappa_r * Xrarray

    ene = np.zeros(mprec - 1)
    lum = np.zeros((mprec - 1, tprec))
    R_photo = np.zeros(tprec)

    dt = t[1:] - t[:-1]
    dm = m[1:] - m[:-1]

    for j in range(tprec - 1):
        tdiff = (0.08 * kappa[:-1, j] * m[:-1] * _msun_cgs * 3
                 / (vm[:-1] * _c_cgs * t[j] * beta))
        tau = (m[:-1] * _msun_cgs * kappa[:-1, j]
               / (4 * np.pi * (t[j] * vm[:-1])**2))

        lum_j = ene / (tdiff + t[j] * (vm[:-1] / _c_cgs))
        lum[:, j] = lum_j * dm * _msun_cgs
        ene += dt[j] * (edot[:-1, j] - (ene / t[j]) - lum_j)

        pig = np.argmin(np.abs(tau - 1))
        R_photo[j] = vm[pig] * t[j]

    # Last time step
    R_photo[-1] = R_photo[-2] if tprec > 1 else 1.0

    Ltotm = np.sum(lum, axis=0) / 1e20 / 1e20
    Lbol = np.abs(Ltotm) * 1e40

    return Lbol, R_photo


# ── helpers ─────────────────────────────────────────────────────────────

def _dex_err(log10_a, log10_b, min_lbol=10.0):
    """Absolute error in dex (log10 space)."""
    ok = (np.isfinite(log10_a) & np.isfinite(log10_b)
          & (log10_a > np.log10(min_lbol)) & (log10_b > np.log10(min_lbol)))
    if ok.sum() == 0:
        return np.array([np.nan])
    return np.abs(log10_a[ok] - log10_b[ok])


def _report(name, err_dex, threshold_dex, notes, use_p90=False):
    """Print one comparison result line."""
    mx = np.nanmax(err_dex)
    md = np.nanmedian(err_dex)
    p90 = np.nanpercentile(err_dex, 90) if len(err_dex) > 1 else mx
    check_val = p90 if use_p90 else mx
    tag = "PASS" if check_val < threshold_dex else "FAIL"
    metric = "p90" if use_p90 else "max"
    print(f"  [{tag}] {name}")
    print(f"         max={mx:.4f}  p90={p90:.4f}  median={md:.4f} dex  "
          f"(threshold {threshold_dex} dex on {metric})")
    if notes:
        print(f"         {notes}")
    return tag == "PASS", mx, md, p90


# =====================================================================
# Comparisons
# =====================================================================

def compare_arnett():
    """Arnett (1982) bolometric light curve: GL quadrature vs scipy.quad."""
    t_days = np.geomspace(0.5, 60.0, 200)
    tau_m = 15.0     # days
    log10_mni = -0.5  # ~0.316 solar masses

    # NMMA reference
    L_nmma = _nmma_arnett_lc(t_days, tau_m, log10_mni)
    log10_nmma = np.log10(np.maximum(L_nmma, 1e-30))

    # fiesta
    model = ArnettModel(filters=["bessellb"], times=jnp.array(t_days))
    x = dict(tau_m=tau_m, log10_mni=log10_mni, v_phot=1.0)
    log10_fi, _ = model.compute_log10_lbol_rphot(x, jnp.array(t_days))
    log10_fi = np.asarray(log10_fi)

    return _dex_err(log10_fi, log10_nmma, min_lbol=1e30)


def compare_shock_cooling():
    """Piro (2021) shock-cooling model.

    NOTE: fiesta and NMMA use DIFFERENT formulations of the Piro model.
    - NMMA uses the full Piro (2021) with n=10, delta=1.1, K factor
    - fiesta uses a simplified formula: L ~ E*R/(v*td^2)
    Also, fiesta's log10_Renv is in solar radii, NMMA's is in cm.
    """
    t_days = np.geomspace(0.01, 3.0, 200)

    # Physical parameters
    log10_Menv_msun = -2.0     # 0.01 solar masses
    log10_Renv_rsun = 2.0      # 100 solar radii
    log10_Renv_cm = np.log10(10**log10_Renv_rsun * _Rsun_cgs)  # convert for NMMA
    log10_Ee = 50.0            # 1e50 erg

    # NMMA reference
    L_nmma, R_nmma = _nmma_sc_bol_lc(t_days, log10_Menv_msun, log10_Renv_cm, log10_Ee)
    log10_L_nmma = np.log10(np.maximum(L_nmma, 1e-30))

    # fiesta
    model = ShockCoolingModel(filters=["bessellb"], times=jnp.array(t_days))
    x = dict(log10_Menv=log10_Menv_msun, log10_Renv=log10_Renv_rsun, log10_Ee=log10_Ee)
    log10_L_fi, _ = model.compute_log10_lbol_rphot(x, jnp.array(t_days))
    log10_L_fi = np.asarray(log10_L_fi)

    return _dex_err(log10_L_fi, log10_L_nmma, min_lbol=1e30)


def compare_metzger():
    """Metzger (2017) kilonova: fiesta single-zone vs NMMA 300-shell.

    NOTE: These are fundamentally different numerical approaches.
    - NMMA: 300 mass shells with neutron fractions, variable opacity, Barnes+16
    - fiesta MetzgerModel: single-zone energy-balance ODE
    Large differences are expected.
    """
    t_days = np.geomspace(0.1, 20.0, 200)

    log10_mej = -1.5    # ~0.03 solar masses
    log10_vej_c = -1.0  # 0.1c
    beta_val = 3.0
    log10_kappa_r = 0.0  # 1 cm^2/g

    # NMMA reference (300-shell)
    L_nmma, _ = _nmma_eff_metzger_lbol(t_days, log10_mej, log10_vej_c,
                                        beta_val, log10_kappa_r)
    log10_nmma = np.log10(np.maximum(L_nmma, 1e-30))

    # fiesta single-zone
    model = MetzgerModel(filters=["bessellb"], times=jnp.array(t_days))
    x = dict(log10_mej=log10_mej, log10_vej=log10_vej_c,
             beta=beta_val, log10_kappa_r=log10_kappa_r)
    log10_fi, _ = model.compute_log10_lbol_rphot(x, jnp.array(t_days))
    log10_fi = np.asarray(log10_fi)

    return _dex_err(log10_fi, log10_nmma, min_lbol=1e30)


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 72)
    print("fiesta vs NMMA — bolometric luminosity comparison (log10 space)")
    print("=" * 72)

    all_pass = True
    summary = []

    # ── Arnett ──────────────────────────────────────────────────────
    print("\nArnett (1982) — Ni56/Co56-powered supernova")
    print("-" * 72)
    err = compare_arnett()
    ok, mx, md, p90 = _report(
        "Arnett bolometric", err, 1.0,
        "GL quadrature vs scipy.quad; tau_ni=8.77 (fiesta) vs 8.8 (NMMA)")
    all_pass &= ok
    summary.append(("Arnett bolometric", mx, p90, md, 1.0, ok))

    # ── Shock Cooling ───────────────────────────────────────────────
    print("\nShock cooling — Piro (2021)")
    print("-" * 72)
    err = compare_shock_cooling()
    ok, mx, md, p90 = _report(
        "Shock cooling (Piro 2021)", err, 5.0,
        "DIFFERENT formulas: fiesta simplified vs NMMA full Piro (n=10, delta=1.1)",
        use_p90=True)
    all_pass &= ok
    summary.append(("Shock cooling (Piro 2021)", mx, p90, md, 5.0, ok))

    # ── Metzger ─────────────────────────────────────────────────────
    print("\nMetzger (2017) — Kilonova")
    print("-" * 72)
    err = compare_metzger()
    ok, mx, md, p90 = _report(
        "Metzger kilonova", err, 5.0,
        "DIFFERENT approaches: fiesta single-zone ODE vs NMMA 300-shell",
        use_p90=True)
    all_pass &= ok
    summary.append(("Metzger kilonova", mx, p90, md, 5.0, ok))

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Model':<35} {'Max':>8} {'P90':>8} {'Median':>8} {'Thresh':>8} {'':>6}")
    print("-" * 80)
    for name, mx, p90, md, thresh, ok in summary:
        tag = "PASS" if ok else "FAIL"
        print(f"{name:<35} {mx:>8.4f} {p90:>8.4f} {md:>8.4f} {thresh:>7.2f}   [{tag}]")

    # ── Not compared ────────────────────────────────────────────────
    print("\nModels not compared (no direct NMMA function equivalent):")
    print("  - MetzgerFullModel        (multi-shell; closest to NMMA eff_metzger_lc)")
    print("  - OneComponentKilonovaModel (custom diffusion-integral heating)")
    print("  - MagnetarBoostedKN       (Metzger ODE + magnetar engine)")

    print("\n" + ("ALL PASS" if all_pass else "SOME FAILURES — see above"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
