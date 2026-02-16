import jax.numpy as jnp

import dust_extinction.shapes as dustShp

from fiesta.constants import c

# coefficients from Pei 1992
abav = dustShp.P92.AbAv
extModel = dustShp.P92(
        BKG_amp=185.0 * abav,
        BKG_lambda=0.042,
        BKG_b=90.0,
        BKG_n=2.0,
        FUV_amp=27 * abav,
        FUV_lambda=0.08,
        FUV_b=5.5,
        FUV_n=4.0,
        NUV_amp=0.005 * abav,
        NUV_lambda=0.22,
        NUV_b=-1.95,
        NUV_n=2.0,
        SIL1_amp=0.010 * abav,
        SIL1_lambda=9.7,
        SIL1_b=-1.95,
        SIL1_n=2.0,
        SIL2_amp=0.012 * abav,
        SIL2_lambda=18.0,
        SIL2_b=-1.80,
        SIL2_n=2.0,
        FIR_amp=0.030 * abav,
        FIR_lambda=25.0,
        FIR_b=0.0,
        FIR_n=2.0,
    )

nu_val = jnp.geomspace(3e11, 2.9e17, 200)
k_val = nu_val / (c * 1e6) # in microns
Ax_o_Av_val = extModel(k_val)



def extinctionFactorP92SMC(nu, Ebv, redshift):
    
    nu_host = (1+redshift) * nu

    Ax_o_Av = jnp.interp(nu, nu_val, Ax_o_Av_val)
    Av = 2.93 * Ebv
    ext = jnp.ones(nu.shape)
    mask = (nu_host > 3e11) & (nu_host < 2.9e17) # range of the dustShp model
    ext = ext.at[mask].set(jnp.power(10, -0.4 * Ax_o_Av[mask] * Av))
    
    return ext


