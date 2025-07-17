import re

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS
from sncosmo import get_bandpass


from fiesta.conversions import monochromatic_AB_mag, bandpass_AB_mag, integrated_AB_mag
import fiesta.constants as constants


#########################
### Filters           ###
#########################


class Filter:

    def __init__(self,
                 name: str,):
        """
        Filter class that uses the bandpass properties from sncosmo or just a simple monochromatic filter based on the name.
        The necessary attributes are stored as jnp arrays.

        Args: 
            name (str): Name of the filter. Will be either passed to sncosmo to get the optical bandpass, or the unit at the end will be used to create a monochromatic filter. Supported units are keV and GHz.
        """
        self.name = name

        if self.name in list(map(lambda x: x[0], _BANDPASSES._primary_loaders)):
            bandpass = get_bandpass(self.name) # sncosmo bandpass
            self.nu = constants.c / (bandpass.wave_eff*1e-10)
            self.nus = constants.c / (bandpass.wave[::-1]*1e-10)
            self.trans = bandpass.trans[::-1] # reverse the array to get the transmission as function of frequency (not wavelength)

            if len(self.nus)>100: # to avoid memory issues later
                self.nus = jnp.linspace(self.nus[0], self.nus[-1], 100)
                self.trans = bandpass(constants.c / self.nus * 1e10)

            self.filt_type = "bandpass"
            
        elif self.name in list(map(lambda x: x[0], _BANDPASS_INTERPOLATORS._primary_loaders)):
            bandpass = get_bandpass(self.name, 0) # these bandpass interpolators require a radius (here by default 0 cm)
            self.nu = constants.c/(bandpass.wave_eff*1e-10)
            self.nus = constants.c / (bandpass.wave[::-1]*1e-10)
            self.trans = bandpass.trans[::-1] # reverse the array to get the transmission as function of frequency (not wavelength)

            if len(self.nus)>100: # to avoid memory issues later
                self.nus = jnp.linspace(self.nus[0], self.nus[-1], 100)
                self.trans = bandpass(constants.c / self.nus * 1e10)
                
            self.filt_type = "bandpass"

        elif self.name.endswith("GHz"):
            freq = re.findall(r"[-+]?(?:\d*\.*\d+)", self.name.replace("-",""))
            freq = float(freq[-1])
            self.nu = freq*1e9
            self.nus = jnp.array([self.nu])
            self.trans = jnp.ones(1)
            self.filt_type = "monochromatic"

        elif self.name.endswith("keV"):
            if bool(re.match(r'^.*[^0-9.]-\d+(\.\d*)?keV$', self.name)):
                energy = float(re.findall(r"\d+(?:\.\d*)?", self.name)[-1])
                self.nu = energy*1000*constants.eV / constants.h
                self.nus = jnp.array([self.nu])
                self.trans = jnp.ones(1)
                self.filt_type = "monochromatic"

            elif bool(re.match(r'^.*[^0-9.]-\d+(\.\d*)?-\d+(\.\d*)?keV$', self.name)):
                energy1, energy2 = re.findall(r"\d+(?:\.\d*)?", self.name)
                nu1 = float(energy1)*1000*constants.eV / constants.h
                nu2 = float(energy2)*1000*constants.eV / constants.h
                self.nus = jnp.linspace(nu1, nu2, 20)
                self.trans = jnp.ones_like(self.nus)
                self.nu = jnp.mean(self.nus)
                self.filt_type = "integrated"
            
            else: 
                raise ValueError(f"X-ray filter {self.name} must either be in format 'X-ray-*-keV' or 'X-ray-*-*-keV' ")

        else:
            raise ValueError(f"Filter {self.name} not recognized")
                    
        self.wavelength = constants.c/self.nu*1e10
        self._calculate_ref_flux()

        if self.filt_type=="bandpass":
            self.get_mag = lambda Fnu, nus: bandpass_AB_mag(Fnu, nus, self.nus, self.trans, self.ref_flux)
        elif self.filt_type=="monochromatic":
            self.get_mag = lambda Fnu, nus: monochromatic_AB_mag(Fnu, nus, self.nus, self.trans, self.ref_flux)
        elif self.filt_type=="integrated":
            self.get_mag = lambda Fnu, nus: integrated_AB_mag(Fnu, nus, self.nus, self.trans)

    
    def _calculate_ref_flux(self,):
        """method to determine the reference flux for the magnitude conversion."""
        if self.filt_type in ["monochromatic", "integrated"]:
            self.ref_flux = 3631000. # mJy
        elif self.filt_type=="bandpass":
            integrand = self.trans / (constants.h_erg_s * self.nus) # https://en.wikipedia.org/wiki/AB_magnitude
            integral = jnp.trapezoid(y = integrand, x = self.nus)
            self.ref_flux = 3631000. * integral.item() # mJy
    
    def get_mags(self, fluxes: Float[Array, "n_samples n_nus n_times"], nus: Float[Array, "n_nus"]) -> Float[Array, "n_samples n_times"]:

        def get_single(flux):
            return self.get_mag(flux, nus)
        
        mags = jax.vmap(get_single)(fluxes)
        return mags
