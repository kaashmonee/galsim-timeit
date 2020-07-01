import sys
import os
import math
import logging
import galsim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Timer:

    GALAXY_NAMES = {
        "exponential": "Exponential",
        "gaussian": "Gaussian",
        "devaucouleurs": "DeVaucouleurs",
        "sersic": "Sersic"
    }

    PSFS = {
        "vonkarman": "VonKarman PSF",
        "airy": "Airy PSF",
        "moffat": "Moffat PSF",
        "kolmogorov": "Kolmogorov PSF",
        "optical": "Optical PSF"
    }
    
    def __init__(self, galaxy, flux_range : tuple, psf, num_intervals=15, debug=False):
        """
        Timer object constructor. Takes in a type of galaxy and the flux range
        to vary. The flux range is a tuple that takes in the min flux and the max flux.
        flux_range : (min_flux, max_flux)
        """
        # Setting the galaxy
        self.set_galaxy(galaxy)

        # Setting the PSF to convolve with
        self.set_psf(psf)

        # Starting and ending indices
        (start, end) = flux_range

        # Debug flag. We compute and plot fewer fluxes if the debug flag is set to true
        self.debug = debug

        # Creating the flux range
        self.fluxs = np.linspace(start, end, num_intervals)
        self.log_fluxs = np.logspace(np.log(start), np.log(end), num_intervals)


    def toggle_debug(self):
        self.debug = not self.debug


    def time_init(self):
        pass

    def plot_init_times(self):
        pass

    def set_galaxy(self, gal : str):
        if gal in {"exponential", "gaussian", "devaucouleurs", "sersic"}:
            self.cur_gal = Timer.GALAXY_NAMES[gal]
        else:
            raise ValueError("Please choose a valid galaxy profile.")
        

    def set_psf(self, psf : str):
        if psf in {"vonkarman", "airy", "moffat", "kolmogorov", "optical"}:
            self.cur_psf = Timer.PSFS[psf]
        else:
            raise ValueError("Please choose a valid PSF name.")

    def compute_draw_times(self, psf):
        pass

    def plot_draw_times(self):
        pass

    def __repr__(self):
        pass

    def compute_all(self):
        pass

    def draw_all(self):
        pass



