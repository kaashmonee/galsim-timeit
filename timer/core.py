import sys
import os
import math
import logging
import galsim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .helpers import timeit
import copy


class Timer:

    GALAXY_NAMES = {
        "exponential": "Exponential",
        "gaussian": "Gaussian",
        "devaucouleurs": "DeVaucouleurs",
        "sersic": "Sersic"
    }

    GALAXY_CONSTRUCTORS = {
        "exponential": galsim.Exponential,
        "gaussian": galsim.Gaussian,
        "devaucouleurs": galsim.DeVaucouleurs,
        "sersic": galsim.Sersic
    }

    GALAXY_CONSTRUCTOR_DEFAULT_PARAMS = {
        "exponential": {
            "half_light_radius": 1,
            "flux": 0
        },
        "gaussian": {
            "half_light_radius": 1,
            "flux": 0
        },
        "devaucouleurs": {
            "half_light_radius": 1,
            "flux": 0
        },
        "sersic": {
            "half_light_radius": 1,
            "flux": 0,
            "n": 2.5
        }
    }

    PSF_DEFAULT_CONFIG = {
        "lam": DEFAULT_LAMBDA,
        "diam": DEFAULT_DIAMETER,
        "r0": DEFAULT_R0,
        "psf_beta": PSF_BETA,
        "psf_re": PSF_RE
    }


    PSF_CONSTRUCTOR_DEFAULT_PARAMS = {
        "kolmogorov": {
            "lam": PSF_DEFAULT_CONFIG["lam"],
            "r0": PSF_DEFAULT_CONFIG["r0"],
            "scale_unit": galsim.arcsec
        },
        "vonkarman": {
            "lam": PSF_DEFAULT_CONFIG["lam"],
            "r0": PSF_DEFAULT_CONFIG["r0"]
        },
        "moffat": {
            "beta": PSF_DEFAULT_CONFIG["psf_beta"],
            "flux": 1.0,
            "half_light_radius": PSF_DEFAULT_CONFIG["psf_re"]
        },
        "optical": {
            "lam": PSF_DEFAULT_CONFIG["lam"],
            "diam": PSF_DEFAULT_CONFIG["diam"]
        }
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

        # Default scale is linear
        # Change this using the change_flux_scale routine.
        self.flux_scale = self.fluxs

        # The current galaxy associated with this Timer class.
        self.cur_gal_objs = []

        # The timing after the convolution
        self.final_times = []

        self.rendered_images = []


    def toggle_debug(self):
        self.debug = not self.debug


    def time_init(self, random_offset_range=0, repeat=1):
        """
        Runs the initialization routine with a random offset in the half light radius variable.
        If, however, we're using a Sersic profile, the random offset is instead introduced in
        the Sersic profile n value.
        """

        # Copies the default galaxy arguments and updates it
        # in the loop
        temp_params = copy.deepcopy(self.default_gal_args)

        for i, gal_flux in enumerate(self.flux_scale):
            rand_offset = np.random.random_sample() / (1 / random_offset) if random_offset_range == 0 else 0

            temp_params["flux"] = gal_flux

            # Update n value for Sersic profiles, if not a sersic profile,
            # update the half_light_radius.
            if self.cur_gal_name == "Sersic":
                temp_params["n"] += rand_offset
            else:
                temp_params["half_light_radius"] += rand_offset

            gal, time_gal = timeit(self.cur_gal_name_constructor, repeat=repeat)(**temp_params)
            
            self.init_times.append(time_gal)
            self.cur_gal_objs.append(gal)

        # Save the number of repetitions that the user specified so that we have access to this 
        # variable in the plotting routine.
        self.init_times_repeated = repeat


    def plot_init_times(self):
        plt.title("Setup Time vs. Flux (Averaged over %d runs" % self.repeat)
        plt.xlabel("Flux")
        plt.ylabel(r"Setup Time ($\mu$s)")

        plt.plot(self.flux_scale, self.setup_times * 10**6, label=self.cur_gal_name)

        plt.legend()
        plt.show()
        plt.figure()


    def set_galaxy(self, gal : str, **kwargs):
        if gal in {"exponential", "gaussian", "devaucouleurs", "sersic"}:
            self.cur_gal_name = Timer.GALAXY_NAMES[gal]
            self.cur_gal_name_constructor = Timer.GALAXY_CONSTRUCTORS[gal]

            if not bool(kwargs):
                self.default_gal_args = kwargs
            else:
                self.default_gal_args = Timer.GALAXY_CONSTRUCTOR_DEFAULT_PARAMS[gal]
        else:
            raise ValueError("Please choose a valid galaxy profile.")
        

    def set_psf(self, psf : str, **kwargs):
        if psf in {"vonkarman", "airy", "moffat", "kolmogorov", "optical"}:
            self.cur_psf = Timer.PSFS[psf]
            self.cur_psf_constructor = Timer.PSF_CONSTRUCTORS[psf]

            if not bool(kwargs):
                self.cur_psf_args = Timer.PSF_CONSTRUCTOR_DEFAULT_PARAMS[psf]
            else:
                self.cur_psf_args = kwargs
            
            # Creates a PSF object
            self.cur_psf_obj = self.cur_psf_constructor(**self.cur_psf_args)
        else:
            raise ValueError("Please choose a valid PSF name.")

    def compute_draw_times(self, psf, **kwargs):
        """
        Takes in a PSF and its parameters. If the **kwargs is left blank,
        it uses a default set of parameters already defined. 
        """
        self.set_psf(psf, kwargs)
        convolution_times = np.zeros(len(flux_scale))
        final_times = np.zeros(len(flux_scale))

        for gal_ind, gal in enumerate(self.cur_gal_objs):
            convolved_img_final = galsim.Convolve([gal, self.cur_psf_obj])

            img, draw_img_time = timeit(cnvl_img_final.drawImage) (phot_image, method="phot", rng=rng)

            self.images.append(img)
            self.final_times.append(draw_img_time)




    def plot_draw_times(self):
        pass

    def __repr__(self):
        pass

    def compute_all(self):
        pass

    def draw_all(self):
        pass



