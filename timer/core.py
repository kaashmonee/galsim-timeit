import sys
import os
import math
import logging
import galsim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from helpers import timeit
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

    DEFAULT_LAMBDA = 700
    DEFAULT_DIAMETER = 3
    DEFAULT_R0 = 0.15 * (DEFAULT_LAMBDA/500)**1.2 
    PSF_BETA = 5
    PSF_RE = 1.0 # arcsec

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
    
    def __init__(self, galaxy, flux_range : tuple, num_intervals=15, debug=False, **kwargs):
        """
        Timer object constructor. Takes in a type of galaxy and the flux range
        to vary. The flux range is a tuple that takes in the min flux and the max flux.
        flux_range : (min_flux, max_flux)
        """
        # Setting the galaxy
        self.set_galaxy(galaxy)

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

        # Setup times
        self.init_times = []

        # The timing after the convolution
        self.final_times = []

        self.rendered_images = []

        # Initializing the galaxy
        self.set_galaxy(galaxy, **kwargs)


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
        print("temp_params:", temp_params)

        for i, gal_flux in enumerate(self.flux_scale):
            rand_offset = np.random.random_sample() / (1 / random_offset_range) if random_offset_range != 0 else 0

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
        plt.title("Setup Time vs. Flux (Averaged over %d runs" % self.init_times_repeated)
        plt.xlabel("Flux")
        plt.ylabel(r"Setup Time ($\mu$s)")

        plt.plot(self.flux_scale, np.array(self.init_times) * 10**6, label=self.cur_gal_name)

        plt.legend()
        plt.show()
        plt.figure()


    def set_galaxy(self, gal : str, **kwargs):
        if gal in {"exponential", "gaussian", "devaucouleurs", "sersic"}:
            self.cur_gal_name = Timer.GALAXY_NAMES[gal]
            self.cur_gal_name_constructor = Timer.GALAXY_CONSTRUCTORS[gal]

            # If kwargs are empty...
            if not bool(kwargs):
                self.default_gal_args = Timer.GALAXY_CONSTRUCTOR_DEFAULT_PARAMS[gal]
            else:
                self.default_gal_args = kwargs

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


    def plot_draw_times(self, axis=None):
        if axis is None:
            fig, axis = plt.subplot(1, 1)

        axis.set_ylim(-0.05, 6.25)
        axis.set_title(self.cur_gal_name + " " + self.cur_psf + " " + "\nTime (s) vs. Flux")
        axis.set(xlabel="Flux", ylabel="Time (s)")

        axis.scatter(self.flux_scale, self.final_times, label=self.cur_gal_name)
        slope, intercept, r_value, p_value, stderr = stats.linregress(self.flux_scale, self.final_times)
        axis.plot(self.flux_scale, intercept + slope * flux_scale, 'tab:orange', label=self.cur_gal_name)

        annotation = "y=" + str(round(slope, 10)) + "x" + "+" + str(round(intercept, 5))
        axis.annotate(annotation, (5, 5))

    def __repr__(self):
        output = ("""
        Galaxy Name: %s,
        PSF Used: %s,
        """ % (self.cur_gal_name, self.cur_psf))

        return output


    def compute_all(self):
        pass

    def draw_all(self):
        pass



