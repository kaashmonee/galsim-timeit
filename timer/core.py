import sys
import os
import math
import logging
import galsim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from timer.helpers import timeit
import copy
import pathlib


# Initializing logger...
# This is the only thing that should be done globally.
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("Timer")


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
        },
        "gaussian": {
            "half_light_radius": 1,
        },
        "devaucouleurs": {
            "half_light_radius": 1,
        },
        "sersic": {
            "half_light_radius": 1,
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
        },
        "airy": {
            "lam": PSF_DEFAULT_CONFIG["lam"],
            "diam": PSF_DEFAULT_CONFIG["diam"],
            "scale_unit": galsim.arcsec
        }
    }

    PSF_CONSTRUCTORS = {
        "vonkarman": galsim.VonKarman,
        "kolmogorov": galsim.Kolmogorov,
        "airy": galsim.Airy,
        "moffat": galsim.Moffat,
        "optical": galsim.OpticalPSF
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
        self.set_galaxy(galaxy, **kwargs)

        # Starting and ending indices
        (self.start, self.end) = flux_range

        # Setting up debug mode functionality
        # This indicates the number of intervals we're going to have. 
        # This will depend on whether or not we're running in debug mode.
        # By default, we are not going to be running in debug mode.

        # The number of intervals that the user specified.
        self.num_intervals = num_intervals

        # In debug mode, the number of intervals are going to be a lot less.
        self.num_debug_intervals = 5

        # Debug flag. We compute and plot fewer fluxes if the debug flag is set to true
        self.set_debug(debug)

        # The current galaxy associated with this Timer class.
        self.cur_gal_objs = []

        # Setup times
        self.init_times = []

        # The timing after the convolution
        self.final_times = []

        # The drawn images
        self.rendered_images = []

        # Initializing the galaxy
        self.set_galaxy(galaxy, **kwargs)

        # rng
        self.random_seed = 15434225
        self.rng = galsim.BaseDeviate(self.random_seed + 1)
        


    def set_debug(self, debug):
        """
        Sets an internal debug flag. If true, then changes the number of flux levels and plots
        fewer flux points.
        """
        self.debug = debug
        if debug:
            self.cur_num_intervals = self.num_debug_intervals
        else:
            self.cur_num_intervals = self.num_intervals

        # Creating the flux range
        self.fluxs = np.linspace(self.start, self.end, self.cur_num_intervals)
        self.log_fluxs = np.logspace(np.log(self.start), np.log(self.end), self.cur_num_intervals)

        # Default scale is linear
        # Change this using the change_flux_scale routine.
        self.flux_scale = self.fluxs

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


    def plot_init_times(self, axis=None):
        """
        This routine should be called AFTER the time_init() routine is called.
        This is the view for the `time_init` function. After time_init is called,
        this routine plots the time taken to initialize the specified galaxy profile
        vs. the flux. 
        """

        axis_is_none = not bool(axis)

        if axis is None:
            fig, axis = plt.subplots(1, 1)

        # If the user attempts to run this function before actually running time_init, it 
        # should catch the exception and re-raise it with a helpful error message that tells
        # the user to run the time_init function first.
        try:
            title = "Setup Time vs. Flux (Averaged over %d runs)" % self.init_times_repeated
            axis.set_title(title)
        except AttributeError as e:
            raise AttributeError(str(e) + "\nPlease run the time_init routine first.")

        axis.set(xlabel="Flux", ylabel=r"Setup Time ($\mu$s)")

        # Omit the first point 
        # This is because the first point dominates the time taken
        axis.plot(self.flux_scale[1:], np.array(self.init_times)[1:] * 10**6, label=self.cur_gal_name)

        logger.info("Done plotting init...")

        # If the user specifies an axis, this means they want to manage the plotting themselves.
        # We then do not want to call show() prematurely, because the user will be responsible for
        # calling show when they've plotted everything they want to on the axes they want to.
        if axis_is_none:
            fig.canvas.set_window_title(title)
            plt.show()


    def set_galaxy(self, gal : str, **kwargs):
        """
        An internal routine used by this function to choose a galaxy constructor with a 
        specified set of parameters. If kwargs is empty, then it uses the default set
        of parameters initialized above.
        """

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
        """
        Takes in the name of a PSF as a string and optional keyword arguments.
        If additional kwargs are provided, then it constructor is used on the 
        optional arguments provided.
        """

        if psf in {"vonkarman", "airy", "moffat", "kolmogorov", "optical"}:
            self.cur_psf_name = psf
            self.cur_psf_disp_name = Timer.PSFS[psf]
            self.cur_psf_constructor = Timer.PSF_CONSTRUCTORS[psf]

            # Reset to empty lists to make sure that previous
            # runs with a different PSF aren't preserved.
            self.rendered_images = []
            self.final_times = []

            if not bool(kwargs):
                self.cur_psf_args = Timer.PSF_CONSTRUCTOR_DEFAULT_PARAMS[psf]
            else:
                self.cur_psf_args = kwargs
            
            # Creates a PSF object
            self.cur_psf_obj = self.cur_psf_constructor(**self.cur_psf_args)
        else:
            raise ValueError("Please choose a valid PSF name.")

    def compute_phot_draw_times(self, **kwargs):
        """
        Takes in a PSF and its parameters. If the **kwargs is left blank,
        it uses a default set of parameters already defined. 
        """
        try:
            logger.info("Computing draw times for the %s profile convolved with %s for %d flux levels." % (self.cur_gal_name, self.cur_psf_disp_name, self.cur_num_intervals))
        except AttributeError as e:
            raise AttributeError(str(e) + "\nPlease set the psf first using set_psf.")
        

        if self.debug:
            logger.info("NOTE: running in debug mode.")

        for gal_ind, gal in enumerate(self.cur_gal_objs):
            convolved_img_final = galsim.Convolve([gal, self.cur_psf_obj])

            img, draw_img_time = timeit(convolved_img_final.drawImage) (method="phot", rng=self.rng, **kwargs)

            img_metadata = {
                "galaxy": self.cur_gal_name,
                "psf": self.cur_psf_name,
                "flux": self.flux_scale[gal_ind],
                "method": "photon_shooting"
            }
            self.rendered_images.append((img, img_metadata))
            self.final_times.append(draw_img_time)

            logger.info("Drawing %d/%d" % (gal_ind+1, self.cur_num_intervals))


    def save_phot_shoot_images(self, directory="", save=True, show=False):
        """
        If this function is called after compute_phot_draw_images,
        then it saves all the generated images to a directory in 
        examples/output. The user can also choose a directory by
        populating the directory parameter.
        """

        # Gets the parent of the directory where the current file is in.
        # This is guaranteed to be the root, since the structure of this project
        # should not change.
        root = pathlib.Path(__file__).parent.parent.resolve() 

        default_dir = os.path.join(root, "examples", "output")

        save_dir = default_dir if directory == "" else directory

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if save:
            for (ind, (img, imgdata)) in enumerate(self.rendered_images):
                debug = "debug" if self.debug else ""
                img_name = imgdata["galaxy"] + "_conv_" + imgdata["psf"] + "_" + str(imgdata["flux"]) + "_" + imgdata["method"] + debug + ".fits"
                file_name = os.path.join(save_dir, img_name)
                img.write(file_name)
                logger.info("Wrote image %d/%d to %r" % (ind+1, len(self.rendered_images), file_name))
        
        if show:
            for (ind, (img, imgdata)) in enumerate(self.rendered_images):
                logger_text = "Image %d/%d\nGalaxy: %s, PSF: %s, Flux: %s" % (ind+1, len(self.rendered_images), imgdata["galaxy"], imgdata["psf"], str(imgdata["flux"]))
                logger.info(logger_text)
                plt.figure(logger_text)
                plt.imshow(img.array, cmap="gray")



    def plot_draw_times(self, axis=None):
        """
        A plotting routine to draw the times taken to do photon shooting.
        """
        
        axis_is_none = not bool(axis)

        if axis is None:
            fig, axis = plt.subplots()


        # If this fails, this means that the user has not run the compute_phot_draw_times
        # routine. This catches the exception and raises another one suggesting that the
        # user do that first.

        try:
            title = self.cur_gal_name + " Profile Convolved with " + self.cur_psf_disp_name + " " + "\nTime (s) vs. Flux"
            axis.set_title(title) 
        except AttributeError as e:
            raise AttributeError(str(e) + "\nPlease run the compute_phot_draw_times routine first.")

        axis.set(xlabel="Flux", ylabel="Time (s)")

        axis.scatter(self.flux_scale[1:], self.final_times[1:], label=self.cur_gal_name)
        slope, intercept, r_value, p_value, stderr = stats.linregress(self.flux_scale, self.final_times)
        axis.plot(self.flux_scale[1:], intercept + slope * self.flux_scale[1:], 'tab:orange', label=self.cur_gal_name)

        annotation = "y=" + str(round(slope, 10)) + "x" + "+" + str(round(intercept, 5))

        top_right = (max(self.flux_scale) * 0.75, max(self.final_times) * 0.75)
        axis.annotate(annotation, top_right)

        # If the user specifies an axis, this means they want to manage the plotting themselves.
        # We then do not want to call show() prematurely, because the user will be responsible for
        # calling show when they've plotted everything they want to on the axes they want to.
        if axis_is_none: 
            fig.canvas.set_window_title(title)
            plt.show()

    def __repr__(self):
        output = ("""
        Galaxy Name: %s,
        PSF Used: %s,
        """ % (self.cur_gal_name, self.cur_psf_disp_name))

        return output


    def compute_all(self):
        pass
    
    

