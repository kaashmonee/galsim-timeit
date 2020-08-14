from timer import Timer
from timer.helpers import get_axis_legend_labels
import matplotlib.pyplot as plt
import numpy as np
import galsim
from scipy import stats
import os
from pathlib import Path

# DEBUGGING TOOLS #
import pdb
# END DEBUGING TOOLS #

class Experiment:
    """
    This class will store all the experiments.
    """

    def __init__(self, save=True, show=False, exp_dat_dir="", 
                 default_flux_range=None):

        self.save = save
        self.show = show
        self.exp_dat_dir = exp_dat_dir
        self.cwd = os.getcwd()

        # Define the save locations for all the plots and the images.
        self.plot_save_dir = os.path.join(
            self.cwd,
            "results",
            self.exp_dat_dir, 
            "experiment_plots"
        )

        self.img_save_dir = os.path.join(
            self.cwd,
            "results",
            self.exp_dat_dir,
            "generated_images"
        )

        self.default_flux_range = default_flux_range


    def time_vs_flux_on_gal_size(self, method="phot", plot:tuple=None, legend_labels=[]):
        """
        experiment_1
        Experiment: Time to do photon shooting while varying the galaxy
        size and shape and keeping flux constant. plot is an input variable
        that takes in 2tuple that should contain a (fig, axes:tuple).

        Expected Results: should not expect a significant change between runs

        Procedure:
            - Using an exponential galaxy profile
            - Using a Kolmogorov PSF
            - No randomness
            - One repetition
            - Varying half_light_radius parameter over 5 possible values
            - Plotting initialization time and convolution time for each 
              half_light_radius value.
        """

        half_light_radii = np.linspace(0.5, 1.5, 5)

        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_ax = axs[0]
        draw_ax = axs[1]

        best_fit_equations = []

        galaxy = "exponential"
        psf = "kolmogorov"

        fft_draw_times = []
        fft_draw_time_stdev = []
        fft_image_sizes = []

        for r in half_light_radii:

            params = Timer.GALAXY_CONSTRUCTOR_DEFAULT_PARAMS[galaxy]
            params["half_light_radius"] = r

            t = Timer(galaxy, flux_range=self.default_flux_range, **params)
            t.time_init()
            t.set_psf(psf)
            t.compute_phot_draw_times(method=method)

            best_fit_equations.append(t.draw_time_line_annotation)


            t.plot_init_times(axis=init_ax)
            t.plot_draw_times(axis=draw_ax)

            # algorithm:
            # run t.compute_phot_draw_times(method="fft")
            # - get the fft times
            #   - compute the average and std deviation
            # - compute the image size
            # - add the fft average time to a list of average times
            # - add the fft std dev to a list of std dev times
            # - add the image size to a list of image times

            fft_draw_stats = self.compute_fft_draw_time_stats(t)
            fft_draw_times.append(fft_draw_stats["mean_draw_time"])
            fft_draw_time_stdev.append(fft_draw_stats["draw_time_stdev"])
            fft_image_sizes.append(fft_draw_stats["image_size"])

        # plot the image times vs image size using the std dev list as the 
        # error bars

        self.plot_fft_draw_time_vs_image_size(
            fft_draw_times,
            fft_draw_time_stdev,
            fft_image_sizes,
            1
        )

        legend_labels.extend(["r = %f\n%s %s" % (r, annotation, method) for (r, annotation) in zip(half_light_radii, best_fit_equations)])

        title0 = axs[0].get_title() + "\nVarying half_light_radius"
        title1 = axs[1].get_title() + "\nVarying half_light_radius"
        axs[0].set_title(title0)
        axs[1].set_title(title1)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 1)


    def time_vs_flux_on_gal_shape(self, method="phot", plot:tuple=None, legend_labels=[]):
        """
        experiment_2
        Experiment: Measure the time to do photon shooting while varying the galaxy
        shape, as demonstrated in demo2.py and demo3.py in the GalSim repo.

        Expected results: we do not expect differences between runs, since the 
        galaxy shape should not have an effect on the time taken to do photon shooting

        Procedure:
            - Using an Sersic galaxy profile
            - Using a Kolmogorov PSF
            - No randomness
            - One repetition
            - Varying the galaxy shear
            - Plotting initialization time and convolution time vs. flux for each shear value
        """
        # Obtained from demo3.py
        # The lower the gal_q the greater the shear.
        gal_qs = np.linspace(0.2, 1, 5)
        gal_beta = 23

        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_axis = axs[0]
        draw_axis = axs[1]

        best_fit_line_equations = []

        galaxy, psf = "sersic", "kolmogorov"

        for gal_q in gal_qs:
            t = Timer(galaxy, flux_range=self.default_flux_range)
            t.time_init()
            t.plot_init_times(axis=init_axis)

            mod_gal_objs = []
            shape = galsim.Shear(q=gal_q, beta=gal_beta * galsim.degrees)

            for gal in t.cur_gal_objs:
                mod_gal_objs.append(gal.shear(shape))

            t.cur_gal_objs = mod_gal_objs

            t.set_psf(psf)
            t.compute_phot_draw_times(method=method)
            
            best_fit_line_equations.append(t.draw_time_line_annotation)

            t.plot_draw_times(axis=draw_axis)

        legend_labels.extend(["q = %f\n%s %s" % (q, annot, method) for (q, annot) in zip(gal_qs, best_fit_line_equations)])

        title0 = axs[0].get_title() + "\nVarying q value (shear)"
        title1 = axs[1].get_title() + "\nVarying q value (shear)"

        axs[0].set_title(title0)
        axs[1].set_title(title1)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 2)


    def time_vs_flux_on_profile(self, method="phot", plot:tuple=None):
        """
        experiment_3
        Experiment: Measure the time to do photon shooting vs. flux while varying
        the galaxy profile.

        Expected results: we do not expect that the galaxy profile is a 
        confounding factor in the time vs. flux measurements when performing
        photon shooting.

        Procedure:
            - Vary galaxy profile
            - Use Kolmogorov PSF
            - No randomness
            - One repetition
            - Plot initialization time and convolution time for each flux value
              for each galaxy.
        """

        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_axis = axs[0]
        draw_axis = axs[1]
        psf = "kolmogorov"

        best_fit_line_equations = []

        for gal_name in Timer.GALAXY_NAMES:
            t = Timer(gal_name, flux_range=self.default_flux_range)
            t.time_init()
            t.plot_init_times(axis=init_axis)

            t.set_psf(psf)
            t.compute_phot_draw_times(method=method)

            t.plot_draw_times(axis=draw_axis)
            best_fit_line_equations.append(t.draw_time_line_annotation)

        axs[0].set_title("Init Time for Different Galaxy Profiles")
        axs[1].set_title("Time vs. Photon Shooting for Different Profiles Convolved with %s PSF" % psf)
        axs[0].legend()

        # This is done because of the way get_legend_handles_lables() returns and because
        # of the fact that the Timer class has a default method of setting labels by galaxy name.
        # It first produces a tuple where the first element is a list of matplotlib.lines.Line2D objects
        # The 2nd element is a list of the legend labels. We furthermore only want to 
        # modify the first 4 labels since they represent hte labels for the plot, so we 
        # split up the original legend_labels list into 2.
        line_legend_labels_to_modify = axs[1].get_legend_handles_labels()[1][0:5]
        rest = axs[1].get_legend_handles_labels()[1][5:]

        labels_annotated_half = [label + "\n%s" % annot for (label, annot) in zip(line_legend_labels_to_modify, best_fit_line_equations)]
        ax1labels = list(labels_annotated_half) + list(rest)

        # import pdb; pdb.set_trace()
        axs[1].legend(ax1labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 3)
    
    def time_vs_flux_on_psf(self, method="phot", plot:tuple=None, legend_labels=[]):
        """
        experiment_4
        Experiment: Measure the time to do photon shooting vs. flux while varying the PSF.

        Expected results: We do not expect substantial differences between runs, since the PSF 
        shouldn't drastically affect the runtime of the photon shooting routine.

        Procedure:
            - Use Sersic galaxy profile
            - Vary PSF
            - No randomness
            - One repetition
            - Plot instantiation time and convolution time for each flux value on different convolutions
              with different PSFs.
        """
        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        lines = []

        for psf in Timer.PSFS:
            t = Timer(galaxy, flux_range=self.default_flux_range)

            t.time_init()
            t.plot_init_times(axis=init_axis)

            t.set_psf(psf)

            t.compute_phot_draw_times(method=method)
            lines.append(t.draw_time_line_annotation)

            t.plot_draw_times(axis=draw_axis)

        temp_labels = [(psf+" %s" % method) for psf in Timer.PSFS]
        
        title1 = "Time vs. Photon Shooting for Sersic Profile Convolved with Various PSFs"
        axs[1].set_title(title1)

        temp_labels = [label+"\n%s" % annot for (label, annot) in zip(temp_labels, lines)]

        legend_labels.extend(temp_labels)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 4)


    def time_vs_flux_on_optical_psf_params(self, method="phot", plot:tuple=None, legend_labels=[]):
        """
        experiment_5
        Experiment: Measure the time to do photon shooting vs. flux by varying various parameters of the Optical PSF.

        Expected results: Not sure

        Procedure:
            - Use sersic galaxy profile
            - Use Optical PSF
            - Vary parameters using the Noll index to include different optical aberrations.
            - One repetition
            - Plot instantiation time and convolution time for each flux value on different convolutions with different
              types of Optical PSFs

        Results: No dependence on aberration
        """

        # Include defocus, astigmatism, coma, and trefoil

        defocus = [0.0] * 12
        defocus[4] = 0.06                         # Noll index 4 = defocus

        astigmatism = [0.0] * 12
        astigmatism[5:7] = [0.12, -0.08]          # Noll index 5, 6 = astigmatism

        coma = [0.0] * 12
        coma[7:9] = [0.07, 0.04]                  # Noll index 7, 8 = coma

        spherical = [0.0] * 12
        spherical[11] = -0.13                     # Noll index 11 = spherical


        aberrations_list = [
            [0.0] * 12,
            defocus,
            astigmatism,
            coma,
            spherical
        ]

        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        psf = "optical"

        lines = []

        for aberrations in aberrations_list:

            params = Timer.get_PSF_default_params(psf)
            params["aberrations"] = aberrations

            t = Timer(galaxy, flux_range=self.default_flux_range)
            t.time_init()

            t.plot_init_times(axis=init_axis)

            t.set_psf(psf, **params)

            t.compute_phot_draw_times(method=method)
            lines.append(t.draw_time_line_annotation)

            t.plot_draw_times(axis=draw_axis)


        temp_labels = [
            "none",
            "defocus=%f" % defocus[4],
            "astigmatism=%f,%f" % (astigmatism[5], astigmatism[6]),
            "coma=%f,%f" % (coma[7], coma[8]),
            "spherical=%f" % spherical[11]
        ]
        temp_labels = [label + " %s" % method for label in temp_labels]


        title1 = "Time for Photon Shooting vs. Flux with Sersic Profile Convolved with Optical PSF"

        axs[1].set_title(title1)

        temp_labels = [label + "\n%s" % annot for (label, annot) in zip(temp_labels, lines)]

        legend_labels.extend(temp_labels)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 5)


    def time_vs_flux_on_optical_psf_vary_obscuration(self, method="phot", plot:tuple=None, legend_labels=[]):
        """
        experiment_6
        Experiment: Measure the time to do photon shooting vs. flux while changing the lam_over_diam
        parameter for the OpticalPSF.

        Expected Results: Since the OpticalPSF does do a Fourier transform, we expect that the 
        time taken should change if we double the lam_over_diam parameter since the image size
        also changes.

        Procedure:
            - Use sersic galaxy profile
            - Use Optical PSF
            - Vary obscuration for obscuration values in [0, 0.25, 0.5, 0.75, 1]
            - One repetition
            - Plot instantiation time and convolution time for each flux value on different convolutions
              for 2 different lam_over_diam parameters.
        """

        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        psf = "optical"

        obscurations = np.linspace(0, 0.5, 5)

        lines = []

        for obscuration in obscurations:

            params = Timer.get_PSF_default_params(psf)
            params["obscuration"] = obscuration

            t = Timer(galaxy, flux_range=self.default_flux_range)
            t.time_init()

            t.plot_init_times(axis=init_axis)

            t.set_psf(psf, **params)

            t.compute_phot_draw_times(method=method)
            lines.append(t.draw_time_line_annotation)

            t.plot_draw_times(axis=draw_axis)


        temp_labels = ["obscuration = %f %s" % (o, method) for o in obscurations]

        init_axis.legend(legend_labels)

        temp_labels = [label+"\n%s" % annot for (label, annot) in zip(temp_labels, lines)]

        legend_labels.extend(temp_labels)

        draw_axis.legend(legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 6)


    def time_vs_flux_on_optical_psf_vary_lam_over_diam(self, method="phot", plot:tuple=None, legend_labels=[]):
        """
        experiment_7
        Experiment: Measure the time to do photon shooting vs. flux while changing the lam_over_diam
        parameter for the OpticalPSF.

        Expected Results: Since the OpticalPSF does do a Fourier transform, we expect that the 
        time taken should change if we double the lam_over_diam parameter since the image size
        also changes.

        Procedure:
            - Use sersic galaxy profile
            - Use Optical PSF
            - One repetition
            - Plot instantiation time and convolution time for each flux value on different convolutions
              for 2 different lam_over_diam parameters.
        """                
        if plot is None:
            fig, axs = plt.subplots(1, 2)
        else:
            (fig, axs) = plot

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        psf = "optical"

        # Obtained from GalSim documentation:
        # http://galsim-developers.github.io/GalSim/_build/html/psf.html#optical-psf
        # Last multiplication operation converts to arcseconds.
        lod = ((Timer.DEFAULT_LAMBDA * 1.e-9) / Timer.DEFAULT_DIAMETER) * 206265

        lam_over_diams = np.linspace(0.1, 5., 5) * lod

        lines = []

        for lam_over_diam in lam_over_diams:
            
            # Update params
            # Have to get rid of lam and diam keys because 
            # we're specifying lam_over_diam, and GalSim will only accept
            # either a lam and diam value or a lam_over_diam value in its PSF
            # constructors.
            params = Timer.get_PSF_default_params(psf)
            del params["lam"]
            del params["diam"]
            params["lam_over_diam"] = lam_over_diam

            t = Timer(galaxy, flux_range=self.default_flux_range)
            t.time_init()

            t.plot_init_times(axis=init_axis)

            t.set_psf(psf, **params)

            t.compute_phot_draw_times(method=method)
            lines.append(t.draw_time_line_annotation)

            t.plot_draw_times(axis=draw_axis)


        temp_labels = ["lam_over_diam = %f arcsecs %s" % (lod, method) for lod in lam_over_diams]


        temp_labels = [label+"\n%s" % annot for (label, annot) in zip(temp_labels, lines)]

        legend_labels.extend(temp_labels)

        init_axis.legend(legend_labels)
        draw_axis.legend(legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 7)


    def fft_image_size_vs_flux_vary_lam_over_diam(self, method="phot", plot:tuple=None):
        """
        experiment_8
        Experiment: Determine relationship of lam_over_diam parameter to image size.

        Expected Results: On various runs with different lam_over_diam parameters, we expect
        a linear relationship between the lam_over_diam parameter to the size of the image computing
        an FFT. 

        Procedure: 
            - Choose a Sersic galaxy profile.
            - Choose an OpticalPSF profile. (This is because the OpticalPSF routine does an underlying FFT)
              Please see the first paragraph in Rowe et al., 2015 (https://arxiv.org/pdf/1407.7676.pdf) page 6 
              for more information.
            - One repetition.
            - Perform photon shooting using the compute_phot_draw_times() routine without any parameters on a Timer object.
            - Plot image size vs. flux.
        """

        if plot is None:
            fig, [ax, ax2] = plt.subplots(1, 2)
        else:
            (fig, [ax, ax2]) = plot

        galaxy = "point"
        psf = "optical"

        # Obtained from GalSim documentation:
        # http://galsim-developers.github.io/GalSim/_build/html/psf.html#optical-psf
        # Last multiplication operation converts to arcseconds.
        lod = ((Timer.DEFAULT_LAMBDA * 1.e-9) / Timer.DEFAULT_DIAMETER) * 206265 

        resolution = 5
        start_scale = 0.1
        end_scale = 5.0

        lam_over_diams = np.linspace(start_scale, end_scale, resolution) * lod

        # Setting up plotting...
        title = "Image Size vs. lam_over_diam Parameter with \n%s Galaxy Profile Convolved with %s PSF" % (galaxy, psf)
        ax.set_title(title)
        ax.set_xlabel("lambda/diam (arcseconds)")
        ax.set_ylabel("Image Sizes")

        ax2.set_title("Image Size vs. Flux Response on lam_over_diam\n%s Galaxy Convolved with %s PSF" % (galaxy, psf))
        ax2.set_xlabel("Flux")
        ax2.set_ylabel("Image Sizes")

        avg_img_sizes = []
        img_size_std_devs = []

        lines = []

        for ctr, lam_over_diam in enumerate(lam_over_diams):
            
            # Updating custom params. Using default params and modifying it.
            # This is done for the same reason as in experiment 7.
            params = Timer.get_PSF_default_params(psf)
            del params["lam"]
            del params["diam"]
            params["lam_over_diam"] = lam_over_diam

            t = Timer(galaxy, flux_range=self.default_flux_range)
            t.time_init()

            t.set_psf(psf, **params)

            t.compute_phot_draw_times(method=method)
            lines.append(t.draw_time_line_annotation)

            img_sizes = [img[1]["image_size"] for img in t.rendered_images]

            ax2.plot(t.flux_scale, img_sizes)

            avg_img_size = np.mean(img_sizes)
            img_size_std_dev = np.std(img_sizes)

            avg_img_sizes.append(avg_img_size)
            img_size_std_devs.append(img_size_std_dev)

            t.save_phot_shoot_images(directory=self.img_save_dir)

            print("%d/%d" % (ctr+1, len(lam_over_diams)))
            

        # Plotting
        ax.errorbar(lam_over_diams, avg_img_sizes, fmt="o")

        ax2_legend_labels = ["lam/diam = %f arcseconds\n%s" % (lod, annot) for (lod, annot) in zip(lam_over_diams, lines)]
        ax2.legend(ax2_legend_labels)

        if self.show:
            plt.show()

        if self.save:
            self.save_figure(fig, 8)


    def run_all(self, method="phot"):
        self.time_vs_flux_on_gal_size(method=method)
        self.time_vs_flux_on_gal_shape(method=method)
        self.time_vs_flux_on_profile(method=method)
        self.time_vs_flux_on_psf(method=method)
        self.time_vs_flux_on_optical_psf_params(method=method)
        self.time_vs_flux_on_optical_psf_vary_obscuration(method=method)
        self.time_vs_flux_on_optical_psf_vary_lam_over_diam(method=method)
        self.fft_image_size_vs_flux_vary_lam_over_diam(method=method)
        self.get_PSF_FWHM()


    def get_PSF_FWHM(self):
        """
        experiment_9
        This function just outputs the FWHM values for each PSF.
        Instantiates a dummy galaxy with a dummy start and end 
        simply for the purposes of obtaining a FWHM value.
        """

        galaxy = "sersic"
        for psf in Timer.PSFS:

            t = Timer(galaxy, flux_range=self.default_flux_range)
            t.time_init()

            t.set_psf(psf)

            print("%s fwhm (arcseconds): %f" % (psf, t.cur_psf_obj.calculateFWHM()))


    def save_figure(self, figure, experiment_number, filename_prefix=""):
        """
        This saves the image to the ./experiment_results directory as a PNG.
        """
        save_dir = self.plot_save_dir

        # Make the directory if it is not already there.
        if not os.path.isdir(save_dir):
            path = Path(save_dir)
            path.mkdir(parents=True)

        filename = "experiment_%d.png" % experiment_number

        # Add the option to include a prefix in case we want to call this method
        # multiple times within the same routine.
        filename = filename_prefix + filename

        save_loc = os.path.join(save_dir, filename)

        width = 20 # inches 
        height = 15 # inches
        figure.set_size_inches(width, height, forward=True)
        figure.savefig(save_loc)

        print("Saving %s" % filename)


    def compute_fft_draw_time_stats(self, t):
        """
        This routine takes in timer:Timer object that contains the results of 
        having completed the FFT drawing routines. We use this to compute
        """
        t.compute_phot_draw_times(method="fft")
        draw_times = t.final_times
        image_sizes = [rendered_image[1]["image_size"] for rendered_image in t.rendered_images]
        
        # Safety check to ensure that only one image size is generated
        assert len(set(image_sizes)) == 1

        dat = dict()
        dat["image_size"] = image_sizes[0]

        mean_draw_time = np.mean(draw_times)
        dat["mean_draw_time"] = mean_draw_time

        draw_time_stdev = np.std(draw_times)
        dat["draw_time_stdev"] = draw_time_stdev

        return dat


    def plot_fft_draw_time_vs_image_size(self, draw_times, stdevs, image_sizes, exp_number):
        """
        This routine takes in a list of draw times, stdevs, and image sizes and 
        plots them.
        """
        fig, ax = plt.subplots()

        ax.errorbar(image_sizes, draw_times, yerr=stdevs, fmt="o")
        self.save_figure(fig, exp_number, filename_prefix="fft_draw_times")


        


        



class PhotonAndFFTPlottingExperiment(Experiment):

    def run_fft_times_on_changing_flux(self):
        
        # plots: (figure object * (ax1 * ax2 * ... * axn)) * ... *
        plots = [plt.subplots(1, 2) for _ in range(8)]

        # experiment_1 
        self.save = False
        self.time_vs_flux_on_gal_size(method="phot", plot=plots[0])
        labels = get_axis_legend_labels(plots[0][1][1])
        self.save = True
        self.time_vs_flux_on_gal_size(method="fft", plot=plots[0], legend_labels=labels)

        # experiment_2
        self.save = False
        self.time_vs_flux_on_gal_shape(method="phot", plot=plots[1])
        labels = get_axis_legend_labels(plots[1][1][1])
        self.save = True
        self.time_vs_flux_on_gal_shape(method="fft", plot=plots[1], legend_labels=labels)

        # experiment_3
        self.save = False
        self.time_vs_flux_on_profile(method="phot", plot=plots[2])
        self.save = True
        self.time_vs_flux_on_profile(method="fft", plot=plots[2])

        # experiment_4
        self.save = False
        self.time_vs_flux_on_psf(method="phot", plot=plots[3])
        labels = get_axis_legend_labels(plots[3][1][1])
        self.save = True
        self.time_vs_flux_on_psf(method="fft", plot=plots[3], legend_labels=labels)

        # experiment_5
        self.save = False
        self.time_vs_flux_on_optical_psf_params(method="phot", plot=plots[4])
        labels = get_axis_legend_labels(plots[4][1][1])
        self.save = True
        self.time_vs_flux_on_optical_psf_params(method="fft", plot=plots[4], legend_labels=labels)

        # experiment_6
        self.save = False
        self.time_vs_flux_on_optical_psf_vary_obscuration(method="phot", plot=plots[5])
        labels = get_axis_legend_labels(plots[5][1][1])
        self.save = True
        self.time_vs_flux_on_optical_psf_vary_obscuration(method="fft", plot=plots[5], legend_labels=labels)

        # experiment_7
        self.save = False
        self.time_vs_flux_on_optical_psf_vary_lam_over_diam(method="phot", plot=plots[6])
        labels = get_axis_legend_labels(plots[6][1][1])
        self.save = True
        self.time_vs_flux_on_optical_psf_vary_lam_over_diam(method="fft", plot=plots[6], legend_labels=labels)



def main():
    e = Experiment(exp_dat_dir="test_time_v_image_size")
    e.time_vs_flux_on_gal_size()
    # e = PhotonAndFFTPlottingExperiment(exp_dat_dir="testing_horizontals")
    # e.run_fft_times_on_changing_flux()
    # e.run_all()

    # e1 = Experiment(exp_dat_dir="flux_v_time_fft_on_phot_shooting_experiments_entire_flux_range")
    # e1.run_all(method="fft")

    # e2 = Experiment(exp_dat_dir="flux_v_time_phot_shooting_dat_10_3_max_flux",
    #                 default_flux_range=(1.e1, 1.e3))
    # e2.run_all()


if __name__ == "__main__":
    main()
