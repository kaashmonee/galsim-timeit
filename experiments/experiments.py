from timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import galsim
from scipy import stats

class Experiment:
    """
    This class will store all the experiments.
    """

    def time_phot_shooting_vs_gal_size(self):
        """
        Experiment: Time to do photon shooting while varying the galaxy
        size and shape and keeping flux constant.

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

        start, end = 1.e3, 1.e5
        half_light_radii = np.linspace(0.5, 1.5, 5)

        fig, axs = plt.subplots(1, 2)
        init_ax = axs[0]
        draw_ax = axs[1]

        for r in half_light_radii:
            params = {
                "half_light_radius": r
            }
            t = Timer("exponential", (start, end), **params)
            t.time_init()
            t.set_psf("kolmogorov")
            t.compute_phot_draw_times()

            t.plot_init_times(axis=init_ax)
            t.plot_draw_times(axis=draw_ax)

        legend_labels = ["r = %f" % r for r in half_light_radii]

        title0 = axs[0].get_title() + "\nVarying half_light_radius"
        title1 = axs[1].get_title() + "\nVarying half_light_radius"
        axs[0].set_title(title0)
        axs[1].set_title(title1)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        plt.show()


    def time_phot_shooting_vs_gal_shape(self):
        """
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
        start, end = 1.e3, 1.e5

        # Obtained from demo3.py
        # The lower the gal_q the greater the shear.
        gal_qs = np.linspace(0.2, 1, 5)
        gal_beta = 23

        fig, axs = plt.subplots(1, 2)
        init_axis = axs[0]
        draw_axis = axs[1]

        for gal_q in gal_qs:
            t = Timer("sersic", (start, end))
            t.time_init()
            t.plot_init_times(axis=init_axis)

            mod_gal_objs = []
            shape = galsim.Shear(q=gal_q, beta=gal_beta * galsim.degrees)

            for gal in t.cur_gal_objs:
                mod_gal_objs.append(gal.shear(shape))

            t.cur_gal_objs = mod_gal_objs

            t.set_psf("kolmogorov")
            t.compute_phot_draw_times()

            t.plot_draw_times(axis=draw_axis)

        legend_labels = ["q = %f" % q for q in gal_qs]

        title0 = axs[0].get_title() + "\nVarying q value (shear)"
        title1 = axs[1].get_title() + "\nVarying q value (shear)"

        axs[0].set_title(title0)
        axs[1].set_title(title1)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        plt.show()



    def time_phot_shooting_vs_profile(self):
        """
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
        start, end = 1.e3, 1.e5

        fig, axs = plt.subplots(1, 2)
        init_axis = axs[0]
        draw_axis = axs[1]
        psf = "kolmogorov"

        for gal_name in Timer.GALAXY_NAMES:
            t = Timer(gal_name, (start, end))
            t.time_init()
            t.plot_init_times(axis=init_axis)

            t.set_psf(psf)
            t.compute_phot_draw_times()
            if gal_name == "point":
                t.save_phot_shoot_images()

            t.plot_draw_times(axis=draw_axis)

        axs[0].set_title("Init Time for Different Galaxy Profiles")
        axs[1].set_title("Time vs. Photon Shooting for Different Profiles Convolved with %s PSF" % psf)
        axs[0].legend()
        axs[1].legend()

        plt.show()

    
    def time_phot_shooting_vs_psf(self):
        """
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
        start, end = 1.e3, 1.e5

        fig, axs = plt.subplots(1, 2)

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"

        for psf in Timer.PSFS:
            t = Timer(galaxy, (start, end))

            t.time_init()
            t.plot_init_times(axis=init_axis)

            t.set_psf(psf)

            t.compute_phot_draw_times()

            t.plot_draw_times(axis=draw_axis)

        legend_labels = [psf for psf in Timer.PSFS]
        
        title1 = "Time vs. Photon Shooting for Sersic Profile Convolved with Various PSFs"
        axs[1].set_title(title1)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)


        plt.show()


    def time_phot_shooting_vs_optical_psf_params(self):
        """
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

        fig, axs = plt.subplots(1, 2)

        start, end = 1.e3, 1.e5

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        psf = "optical"

        for aberrations in aberrations_list:

            params = {
                "lam": Timer.DEFAULT_LAMBDA,
                "diam": Timer.DEFAULT_DIAMETER,
                "aberrations": aberrations
            }

            t = Timer(galaxy, (start, end))
            t.time_init()

            t.plot_init_times(axis=init_axis)

            t.set_psf(psf, **params)

            t.compute_phot_draw_times()

            t.plot_draw_times(axis=draw_axis)


        legend_labels = [
            "none",
            "defocus=%f" % defocus[4],
            "astigmatism=%f,%f" % (astigmatism[5], astigmatism[6]),
            "coma=%f,%f" % (coma[7], coma[8]),
            "spherical=%f" % spherical[11]
        ]

        title1 = "Time for Photon Shooting vs. Flux with Sersic Profile Convolved with Optical PSF"

        axs[1].set_title(title1)

        axs[0].legend(legend_labels)
        axs[1].legend(legend_labels)

        plt.show()


    def time_phot_shooting_vs_optical_psf_vary_obscuration(self):
        """
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

        fig, axs = plt.subplots(1, 2)

        start, end = 1.e3, 1.e5

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        psf = "optical"

        obscurations = np.linspace(0, 0.5, 5)

        for obscuration in obscurations:
            params = {
                "lam": Timer.DEFAULT_LAMBDA,
                "diam": Timer.DEFAULT_DIAMETER,
                "obscuration": obscuration
            }

            t = Timer(galaxy, (start, end))
            t.time_init()

            t.plot_init_times(axis=init_axis)

            t.set_psf(psf, **params)

            t.compute_phot_draw_times()

            t.plot_draw_times(axis=draw_axis)


        title0 = init_axis.get_title() + "\nVarying obscuration parameter in OpticalPSF"
        title1 = draw_axis.get_title() + "\nVarying obscuration parameter in OpticalPSF"

        legend_labels = ["obscuration = %f" % o for o in obscurations]

        init_axis.legend(legend_labels)
        draw_axis.legend(legend_labels)

        plt.show()


    def time_phot_shooting_vs_optical_psf_vary_lam_over_diam(self):
        """
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
        fig, axs = plt.subplots(1, 2)

        start, end = 1.e3, 1.e5

        init_axis = axs[0]
        draw_axis = axs[1]

        galaxy = "sersic"
        psf = "optical"

        # Obtained from GalSim documentation:
        # http://galsim-developers.github.io/GalSim/_build/html/psf.html#optical-psf
        # Last multiplication operation converts to arcseconds.
        lod = ((Timer.DEFAULT_LAMBDA * 1.e-9) / Timer.DEFAULT_DIAMETER) * 206265

        lam_over_diams = np.linspace(0.1, 5., 5) * lod

        for lam_over_diam in lam_over_diams:
            params = {
                "lam_over_diam": lam_over_diam
            }

            t = Timer(galaxy, (start, end))
            t.time_init()

            t.plot_init_times(axis=init_axis)

            t.set_psf(psf, **params)

            t.compute_phot_draw_times()

            t.plot_draw_times(axis=draw_axis)


        title0 = init_axis.get_title() + "\nVarying lam_over_diam in OpticalPSF"
        title1 = draw_axis.get_title() + "\nVarying lam_over_diam in OpticalPSF"

        legend_labels = ["lam_over_diam = %f arcsecs" % lod for lod in lam_over_diams]

        init_axis.legend(legend_labels)
        draw_axis.legend(legend_labels)

        plt.show()


    def fft_image_size_vs_flux_vary_lam_over_diam(self):
        """
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

        fig, [ax, ax2] = plt.subplots(1, 2)

        galaxy = "sersic"
        psf = "optical"

        start, end = 1.e3, 1.e5

        # Obtained from GalSim documentation:
        # http://galsim-developers.github.io/GalSim/_build/html/psf.html#optical-psf
        # Last multiplication operation converts to arcseconds.
        lod = ((Timer.DEFAULT_LAMBDA * 1.e-9) / Timer.DEFAULT_DIAMETER) * 206265 

        resolution = 5
        start_scale = 0.1
        end_scale = 5.0

        lam_over_diams = np.linspace(start_scale, end_scale, resolution) * lod
        image_sizes = []

        # Setting up plotting...
        title = "Image Size vs. lam_over_diam Parameter with \nSersic Galaxy Profile Convolved with Optical PSF"
        ax.set_title(title)
        ax.set_xlabel("lambda/diam (arcseconds)")
        ax.set_ylabel("Image Sizes")

        ax2.set_title("Image Size vs. Flux Response on lam_over_diam\nSersic Galaxy Convolved with Optical PSF")
        ax2.set_xlabel("Flux")
        ax2.set_ylabel("Image Sizes")

        avg_img_sizes = []
        img_size_std_devs = []

        for ctr, lam_over_diam in enumerate(lam_over_diams):
            params = {
                "lam_over_diam": lam_over_diam
            }

            t = Timer(galaxy, (start, end))
            t.time_init()

            t.set_psf(psf, **params)

            t.compute_phot_draw_times()

            img_sizes = [img[1]["image_size"] for img in t.rendered_images]

            ax2.plot(t.flux_scale, img_sizes)

            avg_img_size = np.mean(img_sizes)
            img_size_std_dev = np.std(img_sizes)

            avg_img_sizes.append(avg_img_size)
            img_size_std_devs.append(img_size_std_dev)

            print("%d/%d" % (ctr+1, len(lam_over_diams)))
            

        # Plotting
        ax.errorbar(lam_over_diams, avg_img_sizes, fmt="o")
        ax2_legend_labels = [("lam/diam = %f arcseconds" % lod) for lod in lam_over_diams]
        ax2.legend(ax2_legend_labels)

        plt.show()



    def get_PSF_FWHM(self):
        """
        This function just outputs the FWHM values for each PSF.
        Instantiates a dummy galaxy with a dummy start and end 
        simply for the purposes of obtaining a FWHM value.
        """

        # These values do not matter!
        s, e = 1, 2   
        galaxy = "sersic"
        for psf in Timer.PSFS:

            t = Timer(galaxy, (s, e))
            t.time_init()

            t.set_psf(psf)

            print("%s fwhm (arcseconds): %f" % (psf, t.cur_psf_obj.calculateFWHM()))



def main():
    e = Experiment()
    # e.time_phot_shooting_vs_gal_size()
    # e.time_phot_shooting_vs_gal_shape()
    # e.time_phot_shooting_vs_profile()
    # e.time_phot_shooting_vs_psf()
    # e.time_phot_shooting_vs_optical_psf_params()
    # e.time_phot_shooting_vs_optical_psf_vary_obscuration()
    # e.time_phot_shooting_vs_optical_psf_vary_lam_over_diam()
    e.fft_image_size_vs_flux_vary_lam_over_diam()
    e.get_PSF_FWHM()

if __name__ == "__main__":
    main()
