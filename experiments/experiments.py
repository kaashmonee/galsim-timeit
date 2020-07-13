from timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import galsim

class Experiment:
    """
    This class will store all the experiments.
    """

    def plot_setup_times(self):
        start, end = 1.e3, 1.e5

        for galaxy in Timer.GALAXY_NAMES:
            fig, axs = plt.subplots(1, 2)

            for psf in Timer.PSFS:
                t = Timer(galaxy, (start, end), debug=False)
                t.time_init()
                t.plot_init_times(axis=axs[0])

                t.set_psf(psf)
                t.compute_phot_draw_times()
                t.plot_draw_times(axis=axs[1])

            plt.show()


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


def main():
    e = Experiment()
    e.time_phot_shooting_vs_gal_size()
    e.time_phot_shooting_vs_gal_shape()
    e.time_phot_shooting_vs_profile()
    e.time_phot_shooting_vs_psf()

if __name__ == "__main__":
    main()