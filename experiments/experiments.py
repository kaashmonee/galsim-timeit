from timer import Timer
import matplotlib.pyplot as plt
import numpy as np

class Experiment:
    """
    This class will store all the experiments.
    """

    def change_galaxy_shape(self):
        pass            

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

        plt.show()


    def time_phot_shooting_vs_gal_shape(self):
        """
        Experiment: Measure the time to do photon shooting while varying the galaxy
        shape, as demonstrated in demo2.py and demo3.py in the GalSim repo.

        Expected results: we do not expect differences between runs, since the 
        galaxy shape should not have an effect on the time taken to do photon shooting

        Procedure:
            - Using an exponential galaxy profile
            - Using a Kolmogorov PSF
            - No randomness
            - One repetition
            - Varying the galaxy shear
            - Plotting initialization time and convolution time vs. flux for each shear value
        """
        start, end = 1.e3, 1.e5

        for shear in shears:
            t = Timer("sersic", (start, end))
            t.time_init()

            mod_gal_objs = []
            shape = galsim.Shear(q=q, beta=beta)

            for gal in t.cur_gal_objs:
                mod_gal_objs.append(gal.shear(gal_shape))

            t.cur_gal_objs = mod_gal_objs

            t.set_psf("kolmogorov")
            t.compute_phot_draw_times()
            t.plot_draw_times()



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
        pass



    def run_experiments(self):
        self.plot_setup_times()



def main():
    Experiment().time_phot_shooting_vs_gal_size()
    Experiment().time_phot_shooting_vs_gal_shape()

if __name__ == "__main__":
    main()
