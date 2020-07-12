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
        pass


    def run_experiments(self):
        self.plot_setup_times()



def main():
    Experiment().time_phot_shooting_vs_gal_size()
    Experiment().time_phot_shooting_vs_gal_shape()

if __name__ == "__main__":
    main()
