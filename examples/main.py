from timer import Timer


def main():
    """
    An example main routine. This main function should function as a
    series of examples of how to use this utility. It also serves
    to test all the routines and plot them to ensure that they are
    working properly. This can be changed to suit your needs.
    """


    start, end = 1.e3, 1.e5
    exp_kolm_t = Timer("exponential", (start, end), debug=False)
    exp_kolm_t.time_init()
    exp_kolm_t.plot_init_times()

    exp_kolm_t.set_psf("kolmogorov")
    exp_kolm_t.compute_phot_draw_times()
    exp_kolm_t.save_phot_shoot_images()

    exp_kolm_t.plot_draw_times()


if __name__ == "__main__":
    main()

