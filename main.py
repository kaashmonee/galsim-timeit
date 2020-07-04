from timer import Timer

def main():
    """
    An example main routine. This main function should function as a 
    series of examples of how to use this utility. It also serves
    to test all the routines and plot them to ensure that they are 
    working properly. This can be changed to suit your needs.
    """


    start, end = 1.e3, 1.e7
    exp_kolm_t = Timer("exponential", (start, end), debug=True)
    exp_kolm_t.time_init()
    exp_kolm_t.plot_init_times()

    exp_kolm_t.compute_phot_draw_times("kolmogorov")
    exp_kolm_t.plot_draw_times()


    Timer.draw_all()



if __name__ == "__main__":
    main()

