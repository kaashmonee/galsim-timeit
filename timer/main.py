from core import Timer

def main():

    # Simply create a timer for a galaxy with a specicfic PSF
    start, end = 1.e3, 1.e7
    exp_kolm_t = Timer("exponential", (start, end))
    exp_kolm_t.time_init()
    exp_kolm_t.plot_init_times()
    exp_kolm_t.compute_draw_times("kolmogorov")
    exp_kolm_t.plot_draw_times()
    print(exp_kolm_t)



if __name__ == "__main__":
    main()
