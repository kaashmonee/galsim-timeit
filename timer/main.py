def main():

    # Initialize the logger
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo7")

    random_seed = 1534225

    gal_flux = 1.e5 # counts
    rng = galsim.BaseDeviate(random_seed + 1)
    gal_r0 = 2.7 # arcseconds
    psf_beta = 5
    pixel_scale = 0.2
    psf_re = 1.0 # arcsec


    # TODO:
    # These values were chosen for a specific galaxy and PSF.
    # When you draw images, try to look at them and make sure they look reasonable
    # Let GalSim choose the image size
    # Simplify drawImage call to avoid specifying image size
    nx = 64
    ny = 64

    # Obtaining setup start time to obtain fixed costs for setting up
    # a particular type of profile.

    num_gals = 4

    # Creates linearly and logarithmically spaced flux values
    fluxs = np.linspace(1.e3, 1.e7, 15)
    log_fluxs = np.logspace(3, 7, 15)

    # Identify the flux scale used
    # Uncomment one or the other to use one or the other throughout the code.

    flux_scale = fluxs
    # flux_scale = log_fluxs


    # Plotting setup times
    galaxy_names = ["Exponential", "Gaussian", "DeVaucouleurs", "Sersic"]

    # Creats a 4x15 array where we store the setup times for each galaxy at each flux
    setup_times_vary_flux = np.zeros((num_gals, len(flux_scale)))



if __name__ == "__main__":
    main()

