import time
import galsim
from astropy.io import fits
import sys


def timeit(func, min_of_n : int = 3):
    """
    Takes in a function func and rusn it on the arguments 
    provided. Outputs the output of func(*args) and the time taken.
    If min_of_n is an integer > 1, then it repeats
    `min_of_n` number of times and outputs the minimum amount of time 
    taken.
    """

    if min_of_n < 1:
        raise ValueError("min_of_n parameter cannot be less than 1.")

    def timeit_wrapper(*args, **kwargs):
        min_time = sys.maxsize # big, unfeasible number for a timeit experiment

        for _ in range(min_of_n):

            tstart = time.time()
            res = func(*args, **kwargs)
            tend = time.time()
            duration = tend - tstart

            if duration <= min_time:
                min_time = duration

        return res, min_time

    return timeit_wrapper
