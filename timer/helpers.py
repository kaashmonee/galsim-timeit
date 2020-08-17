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


def get_axis_legend_labels(axis):
    """
    A simple wrapper to an alread existing matplotlib routine to make obtaining
    axis legends easier.
    This is done because of the way get_legend_handles_lables() returns.
    It first produces a tuple where the first element is a list of matplotlib.lines.Line2D objects
    The 2nd element is a list of the legend labels. 
    """
    return [text.get_text() for text in axis.get_legend().texts]

def get_plotted_colors(axis):
    """
    Returns a list of colors used for each Line2D object in the axis.
    """
    return [line.get_color() for line in axis.get_lines()]

def get_most_recently_drawn_color(axis):
    """
    Returns the color of the most recent line drawn.
    """
    return axis.get_lines()[-1].get_color()
