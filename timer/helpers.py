import time

def timeit(func, repeat : int = 1):
    """
    Takes in a function func and rusn it on the arguments 
    provided. Outputs the output of func(*args) and the time taken.
    If repeat is an integer > 1, then it repeats the routine
    `repeat` number of times and outputs the average amount of time 
    taken.
    """

    if repeat < 1:
        raise ValueError("repeat parameter cannot be less than 1.")

    def timeit_wrapper(*args, **kwargs):
        average = 0

        for _ in range(repeat):

            tstart = time.time()
            res = func(*args, **kwargs)
            tend = time.time()
            duration = tend - tstart

            average += duration

        average /= repeat

        return res, average

    return timeit_wrapper
