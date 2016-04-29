import time
import helper


def time_measure(func):
    """
    Decorator function to measure time
    """
    def inner(*args_, **kwargs_):
        """
        args_ contains: [team_id, ...]
        """
        t0_ = time.time()
        output = func(*args_, **kwargs_)
        print("[{0}] Execution time of '{1}': {2} [s]".format(helper.get_date_time(), func.__name__, time.time() - t0_))
        return output
    return inner
