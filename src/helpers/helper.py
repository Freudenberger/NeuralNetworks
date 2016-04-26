import time


def get_date_time():
    """
    Returns current Date Time with zone information.
    """
    return time.strftime('%H:%M:%S', time.localtime())