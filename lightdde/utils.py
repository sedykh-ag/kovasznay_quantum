from functools import wraps
from time import perf_counter


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        value = func(*args, **kwargs)
        t1 = perf_counter()
        print(f"Executed in {t1 - t0:.2f} sec.")
        return value
    return wrapper