import os
from functools import wraps
from time import perf_counter

from torch.distributed import init_process_group, destroy_process_group 

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

def ddp_exit():
    destroy_process_group()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        value = func(*args, **kwargs)
        t1 = perf_counter()
        print(f"Executed in {t1 - t0:.2f} sec.")
        return value
    return wrapper