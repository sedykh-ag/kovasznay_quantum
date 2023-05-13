import os

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group 

from essential import *


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

def ddp_exit():
    destroy_process_group()

def main(rank, world_size):
    ddp_setup(rank, world_size)

    data = dde.data.PDEData(
        domain=domain,
        pde=pde,
        bcs=[boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
        exact_solution=[u_func, v_func, p_func],
        num_domain=2601,
        num_bc=400,
        num_test=10000,
    )

    model = dde.Model(
        rank=rank,
        world_size=world_size,
        data=data,
        model=FNN(),
        save_every=10,
        log_every=5,
    )

    model.compile()
    model.train(epochs=100)

    ddp_exit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", "-t", required=True, type=int, help="How many threads")
    args = parser.parse_args()

    mp.spawn(main, args=(args.threads,), nprocs=args.threads)