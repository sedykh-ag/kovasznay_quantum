import os

from torch import nn
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group 

from essential import *
from qmodels import *

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
        num_test=5000,
    )

    net = ClassicNet2(in_dim=2, out_dim=3, hidden_dim=12, activation=nn.SiLU)
    # net = QuantumNet(in_dim=2, out_dim=3, activation=nn.SiLU)
    # net = FNN()
    
    model = dde.Model(
        data=data,
        model=net,
        save_path="models/classic2_1000e_12h_silu",
        log_every=20, # log_every implies testing, so frequent logging slows down training
        save_every=10,
    )

    model.compile()
    model.train(epochs=1000)
    
    ddp_exit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", "-t", required=True, type=int, help="How many threads")
    args = parser.parse_args()
    
    mp.spawn(main, args=(args.threads,), nprocs=args.threads)