import os

from essential import *
from qmodels import *

def main():

    data = dde.data.PDEData(
        domain=domain,
        pde=pde,
        bcs=[boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
        exact_solution=[u_func, v_func, p_func],
        num_domain=2601,
        num_bc=400,
        num_test=5000,
    )
    
    net = FNN()
    
    model = dde.Model(
        data=data,
        model=net,
        save_path="models/FNN_1000e_nodistr_tanh",
        log_every=100,
        save_every=100,
    )

    model.compile()
    model.train(epochs=1000)


if __name__ == "__main__":
    main()