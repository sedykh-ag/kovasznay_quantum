from typing import Callable

import torch

from . import geometry
from . import gradients as grad


class PDEData:
    def __init__(
            self,
            domain: geometry.Hypercube,
            pde: Callable,
            bcs: list[Callable],
            exact_solution: list[Callable], # list of components of exact solution
            num_domain: int,
            num_bc: int,
            num_test: int,
    ):
        self.domain = domain
        self.pde = pde
        self.bcs = bcs
        self.exact_solution = exact_solution
        self.num_domain = num_domain
        self.num_bc = num_bc
        self.num_test = num_test

        self.MSE = torch.nn.functional.mse_loss

        self.points_pde = torch.tensor([])
        self.points_bc = torch.tensor([])
        self.points_test = self.domain.uniform_points(num_test) # no need to resample these, why not sample from the start

    def get_points_test(self):
        return self.points_test

    def get_points_train(self, resample=False):
        # resamples points only if necessary and stores them in memory for further usage (e.g. loss computation)
        if len(self.points_pde) == 0 or resample:
            self.points_pde = self.domain.random_points(self.num_domain)
            self.points_bc = self.domain.random_boundary_points(self.num_bc)
            return self.points_pde, self.points_bc
        else:
            return self.points_pde, self.points_bc
    
    def test(self, pred, metric="rmse"):
        """Compares pred solution to exact solution via calculating 'metric' on a uniform point grid.

        Args:
            pred: Predictions output.

        Returns:
            list of errors for each component [n_points, n_components]
        """
        if metric == "rmse":
            metric = lambda input, target: torch.sqrt(torch.nn.functional.mse_loss(input, target))
        else:
            raise ValueError("No such metric!")
        
        err = [None] * pred.shape[1]
        for component, func in enumerate(self.exact_solution):
            u = func(self.points_test) # exact solution
            v = pred[:, component:component+1] # pred solution
            err[component] = metric(u, v).item()
        
        return err


    def loss(self, x_pde, x_bc, pred_pde, pred_bc, C=[]):
        # pde loss
        loss_pde = []
        for residual in self.pde(x_pde, pred_pde):
            loss_pde.append(self.MSE(residual, torch.zeros_like(residual))) # < (pde_residual)**2 >
        
        # bc loss
        loss_bc = []
        for cond in self.bcs:
            residual = cond(x_bc, pred_bc)
            loss_bc.append(self.MSE(residual, torch.zeros_like(residual))) # < (u - u_0)**2 >
        
        losses = loss_pde + loss_bc # list concatenation
        if len(C) == 0: # equal weights for all losses
            C = [1.0] * len(losses)

        loss_total = torch.tensor(0.0)
        for i in range(len(losses)):
            loss_total += C[i] * losses[i]
        
        return loss_total