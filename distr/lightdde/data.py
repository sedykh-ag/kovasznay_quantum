from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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

        # will be different for each model instance
        self.ds_pde = TensorDataset(self.domain.random_points(self.num_domain))
        self.ds_bc = TensorDataset(self.domain.random_boundary_points(self.num_bc))
        self.ds_test = TensorDataset(self.domain.uniform_points(num_test))

        # prepare dataloaders
        self.dl_pde = self._prepare_dataloader(self.ds_pde)
        self.dl_bc = self._prepare_dataloader(self.ds_bc)
        self.dl_test = self._prepare_dataloader(self.ds_test)
        # это нужно чтобы самому не заниматься сплиттингом датасета между потоками

    def get_points_train(self):
        x_pde = next(iter(self.dl_pde))[0]
        x_bc = next(iter(self.dl_bc))[0]

        # rank = dist.get_rank()
        # print(f"DEBUG: [CPU {rank}] x_pde = ", x_pde)
        # print(f"DEBUG: [CPU {rank}] len(x_pde) = ", len(x_pde))
        # assert False

        return x_pde, x_bc
    
    def get_points_test(self):
        x_test = next(iter(self.dl_test))[0]
        return x_test

    def _prepare_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            sampler=DistributedSampler(dataset, shuffle=False, drop_last=False)
        )

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
            u = func(self.get_points_test()) # exact solution
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