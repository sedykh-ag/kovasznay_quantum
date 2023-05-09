from typing import Callable

import numpy as np
import torch

from .geometry import Hypercube

class DirichletBC:
    def __init__(
        self,
        domain: Hypercube,
        func: Callable,        # u_0
        on_boundary: Callable, # boundary_condition(x, on_boundary: bool)
        component: int,
    ):
        self.domain = domain
        self.func = func
        self.on_boundary = lambda x, on: torch.tensor( # was np.array
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

        self.stored_err = None

    def __call__(self, points, pred, resample=False):
        # returns (u - u_0) residual error
        if (self.stored_err != None) and (resample == False):
            return self.stored_err # return cached value
        
        idx = self.on_boundary(points, self.domain.on_boundary(points))
        points = points[idx]
        pred = pred[idx]
        err = pred[:, self.component:self.component+1] - self.func(points)
        return err
    
    def filter(self, x):
        # filters only points which satisfy on_boundary from all input points
        return x[self.on_boundary(x, self.domain.on_boundary(x))]
