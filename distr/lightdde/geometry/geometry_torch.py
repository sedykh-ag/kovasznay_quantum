import itertools

import numpy as np
import torch

from .sampler import sample
from .. import config


class Hypercube():
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        
        self.dim = len(xmin)
        self.xmin = torch.tensor(xmin, dtype=config.real(torch))
        self.xmax = torch.tensor(xmax, dtype=config.real(torch))
        if torch.any(self.xmin >= self.xmax):
            raise ValueError("xmin >= xmax")

        self.side_length = self.xmax - self.xmin
        self.volume = torch.prod(self.side_length)

    def inside(self, x):
        return torch.logical_and(
            torch.all(x >= self.xmin, axis=-1), torch.all(x <= self.xmax, axis=-1)
        )
    
    def on_boundary(self, x):
        _on_boundary = torch.logical_or(
            torch.any(torch.isclose(x, self.xmin), axis=-1),
            torch.any(torch.isclose(x, self.xmax), axis=-1),
        )
        return torch.logical_and(self.inside(x), _on_boundary)

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(torch.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    torch.linspace(
                        self.xmin[i], self.xmax[i], steps=ni, dtype=config.real(torch)
                    )
                )
            else:
                xi.append(
                    torch.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        steps=ni + 1,
                        dtype=config.real(torch),
                    )[1:-1]
                )
        x = torch.tensor(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        x = torch.tensor(sample(n, self.dim, random), dtype=config.real(torch))
        return self.xmin + (self.xmax - self.xmin) * x

    def random_boundary_points(self, n, random="pseudo"):
        x = torch.tensor(sample(n, self.dim, random), dtype=config.real(torch))
        # Randomly pick a dimension
        rand_dim = torch.tensor(np.random.randint(self.dim, size=n))
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[torch.arange(n), rand_dim] = torch.round(x[torch.arange(n), rand_dim])
        return (self.xmax - self.xmin) * x + self.xmin