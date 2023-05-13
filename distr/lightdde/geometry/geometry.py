import itertools

import numpy as np

from .sampler import sample
from .. import config


class Hypercube():
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        
        self.dim = len(xmin)
        self.xmin = np.array(xmin, dtype=config.real(np))
        self.xmax = np.array(xmax, dtype=config.real(np))
        if np.any(self.xmin >= self.xmax):
            raise ValueError("xmin >= xmax")

        self.side_length = self.xmax - self.xmin
        self.volume = np.prod(self.side_length)

    def inside(self, x):
        return np.logical_and(
            np.all(x >= self.xmin, axis=-1), np.all(x <= self.xmax, axis=-1)
        )
    
    def on_boundary(self, x):
        _on_boundary = np.logical_or(
            np.any(np.isclose(x, self.xmin), axis=-1),
            np.any(np.isclose(x, self.xmax), axis=-1),
        )
        return np.logical_and(self.inside(x), _on_boundary)

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    np.linspace(
                        self.xmin[i], self.xmax[i], num=ni, dtype=config.real(np)
                    )
                )
            else:
                xi.append(
                    np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype=config.real(np),
                    )[1:]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        return self.xmin + (self.xmax - self.xmin) * x

    def random_boundary_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        # Randomly pick a dimension
        rand_dim = np.random.randint(self.dim, size=n)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
        return (self.xmax - self.xmin) * x + self.xmin