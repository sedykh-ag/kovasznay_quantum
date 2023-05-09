import numpy as np
import torch

class Real:
    def __init__(self, precision):
        self.precision = precision
        self.reals = None

        if precision == 16:
            self.reals = {np: np.float16, torch: torch.float16}
        if precision == 32:
            self.reals = {np: np.float32, torch: torch.float32}
        if precision == 64:
            self.reals = {np: np.float64, torch: torch.float64}

    def __call__(self, package):
        return self.reals[package]
