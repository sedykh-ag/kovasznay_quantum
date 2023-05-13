import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .model import Model

def scatter_plot(model: Model, components: dict = None, num_points: int = None, s: int = 7, **fig_kw):
    """Plots each component individually
    """
    if num_points == None:
        num_points = model.data.num_test

    x = model.data.domain.uniform_points(num_points)
    y = model.predict(x)

    x = x.numpy()
    y = y.numpy()

    if components == None:
        components = {0: "u", 1: "v", 2: "p"}

    fig, axs = plt.subplots(ncols=len(components), nrows=1, figsize=(16, 9), **fig_kw)
    for i in components:
        mappable = axs[i].scatter(x[:, 0], x[:, 1], c=y[:, i], s=s)
        axs[i].set_aspect("equal")
        fig.colorbar(mappable, ax=axs[i], location="bottom", label=components[i])
    
    # plt.tight_layout()
    # plt.legend()   
    plt.show()

    return x, y