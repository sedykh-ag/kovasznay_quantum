from typing import List, Callable

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


def compare_plot(models: List[Model], solution_fn: List[Callable], component: int = 0, num_points: int = None, s: int = 7, **fig_kw):
    """Creates 3 plots of one component, comparing each of models in models list.
    """
    n_models = len(models)
    if num_points == None:
        num_points = models[0].data.num_test

    x = models[0].data.domain.uniform_points(num_points)
    ys = [models[i].predict(x)[:, component] for i in range(n_models)]

    x = x.cpu().numpy()
    ys = [y.cpu().numpy() for y in ys]

    # define plot
    fig, axs = plt.subplots(ncols=n_models+1, nrows=1, figsize=(10, 5), **fig_kw)

    # plot models
    for i in range(n_models):
        mappable = axs[i].scatter(x[:, 0], x[:, 1], c=ys[i], s=s, cmap="seismic", vmin=-3.5, vmax=3.5)
        # fig.colorbar(mappable, ax=axs[i], location="bottom", label=f"c {component} m {i}")

    # plot exact solution
    y_exact = solution_fn[component](torch.from_numpy(x).float()).cpu().numpy()
    mappable_exact = axs[n_models].scatter(x[:, 0], x[:, 1], c=y_exact, s=s, cmap="seismic", vmin=-3.5, vmax=3.5)
    # fig.colorbar(mappable_exact, ax=axs[n_models], location="bottom", label=f"c {component} exact")

    

    # aux
    for ax in axs:
        ax.set_aspect("equal")
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
    # titles
    axs.flat[0].set_title("quantum")
    axs.flat[1].set_title("classical")
    axs.flat[2].set_title("exact")
    
    # show plot
    plt.tight_layout()

    # add shared colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mappable_exact, cax=cbar_ax, shrink=0.5, aspect=30)
    
    # plt.savefig("compare_plot.pdf")
    plt.show()

    return x, [y_exact, ys]