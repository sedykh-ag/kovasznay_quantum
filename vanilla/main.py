import torch

import lightdde as dde


Re = 20
nu = 1 / Re
l = 1 / (2 * nu) - (1 / (4 * nu ** 2) + 4 * torch.pi ** 2)**0.5

def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]

def u_func(x):
    return 1 - torch.exp(l * x[:, 0:1]) * torch.cos(2 * torch.pi * x[:, 1:2])

def v_func(x):
    return l / (2 * torch.pi) * torch.exp(l * x[:, 0:1]) * torch.sin(2 * torch.pi * x[:, 1:2])

def p_func(x):
    return 1 / 2 * (1 - torch.exp(2 * l * x[:, 0:1]))


""" DEFINE GEOMETRY """
domain = dde.geometry.Hypercube(xmin=[-0.5, -0.5], xmax=[1., 1.5])


""" DEFINE BC AND DDE DATA """
boundary_condition_u = dde.bc.DirichletBC(domain,
                                          u_func,
                                          lambda _, on_boundary: on_boundary,
                                          component=0)

boundary_condition_v = dde.bc.DirichletBC(domain,
                                          v_func,
                                          lambda _, on_boundary: on_boundary,
                                          component=1)

def on_boundary_p(x, on_boundary):
    return on_boundary and torch.isclose(x[0], torch.tensor(1.0))

boundary_condition_right_p = dde.bc.DirichletBC(domain,
                                                p_func,
                                                on_boundary_p,
                                                component=2)

data = dde.data.PDEData(
    domain=domain,
    pde=pde,
    bcs=[boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
    exact_solution=[u_func, v_func, p_func],
    num_domain=2601,
    num_bc=400,
    num_test=10000,
)


""" DEFINE NEURAL NETWORK """
class FNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.Tanh()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 50),
            self.activation,
            torch.nn.Linear(50, 50),
            self.activation,
            torch.nn.Linear(50, 50),
            self.activation,
            torch.nn.Linear(50, 50),
            self.activation,
            torch.nn.Linear(50, 3),
        )

    def forward(self, x):
        x = x.view(-1, 2)
        return self.net(x)