import torch
from torch.utils.data import TensorDataset
from torch import nn
import pennylane as qml
import numpy as np


def get_circuit(n_qubits=4, n_depth=4, n_layers=1):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w):
        qml.StronglyEntanglingLayers(w[0], wires=range(n_qubits))
        
        for i in range(n_depth):
            qml.AngleEmbedding(features=inputs[i*n_qubits : (i+1)*n_qubits],
                            wires=range(n_qubits),
                            rotation="X")
            qml.StronglyEntanglingLayers(w[i+1], wires=range(n_qubits))
            
        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

    weight_shapes = {"w": (n_depth+1, n_layers, n_qubits, 3)}

    return circuit, weight_shapes


def draw_circuit(circuit, weight_shapes):
    in_dim = weight_shapes["w"][2] * (weight_shapes["w"][0] - 1)
    qml.draw_mpl(circuit, decimals=2, expansion_strategy='device')(inputs=torch.arange(in_dim), w=torch.rand(weight_shapes["w"]))


class QScalar(nn.Module):
    def __init__(self, in_dim, n_qubits=2, n_depth=1, n_layers=1):
        super().__init__()
        self.in_dim = in_dim

        self.n_qubits = n_qubits
        self.n_depth = n_depth
        self.n_layers = n_layers

        self.circuit, self.weight_shapes = get_circuit(self.n_qubits, self.n_depth, self.n_layers)

        self.net = nn.Sequential(
            qml.qnn.TorchLayer(self.circuit, self.weight_shapes), # outputs a tensor of shape (N, n_qubits)
            nn.Linear(self.n_qubits, 1)
        )
    
    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = x.repeat(1, self.n_qubits * self.n_depth // self.in_dim)
        return self.net(x)

class CScalar(nn.Module):
    def __init__(self, in_dim, hidden_dim=10, activation=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            self.activation(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, x):
        x = x.view(-1, self.in_dim)
        return self.net(x)
    
class QuantumNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=10, activation=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.qlayers = nn.ModuleList([QScalar(self.in_dim) for _ in range(self.out_dim)])
        self.clayers = nn.ModuleList([CScalar(self.in_dim, self.hidden_dim, self.activation) for _ in range(self.out_dim)])
        self.lin_out = nn.ModuleList([nn.Linear(3, 1) for _ in range(self.out_dim)])

    def forward(self, x):
        x = x.view(-1, self.in_dim)

        y = [None] * self.out_dim
        for i in range(self.out_dim):
            y[i] = self.lin_out[i](
                torch.hstack([
                    self.qlayers[i](x * 2 * torch.pi) * self.clayers[i](x),
                    self.qlayers[i](x * 2 * torch.pi),
                    self.clayers[i](x),
                ])
            )
        y = torch.hstack(y)

        return y

class ClassicNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16): # does not require activation function
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        self.clayers = nn.ModuleList([CScalar(self.in_dim, self.hidden_dim) for _ in range(self.out_dim)])
        
    def forward(self, x):
        x = x.view(-1, self.in_dim)
        
        y = [None] * self.out_dim
        for i in range(self.out_dim):
            y[i] = self.clayers[i](x)
        y = torch.hstack(y)
        
        return y