import os

import torch

from . import data
from . import gradients as grad
from .utils import timeit

class Model:
    def __init__(
            self,
            data: data.PDEData,
            model: torch.nn.Module,
            save_path: str,
            save_every: int,
    ):
        self.data = data
        self.model = model
        self.save_path = save_path
        self.save_every = save_every

        self.epochs_run = 0
        self._compiled = False

        os.makedirs(self.save_path, exist_ok=True)

    def save_snapshoot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch
        }
        torch.save(snapshot, os.path.join(self.save_path, "ckpt.pt"))
        print(f"Saved checkpoint at {self.save_path}")

    def load_snapshot(self, snapshot_path=None):
        if snapshot_path == None:
            snapshot_path = os.path.join(self.save_path, "ckpt.pt")
        snapshot = torch.load(snapshot_path)
        
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Loaded snapshot at epoch {self.epochs_run}")

    def compile(self, optimizer="adam", lr=0.001, scheduler=None):
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            raise NotImplementedError("L-BFGS is not implemented yet!")
        else:
            raise ValueError("Unknown optimizer!")
        
        if os.path.exists(os.path.join(self.save_path, "ckpt.pt")):
            self.load_snapshot()

        self._compiled = True
        
        # TODO: Distributed training goes here (probably)

    @timeit
    def predict(self, x):
        with torch.no_grad():
            return self.model(x)

    def _run_epoch(self, epoch):
        self.optimizer.zero_grad()
        x_pde, x_bc = [x.requires_grad_() for x in self.data.get_points_train()]
        pred_pde = self.model(x_pde)
        pred_bc = self.model(x_bc)
        loss = self.data.loss(x_pde, x_bc, pred_pde, pred_bc)
        loss.backward()
        self.optimizer.step()
        
        return loss

    def _run_test_epoch(self, epoch):
        with torch.no_grad():
            X = self.data.get_points_test()
            pred = self.model(X)
            # calculate RMSE against exact solution
            return self.data.test(pred)
    
    @timeit
    def train(self, epochs, resample=False):
        if not self._compiled:
            raise Exception("Compile before training!")
        
        print("Started training...")
        for epoch in range(self.epochs_run+1, epochs+1):
            loss = self._run_epoch(epoch)
            metric = self._run_test_epoch(epoch)
            print(f"epoch: {epoch}, loss: {loss.item()}, metric: {metric}")

            grad.clear() # clear cached gradients

            if epoch % self.save_every == 0:
                self.save_snapshoot(epoch)
                
        print("Finished training")