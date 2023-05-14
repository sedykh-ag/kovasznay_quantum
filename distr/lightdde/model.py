import os
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd

from . import data
from . import gradients as grad
from .utils import timeit

        
class Model:
    def __init__(
            self,
            data: data.PDEData,
            model: torch.nn.Module,
            save_path: str = None,
            save_every: int = None,
            log_every: int = None,
    ):
        self.distributed = dist.is_initialized()
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else: # backward compatibility for print statements in non-distributed training
            self.rank = 0
            self.world_size = 1

        self.data = data
        self.model = model
        self.save_path = save_path
        
        if self.save_path == None:
            timestamp = datetime.today().strftime("%d-%m-%Y--%H-%M-%S")
            self.save_path = os.path.join("models", timestamp)
    
        self.save_every = save_every
        self.log_every = log_every

        self.epochs_run = 0
        self._compiled = False

    def log(self, epoch: int, loss_train: float, loss_test: list[float]):
        print(f"epoch: {epoch}, loss_train: {loss_train}, loss_test: {loss_test}")

        data = {
            "loss_train": loss_train,
            "u_err": loss_test[0],
            "v_err": loss_test[1],
            "p_err": loss_test[2],
        }

        PATH = os.path.join(self.save_path, "log.csv")

        pd.DataFrame(data, index=[epoch]).to_csv(PATH, mode="a", header=False)
        

    def save_snapshoot(self, epoch):
        assert self.rank == 0, "Tried to save snapshot from non-zero rank!"
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, os.path.join(self.save_path, "ckpt.pt"))
        print(f"Saved checkpoint at {self.save_path}")

    def load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.rank == 0:
            print(f"Loaded snapshot at epoch {self.epochs_run}")

    def compile(self, optimizer="adam", lr=0.001): 
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            raise NotImplementedError("L-BFGS is not implemented yet!")
        else:
            raise ValueError("Unknown optimizer!")

        # load last checkpoint (if exists)
        if os.path.exists(os.path.join(self.save_path, "ckpt.pt")):
            self.load_snapshot(snapshot_path=os.path.join(self.save_path, "ckpt.pt"))
            
        if self.distributed:
            self.model = DDP(self.model)
        
        # make parent directory (if not exists)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # make log file with headers
        if not os.path.exists(os.path.join(self.save_path, "log.csv")):
            header = pd.DataFrame(columns=["epoch", "loss_train", "u_err", "v_err", "p_err"]).set_index("epoch")
            header.to_csv(os.path.join(self.save_path, "log.csv"), header=True)

        self._compiled = True

    @timeit
    def predict(self, x):
        with torch.no_grad():
            return self.model(x)

    def _run_train_epoch(self, epoch):
        self.optimizer.zero_grad()
        x_pde, x_bc = [x.requires_grad_() for x in self.data.get_points_train()]
        pred_pde = self.model(x_pde)
        pred_bc = self.model(x_bc)
        loss = self.data.loss(x_pde, x_bc, pred_pde, pred_bc)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _run_test_epoch(self, epoch):
        with torch.no_grad():
            X = self.data.get_points_test()
            pred = self.model(X)
            # calculate RMSE against exact solution
            return self.data.test(pred)
    
    @timeit
    def train(self, epochs):
        if not self._compiled:
            raise Exception("Compile before training!")
        
        if self.save_every == None:
            self.save_every = int(0.1 * epochs) if int(0.1 * epochs) > 1 else 1
        if self.log_every == None:
            self.log_every = int(0.01 * epochs) if int(0.01 * epochs) > 1 else 1
        
        if self.rank == 0:
            print(f"Started training {self.save_path} ...")

        for epoch in range(self.epochs_run+1, epochs+1):
            loss_train = self._run_train_epoch(epoch)

            if self.rank == 0 and epoch % self.log_every == 0:
                loss_test = self._run_test_epoch(epoch)
                self.log(epoch, loss_train, loss_test)

            if self.rank == 0 and epoch % self.save_every == 0:
                self.save_snapshoot(epoch)
            
            grad.clear() # clear cached gradients
                
        if self.rank == 0:
            print("Finished training")