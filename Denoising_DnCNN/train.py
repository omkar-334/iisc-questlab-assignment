import datetime
import os
import time

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class SumSquaredError(_Loss):
    """Sum Squared Error Loss
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, reduction="sum"):
        super().__init__(reduction=reduction)

    def forward(self, inputs, target):
        return torch.nn.functional.mse_loss(inputs, target, reduction="sum").div_(2)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, save_dir="models", batch_size=4, lr=2e-3, n_epoch=180, patience=10, wandb_project="dncnn-training", wandb_entity=None):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.lr = lr
        self.n_epoch = n_epoch
        self.patience = patience
        self.cuda = torch.cuda.is_available()
        # self.criterion = torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction="sum").div_(2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=self.patience, verbose=True)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.cuda:
            self.model = self.model.cuda()

        if wandb.run is None:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "epochs": self.n_epoch,
                    "patience": self.patience,
                },
            )

    def log(self, epoch, train_loss, val_loss, elapsed_time, wandb_log=True):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:")

        message = f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {elapsed_time:.2f}s"
        print(timestamp, message)

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": elapsed_time,
            })
            np.savetxt(
                os.path.join(self.save_dir, "train_result.txt"),
                np.array([epoch, train_loss, val_loss, elapsed_time]),
                fmt="%2.4f",
            )

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                if self.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                output = self.model(batch_y)
                loss = self.criterion(output, batch_x)
                val_loss += loss.item()
        return val_loss / len(self.test_loader)

    def train_epoch(self):
        epoch_loss = 0
        start_time = time.time()
        self.model.train()

        for _, (batch_x, batch_y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if self.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            output = self.model(batch_y)
            loss = self.criterion(output, batch_x)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        elapsed_time = time.time() - start_time
        epoch_loss /= len(self.train_loader)
        return epoch_loss, elapsed_time

    def train(self):
        for epoch in range(self.n_epoch):
            train_loss, elapsed_time = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)

            self.log(epoch + 1, train_loss, val_loss, elapsed_time)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch + 1}.pth"))

        wandb.finish()
