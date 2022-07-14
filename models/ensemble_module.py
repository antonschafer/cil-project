import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from models.base_module import BaseModule


class EnsembleModule(BaseModule):

    def __init__(self, in_dim, config):  # TODO put in_dim in config?
        super().__init__(config=config)
        self.in_dim = in_dim
        self.save_hyperparameters()
        self.config = config

        self.prediction_head = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_dim, config["hidden_size"]),
            nn.BatchNorm1d(config["hidden_size"]),
            nn.LeakyReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config['hidden_size'], 1),
        )

    def forward(self, x):
        return self.prediction_head(x)

    def preds_labels_loss(self, batch):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)
        return preds, y, loss

    def preds(self, batch):
        return torch.sigmoid(self(batch)).view(-1).cpu()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler_config = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=2, verbose=True,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return [optimizer], [lr_scheduler_config]
