import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from models.base_module import BaseModule


class EnsembleModule(BaseModule):

    def __init__(self, submodels, in_dim, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.submodels = submodels  # no module list s.t. not registered as submodules
        for m in self.submodels:
            m.eval()
            m.to(self.device)

        self.prediction_head = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_dim, config["hidden_size"]),
            nn.BatchNorm1d(),
            nn.LeakyReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config['hidden_size'], 1),
        )

    def forward(self, x):
        with torch.no_grad():
            sub_output_list = [sub(x_sub)
                               for sub, x_sub in zip(self.submodels, x)]
            sub_outputs = torch.stack(sub_output_list, dim=1)

        return self.prediction_head(sub_outputs)

    def preds_labels_loss(self, batch):
        x, y_bin = batch
        y = y_bin[:, 1]
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.where(logits > 0, 1, 0)
        return loss, preds, y

    def test_step(self, batch, batch_idx):
        return (self(batch) > 0).cpu()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler_config = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=2, verbose=True,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
