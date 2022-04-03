import pytorch_lightning as pl
from torch.nn import functional as F
from torch import optim
from transformers import AutoModelForSequenceClassification
import torch

class BaseModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2,
                                                                        ignore_mismatched_sizes=True)

    def load_ckpt(self,path):
        self.model.load_state_dict(torch.load(path)['state_dict'])
    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y.long())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y.long())
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]
