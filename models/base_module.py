import os
import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
from sklearn.metrics import classification_report
import wandb


class BaseModule(pl.LightningModule):
    """
    BaseModule for task, superclass for all models that automates logging etc

    To implement: init, preds_labels_loss, forward, test_step, configure_optimizers
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = None

    def load_ckpt(self, path):
        model_dict = torch.load(path)['state_dict']
        model_dict = {k.replace('model.', ''): v for k,
                      v in model_dict.items() if 'model' in k}
        self.model.load_state_dict(model_dict)

    def preds_labels_loss(self, batch):
        """
        Returns predictions (0/1), labels (0/1), and loss (not detached) for a batch.
        """
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self.preds_labels_loss(batch)
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", (preds == labels).float().mean(), on_step=True,  # TODO check train acc correct
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self.preds_labels_loss(batch)
        return {
            "preds": preds.tolist(),
            "labels": labels.tolist(),
            "loss": loss.item()
        }

    def validation_epoch_end(self, outputs):
        def aggregate(metric):
            return [x for xs in outputs for x in xs[metric]]

        preds = aggregate("preds")
        labels = aggregate("labels")
        loss = np.mean([x["loss"] for x in outputs])

        self.log("val_loss", loss)
        self.log("val_accuracy", np.mean(preds == labels))

        print("Validation Classification Report:")
        print(classification_report(labels, preds))

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        if preds.device.type == "cuda":  # TODO remove
            print("\n\nPREDS ON CUDA"*10)
        preds = torch.vstack(outputs).cpu().numpy()
        outputs = np.where(preds == 1, 1, -1)
        ids = np.arange(1, outputs.shape[0]+1)
        outdf = pd.DataFrame({"Id": ids, 'Prediction': preds})
        outdf.to_csv(os.path.join(wandb.run.dir, 'output.csv'), index=False)

    def configure_optimizers(self):
        pass
