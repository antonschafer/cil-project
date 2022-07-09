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

        # for hack to run val_final and val set and log correct name
        self._val_set_name = "val"
        self._save_preds = False

    def load_ckpt(self, path):
        model_dict = torch.load(path)['state_dict']
        model_dict = {k.replace('model.', ''): v for k,
                      v in model_dict.items() if 'model' in k}
        self.model.load_state_dict(model_dict)

    def preds_labels_loss(self, batch):
        """
        Returns prediction probabilities [0,1], labels {0,1}, and loss (not detached) for a batch.
        """
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self.preds_labels_loss(batch)
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", ((preds > 0.5) == labels).float().mean(), on_step=True,  # TODO check train acc correct
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self.preds_labels_loss(
            batch)
        return {
            "preds": preds.tolist(),
            "labels": labels.tolist(),
            "loss": loss.item()
        }

    def validation_epoch_end(self, outputs):
        def aggregate(metric):
            return np.array([x for xs in outputs for x in xs[metric]])

        # log with correct name (val_final set vs val set)
        def log_metric(metric, value):
            self.log("{}_{}".format(self._val_set_name, metric), value)

        preds = aggregate("preds")
        labels = aggregate("labels")
        loss = np.mean([x["loss"] for x in outputs])
        correct_preds = (preds > 0.5) == labels

        log_metric("loss", loss)
        log_metric("accuracy", correct_preds.mean())

        print("Classification Report {} set:".format(self._val_set_name))
        print(classification_report(labels, preds > 0.5))

        if self._save_validation_preds:
            np.save(os.path.join(wandb.run.dir,
                    "{}_preds.npy".format(self._val_set_name)), preds)
            np.save(os.path.join(wandb.run.dir,
                    "{}_correct_preds.npy".format(self._val_set_name)), correct_preds)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        preds = np.concatenate(outputs)
        outputs = np.where(preds > 0.5, 1, -1)
        ids = np.arange(1, outputs.shape[0]+1)
        outdf = pd.DataFrame({"Id": ids, 'Prediction': outputs})
        outdf.to_csv(os.path.join(wandb.run.dir, "output.csv"), index=False)

        if self._save_preds:
            np.save(os.path.join(wandb.run.dir, "test_preds.npy"), preds)

    def configure_optimizers(self):
        pass

    # TODO solve cleaner, this is a hack (In separate function to take care of setting val set name)
    def run_final_eval(self, *, trainer, ckpt_path, test_loader, val_loader=None, val_final_loader=None, save_preds=True):
        """
        Run evaluation checkpoint, save predictions
        """
        self._save_validation_preds = save_preds

        if val_loader is not None:
            self._val_set_name = "val"
            trainer.validate(self, val_loader, ckpt_path=ckpt_path)

        if val_final_loader is not None:
            self._val_set_name = "val_final"
            trainer.validate(self, val_final_loader, ckpt_path=ckpt_path)

        trainer.test(self, test_loader, ckpt_path=ckpt_path)

        # reset to default configs
        self._val_set_name = "val"
        self._save_validation_preds = False
