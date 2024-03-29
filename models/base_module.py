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
        model_dict = torch.load(path)["state_dict"]
        model_dict = {
            k.replace("model.", ""): v for k, v in model_dict.items() if "model" in k
        }
        self.model.load_state_dict(model_dict)

    def preds_labels_loss(self, batch):
        """
        Returns prediction probabilities [0,1], labels {0,1}, and loss (not detached) for a batch.
        """
        raise NotImplementedError("Override this method in your model")

    def preds(self, batch):
        """
        Returns prediction probabilities [0,1]
        given a test batch w/o labels
        """
        raise NotImplementedError("Override this method in your model")

    def forward(self, x):
        raise NotImplementedError("Override this method in your model")

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self.preds_labels_loss(batch)
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_accuracy",
            ((preds > 0.5) == labels).float().mean(),
            on_step=True,  # TODO check train acc correct
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        preds, labels, loss = self.preds_labels_loss(batch)
        return {"preds": preds.tolist(), "labels": labels.tolist(), "loss": loss.item()}

    @staticmethod
    def aggregate_outputs(outputs):
        def aggregate(metric):
            return np.array([x for xs in outputs for x in xs[metric]])

        preds = aggregate("preds")
        labels = aggregate("labels")
        loss = np.mean([x["loss"] for x in outputs])
        return preds, labels, loss

    def validation_epoch_end(self, outputs):
        def process_val_output(val_outputs, val_idx):
            preds, labels, loss = self.aggregate_outputs(val_outputs)
            bin_preds = preds > 0.5
            acc = np.mean(bin_preds == labels)

            self.log("val_loss" + (("_" + str(val_idx)) if val_idx > 0 else ""), loss)
            self.log(
                "val_accuracy" + (("_" + str(val_idx)) if val_idx > 0 else ""), acc
            )

            print(
                "Validation Classification Report for validation loader"
                + (("_" + str(val_idx)) if val_idx > 0 else "")
                + ":"
            )
            print(classification_report(labels, bin_preds, zero_division=0, digits=4))

        if isinstance(outputs[0], dict):
            process_val_output(outputs, 0)
        else:
            # For multiple validation loaders
            for output_idx, output in enumerate(outputs):
                process_val_output(output, output_idx)

    def predict_step(self, batch, batch_idx):
        preds, labels, loss = self.preds_labels_loss(batch)
        return {"preds": preds.tolist(), "labels": labels.tolist(), "loss": loss.item()}

    def test_step(self, batch, batch_idx):
        return self.preds(batch)

    def test_epoch_end(self, outputs):
        preds = np.concatenate(outputs)
        outputs = np.where(preds > 0.5, 1, -1)
        ids = np.arange(1, outputs.shape[0] + 1)
        outdf = pd.DataFrame({"Id": ids, "Prediction": outputs})
        outdf.to_csv(os.path.join(wandb.run.dir, "output.csv"), index=False)
        np.save(os.path.join(wandb.run.dir, "test_preds.npy"), preds)

    def configure_optimizers(self):
        raise NotImplementedError("Override this method in your model")
