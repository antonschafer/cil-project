from transformers import AutoModelForSequenceClassification
from models.base_module import BaseModule
import numpy as np
import torch
import torch.optim as optim


class ThreeClassHFModule(BaseModule):
    """
    Used for models that are pretrained to predict three classes: negative, neutral, positive
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'])

    @staticmethod
    def fix_labels(y):
        """
        Transform labels such that they are compatible with the model
        (introduce neutral labels as zeros)  # TODO move to dataset for efficiency
        """
        return torch.cat([y[:, 0].unsqueeze(1), torch.zeros(
            y.shape[0], 1, device=y.device), y[:, 1].unsqueeze(1)], dim=1)

    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits[:, [0, 2]]  # ignore "neutral" class

    def preds_labels_loss(self, batch):
        x, y = batch
        y = self.fix_labels(y)
        output = self.model(x, labels=y)
        return output.logits[:, [0, 2]].argmax(axis=1), y[:, 2], output.loss

    def test_step(self, batch, batch_idx):
        return torch.argmax(self(batch), axis=1).cpu()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1, 2], gamma=0.1)
        return [optimizer], [lr_scheduler]


""" For rerieving stats about neutral class

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = self.fix_labels(y)

        output = self.model(x, labels=y)

        return {
            "raw_preds": output.logits.argmax(axis=1).tolist(),
            # ignore "neutral" class
            "preds": output.logits[:, [0, 2]].argmax(axis=1).tolist(),
            "labels": y[:, 2].tolist(),
            "loss": output.loss.item()
        }

    def validation_epoch_end(self, outputs):
        def aggregate(metric):
            return [x for xs in outputs for x in xs[metric]]

        preds = aggregate("preds")
        labels = aggregate("labels")
        loss = np.mean([x["loss"] for x in outputs])

        self.log("val_loss", loss)
        self.log("val_accuracy", np.mean(np.array(preds) == np.array(labels)))

        print("Validation Classification Report:")
        print(classification_report(labels, preds))

        raw_preds = aggregate("raw_preds")
        c_preds = Counter(raw_preds)
        print("Stats with neutral class:")
        print("\tPrediction counts:", list(c_preds.items()))
        c_pred_label = Counter(list(zip(raw_preds, labels)))
        print("\tPrediction-Label pairs:", list(c_pred_label.items()))
        print()

"""
