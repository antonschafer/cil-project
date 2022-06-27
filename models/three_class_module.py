from transformers import AutoModelForSequenceClassification
from models.base_module import BaseModule
import numpy as np
import torch
from sklearn.metrics import classification_report
from collections import Counter


class ThreeClassModule(BaseModule):
    """
    Used for models that are pretrained to predict three classes: negative, neutral, positive
    """

    def __init__(self, config):
        super().__init__(config)
        # overwrite model TODO clean
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'])

    def training_step(self, batch, batch_idx):
        x, y = batch

        # transform labels such that they are compatible with the model
        #     (introduce neutral labels as zeros)
        # TODO move to dataset for efficiency
        y = torch.cat([y[:, 0].unsqueeze(1), torch.zeros(
            y.shape[0], 1, device=y.device), y[:, 1].unsqueeze(1)], dim=1)

        output = self.model(x, labels=y)  # TODO fix labels

        #accuracy = (output.logits.argmax(axis=0) == y).mean()
        self.log("train_loss", output.loss.item(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return output.loss

    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits[:, [0, 2]]  # ignore "neutral" class

    def validation_step(self, batch, batch_idx):
        x, y, texts = batch

        output = self.model(x)

        return {
            "raw_preds": output.logits.argmax(axis=1).tolist(),
            # ignore "neutral" class
            "preds": output.logits[:, [0, 2]].argmax(axis=1).tolist(),
            "labels": y[:, 1].tolist(),
            # don't have labels for three classes model was trained for
            "loss": float("NaN"),


            "texts": texts
        }

    def validation_epoch_end(self, outputs):
        def aggregate(metric):
            return np.array([x for xs in outputs for x in xs[metric]])

        preds = aggregate("preds")
        labels = aggregate("labels")
        loss = np.mean([x["loss"] for x in outputs])

        self.log("val_loss", loss)
        self.log("val_accuracy", np.mean(preds == labels))

        print("Validation Classification Report:")
        print(classification_report(labels, preds))

        raw_preds = aggregate("raw_preds")
        c_preds = Counter(raw_preds)
        print("Stats with neutral class:")
        print("\tPrediction counts:", list(c_preds.items()))
        c_pred_label = Counter(list(zip(raw_preds, labels)))
        print("\tPrediction-Label pairs:", list(c_pred_label.items()))
        print()

        # TODO
        false_positives = (raw_preds == 2) & (labels == 0)
        false_negatives = (raw_preds == 0) & (labels == 1)
        texts = aggregate("texts")
        print("\nFalse positives:")
        print("".join(texts[false_positives][:20]))
        print("\nFalse negatives:")
        print("".join(texts[false_negatives][:20]))
