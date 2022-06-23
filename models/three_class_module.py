from transformers import AutoModelForSequenceClassification
from models.base_module import BaseModule
import numpy as np
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
        raise NotImplementedError("TODO")

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.model(x)

        return {
            "raw_preds": output.logits.argmax(axis=1).tolist(),
            # ignore "neutral" class
            "preds": output.logits[:, [0, 2]].argmax(axis=1).tolist(),
            "labels": y[:, 1].tolist(),
            # don't have labels for three classes model was trained for
            "loss": float("NaN")
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

    def test_step(self, batch, batch_idx):
        x = batch
        logits = self(x)[:, [0, 2]]  # ignore "neutral" class
        self.test_list.append(logits)
