from transformers import AutoModelForSequenceClassification
from modeling.base_module import BaseModule


class TwitterRobertaModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        # overwrite model TODO clean
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'])

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("TODO")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("TODO")

    def test_step(self, batch, batch_idx):
        x = batch
        logits = self(x)[:, [0, 2]]  # ignore "neutral" class
        self.test_list.append(logits)

