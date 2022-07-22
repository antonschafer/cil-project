from torch import optim
from transformers import AutoModelForSequenceClassification
import torch

from models.base_module import BaseModule


class BinaryHFModule(BaseModule):

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2,revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                                        ignore_mismatched_sizes=True,output_hidden_states= config.get("output_hidden_states",False))
        if "gpt" in config['model_name'].lower():
            self.model.config.pad_token_id = self.model.config.eos_token_id


    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits

    def preds_labels_loss(self, batch):
        x, y = batch
        output = self.model(x, labels=y)
        return output.logits.softmax(axis=1)[:, 1], y[:, 1], output.loss

    def preds(self, batch):
        return torch.softmax(self(batch), axis=1)[:, 1].cpu()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1, 2], gamma=0.1)
        return [optimizer], [lr_scheduler]
