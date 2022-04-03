from statistics import mode
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import optim
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd

class BaseModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2,
                                                                        ignore_mismatched_sizes=True)
        self.test_list = []

    def load_ckpt(self,path):
        model_dict = torch.load(path)['state_dict']
        model_dict = {k.replace('model.',''):v for k,v in model_dict.items() if 'model' in k}
        self.model.load_state_dict(model_dict)
        
    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        loss = self.model(x,labels=y).loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss= self.model(x,labels=y).loss
        self.log("val_loss", loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        self.test_list.append(logits)

    def test_epoch_end(self, outputs):
        print(self.test_list)
        test_outputs = torch.vstack(self.test_list).cpu().numpy()
        test_outputs = test_outputs.argmax(axis=1)
        test_outputs[test_outputs == 0] = -1
        outdf = pd.DataFrame({'Prediction':test_outputs})
        outdf.to_csv('output.csv')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]
