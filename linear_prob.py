import torch
import math
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F

from util.utils_eval import get_metrics
from util.loadEEG import get_data, get_IMWUTdata

from model.SensorTransformer import SensorTransformerEncoder
from model.module import Conv1dWithConstraint, LinearWithConstraint
from model.common import seed_torch, create_1d_absolute_sin_cos_embedding
seed_torch(7)

class LitSensorPT(pl.LightningModule):
    
    def __init__(self):
        super().__init__()    
        
        #use_channels_names = [      'FP1', 'FPZ', 'FP2', 
        #                       'AF3', 'AF4', 
        #    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        #'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
        #    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        #'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        #     'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
        #              'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
        #                       'O1', 'OZ', 'O2', ]
        use_channels_names = ['S1_D1 hbo', 'S1_D1 hbr', 'S2_D1 hbo', 'S2_D1 hbr', 'S3_D1 hbo', 'S3_D1 hbr',
                     'S1_D2 hbo', 'S1_D2 hbr', 'S3_D2 hbo', 'S3_D2 hbr', 'S4_D2 hbo', 'S4_D2 hbr', 'S2_D3 hbo',
                     'S2_D3 hbr', 'S3_D3 hbo', 'S3_D3 hbr', 'S5_D3 hbo', 'S5_D3 hbr', 'S3_D4 hbo', 'S3_D4 hbr',
                     'S4_D4 hbo', 'S4_D4 hbr', 'S5_D4 hbo', 'S5_D4 hbr', 'S6_D5 hbo', 'S6_D5 hbr', 'S7_D5 hbo',
                     'S7_D5 hbr', 'S8_D5 hbo', 'S8_D5 hbr', 'S6_D6 hbo', 'S6_D6 hbr', 'S8_D6 hbo', 'S8_D6 hbr',
                     'S9_D6 hbo', 'S9_D6 hbr', 'S7_D7 hbo', 'S7_D7 hbr', 'S8_D7 hbo', 'S8_D7 hbr', 'S10_D7 hbo',
                     'S10_D7 hbr', 'S8_D8 hbo', 'S8_D8 hbr', 'S9_D8 hbo', 'S9_D8 hbr', 'S10_D8 hbo', 'S10_D8 hbr']
        self.chans_num = len(use_channels_names)
        self.num_class = 2

        # init model
        target_encoder = SensorTransformerEncoder(
            img_size=[len(use_channels_names), 256*2],
            patch_size=64,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        #load_path="./logs/sensor_large_1.ckpt"
        load_path="./logs/sensorPT_nemo_tb/version_1/checkpoints/epoch=199-step=5600.ckpt"
        pretrain_ckpt = torch.load(load_path, weights_only=False, map_location=torch.device("cpu"))
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
        
                
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv       = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1   =   LinearWithConstraint(2048, 8, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(8*8, self.num_class, max_norm=0.25)
        
        self.drop           = torch.nn.Dropout(p=0.2)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
    
    def forward(self, x):
        # print(x.shape) # B, C, T
        #B, C, T = x.shape
        #x = x/10
        x = self.chan_conv(x)
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        
        h = self.linear_probe1(self.drop(h))
        
        h = h.flatten(1)
        
        h = self.linear_probe2(h)
        
        return x, h
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        

if __name__=="__main__":
    # load data
    #data_path = "data/BCIC_2b_0_38HZ/"
    #data_path = "data/NEMO_exp/"
    data_path = "data/IMWUT_exp/"
    acc = []
    for i in reversed(range(1,73)):
        all_subjects = [i]
        all_datas = []
        #train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1, is_few_EA = False, target_sample=256*2)
        train_dataset,valid_dataset,test_dataset = get_IMWUTdata(i,data_path,1, is_few_EA = False, target_sample=256*2)
        global max_epochs
        global steps_per_epoch
        global max_lr
        print(train_dataset.y)
        print(valid_dataset.y)
        batch_size=64
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        
        max_epochs = 20
        steps_per_epoch = math.ceil(len(train_loader) )
        max_lr = 4e-4
    
        # init model
        model = LitSensorPT()
    
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks = [lr_monitor]
        
        trainer = pl.Trainer(accelerator='cuda',
                             precision=16,
                             max_epochs=max_epochs, 
                             callbacks=callbacks,
                             enable_checkpointing=True,
                             logger=[pl_loggers.TensorBoardLogger('./logs/', name="IMWUTtest_tb", version=f"subject{i}"), 
                                     pl_loggers.CSVLogger('./logs/', name="IMWUTtest_csv")])
                             #logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_BCIC2B_tb", version=f"subject{i}"), 
                             #        pl_loggers.CSVLogger('./logs/', name="EEGPT_BCIC2B_csv")])
    
        trainer.fit(model, train_loader, valid_loader, ckpt_path='last')

        # predict
        _, logit = model(test_dataset.x)
        print('Y hat',torch.argmax(logit,  dim=-1))
        # accuracy
        y = test_dataset.y
        print('Y',y)
        label = y.long()
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        
        #print('Y:', test_dataset.y)
        print('accuracy',accuracy)
        acc.append(accuracy)
        print('AVERAGE accuracy',np.mean(np.array(acc)))

        