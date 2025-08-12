import torch
import math
from torch import nn
import pytorch_lightning as pl
import sys
import os
import pandas as pd

from functools import partial
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
import random

from util.utils_eval import get_metrics
from util.loadSensor import get_data, get_Mydata, temporal_interpolation, get_leaveOneDatasetOut

from model.SensorTransformer import SensorTransformerEncoder
from model.module import Conv1dWithConstraint, LinearWithConstraint
from model.common import seed_torch, create_1d_absolute_sin_cos_embedding
seed_torch(7)

class LitSensorPT(pl.LightningModule):
    
    def __init__(self, ckptPATH):
        super().__init__()    

        #use_channels_names = ['S1_D1 hbo', 'S1_D1 hbr', 'S2_D1 hbo', 'S2_D1 hbr', 'S3_D1 hbo', 'S3_D1 hbr',
        #             'S1_D2 hbo', 'S1_D2 hbr', 'S3_D2 hbo', 'S3_D2 hbr', 'S4_D2 hbo', 'S4_D2 hbr', 'S2_D3 hbo',
        #             'S2_D3 hbr', 'S3_D3 hbo', 'S3_D3 hbr', 'S5_D3 hbo', 'S5_D3 hbr', 'S3_D4 hbo', 'S3_D4 hbr',
        #             'S4_D4 hbo', 'S4_D4 hbr', 'S5_D4 hbo', 'S5_D4 hbr', 'S6_D5 hbo', 'S6_D5 hbr', 'S7_D5 hbo',
        #             'S7_D5 hbr', 'S8_D5 hbo', 'S8_D5 hbr', 'S6_D6 hbo', 'S6_D6 hbr', 'S8_D6 hbo', 'S8_D6 hbr',
        #             'S9_D6 hbo', 'S9_D6 hbr', 'S7_D7 hbo', 'S7_D7 hbr', 'S8_D7 hbo', 'S8_D7 hbr', 'S10_D7 hbo',
        #             'S10_D7 hbr', 'S8_D8 hbo', 'S8_D8 hbr', 'S9_D8 hbo', 'S9_D8 hbr', 'S10_D8 hbo', 'S10_D8 hbr']
        #use_channels_names = ['S2_D1 hbo', 'S3_D1 hbo', 'S1_D2 hbo', 'S3_D2 hbo',
        #             'S4_D2 hbo', 'S3_D3 hbo', 'S5_D3 hbo', 'S3_D4 hbo', 'S4_D4 hbo', 'S5_D4 hbo', 'S6_D5 hbo', 'S8_D5 hbo', 'S6_D6 hbo',
        #             'S8_D6 hbo', 'S9_D6 hbo', 'S8_D7 hbo', 'S7_D7 hbo', 'S8_D8 hbo', 'S9_D8 hbo', 'S10_D8 hbo', 'S2_D1 hbr', 'S3_D1 hbr',
        #             'S1_D2 hbr', 'S3_D2 hbr', 'S4_D2 hbr', 'S3_D3 hbr', 'S5_D3 hbr', 'S3_D4 hbr', 'S4_D4 hbr', 'S5_D4 hbr', 'S6_D5 hbr',
        #             'S8_D5 hbr', 'S6_D6 hbr', 'S8_D6 hbr', 'S9_D6 hbr', 'S8_D7 hbr', 'S7_D7 hbr', 'S8_D8 hbr', 'S9_D8 hbr', 'S10_D8 hbr']
        use_channels_names = ['ACC0','ACC1','ACC2','BVP','HR','EDA','TEMP']
        #use_channels_names = ['EDA','HR','TEMP','EMG','Resp','EDA1','HR1','TEMP1','EMG1','Resp1',
        #                        'EDA2','HR2','TEMP2','EMG2','Resp2']
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
        load_path=ckptPATH
        pretrain_ckpt = torch.load(load_path, weights_only=False, map_location=torch.device("cuda"))
        
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
        #x = temporal_interpolation(x, 256*2)
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
        #print(label.shape, y_score.shape)
        
        #metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        #results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
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

#        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
#        return loss

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

from sklearn.metrics import precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import itertools
if __name__=="__main__":
    ACCURACY = []
    PRECISION = []
    AUC = []
    SPECIFICITY = []
    KAPPA = []
    
    LOOs = ['LOO1','LOO2','LOO3','LOO4','LOO5','LOO6','LOO7','LOO8','LOO9','LOO10','LOO11','LOO12']
    LOO_lr = []
    for l,LOO in enumerate(LOOs):
        for root, dirs, files in os.walk(f"./logs/sensorPT_empatica_{LOO}_finetune_tb"):
            if 'checkpoints' in root:
                ckpt = root+'/'+files[0]
        print('------CHECKPOINT:', ckpt)
    
        test_path = f"/scratch/project_2014260/data/LOO_finetune/{LOO}/test"
        train_path = f"/scratch/project_2014260/data/LOO_finetune/{LOO}/finetune"
        train_dataset, valid_dataset, test_dataset = get_leaveOneDatasetOut(test_path, train_path, 0, target_sample=256*2)
        print('train size', len(train_dataset))
        global max_epochs
        global steps_per_epoch
        global max_lr
    
        batch_size = 64
        max_epochs = 200
        max_lr = 5e-4
            
        accs = []
        precisions = []
        aucs = []
        specificities = []
        kappas = []
        lrs = list(itertools.product([1e-3,1e-4,2e-4,3e-4,4e-4,5e-4,5e-5,4e-5], [0.3,0.2]))
        
        
        for lr in lrs:
            print('learning rate', lr)
            max_lr = lr[0]
        
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            steps_per_epoch = math.ceil(len(train_loader))
        
            # init model
            model = LitSensorPT(ckptPATH=ckpt)
            model.drop = torch.nn.Dropout(p=lr[1])
        
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            callbacks = [lr_monitor]
        
            trainer = pl.Trainer(
                accelerator='cuda',
                precision='16-mixed',
                max_epochs=max_epochs, 
                callbacks=callbacks,
                enable_checkpointing=False,
                logger=[
                    pl_loggers.TensorBoardLogger('./logs/', name="test_tb", version=f"subject{LOO}"), 
                    pl_loggers.CSVLogger('./logs/', name="test_csv")
                ]
            )
        
            trainer.fit(model, train_loader, valid_loader, ckpt_path='last')
    
            # predict
            _, logit = model(test_dataset.x)
            y_true = test_dataset.y.cpu().numpy()   # assuming tensor, change if not
            y_pred = torch.argmax(logit, dim=-1).cpu().numpy()
            y_prob = torch.softmax(logit, dim=-1)[:,1].detach().numpy()  # probability for class 1
    
            # ACCURACY
            accuracy = (y_pred == y_true).mean()
            accs.append(accuracy)
    
            # PRECISION
            precision = precision_score(y_true, y_pred, zero_division=0)
            precisions.append(precision)
    
            # ROC AUC
            auc = roc_auc_score(y_true, y_prob)
            aucs.append(auc)
    
            # SPECIFICITY
            cm = confusion_matrix(y_true, y_pred)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-7)
            specificities.append(specificity)
    
            # KAPPA
            kappa = cohen_kappa_score(y_true, y_pred)
            kappas.append(kappa)

            print(f"[drop={lr[1]},lr={lr[0]}] acc:{accuracy:.3f} prec:{precision:.3f} auc:{auc:.3f} spec:{specificity:.3f} kappa:{kappa:.3f}")
    
        # Select the best metrics for the learning rate with highest accuracy
        idx = int(np.argmax(accs))
        LOO_lr.append(lrs[idx])
        ACCURACY = np.append(ACCURACY, accs[idx])
        PRECISION = np.append(PRECISION, precisions[idx])
        AUC = np.append(AUC, aucs[idx])
        SPECIFICITY = np.append(SPECIFICITY, specificities[idx])
        KAPPA = np.append(KAPPA, kappas[idx])
    
        print(f"Best [acc={accs[idx]:.3f}], [prec={precisions[idx]:.3f}], [auc={aucs[idx]:.3f}], [spec={specificities[idx]:.3f}], [kappa={kappas[idx]:.3f}]")
        print('Accuracy so far:', ACCURACY)
        print('Precision so far:', PRECISION)
        print('AUROC so far:', AUC)
        print('Specificity so far:', SPECIFICITY)
        print('Kappa so far:', KAPPA)
        print('AVERAGE accuracy', np.mean(ACCURACY))
        print('AVERAGE precision', np.mean(PRECISION))
        print('AVERAGE auc', np.mean(AUC))
        print('AVERAGE specificity', np.mean(SPECIFICITY))
        print('AVERAGE kappa', np.mean(KAPPA))
        print('LEARNING RATES', LOO_lr)

    ## save results to csv
    results_df = pd.DataFrame({
        "LOO": LOOs,
        "LearningRate": LOO_lr,
        "Accuracy": ACCURACY,
        "Precision": PRECISION,
        "AUC": AUC,
        "Specificity": SPECIFICITY,
        "Kappa": KAPPA
    })
    
    results_df.to_csv("loo_foundation_results.csv", index=False)
    print("Results saved to loo_foundation_results.csv")
    print(results_df)
