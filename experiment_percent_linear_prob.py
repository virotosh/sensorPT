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
from util.loadSensor import get_data, get_Mydata, get_percent_Mydata, temporal_interpolation

from model.SensorTransformer import SensorTransformerEncoder
from model.module import Conv1dWithConstraint, LinearWithConstraint
from model.common import seed_torch, create_1d_absolute_sin_cos_embedding
seed_torch(7)

class LitSensorPT(pl.LightningModule):
    
    def __init__(self, ckptPATH):
        super().__init__()    

        use_channels_names = ['ACC0','ACC1','ACC2','BVP','HR','EDA','TEMP']
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


from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, cohen_kappa_score
import itertools

if __name__=="__main__":
    for percent in [60,70,80,90]:
        data_accuracy = []
        data_precision = []
        data_kappa = []
        data_specificity = []
        data_auc = []
        
        LOO_list = [f'LOO{i}' for i in range(1, 13)]
        
        for loo in LOO_list:
            # Load data
            data_path = f"/scratch/project_2014260/data/LOO/{loo}/test"
            subs = []
            for root, dirs, files in os.walk(data_path):
                subs = sorted(files)
        
            # Find checkpoint path
            ckpt = None
            log_path = f"./logs/sensorPT_empatica_{loo}_tb"
            for root, dirs, files in os.walk(log_path):
                if 'checkpoints' in root and len(files) > 0:
                    ckpt = os.path.join(root, files[0])
        
            print('CHECKPOINT:', ckpt)
            loo_best_lrs = []
            session_accuracies = np.array([])
            session_precisions = np.array([])
            session_kappas = np.array([])
            session_specificities = np.array([])
            session_aucs = np.array([])
        
            for subfile in subs[:5]:
                print('-----', subfile)
                global max_epochs
                global steps_per_epoch
                global max_lr
                train_dataset, valid_dataset, test_dataset = get_percent_Mydata(subfile, data_path, 0, target_sample=512, percent=percent)
                batch_size = 64
                if percent <=70 and percent>=40:
                    batch_size = 32
                if percent>=20 and percent <40:
                    batch_size = 16
                if percent>=10 and percent <20:
                    batch_size = 8
                max_epochs = 200
                lrs = list(itertools.product([1e-3,1e-4,2e-4,3e-4,4e-4,5e-4,5e-5,4e-5], [0.3,0.2]))
                accs = []
                precisions = []
                kappas = []
                specificities = []
                aucs = []
        
                for lr in lrs:
                    print('Learning rate:', lr)
                    max_lr = lr[0]
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
                    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
                    
                    steps_per_epoch = math.ceil(len(train_loader))
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
                            pl_loggers.TensorBoardLogger('./logs/', name="test_tb", version=f"subject{subfile}"),
                            pl_loggers.CSVLogger('./logs/', name="test_csv")
                        ]
                    )
                    trainer.fit(model, train_loader, valid_loader)
        
                    # Prediction and Accuracy
                    _, logits = model(test_dataset.x)
                    y_true = test_dataset.y.long().cpu().numpy()
                    y_pred = torch.argmax(logits, dim=-1).cpu().numpy()
                    y_proba = logits.softmax(dim=-1).detach().numpy()
        
                    # Accuracy
                    accuracy = (y_pred == y_true).mean()
                    precision = precision_score(y_true, y_pred, average='weighted')
                    kappa = cohen_kappa_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred)
                    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        
                    print('Y hat:', y_pred)
                    print('Accuracy:', accuracy)
                    print('Precision:', precision)
                    print('Kappa:', kappa)
                    print('Specificity:', specificity)
                    print('ROC AUC:', roc_auc)
        
                    accs.append(accuracy)
                    precisions.append(precision)
                    kappas.append(kappa)
                    specificities.append(specificity)
                    aucs.append(roc_auc)
        
                # Use metrics from the best accuracy epoch
                best_idx = np.argmax(accs)
                loo_best_lrs.append(lrs[best_idx])
        
                session_accuracies = np.append(session_accuracies, accs[best_idx])
                session_precisions = np.append(session_precisions, precisions[best_idx])
                session_kappas = np.append(session_kappas, kappas[best_idx])
                session_specificities = np.append(session_specificities, specificities[best_idx])
                session_aucs = np.append(session_aucs, aucs[best_idx])
        
                print('Accuracies so far:', session_accuracies)
                print('Average accuracy:', np.mean(session_accuracies))
                print('Best learning rates:', loo_best_lrs)
        
            # Store average metrics for this LOO
            data_accuracy.append(np.mean(session_accuracies))
            data_precision.append(np.mean(session_precisions))
            data_kappa.append(np.mean(session_kappas))
            data_specificity.append(np.mean(session_specificities))
            data_auc.append(np.mean(session_aucs))
        
            print(f'---METRICS for {loo}---')
            print('Average Accuracy:', data_accuracy[-1])
            print('Average Precision:', data_precision[-1])
            print('Average Kappa:', data_kappa[-1])
            print('Average Specificity:', data_specificity[-1])
            print('Average AUC:', data_auc[-1])
            print('--- ALL DATA SO FAR ---')
            print('ALL Accuracies:', data_accuracy)
            print('ALL Precisions:', data_precision)
            print('ALL Kappa:', data_kappa)
            print('ALL Specificities:', data_specificity)
            print('ALL AUCs:', data_auc)
            print('MEAN Accuracy:', np.mean(data_accuracy))
            print('MEAN Kappa:', np.mean(data_kappa))
            print('MEAN Precision:', np.mean(data_precision))
            print('MEAN Specificity:', np.mean(data_specificity))
            print('MEAN AUC:', np.nanmean(data_auc))
    
        results_df = pd.DataFrame({
            'LOO': LOO_list,
            'Accuracy': data_accuracy,
            'Precision': data_precision,
            'Kappa': data_kappa,
            'Specificity': data_specificity,
            'AUC': data_auc
        })
        
        print(results_df)
        results_df.to_csv(f'percent_{percent}_louo_foundation_results.csv', index=False)
        print(f"Results saved to percent_{percent}_louo_foundation_results.csv")

        
