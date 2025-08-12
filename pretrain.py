# Training in 256Hz data and 4s
import torch
from pytorch_lightning import loggers as pl_loggers

from model.sensorPT import *
from model.configs import *
from model.common import seed_torch
seed_torch(7)
torch.set_float32_matmul_precision("medium")

if __name__=="__main__":

    for loo in ['LOO9','LOO10','LOO11','LOO12']:
    
        #pretrain_dir = f"/scratch/project_2014260/data/LOO/{loo}/full"
        pretrain_dir = f"/scratch/project_2014260/data/LOO_finetune/{loo}/full"
        print(pretrain_dir)
        
        train_dataset = torchvision.datasets.DatasetFolder(root=f"{pretrain_dir}/TrainFolder/", loader=load_fn,  extensions=['.edf'])
        valid_dataset = torchvision.datasets.DatasetFolder(root=f"{pretrain_dir}/ValidFolder/", loader=load_fn, extensions=['.edf'])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        
        print(len(train_loader), len(valid_loader))
        
        steps_per_epoch = math.ceil(len(train_loader)/len(devices))
    
        
        # init model
        model = sensorPT(get_config(**(MODELS_CONFIGS[tag])), 
                         USE_LOSS_A =(variant != "A"),
                         USE_LN     =(variant != "B"),
                         USE_SKIP   =(variant != "C"))
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks = [lr_monitor]
        
        #trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=callbacks,
        #                     logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"sensorPT_{tag}_{variant}_tb"), 
        #                             pl_loggers.CSVLogger('./logs/', name=f"sensorPT_{tag}_{variant}_csv")])
        trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=callbacks,
                             logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"sensorPT_empatica_{loo}_finetune_tb"), 
                                     pl_loggers.CSVLogger('./logs/', name=f"sensorPT_empatica_{loo}_finetune_csv")])
        trainer.fit(model, train_loader, valid_loader)
