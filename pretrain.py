# Training in 256Hz data and 4s
import torch
from pytorch_lightning import loggers as pl_loggers

from model.sensorPT import *
from model.configs import *
torch.set_float32_matmul_precision("medium")


def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
seed_torch(7)


# init model


model = sensorPT(get_config(**(MODELS_CONFIGS[tag])), 
                 USE_LOSS_A =(variant != "A"),
                 USE_LN     =(variant != "B"),
                 USE_SKIP   =(variant != "C"))
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=callbacks,
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"sensorPT_{tag}_{variant}_tb"), 
                             pl_loggers.CSVLogger('./logs/', name=f"sensorPT_{tag}_{variant}_csv")])
trainer.fit(model, train_loader, valid_loader)