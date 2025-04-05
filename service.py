from pathlib import Path
from linear_prob import LitSensorPT
import torch 


class SensorPTService:
    def __init__(self):
        self.tuned_model =  self._initialize_from_ckpt()

    def _initialize_from_ckpt(self):
        ckpt_path = './logs/EEGPT_BCIC2B_tb/subject1/checkpoints/epoch=99-step=8200.ckpt'
        if Path(ckpt_path).is_file():
            print("load tuned model")
            return LitSensorPT.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        else:
            return None



    