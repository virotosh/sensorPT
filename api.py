from linear_prob import LitSensorPT
import torch
import numpy as np

import os
import json
from typing import Dict, List, Union, Any
from fastapi import FastAPI, Request
from sklearn import preprocessing

app = FastAPI()

# init model
ckpt_path = './logs/EEGPT_BCIC2B_tb/subject1/checkpoints/epoch=99-step=8200.ckpt'
model = LitSensorPT.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
model.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}
    
@app.post("/predict")
async def get_probs(request: Request):
    _req = await request.json()
    req = np.array(_req["eeg"], dtype="float32")
    test_dataset = torch.from_numpy(req)
    _, logit = model(test_dataset)
    probs = logit.detach().numpy()[0]
    probs_norm = (probs - probs.min()) / (probs - probs.min()).sum()
    res = dict(zip(["left hand","right hand",'stress'], probs_norm[:-1]))
    
    return json.dumps(str(res))

if __name__=="__main__":
    
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False,
    )