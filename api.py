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
ckpt_path = './data/epoch=199-step=3400.ckpt'
model = LitSensorPT.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
model.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}
    
@app.post("/predict")
async def get_probs(request: Request):
    _req = await request.json()
    req = np.array(_req["empatica"], dtype="float32")
    test_dataset = torch.from_numpy(req)
    _, logit = model(test_dataset)
    print('Y hat',torch.argmax(logit,  dim=-1))
    probs = logit.detach().numpy()[0]
    probs_norm = (probs - probs.min()) / (probs - probs.min()).sum()
    res = dict(zip(["no stress","stress"], probs_norm))
    
    return json.dumps(str(res))

if __name__=="__main__":
    
    import uvicorn
    print('api intializing')
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False,
    )