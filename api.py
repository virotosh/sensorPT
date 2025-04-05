from service import SensorPTService
from linear_prob import LitSensorPT
from util.loadEEG import get_data
import torch
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


if __name__=="__main__":
    # load data
    data_path = "data/BCIC_2b_0_38HZ/"    
    # init model
    service = SensorPTService()
    model = service.tuned_model
    model.to(torch.device('cpu'))
    model.eval()
    
    for i in range(1,2):
        all_subjects = [i]
        all_datas = []
        train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1, is_few_EA = True, target_sample=256*4)
        print(test_dataset.x[:1].numpy())
        print('predict')
        # predict
        _, logit = model(test_dataset.x[:1])
        #print('Y hat',torch.argmax(logit,  dim=-1))
        # accuracy
        y = test_dataset.y[:1]
        label = y.long()
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        
        #print('Y:', test_dataset.y)
        print('accuracy',accuracy)