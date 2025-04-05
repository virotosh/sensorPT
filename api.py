from service import SensorPTService
from linear_prob import LitSensorPT
from util.loadEEG import get_data
import torch


if __name__=="__main__":
    # load data
    data_path = "data/BCIC_2b_0_38HZ/"
    for i in range(1,2):
        all_subjects = [i]
        all_datas = []
        train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1, is_few_EA = True, target_sample=256*4)
        
        global max_epochs
        global steps_per_epoch
        global max_lr
    
        batch_size=64
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
        # init model
        model = SensorPTService()
        mode.tuned_model.eval()
        
        # predict
        _, logit = model.tuned_model(test_dataset.x)
        #print('Y hat',torch.argmax(logit,  dim=-1))
        # accuracy
        y = test_dataset.y
        label = y.long()
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        
        #print('Y:', test_dataset.y)
        print('accuracy',accuracy)