from util.loadEEG import *



#Load_BCIC_2b_raw_data(data_path='data/BCICIV_2b')

data_path = "data/BCIC_2b_0_38HZ/"
for i in range(1,2):
    all_subjects = [i]
    all_datas = []
    train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1, is_few_EA = True, target_sample=256*4)

print(len(train_dataset[0][0][0]))