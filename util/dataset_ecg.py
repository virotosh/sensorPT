import pandas as pd
import numpy as np
from os import listdir
import torch
import random
import os
from scipy.io import savemat


if __name__=="__main__":
    CLASpath = '../data/CLAS_Database/CLAS/Answers/'
    subs = ['Part'+str(ii) for ii in range(1,51)]
    data = []
    labels = []
    for sub in subs:
        onlyfiles = listdir('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/')
        ecgfiles = []
        if not os.path.exists(CLASpath+sub+'_c_i_answers.csv'):
            continue
        df = pd.read_csv(CLASpath+sub+'_c_i_answers.csv')
        for ind,row in df.iterrows():
            stimulus = row['stimulus'].lower().replace('.','-')
            
            for i,fn in enumerate(onlyfiles):
                if (fn.split('_')[-1].replace('.csv','').lower() == stimulus) and 'ecg' in fn:
                    ecg = pd.read_csv('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/'+fn)
                    if(len(list(ecg['ecg2']))>512):
                        dst="../data/merged_ecg/"
                        if random.random()<0.1:
                            dst+="ValidFolder/0/"
                        else:
                            dst+="TrainFolder/0/"
                        os.makedirs(dst, exist_ok=True)
                        #print(len(list(ecg['ecg2'])), fn)
                        ecgfiles.append(fn)
                        __tmp = []
                        for iii in range (29):
                            __tmp+= [list(ecg['ecg1'])[:512]*2,list(ecg['ecg2'])[:512]*2]
                        data.append(__tmp)
                        _d = torch.from_numpy(np.array(__tmp, dtype='float32') )
                        labels.append(1 if row[1]=='correct' else 0)
                        torch.save(_d, dst + 'CLAS'+f"_{sub}_{i}_{ind}.edf")
                    break

    ## for finetune 
    subs = ['Part'+str(ii) for ii in range(51,56)]
    for sub_id,sub in enumerate(subs):
        data = []
        labels = []
        onlyfiles = listdir('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/')
        if not os.path.exists(CLASpath+sub+'_c_i_answers.csv'):
            continue
        df = pd.read_csv(CLASpath+sub+'_c_i_answers.csv')
        for ind,row in df.iterrows():
            stimulus = row['stimulus'].lower().replace('.','-')
            for i,fn in enumerate(onlyfiles):
                if (fn.split('_')[-1].replace('.csv','').lower() == stimulus) and 'ecg' in fn:
                    ecg = pd.read_csv('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/'+fn)
                    if(len(list(ecg['ecg2']))>512):
                        __tmp = []
                        for iii in range (29):
                            __tmp+= [list(ecg['ecg1'])[:512]*2,list(ecg['ecg2'])[:512]*2]
                        data.append(__tmp)
                        labels.append(1 if row[1]=='correct' else 0)
                    break
        os.makedirs(f"../data/ECGtest/sub{sub_id+1}_train", exist_ok=True)
        savemat(f"../data/ECGtest/sub{sub_id+1}_train/Data.mat", {'x_data':data,'y_data':[labels]}, oned_as='row')

    ## for test 
    subs = ['Part'+str(ii) for ii in range(56,61)]
    for sub_id,sub in enumerate(subs):
        data = []
        labels = []
        onlyfiles = listdir('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/')
        if not os.path.exists(CLASpath+sub+'_c_i_answers.csv'):
            continue
        df = pd.read_csv(CLASpath+sub+'_c_i_answers.csv')
        for ind,row in df.iterrows():
            stimulus = row['stimulus'].lower().replace('.','-')
            for i,fn in enumerate(onlyfiles):
                if (fn.split('_')[-1].replace('.csv','').lower() == stimulus) and 'ecg' in fn:
                    ecg = pd.read_csv('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/'+fn)
                    if(len(list(ecg['ecg2']))>512):
                        __tmp = []
                        for iii in range (29):
                            __tmp+= [list(ecg['ecg1'])[:512]*2,list(ecg['ecg2'])[:512]*2]
                        data.append(__tmp)
                        labels.append(1 if row[1]=='correct' else 0)
                    break
        os.makedirs(f"../data/ECGtest/sub{sub_id+1}_test", exist_ok=True)
        savemat(f"../data/ECGtest/sub{sub_id+1}_test/Data.mat", {'x_data':data,'y_data':[labels]}, oned_as='row')