import pandas as pd
import numpy as np
from os import listdir
import torch
import random



if __name__=="__main__":
    CLASpath = '../data/CLAS_Database/CLAS/Answers/'
    subs = ['Part'+str(ii) for ii in range(1,51)]
    data = []
    labels = []
    for sub in subs:
        onlyfiles = listdir('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/')
        ecgfiles = []
        df = pd.read_csv(CLASpath+sub+'_c_i_answers.csv')
        for ind,row in df.iterrows():
            if (row[1]!='empty'):
                stimulus = row['stimulus'].lower().replace('.','-')
                
                for i,fn in enumerate(onlyfiles):
                    if (fn.split('_')[-1].replace('.csv','').lower() == stimulus) and 'ecg' in fn:
                        ecg = pd.read_csv('../data/CLAS_Database/CLAS/Participants/'+sub+'/all_separate/'+fn)
                        if(len(list(ecg['ecg2']))>500):
                            dst="../data/merged_ecg/"
                            if random.random()<0.1:
                                dst+="ValidFolder/0/"
                            else:
                                dst+="TrainFolder/0/"
                            #print(len(list(ecg['ecg2'])), fn)
                            ecgfiles.append(fn)
                            data.append([list(ecg['ecg1'])[:500],list(ecg['ecg2'])[:500]])
                            _d = torch.from_numpy(np.array([list(ecg['ecg1'])[:500],list(ecg['ecg2'])[:500]], dtype='float32') )
                            labels.append(1 if row[1]=='correct' else 0)
                            torch.save(_d, dst + 'CLAS'+f"_{i}.edf")
                        break