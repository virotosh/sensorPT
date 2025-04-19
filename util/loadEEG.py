import os
import glob
import numpy as np
import scipy.io as sio
import sys
import mne

import scipy.linalg
import scipy.io
import scipy.sparse
import scipy.signal as signal
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset,DataLoader

class LoadBCIC_2b:
    '''A class to load the test and train data of the BICI IV 2b datast'''
    def __init__(self,path,subject,tmin =0,tmax = 4,bandpass = None):
        self.tmin = tmin
        self.tmax = tmax
        self.bandpass = bandpass
        self.subject = subject
        self.path = path
        self.train_name = ['1','2','3']
        self.test_name = ['4','5']
        self.train_stim_code  = ['769','770']
        self.test_stim_code  = ['783']
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        
    def get_train_data(self):
        data = []
        label = []
        for se in self.train_name:
            data_name = r'B0{}0{}T.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}T.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,True,self.tmin,self.tmax,self.bandpass)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
    
    def get_test_data(self):
        data = []
        label = []
        for se in self.test_name:
            data_name = r'B0{}0{}E.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}E.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,False,self.tmin,self.tmax,self.bandpass)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
            
    
    def get_epoch(self,data_path,isTrain = True,tmin =0,tmax = 4,bandpass = None):
        raw_data = mne.io.read_raw_gdf(data_path)
        events,events_id =  mne.events_from_annotations(raw_data)
        if isTrain:
            stims = [values for key,values in events_id.items() if key in self.train_stim_code]
        else:
            stims = [values for key,values in events_id.items() if key in self.test_stim_code]
        epochs = mne.Epochs(raw_data,events,stims,tmin = tmin,tmax = tmax,event_repeated='drop',baseline=None,preload=True, proj=False, reject_by_annotation=False)

        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')

        epochs = epochs.drop_channels(self.channels_to_remove)
        eeg_data = epochs.get_data()*1e6
        return eeg_data
    
    def get_label(self,label_path):
        label_info = sio.loadmat(label_path)
        return label_info['classlabel'].reshape(-1)-1




def Load_BCIC_2b_raw_data(data_path, tmin=0, tmax=4,bandpass = [0,38]):
    '''
    Load all the 9 subjects data,and save it in the fold of r'./Data'.
    '''
    
    if bandpass is None:
        save_path = 'data/BCIC_2b'
    else:
        save_path = os.path.join(r'data','BCIC_2b_{}_{}HZ'.format(bandpass[0],bandpass[1]))

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    for sub in range(1,10):
        load_raw_data = LoadBCIC_2b(data_path,sub,tmin, tmax,bandpass)
        save_train_path = os.path.join(save_path,r'sub{}_train'.format(sub))
        save_test_path = os.path.join(save_path,r'sub{}_test').format(sub)
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        
        train_x,train_y = load_raw_data.get_train_data()
        scipy.io.savemat(os.path.join(save_train_path,'Data.mat'),{'x_data':train_x,'y_data':train_y})
        
        test_x,test_y = load_raw_data.get_test_data()
        scipy.io.savemat(os.path.join(save_test_path,'Data.mat'),{'x_data':test_x,'y_data':test_y})


def get_IMWUTdata(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    target_session_1_path = os.path.join(data_path,r'sub{}_Data.mat'.format(sub))
    session_1_data = sio.loadmat(target_session_1_path)
    session_1_x = session_1_data['x_data']
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(test_x_1[:,use_channels,:],test_y_1)
    else:
        test_dataset = eeg_dataset(test_x_1,test_y_1)
        
    source_train_x = []
    source_train_y = []

    
    for i in range(1,73):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
        session_1_x = train_data['x_data']
        session_1_y = train_data['y_data'].reshape(-1)
        
        source_train_x.extend(session_1_x)
        source_train_y.extend(session_1_y)

    train_x,valid_x,train_y,valid_y = train_test_split(source_train_x,source_train_y,test_size = 0.2,stratify = source_train_y)
    
    #augment
    for i in range(200):
        train_x.extend(np.random.uniform(low=-1.0, high=1.0, size=(1,48,512)))
        train_y.extend(np.random.randint(2, size=(1,1)).reshape(-1))
        
    source_train_x = torch.FloatTensor(np.array(train_x))
    source_train_y = torch.LongTensor(np.array(train_y))

    source_valid_x = torch.FloatTensor(np.array(valid_x))
    source_valid_y = torch.LongTensor(np.array(valid_y))
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y)
    
    return train_dataset,valid_datset,test_dataset
    
def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,6):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)
        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.2,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset
def EA(x,new_R = None):
    # print(x.shape)
    '''
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    '''
    
    xt = np.transpose(x,axes=(0,2,1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x,xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
    new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(new_R,0.5))
    
    return new_x
def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        