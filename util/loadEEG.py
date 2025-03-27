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
        