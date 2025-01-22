import numpy as np
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import  process_Rawboost_feature	
from utils import pad
import pandas as pd
import torch

class EmotionDataset(Dataset):
    def __init__(self, args, csv_path, base_dir, algo, split_set):
        df = pd.read_csv(csv_path)
        self.df = df[df['Split_Set'] == split_set].reset_index(drop=True)
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 66800
        self.emotion_cols = ['Angry', 'Sad', 'Happy', 'Surprise', 
                           'Fear', 'Disgust', 'Contempt', 'Neutral']
        
        # Convert each column individually to float32
        for col in self.emotion_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('float32')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_path = f"{self.base_dir}/{row['FileName']}"
        
        labels = {col: float(row[col]) for col in self.emotion_cols}
        
        X, fs = librosa.load(file_path, sr=16000)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = torch.Tensor(X_pad)
        
        # Convert labels to tensor explicitly
        target = torch.tensor([float(row[col]) for col in self.emotion_cols], dtype=torch.float32)
        
        return x_inp, target

class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=66800      
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad= pad(Y, self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target

class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id  
