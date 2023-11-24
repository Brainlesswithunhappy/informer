import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)#转换成pandas支持的格式
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):#字典形式 
    def __init__(self, root_path, flag='train', size=None, 
                 features='M', data_path='chj.csv', 
                 target='cy_d', scale=True, inverse=False, timeenc=0, freq='h', cols=None):#cols=None
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc#时间序列处理方式
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()#数据标准化 标准化
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('D1'); cols.remove('D2'); cols.remove('D3')#列名，去掉标签，日期 cols.remove(self.target[0]);cols.remove(self.target[1]);cols.remove(self.target[2]);cols.remove(self.target[3]);cols.remove(self.target[4]);
        df_raw = df_raw[['D1','D2','D3']+cols+[self.target]]#数据集合，此处需要再划分训练和测试和验证，如果已经划分，此处可以处理掉
    
            
        num_train = int(len(df_raw)*0.7)#直接做切分做训练、测试，验证集
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]#border1s:[0,24448,27956]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]#border1:0 set_type 用来存数据位置0是训练，1是测试，2是验证
        border2 = border2s[self.set_type]#border2:24544 用来取数据的
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[0:]#不需要时间，所以从1开始，此处序号排列从0开始
            df_data = df_raw[cols_data]
            stamp_data = df_raw.columns[0:3]#by lhj
            df_stamp = df_raw[stamp_data]#by lhj
            
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:#标准化处理
            train_data = df_data[border1s[0]:border2s[0]]#scale数据规模
            self.scaler.fit(train_data.values)#均值和偏差 对每列处理
            data = self.scaler.transform(df_data.values)#数据标准化处理
            data[np.isnan(data)] = 0
            #data.fillna(0, inplace = True)#以0填充 by lhj 这两种都是panda的填充方式，上面是numpy的数组，不能直接用 改用numpy的填充方式
            #data.interpolate(method='linear')  # 使用线性插值填充 NaN 值
            
            train_stamp = df_stamp[border1s[0]:border2s[0]]
            self.scaler.fit( train_stamp.values)#均值和偏差 对每列处理
            data_stamp = self.scaler.transform(df_stamp.values)#数据标准化处理
            #data_stamp.fillna(0, inplace = True)#以0填充 by lhj 同上
            data_stamp[np.isnan(data_stamp)] = 0
        else:
            data = df_data.values
            
        #df_stamp = df_raw[['D1','D2','D3']][border1:border2]
        #df_stamp['D1','D2','D3'] = pd.to_datetime(df_stamp.date)#取时间序列，对时间 取特征  借助pandas的格式，转换格式，方便调用
        #data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        #df_stamp = df_raw[['D1','D2','D3']][border1s[0]:border2s[0]]#by lhj  
        #data_stamp = self.scaler.transform(df_stamp.values) #by lhj  
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]# 之前是没有[border1:border2] 加上的 by lhj
    
    def __getitem__(self, index): #
        s_begin = index  #index随机数据，比如在10000个里随机选取一个
        s_end = s_begin + self.seq_len#+96  #s是训练，r是预测
        r_begin = s_end - self.label_len #+48
        r_end = r_begin + self.label_len + self.pred_len#+48+24

        seq_x = self.data_x[s_begin:s_end]#96长度，12特征个数
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]#72 48是原始有标签的 24是预测
        seq_x_mark = self.data_stamp[s_begin:s_end]#时间特征 96,4 
     
        seq_y_mark = self.data_stamp[r_begin:r_end]#时间特征72,4

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='M', data_path='chj.csv', 
                 target='cy_d', scale=True, inverse=False, timeenc=0, freq='15min',cols=None): #cols=None
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);cols.remove(self.target); cols.remove('D1'); cols.remove('D2'); cols.remove('D3')#列名，去掉标签，日期 cols.remove(self.target[0]);cols.remove(self.target[1]);cols.remove(self.target[2]);cols.remove(self.target[3]);cols.remove(self.target[4]);
        df_raw = df_raw[['D1','D2','D3']+cols+[self.target]]#数据集合，此处需要再划分训练和测试和验证，如果已经划分，此处可以处理掉  
            
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[0:]# 已经处理时间，前三为nowinphase nowincyle pahseincycle，所以从3开始
            df_data = df_raw[cols_data]
            tmp_stamp =df_raw.columns[0:3]
            pred_dates =df_raw[tmp_stamp]
            
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
            data[np.isnan(data)] = 0
            self.scaler.fit(pred_dates.values)
            data_stamp = self.scaler.transform(pred_dates.values)
            data_stamp[np.isnan(data_stamp)] = 0
            
        else:
            data = df_data.values
            
       # tmp_stamp = df_raw[['D1','D2','D3']][border1:border2]
        #tmp_stamp['D1','D2','D3'] = pd.to_datetime(tmp_stamp.date)#取时间序列，对时间 取特征  借助pandas的格式，转换格式，方便调用
       # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
       # pred_dates = pd.date_range( periods=self.pred_len+1, freq=self.freq)#tmp_stamp.date.values[-1],???可能是错的
        
       # df_stamp = pd.DataFrame(columns = ['D1','D2','D3'])
        #df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
       # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])#??? 可能是错的

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
