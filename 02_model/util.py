import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
import random
import math
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from matplotlib import pyplot as plt
from collections import defaultdict
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig


from websockets import Data

def normal_std(x):
    '''This function has been copied from the following repository with no modifications: https://github.com/zaidalmahmoud/Cyber-trend-forecasting'''
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoader(object):
    '''The core logic has been extrapolated by E. Nolan from a function by Zaid.
    Specific additions by E. Nolan are marked directly in the methods.'''
    col = None  # Class variable to hold column names

    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, out=1, graph_file=None):
        self.device = device
        self.h = horizon
        self.P = window
        self.normalize_type = normalize
        self.out_len=out

        # Added by E Nolan
        self.col_names, self.rawdat, self.time_label = self.load_data(file_name)
        self.time_steps, self.num_nodes = self.rawdat.shape
        self.prim_nodes = self.get_prim_nodes(graph_file)

        #  Normalisation
        self.shift=0
        self.min_data=np.min(self.rawdat)
        if(self.min_data<0):
            self.shift=(self.min_data*-1)+1
        elif (self.min_data==0):
            self.shift=1

        self.dat = np.zeros(self.rawdat.shape)
        self.scale = np.ones(self.num_nodes)
        self._normalized(self.normalize_type)
        #  Split Data
        self._split(int(train * self.time_steps), int((train + valid) * self.time_steps), self.time_steps)

        # Tensor prep
        self.scale = torch.from_numpy(self.scale).float().to(device)
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0),self.test[1].size(1), self.num_nodes)
        self.scale = self.scale.to(self.device)
        #  Loss stats
        self.rse = torch.sqrt(torch.mean((tmp - torch.mean(tmp)) ** 2))
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.adj = self.build_predefined_adj(graph_file) 
        DataLoader.col = self.col_names

    def load_data(self, file_name):
        '''Modified by E Nolan'''
        # Read the CSV file of the dataset

        if not os.path.exists(file_name):
            raise FileNotFoundError(f"The file {file_name} does not exist.")
    
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            header = [c.strip() for c in next(reader)]

        col_names = header[1:]
        usecols = range(1, len(header))  # Skip the first column (Date)
        rawdat = np.loadtxt(file_name, delimiter=',', skiprows=1, usecols=usecols)

        time_label = np.loadtxt(file_name, skiprows=1, usecols=0, delimiter=",", dtype=str).tolist()

        return col_names, rawdat, time_label
 
    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.num_nodes):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))
    
    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1,train)
        valid_set = range(train, valid) 
        test_set = range(valid, self.time_steps)
        
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test =  self._batchify(test_set, self.h)
        
        
        self.test_window=torch.from_numpy(self.dat[-(36+self.P):, :]) 

    def _batchify(self, idx_set, horizon):
        n = len(idx_set) 
        X = torch.zeros((n-self.out_len, self.P, self.num_nodes)) #n samples x P time steps lookback x number of columns.
        Y = torch.zeros((n-self.out_len, self.out_len, self.num_nodes)) 

        for i in range(n-self.out_len): 
            end = idx_set[i] - self.h + 1 
            start = end - self.P 
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]) 
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i]:idx_set[i]+self.out_len, :])
            
        return [X, Y]


    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    #  Refactored by E Nolan
    def build_predefined_adj(self, graph_file=None):
        num_nodes = len(self.col_names)
        adj = torch.zeros((num_nodes, num_nodes))

        if not graph_file or not os.path.exists(graph_file):
            print("No valid graph found. Returning zero-adjacency matrix")
            return adj
 
        # Initialize an empty dictionary with default value as an empty list
        graph = defaultdict(list)
        with open(graph_file, 'r') as f:
            reader = csv.reader(f)
            # Iterate over each row in the CSV file
            for row in reader:
                # Extract the key node from the first column
                key_node = row[0]
                # Extract the adjacent nodes from the remaining columns
                adjacent_nodes =  [node for node in row[1:] if node]#does not include empty columns
                
                # Add the adjacent nodes to the graph dictionary
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with',len(graph),'attacks...')
        # Print the column names list
        print(num_nodes, 'columns loaded...')

        index_nodes = {name: i for i, name in enumerate(self.col_names)}
        for key_node, adjacent_nodes in graph.items():
            if key_node in index_nodes:
                i = index_nodes[key_node]
                for adjacent_node in adjacent_nodes:
                    if adjacent_node in index_nodes:
                        j = index_nodes[adjacent_node]
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
        print('Adjacency created...')

        return adj

    def get_prim_nodes(self, graph_file=None):
        '''Added by E Nolan'''
        prim_nodes = {}
        if graph_file:
            key_nodes = set()
            with open(graph_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    key_nodes.add(row[0].strip().lower())
    
            for i, name in enumerate (self.col_names):
                name_lower = name.lower()
                if any(kw in name_lower for kw in key_nodes):
                    prim_nodes[i] = {
                        "node_index": i,
                        "node_name": name
                    }
        if not graph_file or not os.path.exists(graph_file):
            prim_nodes = self.col_names
            print('No primary nodes set. Using all columns as primary nodes.')

        return prim_nodes

''' Utilies and helper functions.
Copied directly from https://github.com/zaidalmahmoud/Cyber-trend-forecasting/blob/main/B-MTGNN/train_test.py
Small adjustments, deletions and transtiion to args over fixed variables performed by E. Nolan
Copied segments will be marked as # By Zaid'''

#  By Zaid
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# By Zaid
def consistent_name(name):

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

# By Zaid
def save_metrics_1d(predict, test, title, type):
    #RRSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator

    #Relative Absolute Error RAE  - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))

    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s) # calculate the mean of each column in test
    diff_r = test_s - mean_all # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    # Added by E Nolan to avoid division by 0
    if root_sum_squared_r == 0:
        rrse = 0
    else:
        #RRSE according to Lai et.al
        rrse = root_sum_squared/root_sum_squared_r

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()


    title=title.replace('/','_')
    with open('02_model/model/'+type+'/'+title+'_'+type+'.txt',"w") as f:
      f.write('rse:'+str(rrse)+'\n')
      f.write('rae:'+str(rae)+'\n')
      f.close()


# By Zaid
def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95):

    #all months
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    for year in range (11,26):   
        for month in months:
            if year==11 and month not in ['Jul','Aug','Sep','Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))   
    M2=[]
    p=[]
    
    #last 3 years
    if type=='Testing':
        M=M[-len(predicted):]
        for index,value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 
    
    else: ##Validation x axis: Oct-22 to Sep-25
        M=M[135:171]
        for index,value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 

    x=range(1,len(predicted)+1)
    plt.plot(x,actual,'b-',label='Actual')
    plt.plot(x,predicted,'--', color='purple',label='Predicted')
    # Plot the confidence interval as a shaded region
    plt.fill_between(x, predicted-confidence_95.numpy(), predicted+confidence_95.numpy(), alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(ticks = p ,labels = M2, rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    title=title.replace('/','_')
    plt.savefig('02_model/model/'+type+'/'+title+'_'+type+'.png', bbox_inches="tight")
    plt.savefig('02_model/model/'+type+'/'+title+'_'+type+".pdf", bbox_inches = "tight", format='pdf')


    plt.show(block=False)
    plt.pause(2)
    plt.close()


# By Zaid
# symmetric mean absolute percentage error (optional)
def s_mape(yTrue,yPred):
  mape=0
  for i in range(len(yTrue)):
    mape+= abs(yTrue[i]-yPred[i])/ (abs(yTrue[i])+abs(yPred[i]))
  mape/=len(yTrue)

  return mape

class ModelWrapper(torch.nn.Module):
            ''' Implementation generated with AI'''
            def __init__(self, inner_model, target_t):
                super().__init__()
                self.inner_model = inner_model
                self.target_t = target_t
                
            def forward(self, x, edge_index=None):
                num_n, num_f = x.shape
                seq_len = self.inner_model.seq_length
                model_input = x.T.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, seq_len)
                preds = self.inner_model(model_input, idx=None)
                if preds.ndim == 4:
                    return preds[0, 0, :, self.target_t]
                return preds[:, self.target_t]