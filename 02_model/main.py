import argparse
import pandas as pd
import ast
import math
import time
import torch
import pickle
import torch.nn as nn
from net import gtnet
import numpy as np
import random
from util import *
from trainer import Optim
import sys
from matplotlib import pyplot as plt, scale
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig


plt.rcParams['savefig.dpi'] = 1200

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--mode', type=str, default=None, help='Please select a mode to run. Options: test_hps, train and forecast.')
parser.add_argument('--data', type=str, default='02_model/data/data.csv', help='location of the data file') # Change implemented by E Nolan, can now handle CSV directly
parser.add_argument('--model_path', type=str, default='02_model/model/model.pt',
            help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=36,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1) 
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=10,help='clip')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')

#  Added by E Nolan 
parser.add_argument('--graph_file', type=str, default='02_model/data/graph.csv', help="Graph file defining primary node and links. Please submit a csv containing list of primary node elements in absence of established links")
parser.add_argument('--iterations', type=int, default=60, help="Number of iterations" )
parser.add_argument('--create_plots', type=bool, default=False, help='Set to "False to avoid craetion of evaluation plot for each node')
parser.add_argument('--seed', type=int, default=123, help='Set a seed for reproducability')
parser.add_argument("--hp_file", type=str, default='02_model/model/hp.txt', help="Path to a hyperparameter file.")
parser.add_argument('--forecast_results_file', type=str, default='02_model/model/Results/forecast_results.pkl', help="Path to save the forecast results.")
parser.add_argument('--plot_forecast', type=bool, default=True, help='Set to True to generate a forecast plot for all primary nodes.')
args = parser.parse_args()
device = torch.device('cpu')

def get_random_hyperparameters():
    ''' Function to randomly generate a combination of hyperparameters for a training iteraton. 
    Returns:
        dict: A dictionary containing radnomly selected hyperparameter values.
    '''
    #Predefined by Zaid
    # model hyper-parameters
    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]#[0.00001,0.0001,0.0002,0.0003]
    convs=[4,8,16]
    ress=[16,32,64]
    skips=[64,128,256]
    ends=[256,512,1024]
    layers=[1,2]
    ks=[20,30,40,50,60,70,80,90,100]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]

    return {
        'gcn_depth': random.choice(gcn_depths),
        'lr': random.choice(lrs),
        'conv': random.choice(convs),
        'res': random.choice(ress), 
        'skip': random.choice(skips),
        'end': random.choice(ends),
        'layer': random.choice(layers), 
        'k': random.choice(ks), 
        'dropout': random.choice(dropouts),
        'dilation_ex': random.choice(dilation_exs),
        'node_dim': random.choice(node_dims),
        'prop_alpha': random.choice(prop_alphas),
        'tanh_alpha': random.choice(tanh_alphas)
        }

def build_model(hps):
    ''' Takes hyperparameters and arguments to load data and build the model
    Args:
        hps (dict): Dictionary of randomly sampled hyperparameters.
    Returns:
        tuple: (model, Data)
    '''
    Data = DataLoader(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize,args.seq_out_len, args.graph_file)
    print('train X:',Data.train[0].shape)
    print('train Y:', Data.train[1].shape)
    print('valid X:',Data.valid[0].shape)
    print('valid Y:',Data.valid[1].shape)
    print('test X:',Data.test[0].shape)
    print('test Y:',Data.test[1].shape)
    print('test window:', Data.test_window.shape)

    print('length of training set=',Data.train[0].shape[0])
    print('length of validation set=',Data.valid[0].shape[0])
    print('length of testing set=',Data.test[0].shape[0])
    print('valid=',int((0.43 + 0.3) * Data.time_steps))

    model = gtnet(
        gcn_true=args.gcn_true, 
        buildA_true=args.buildA_true, 
        gcn_depth = hps['gcn_depth'], 
        num_nodes=Data.num_nodes, 
        device=device, 
        predefined_A=Data.adj,
        dropout=hps['dropout'], 
        subgraph_size=hps['k'],
        node_dim=hps['node_dim'], 
        dilation_exponential=hps['dilation_ex'],
        conv_channels=hps['conv'], 
        residual_channels=hps['res'], 
        skip_channels=hps['skip'], 
        end_channels=hps['end'],
        seq_length=args.seq_in_len, 
        in_dim=args.in_dim, 
        out_dim=args.seq_out_len,
        layers=hps['layer'], 
        propalpha=hps['prop_alpha'], 
        tanhalpha=hps['tanh_alpha'], 
        layer_norm_affline=False)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    print("DEBUG DATA CHECK:")
    print("  - Train X contains NaN:", torch.isnan(Data.train[0]).any().item())
    print("  - Train Y contains NaN:", torch.isnan(Data.train[1]).any().item())
    print("  - Data scale contains zero/NaN:", torch.isnan(Data.scale).any().item(), (Data.scale == 0).any().item())
    return model, Data

def uncertainty_measure(X, model, montecarlo=False):
    ''' Function to measure episteipic uncertainty using montecarlo dropout. Computes multiple forward passwed with active dropout to estimate prediction variance and confidence intevrals.
    The core logic is extracted from a different function by Zaid.

    Args:
        X: Input batch or window
        model: torch model with dropout enabled
        num_runs: number of forward passess
    Returns:
        tuple:
            - torch.Tensor: Mean predicted values.
            - torch.Tensor or None: Variance across runs.
            - torch.Tensor or None: 95% confidence intervals.
    '''
    num_runs = 10 if montecarlo else 1
    outputs = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(X)
            output = torch.squeeze(output)
            if len(output.shape) == 1 or len(output.shape) == 2:
                output = output.unsqueeze(dim=0)
            outputs.append(output)


    outputs = torch.stack(outputs)
    mean = torch.mean(outputs, dim=0)
    if num_runs > 1:
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0) 
        confidence=1.96*std_dev/torch.sqrt(torch.tensor(num_runs))
    else:
        var = torch.zeros_like(mean)
        confidence = torch.zeros_like(mean)

    return mean, var, confidence

def train_step(model, tx, ty, data, criterion, optim):
    ''' Function that trains a single step for a node split. 
    The core logic is extracted from a different function by Zaid
    Args:
        model: Torch model in training
        tx: Input tensor batch slice
        ty: Traget tensor batch slice
        data: The dataset object
        criterion: loss function
        optim: helper function
    Returns:
        tuple:
            loss.item(): value of computes loss for step
            n_sample: number of samples processed'''
    output = model(tx)           
    output = torch.squeeze(output,3)

    scale = data.scale.expand (output.size(0), output.size(1), data.num_nodes)

    scaled_output = output*scale 
    scaled_ty = ty * scale

    loss = criterion(scaled_output, scaled_ty)
    loss.backward()
    grad_norm = optim.step()
    n_sample = output.size(0) * output.size(1) * data.num_nodes

    return loss.item(), n_sample

def train(data, X, Y, model, criterion, optim, batch_size):
    ''' Training function that invokes a training step for each batch slice.
     The core logic is extracted from a different function by Zaid but had been optimised where function compontents were not being used and leverages automatic setting of num.nodes and extraction of shape (n, m) by E. Nolan
     Args:
        data: dataset object
        X: training input data]
        Y: training target labels
        model: torch model
        criterion: loss function
        optim: helper function
        batch_size: size of the batch, passes through arguments
     Returns:
        float: average loss across samples in training set'''
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for batch_X, batch_Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        batch_X = torch.unsqueeze(batch_X,dim=1).transpose(2,3)

        loss_val, n_sample = train_step(model, batch_X, batch_Y, data, criterion, optim)
        total_loss += loss_val
        n_samples += n_sample

    return total_loss / n_samples

def convert_to_numpy(tensors):
    ''' Function to convert tensor into into Numpy 
    Args:
        tensors: a tuple or list of tensors
    Returns: 
        tuples: corresponding arrays'''

    return tuple(t.detach().cpu().numpy() for t in tensors)

# Use NumPy?
def calculate_rrse(predict, test):
    ''' Function to calculate RRSE
    The core logic has been extrapolated by E. Nolan from a function by Zaid, where the RRSE calculation is attributed to Lai et.al
    Args:
        predict: tensor containing predictions
        test: tensor containing target
    Returns:
        float: rrse'''
    sum_squared_diff = np.sum((test - predict) **2)
    root_sum_squared= math.sqrt(sum_squared_diff) 

    mean_all = np.mean(test)
    diff_r = test - mean_all
    denom_sq = np.sum(diff_r ** 2)
    if denom_sq == 0:
        return 0.0
    # root_sum_squared_r=math.sqrt(np.sum(diff_r) ** 2)

    root_sum_squared_r = math.sqrt(denom_sq)
    rrse=root_sum_squared/root_sum_squared_r 

    return rrse

def calculate_rae(predict, test):
    ''' Function to calculate RAE
    The core logic has been extrapolated by E. Nolan from a function by Zaid.
    Args:
        predict: tensor containing predictions
        test: tensor containing target
    Returns:
        float: rae'''
    sum_absolute_diff = np.sum(np.abs(test - predict))
    mean_all = np.mean(test)
    diff_r = test - mean_all
    sum_absolute_r=np.sum(np.abs(diff_r))

    rae=sum_absolute_diff/sum_absolute_r 
    return rae

def calculate_correlation(predict, test): 
    ''' Function to calculate Correlation
    The core logic has been extrapolated by E. Nolan from a function by Zaid.
        predict: tensor containing predictions
        test: tensor containing target
    Returns:
        float: correlation'''
    sigma_p = (predict).std(axis=0)
    sigma_g = (test).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = test.mean(axis=0)

    index = (sigma_g != 0) 
    correlation = ((predict - mean_p) * (test - mean_g)).mean(axis=0)/ (sigma_p * sigma_g) 
    correlation = (correlation[index]).mean()

    return correlation

def calculate_smape(predict, test):
    ''' Function to calculate RRSE
    The core logic has been extrapolated by E. Nolan from a function by Zaid.
        predict: tensor containing predictions
        test: tensor containing target
    Returns:
        float: smape'''

    smape=0
    for x in range(test.shape[0]):
        for z in range(test.shape[2]):
            smape+=s_mape(test[x,:,z],predict[x,:,z])
    smape/=test.shape[0]*test.shape[2]

    return smape

def create_plot(data, predict, test, variance, confidence, r=0):
    '''Function to generate and save individual node plots and metrics.
    Args:
        data: dataset object
        predict: numpy array of predictions
        test: numpy array of target values
        variance: numpy array of prediction variances
        confidence: numpy array of confidence intervals
        r: starting node index for debugging/printing'''

    for v in range(r,r+data.num_nodes):
        col=v%data.num_nodes
        node_name=DataLoader.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
        node_name=consistent_name(node_name)

        save_metrics_1d(torch.from_numpy(predict[-1,:,col]),torch.from_numpy(test[-1,:,col]),node_name,'Validation')
        plot_predicted_actual(predict[-1,:,col],test[-1,:,col],node_name, 'Validation', variance[-1,:,col], confidence[-1,:,col])

# By Zaid
# Addition of plot creation arg by E. Nolan
def evaluate(data, X, Y, model, batch_size, montecarlo= False):
    '''The core logic has been extrapolated by E. Nolan from a function by Zaid.'''
    all_predictions = None
    all_tests = None
    all_variances = None
    all_confidences = None

    batch_num = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        batch_num += 1
        X = torch.unsqueeze(X,dim=1).transpose(2,3)

        pred_mean, pred_var, pred_confidence = uncertainty_measure(X, model, montecarlo=montecarlo)

        scale = data.scale.expand(Y.size(0), Y.size(1), data.num_nodes) #scale will have the max of each column (142 max values)

        batch_prediction = pred_mean * scale
        batch_test = Y * scale
        batch_variance = pred_var * scale
        batch_confidence = pred_confidence * scale

        if all_predictions is None:
            all_predictions = batch_prediction
            all_tests = batch_test
            all_variances = batch_variance
            all_confidences = batch_confidence
        else:
            all_predictions = torch.cat((all_predictions, batch_prediction))
            all_tests = torch.cat((all_tests, batch_test))
            all_variances = torch.cat((all_variances, batch_variance))
            all_confidences = torch.cat((all_confidences, batch_confidence))

    predict_np, test_np, variance_np, confidence_np = convert_to_numpy((all_predictions, all_tests, all_variances, all_confidences))

    rrse = calculate_rrse(predict_np, test_np)
    rae = calculate_rae(predict_np, test_np)
    correlation = calculate_correlation(predict_np, test_np)
    smape = calculate_smape(predict_np, test_np)


    return rrse, rae, correlation, smape

# By Zaid
#for testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
#The sliding window uses the output from previous step as input of the next step.
#In our case, the window was not slided (we predicted 36 months and the model by default predicts 36 months)
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input):
    #model.eval()# To get Bayesian estimation, we must comment out this line
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    predictions = []
    sum_squared_diff=0
    sum_absolute_diff=0
    r=random.randint(0, 141)
    r=0 # we can choose any random node index for printing
    print('testing r=',str(r))
    scale = data.scale.expand(test_window.size(0), data.num_nodes) #scale will have the max of each column (142 max values)
    print('Test Window Feature:',test_window[:,r])
    
    x_input = test_window[0:n_input, :].clone() # Generate input sequence

    for i in range(n_input, test_window.shape[0],data.out_len):

        print('**************x_input*******************')
        print(x_input[:,r])#prints 1 random column in the sliding window
        print('**************-------*******************')

        X = torch.unsqueeze(x_input,dim=0)
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        X = X.to(torch.float)


        y_true = test_window[i: i+data.out_len,:].clone() 


        # Bayesian estimation
        num_runs = 10

        # Create a list to store the outputs
        outputs = []


        # Use model to predict next time step
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)  
                y_pred = output[-1, :, :,-1].clone()
                #if this is the last predicted window and it exceeds the test window range
                if y_pred.shape[0]>y_true.shape[0]:
                    y_pred=y_pred[:-(y_pred.shape[0]-y_true.shape[0]),]
            outputs.append(y_pred)

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)


        y_pred=torch.mean(outputs,dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        z=1.96
        confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))



        #shift the sliding window
        if data.P<=data.out_len:
            x_input = y_pred[-data.P:].clone()
        else:
            x_input = torch.cat([x_input[ -(data.P-data.out_len):, :].clone(), y_pred.clone()], dim=0)


        print('----------------------------Predicted months',str(i-n_input+1),'to',str(i-n_input+data.out_len),'--------------------------------------------------')
        print(y_pred.shape,y_true.shape)
        y_pred_o=y_pred
        y_true_o=y_true
        for z in range(y_true.shape[0]):
            print(y_pred_o[z,r],y_true_o[z,r]) #only one col
        print('------------------------------------------------------------------------------------------------------------')


        if predict is None:
            predict = y_pred
            test = y_true
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance=torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


    scale = data.scale.expand(test.size(0), data.num_nodes) #scale will have the max of each column (142 max values)

    #inverse normalisation
    predict*=scale
    test*=scale
    variance*=scale
    confidence_95*=scale


    #Relative Squared Error RSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    #Relative Absolute Error RAE - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))# numerator


    #Root Relative Squared Error RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #Root Relative Squared Error RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=0) # calculate the mean of each column in test call it Yj-
    diff_r = test_s - mean_all.expand(test_s.size(0), data.num_nodes) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r
    print('rrse=',root_sum_squared,'/',root_sum_squared_r)

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
###########################################################################################################


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for z in range(Ytest.shape[1]):
        smape+=s_mape(Ytest[:,z],predict[:,z])
    smape/=Ytest.shape[1]

    #plot predicted vs actual and save errors to file
    counter=0
    for v in range(r,r+data.num_nodes):
        col=v%data.num_nodes
        
        node_name=DataLoader.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
        node_name=consistent_name(node_name)
        
        #save error to file
        save_metrics_1d(torch.from_numpy(predict[:,col]),torch.from_numpy(Ytest[:,col]),node_name,'Testing')
        #plot
        plot_predicted_actual(predict[:,col],Ytest[:,col],node_name, 'Testing',variance[:,col],confidence_95[:,col])
        counter+=1

    return rrse,rae,correlation, smape

def plot_forecast(Data, mean_raw, conf_raw, hist_start=132):
    hist_start = (Data.time_steps - args.seq_in_len) if args.seq_in_len else hist_start
    if isinstance(mean_raw, torch.Tensor):
        mean_raw = mean_raw.detach().cpu().numpy()

    mean_raw = np.mean(mean_raw, axis=0)

    time_forecast = np.arange(Data.time_steps, Data.time_steps + mean_raw.shape[0])
    time_history = np.arange(hist_start, Data.time_steps) # Plot only a small portion of historic data

    time_combined = np.concatenate([time_history, time_forecast])

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(Data.prim_nodes)))

    for idx, node in enumerate(Data.prim_nodes.values()):
        i = node["node_index"]
        node_name = node["node_name"]

        # Fix 3: Use the actual node index 'i' instead of the loop index 'idx'
        historic = Data.rawdat[hist_start:Data.time_steps, i]
        pred = mean_raw[:, i]
        full_line = np.concatenate([historic, pred])

        color = colors[idx]

        plt.plot(time_combined, full_line, label=node_name, color=color, linestyle="-", linewidth=2)

    all_time = pd.to_datetime(Data.time_label)
    future_dates = pd.date_range(start=all_time[-1], periods=mean_raw.shape[0] +1, freq="ME")[1:]
    future_labels = future_dates.strftime('%Y-%m').tolist()
    all_labels = list(Data.time_label) + future_labels

    time_label = [all_labels[int(t-1)] for t in time_combined]
    interval = 12
    plt.xticks(time_combined[::interval], time_label[::interval], rotation=45, ha="right")
    plt.axvline(x=Data.time_steps, color="pink", linestyle="--", label='Forecast Period')

    plt.title("Forecast Horizon", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time Steps", fontsize=11)
    plt.ylabel("Prevalence", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, frameon=True)
    plt.tight_layout()

    plt.show()
    return 

def get_avg_cv(Data, mean_raw, conf_raw):
    if isinstance(mean_raw, torch.Tensor):
        mean_raw = mean_raw.detach().cpu().numpy()
    if isinstance(conf_raw, torch.Tensor):
        conf_raw = conf_raw.detach().cpu().numpy()

    mean_raw = np.mean(mean_raw, axis=0)
    conf_raw = np.mean(conf_raw, axis=0)

    summary = []

    for idx, node in enumerate(Data.prim_nodes.values()):
        i = node["node_index"]
        node_name = node["node_name"]

        mean_pred = mean_raw[:, i]
        conf_pred = conf_raw[:, i]
        ratio = conf_pred / mean_pred
        avg_ratio = np.mean(ratio) if isinstance(ratio, np.ndarray) else ratio

        summary.append({
            "node_name": node_name,
            "avg_ratio": avg_ratio
        })
    
    uncertainty_df = pd.DataFrame(summary)
    uncertainty_df = uncertainty_df.sort_values(by="avg_ratio", ascending=True).reset_index(drop=True)
    uncertainty_df.to_csv("02_model/model/Results/uncertainty_results.csv", index=False)
    return


def test_hps(experiment):
    set_random_seed(args.seed)
    best_val=float('inf')
    best_rse=float('inf')
    best_rae=float('inf')
    best_corr=-float('inf')
    best_smape=float('inf')
    best_hp=[]
    best_data = None
    best_model_path = args.model_path

    #Iteration arg added by E. Nolan
    #random search
    for q in range(args.iterations):
        hps = get_random_hyperparameters()
        model, Data = build_model(hps)
    
        criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
        evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE

        optim = Optim(
            model.parameters(), 
            args.optim, hps['lr'], 
            args.clip, 
            lr_decay=args.weight_decay
        )

        # iter_best_val=float('inf')
        # iter_best_rse=float('inf')
        # iter_best_rae=float('inf')
        # iter_best_corr=-float('inf')
        # iter_best_smape=float('inf')
        
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:',(experiment+1))
                print('Iter:',q)
                print('epoch:',epoch)
                print('hp=', hps)
                print('best_rse=', best_rse)


                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

                val_rrse, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, args.batch_size, montecarlo=False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_rrse, val_rae, val_corr, val_smape), flush=True)

                # Save the model if the validation loss is the best we've seen so far.
                sum_loss=val_rrse+val_rae-val_corr
                if (not math.isnan(val_corr)) and sum_loss < best_val:
                    with open(best_model_path, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse = val_rrse
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape

                    current_hp = [
                        hps['gcn_depth'], 
                        hps['lr'], 
                        hps['conv'], 
                        hps['res'], 
                        hps['skip'], 
                        hps['end'],  
                        hps['k'], 
                        hps['dropout'], 
                        hps['dilation_ex'], 
                        hps['node_dim'], 
                        hps['prop_alpha'], 
                        hps['tanh_alpha'], 
                        hps['layer'], 
                        epoch
                    ]
                    best_hp=current_hp
                    best_data = Data  

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=',best_val)
    print('best hps=',best_hp)

    #save best hp to desk
    with open(args.hp_file,"w") as f:
        f.write(str(best_hp))
        f.close()

    # Load the best saved model 
    with open(best_model_path, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(data=best_data, X=best_data.valid[0], Y=best_data.valid[1], model=model,
                                         batch_size=args.batch_size, montecarlo=True)

    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(best_data, best_data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len) 
    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape

def train_final():
    '''The core logic has been extrapolated by E. Nolan from a function by Zaid.'''
    set_random_seed(args.seed)

    filename = args.hp_file
    with open(filename, 'r') as file:
        content = file.read()
        hp = ast.literal_eval(content)

    hps = {
    'gcn_depth': hp[0],
    'lr': hp[1],
    'conv': hp[2],
    'res': hp[3],
    'skip': hp[4],
    'end': hp[5],
    'layer': hp[-2],
    'k': hp[6],
    'dropout': hp[7],
    'dilation_ex': hp[8],
    'node_dim': hp[9],
    'prop_alpha': hp[10],
    'tanh_alpha': hp[11],
    'epoch': hp[13]
    }

    model, Data = build_model(hps)

    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    criterion = nn.MSELoss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
    evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE


    optim = Optim(
                model.parameters(), 
                args.optim, hps['lr'], 
                args.clip, 
                lr_decay=args.weight_decay
            )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for e in range(1, hps['epoch'] + 1):
            print(f'epoch {e}/{hps["epoch"]}')
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        with open(args.model_path, 'wb') as f:
            torch.save(model, f)        
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

def forecast(graph_file=None):
    '''The core logic has been extrapolated by E. Nolan from a function by Zaid.'''
    Data = DataLoader(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize,args.seq_out_len, args.graph_file)

    scale = np.ones(Data.num_nodes)
    dat = np.zeros(Data.rawdat.shape)

    for i in range(Data.num_nodes):
        scale[i] = np.max(np.abs(Data.rawdat[:, i]))
        dat[:, i] = Data.rawdat[:, i] / scale[i]

    P=args.seq_in_len 
    X= torch.from_numpy(dat[-P:, :]) 
    X = torch.unsqueeze(X,dim=0)
    X = torch.unsqueeze(X,dim=1)
    X = X.transpose(2,3)
    X = X.to(torch.float)

    with open(args.model_path, 'rb') as f:
        model = torch.load(f)

    mean, var, confidence = uncertainty_measure(X, model, montecarlo=True)

    dat_raw = dat * scale
    mean_raw = mean * scale if isinstance(mean, np.ndarray) else mean * torch.tensor(scale, device=mean.device)
    var_raw = var * scale
    conf_raw = confidence * scale
   
    print('======================================================')
    print("      PRIMARY ATTACK FORECAST OUTCOMES REPORT      ")
    print('======================================================')

    forecast_results = {}
    for node in Data.prim_nodes.values():
        i = node["node_index"]
        node_name = node["node_name"]
        
        node_pred = mean_raw[..., i]
        node_conf = conf_raw[..., i]
        
        forecast_results[node_name] = {
            "prediction": node_pred,
            "confidence": node_conf
        }

        with open(args.forecast_results_file, 'wb') as f:
            pickle.dump(forecast_results, f)
        
        print(f"\nTarget Attack Node: {node_name}")
        print(f"  -> Projected Horizon Values: {np.round(node_pred, 2)}")
        print(f"  -> Confidence Bounds: ± {np.round(node_conf, 2)}")

    print('======================================================')

    if args.plot_forecast:
        plot_forecast(Data, mean_raw, conf_raw)
    get_avg_cv(Data, mean_raw, conf_raw)
    return forecast_results, mean_raw, var_raw, conf_raw

# Explainer
def explanations(graph_file=None):
    ''' Implementation generated with AI'''
    Data = DataLoader(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len, args.graph_file)

    scale = np.max(np.abs(Data.rawdat), axis=0)
    scale[scale == 0] = 1.0
    dat = Data.rawdat / scale

    P = args.seq_in_len 
    X_tensor = torch.tensor(dat[-P:], dtype=torch.float32).T.unsqueeze(0).unsqueeze(0)

    with open(args.model_path, 'rb') as f:
        model = torch.load(f)
    model.eval()

    model_config = ModelConfig(
        mode='regression',
        task_level='node',
        return_type='raw'
    )

    all_records = []
    num_nodes = Data.num_nodes

    for t in range(args.seq_out_len):
        print(f"Explaining Timestep (Horizon) {t+1}/{args.seq_out_len}...")
    
        wrapped_model = ModelWrapper(model, target_t=t)
        
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=40, lr=0.01),
            explanation_type='model',
            model_config=model_config,
            node_mask_type='attributes',
            edge_mask_type=None
        )
        
        real_node_features = X_tensor[0, :, :, -1].T 
        dummy_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        explanation = explainer(x=real_node_features, edge_index=dummy_edge_index)
        node_masks = explanation.node_mask.detach().cpu().numpy()
        
        for node_idx in range(num_nodes):
            node_name = Data.col_names[node_idx]
            for feat_idx, score in enumerate(node_masks[node_idx]):
                all_records.append({
                    'node_id': node_idx,
                    'node_name': node_name,
                    'timestep': t + 1,
                    'feature_index': feat_idx,
                    'importance_score': score
                })

    explanations_df = pd.DataFrame(all_records).sort_values(by='importance_score', ascending=False)
    explanations_df.to_csv("02_model/model/Results/gnn_explanations.csv", index=False)
    return explanations_df

if __name__ == "__main__":
   
    if args.mode == 'test_hps':
        vacc = []
        vrae = []
        vcorr = []
        vsmape=[]
        acc = []
        rae = []
        corr = []
        smape=[]
        for i in range(1):
            val_acc, val_rae, val_corr, val_smape = test_hps(i)
            vacc.append(val_acc)
            vrae.append(val_rae)
            vcorr.append(val_corr)
            vsmape.append(val_smape)
            # acc.append(test_acc)
            # rae.append(test_rae)
            # corr.append(test_corr)
            # smape.append(test_smape)
        print('\n\n')
        print('1 run average')
        print('\n\n')
        print("valid\trse\trae")
        print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
        print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
        print('\n\n')
        print("test\trse\trae")
        print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
        print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))

    elif args.mode == 'train':
        train_final()
    
    elif args.mode == "forecast":
        forecast(args.graph_file)

    elif args.mode == "explain":
        explanations()

    else:
        print("No mode has been selected. Please provide either test_hps, train or forecast as --mode in the terminal" )

