import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import sklearn.metrics
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.utils import resample
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
cross = nn.CrossEntropyLoss()

def get_ens_df(x, scaler, model_dict): ##Gets the predictions of all the different models, given a scaler, and the explanatory variables
    
    X = pd.DataFrame(scaler.transform(x.iloc[:,1:]), index = x['match_api_id'], columns = x.iloc[:,1:].columns) ##data to get predictions from
    x_ens = pd.DataFrame(data =  0, index = x.index, columns = []) #empty df to fill with all the different model predictions

    for model_name in model_dict:
        model = model_dict[model_name].model
        with torch.no_grad():
            pred = model(Tensor(X.values))
            
        x_train_ens = pd.DataFrame(data =  np.array(pred), index = x.index, columns = ['{}_0'.format(model_name), '{}_1'.format(model_name)])
        x_ens = pd.concat([x_ens, x_train_ens], axis = 1)
    return x_ens

def plot_losses(big_losses_list, model_dict, splits): ##input is a list of tuples, train losses = tuple[0], val losses = tuple[1]
    
    fig, axes = plt.subplots(nrows = len(model_dict), ncols=splits, figsize = (splits*4, len(model_dict)*2))
    for i, ax in enumerate(axes.flatten()):
        if i < len(big_losses_list):
            ax.plot(big_losses_list[i][0])
            ax.plot(big_losses_list[i][1])
            ax.set_title(list(model_dict.keys())[math.floor(i/splits)])
            
def save_models(models): ##input is a list of model names which are present in the model_dict
    for model in models:
        torch.save(model_dict[model].model, './NN Models/{}.pt'.format(model))

def ensemble_predict_simple(models, x, y, scaler):
    global pred_ensemble, y_ens
    
    pred_ensemble = 0
    X = pd.DataFrame(scaler.transform(x.iloc[:,1:]), index = x['match_api_id'], columns = x.iloc[:,1:].columns)
    
    for i,model in enumerate(models):
        model.eval()
        with torch.no_grad():
            pred = model(Tensor(X.values))
            pred_ensemble += pred
    
    y_ens = pd.Series(pred_ensemble.max(1).indices)
    print('Ensemble accuracy: {}'.format(accuracy_score(y_ens, y['target_binary'])))
    #print('Ensemble log loss: {}'.format(log_loss(y['target'], y_ens)))
    
def train_many(model_dict, x_train, y_train, batch_size, splits = 1, epochs = 5, learning_rate = 0.01):
    if splits == 1:
        _, __, ___, ____ = train_test_split(x_train, y_train, test_size = 0.15, random_state = 1)
        training_indices = [(_.index, ____.index)]
        _, __, ___, ____ = 0, 0, 0, 0
    
    results = pd.DataFrame(data = 0, index = model_dict.keys(), columns = ['t_acc', 't_loss', 'v_acc', 'v_loss'])
    big_losses_list = [] ##for plotting losses


    for model_name in model_dict:
        model = model_dict[model_name].model
        print('\n' + model_name)
        
        if splits > 1:
            kf = KFold(n_splits=splits, random_state=1, shuffle=True)
            training_indices = kf.split(x_train)
            
        for train_index, val_index in training_indices: 
            
            X_train, X_val = x_train[x_train.index.isin(train_index)], x_train[x_train.index.isin(val_index)]
            Y_train, Y_val = y_train[y_train.index.isin(train_index)], y_train[y_train.index.isin(val_index)]
            
            results_list, losses, all_pred = train(model, X_train, Y_train, X_val, Y_val, epochs, len(X_train), learning_rate)
            results.loc[model_name, :] = results.loc[model_name, :].add(results_list)
            big_losses_list.append(losses)
            print('. ', end=' ')
            

    results /= splits
    return results, big_losses_list




def train(model, X_train, Y_train, X_val, Y_val, epochs, batch_size, learning_rate):
    StandardScaler = preprocessing.StandardScaler().fit(X_train.iloc[:,1:])
    X_train = pd.DataFrame(StandardScaler.transform(X_train.iloc[:,1:]), index = X_train['match_api_id'], columns = X_train.iloc[:,1:].columns)
    X_val = pd.DataFrame(StandardScaler.transform(X_val.iloc[:,1:]), index = X_val['match_api_id'], columns = X_val.iloc[:,1:].columns)
    
    loss, counter = 0, 0
    losses_train, losses_val = [], []

    dataset = TensorDataset(Tensor(X_train.values), torch.Tensor(Y_train['target_binary'].values))
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 25 , gamma = 0.1)
    
    #Train Model
    t = time.time()
    for epoch in range(epochs):
        for x, y in iter(train_loader):
            model.train()
            model.zero_grad()
            
            #forward
            y_pred = model(x)
            loss = cross(y_pred, y.long())
 
            #backward + update
            loss.backward()
            optimizer.step()
            
            #store errors
            with torch.no_grad():
                model.eval()
                y_train_pred = model(Tensor(X_train.values))
                y_val_pred = model(Tensor(X_val.values))
                
                loss_train = cross(y_train_pred, Tensor(Y_train['target_binary'].values).long())
                loss_val = cross(y_val_pred, Tensor(Y_val['target_binary'].values).long())
                
                losses_train.append(loss_train.item())
                losses_val.append(loss_val.item()) 
                
            #if counter % 100 ==0:
                #print('Loss after iteration {}: {}'.format(counter, loss_train.item()))
                
            counter+=1 
        scheduler.step()
    time.time()-t        
    
    ##Returns train/val accuracy/losses, the losses during the training for plotting, the prediction probabilities to be used for ensemble stuff
    return list(evaluate(model, X_train, Y_train)) + list(evaluate(model, X_val, Y_val)), (losses_train, losses_val), (y_train_pred,y_val_pred) 

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(Tensor(x.values))
    global pred_results 
    pred_results = pd.Series(pred.max(1).indices)

    return sklearn.metrics.accuracy_score(pred_results, y['target_binary']), log_loss(y['target_binary'], pred) 



class LinearNN():
    def __init__(self, num_layers, in_out, sizes, activation_type, drop_out):
        self.model = self.n_layer_net(num_layers, in_out, sizes, activation_type, drop_out)
        self.activation_type = activation_type
        self.drop_out_p = drop_out
        self.num_layers = num_layers
        self.activation_type = activation_type
        
    
    def n_layer_net(self, num_layers, in_out, sizes, activation_type, drop_out):
        assert len(sizes) == num_layers
        input_size = in_out[0]
        output_size = in_out[1]
        
        if num_layers ==4:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    output_size), nn.Softmax(dim = 1)
                        )     
        elif num_layers ==5:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], output_size), nn.Softmax(dim = 1)
                        )  
        elif num_layers ==6:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==7:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),  nn.Linear(sizes[6], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==8:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),nn.Linear(sizes[6], sizes[7]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[7]),  nn.Linear(sizes[7], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==9:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),nn.Linear(sizes[6], sizes[7]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[7]),nn.Linear(sizes[7], sizes[8]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[8]),  nn.Linear(sizes[8], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==10:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),nn.Linear(sizes[6], sizes[7]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[7]),nn.Linear(sizes[7], sizes[8]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[8]),nn.Linear(sizes[8], sizes[9]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[9]),  nn.Linear(sizes[9], output_size), nn.Softmax(dim = 1)
                        )
        else:
            print('Invalid num_layers: choose between 4 and 10')
        return model
    
def get_score(model, x, y, scaler):
    x = pd.DataFrame(scaler.transform(x.iloc[:,1:]), index = x['match_api_id'], columns = x.iloc[:,1:].columns)
    model.eval()
    with torch.no_grad():
        pred = model(Tensor(x.values))
    global pred_results 
    pred_results = pd.Series(pred.max(1).indices)

    print(sklearn.metrics.accuracy_score(pred_results, y['target_binary']), log_loss(y['target_binary'], pred) )
    return pred_results