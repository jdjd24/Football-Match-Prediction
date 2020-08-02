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
from pathlib import Path
import datetime
import os
import pickle

now = datetime.datetime.now()
cross = nn.CrossEntropyLoss()
torch.manual_seed(0)

def remove_existing_checkpoints(model_dict):
    for model in model_dict:
        my_file = Path("./NN checkpoints/{}checkpoint.pt".format(model))
        if my_file.is_file(): ##if file exists
            os.remove("./NN checkpoints/{}checkpoint.pt".format(model))
            
def get_ens_df(x, scaler, model_dict, is_models = False, use_odds = True): ##Gets the predictions of all the different models, given a scaler, and the explanatory variables
    
    X = pd.DataFrame(scaler.transform(x.iloc[:,1:]), index = x['match_api_id'], columns = x.iloc[:,1:].columns) ##data to get predictions from
    x_ens = pd.DataFrame(data =  0, index = x.index, columns = []) #empty df to fill with all the different model predictions

    for model_name in model_dict:
        if not is_models:
            model = model_dict[model_name].model
        else:
            model = model_dict[model_name]
            
        with torch.no_grad():
            pred = model(Tensor(X.values))
            
        x_train_ens = pd.DataFrame(data =  np.array(pred), index = x.index, columns = ['{}_0'.format(model_name), '{}_1'.format(model_name), '{}_2'.format(model_name)])
        x_ens = pd.concat([x_ens, x_train_ens], axis = 1)
    
    if use_odds:
        odds = x[['B365H', 'B365D', 'B365A']]
        odds.index = x_ens.index
        x_ens = pd.concat([x_ens, odds], axis = 1)
        
    return x_ens

def plot_losses(model_dict): ##input is a list of tuples, train losses = tuple[0], val losses = tuple[1]
    
    fig, axes = plt.subplots(nrows = len(model_dict), ncols=1, figsize = (4, len(model_dict)*2))
    
    for i, ax in enumerate(axes.flatten()):
        model = model_dict[list(model_dict.keys())[i]]
        ax.plot(model.losses_list[0])
        ax.plot(model.losses_list[1])
        ax.set_title(list(model_dict.keys())[i])
            
def save_models(models, model_dict): ##input is a list of model names which are present in the model_dict
    new_dir = './NN Models/ensemble' + "_" + str(now.hour) + "_"  + str(now.day) + "_" + str(now.month)
    os.mkdir(new_dir)
    
    for model in models:
        torch.save(model_dict[model].model, new_dir + '/{}.pt'.format(model + "_" + str(now.day) + "_" + str(now.month)))


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
    print('Ensemble accuracy: {}'.format(accuracy_score(y_ens, y['target'])))
    #print('Ensemble log loss: {}'.format(log_loss(y['target'], y_ens)))
    
    
def train_one(model_dict, model_name, x_train, y_train, batch_size, epochs = 5, learning_rate = 0.01, use_scheduler = False, scheduler_step_size = 50, random_state = 0):

    _, __, ___, ____ = train_test_split(x_train, y_train, test_size = 0.1, random_state = random_state)
    training_indices = [(_.index, ____.index)]
    _, __, ___, ____ = 0, 0, 0, 0
    
    results = pd.DataFrame(data = 0, index = model_dict.keys(), columns = ['t_acc', 't_loss', 'v_acc', 'v_loss'])
    big_losses_list = [] ##for plotting losses


    model = model_dict[model_name]
    print('\n' + model_name)

 
    for train_index, val_index in training_indices: 

        X_train, X_val = x_train[x_train.index.isin(train_index)], x_train[x_train.index.isin(val_index)]
        Y_train, Y_val = y_train[y_train.index.isin(train_index)], y_train[y_train.index.isin(val_index)]


        results_list, losses = model.train(X_train, Y_train, X_val, Y_val, epochs, get_batch_size(X_train, batch_size), learning_rate, use_scheduler,  scheduler_step_size)
        results.loc[model_name, :] = results.loc[model_name, :].add(results_list)    

    results /= splits
    return results

def train_many(model_dict, x_train, y_train, batch_size, epochs = 5, learning_rate = 0.01, use_scheduler = False, scheduler_step_size = 50, change_trainset = False):

    results = pd.DataFrame(data = 0, index = model_dict.keys(), columns = ['t_acc', 't_loss', 'v_acc', 'v_loss'])
    
    random_states = [0 for x in range(len(model_dict))]
    if change_trainset:
        random_states = [x for x in range(len(model_dict))]
        
    for i, model_name in enumerate(model_dict):
        print('\n' + model_name)
        model = model_dict[model_name]
               
        X_train, X_val, Y_train, Y_val=  train_test_split(x_train, y_train, test_size = 0.1, random_state = random_states[i])

        results_list, losses = model.train(X_train, Y_train, X_val, Y_val, epochs, get_batch_size(X_train, batch_size), learning_rate, use_scheduler,  scheduler_step_size)
        results.loc[model_name, :] = results.loc[model_name, :].add(results_list)
        print('. ', end=' ')

    return results

def get_batch_size(x, size):
    if type(size) != int :
        size = len(x)
    return size 
        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

class LinearNN():
    def __init__(self, model_name, num_layers, in_out, sizes, activation_type, drop_out):
        self.model = self.n_layer_net(num_layers, in_out, sizes, activation_type, drop_out)
        self.activation_type = activation_type
        self.drop_out_p = drop_out
        self.num_layers = num_layers
        self.activation_type = activation_type
        self.name = model_name
        self.losses_list = [[],[]]
        self.scores = 0
        self.predictions_train = []
        self.checkpoints = {}
    
       
    
    def get_score(self, x, y, scaler):
        x = pd.DataFrame(scaler.transform(x.iloc[:,1:]), index = x['match_api_id'], columns = x.iloc[:,1:].columns)
        model = self.model
        model.eval()
        with torch.no_grad():
            pred = model(Tensor(x.values))
        pred_results = pd.Series(pred.max(1).indices)

        print(sklearn.metrics.accuracy_score(pred_results, y['target']), log_loss(y['target'], pred))
        return pred_results.value_counts()
    
    def save_checkpoint(self, state, checkpoint_dir):
        f_path = checkpoint_dir + '/' + '{}checkpoint.pt'.format(self.name)
        torch.save(state, f_path)
    
    def load_checkpoint(self, checkpoint_fpath, model, optimizer):
        my_file = Path(checkpoint_fpath)
        epoch = 0 
        if my_file.is_file(): ##if file exists
            print('Loading saved checkpoint...')
            checkpoint = torch.load(checkpoint_fpath)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
        return model, optimizer, epoch
    

    

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size, learning_rate, use_scheduler = False, scheduler_step_size = 25):
        StandardScaler = preprocessing.StandardScaler().fit(X_train.iloc[:,1:])
        model = self.model
        
        X_train = pd.DataFrame(StandardScaler.transform(X_train.iloc[:,1:]), index = X_train['match_api_id'], columns = X_train.iloc[:,1:].columns)
        X_val = pd.DataFrame(StandardScaler.transform(X_val.iloc[:,1:]), index = X_val['match_api_id'], columns = X_val.iloc[:,1:].columns)

        loss, counter = 0, 0
        losses_train, losses_val = [], []

        dataset = TensorDataset(Tensor(X_train.values), torch.Tensor(Y_train['target'].values))
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.1)
        
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size , gamma = 0.1)
        
        model, optimizer, last_epoch = self.load_checkpoint('./NN checkpoints/{}checkpoint.pt'.format(self.name), model, optimizer)
        
        #Train Model
        t = time.time()
        for epoch in range(last_epoch, epochs+last_epoch):
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

                    loss_train = cross(y_train_pred, Tensor(Y_train['target'].values).long())
                    loss_val = cross(y_val_pred, Tensor(Y_val['target'].values).long())

                    losses_train.append(loss_train.item())
                    losses_val.append(loss_val.item()) 
                    
                    if counter % 100 == 0:
                        print('Loss after iteration {}: {}'.format(counter, loss_train.item()))



                counter+=1 
                
            if use_scheduler:
                scheduler.step()
                
            self.checkpoints[epoch + 1] = model.state_dict()
        time.time()-t
        
        #save checkpoint
        checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        self.save_checkpoint(checkpoint, './NN checkpoints')

        ##Record stuff
        self.losses_list[0].extend(losses_train)
        self.losses_list[1].extend(losses_val)
        self.scores = list(self.evaluate(X_train, Y_train)) + list(self.evaluate(X_val, Y_val))
        self.predictions_train = pd.Series(y_train_pred.max(1).indices)

        return list(self.evaluate(X_train, Y_train)) + list(self.evaluate(X_val, Y_val)), (losses_train, losses_val)

    
    
    def evaluate(self, x, y):
        model = self.model
        model.eval()
        with torch.no_grad():
            pred = model(Tensor(x.values))
        global pred_results 
        pred_results = pd.Series(pred.max(1).indices)
        return sklearn.metrics.accuracy_score(pred_results, y['target']), log_loss(y['target'], pred) 
    

    
    
    def n_layer_net(self, num_layers, in_out, sizes, activation_type, drop_out):
        assert len(sizes) == num_layers
        input_size = in_out[0]
        output_size = in_out[1]
        
        if num_layers ==4:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    output_size), nn.Softmax(dim = 1)
                        )     
        elif num_layers ==5:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], output_size), nn.Softmax(dim = 1)
                        )  
        elif num_layers ==6:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==7:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),  nn.Linear(sizes[6], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==8:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),nn.Linear(sizes[6], sizes[7]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[7]),  nn.Linear(sizes[7], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==9:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
                    activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[1]), nn.Linear(sizes[1], sizes[2]), activation_type, nn.Dropout(p=drop_out), 
                    nn.BatchNorm1d(sizes[2]), nn.Linear(sizes[2], sizes[3]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[3]), nn.Linear(sizes[3],
                    sizes[4]), activation_type, nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[4]), nn.Linear(sizes[4], sizes[5]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[5]), nn.Linear(sizes[5], sizes[6]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[6]),nn.Linear(sizes[6], sizes[7]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[7]),nn.Linear(sizes[7], sizes[8]), activation_type, 
                    nn.Dropout(p=drop_out), nn.BatchNorm1d(sizes[8]),  nn.Linear(sizes[8], output_size), nn.Softmax(dim = 1)
                        )
        elif num_layers ==10:
            model = nn.Sequential(nn.Linear(input_size, sizes[0]), activation_type, nn.BatchNorm1d(sizes[0]), nn.Linear(sizes[0], sizes[1]),
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


        model.apply(init_weights)
        return model
    
