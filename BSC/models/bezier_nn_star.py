import logging
import os
import joblib
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.nn import Module
from scipy.stats import gaussian_kde

from BSC.utils import utils
from BSC.surrogate_model import SurrogateModel

from BSC.bezier_vis import *


def activation(activation_fun=None):
    """Helper for building an activation layer."""
    if activation_fun == "relu":
        return nn.ReLU(inplace=True)
    elif activation_fun == "silu" or activation_fun == "swish":
            return torch.nn.SiLU()
    elif activation_fun == "gelu":
        return torch.nn.GELU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)

class MLPBlock(Module):
    """Transformer MLP block, fc, gelu, fc."""

    def __init__(self, w_in, mlp_d, w_out=32, act="relu"):
        super().__init__()
        self.linear_1 = nn.Linear(w_in, mlp_d, bias=True)
        self.af = activation(act)
        self.linear_2 = nn.Linear(mlp_d, w_out, bias=True)

    def forward(self, x):
        # return self.linear_2(self.af(self.linear_1(x)))
        return self.af(self.linear_1(x))


class NNSurrogateModel(nn.Module):

    def __init__(self, input_shape, hidden_dim, num_layers, out_dim, dropout_p, arch_input_dim=5):
        super().__init__()
        self.arch_input_dim = arch_input_dim
        self.arch_encoding = MLPBlock(self.arch_input_dim, hidden_dim)
        self.reg_encoding = MLPBlock(2, hidden_dim)
        self.combine = MLPBlock(30, 64)
        self.hyp_encoding = MLPBlock(input_shape[1]-arch_input_dim, hidden_dim)

        hidden_dim = 2 * hidden_dim

        self.fclayers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.bnlayers = [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        self.droplayers = [nn.Dropout(dropout_p) for _ in range(num_layers)]
        self.outlayer = nn.Linear(hidden_dim, out_dim)

        self.dvn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p))

        self.out = nn.Linear(int(hidden_dim/2), out_dim - 2)  # no sigmoid
        self.acc = nn.Linear(int(hidden_dim/2), 2)

        self.allout = nn.Linear(int(hidden_dim/2), out_dim)

    def forward(self, x):
        arch_input = x[:, :self.arch_input_dim]
        hyp_input = x[:, self.arch_input_dim:]

        
        # combine_input = x[:, 5:35]


        arch_embding = self.arch_encoding(arch_input)
        # combine_embding = self.combine(combine_input)
        hyp_embding = self.hyp_encoding(hyp_input)

        x = torch.cat([arch_embding, hyp_embding], 1)

        # for fc, bn, drop in zip(self.fclayers, self.bnlayers, self.droplayers):
        #     x = F.relu(fc(x))
        #     x = drop(x)
        # return self.outlayer(x)
        feats = self.dvn(x)
        
        start_end = F.sigmoid(self.acc(feats))
        control_points = self.out(feats)
        
        return torch.cat([100*start_end, control_points], 1)

        # return self.allout(feats)


class BEZIERNNSModel(SurrogateModel):
    def __init__(self, data_root, log_dir, seed, model_config, data_config, search_space, nb101_api):

        super().__init__(data_root, log_dir, seed, model_config, data_config, search_space, nb101_api)

        self.model = None
        self.model_config["param:objective"] = "reg:squarederror"
        self.model_config["param:eval_metric"] = "rmse"

    def parse_param_config(self):
        identifier = "param:"
        param_config = dict()
        for key, val in self.model_config.items():
            if key.startswith(identifier):
                param_config[key.replace(identifier, "")] = val
        return param_config

    def train(self):

        X_train, y_train, E_train, T_train = self.load_dataset(dataset_type='train', use_full_lc=True)
        X_val, y_val, E_val, T_val = self.load_dataset(dataset_type='val', use_full_lc=True)

        # acc = y_train[:,-1]
        # index = np.argsort(acc)
        # rank = np.argsort(index)

        pX = torch.tensor(X_train, dtype=torch.float32)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        # convert lc to cubic bezier
        
        # self.name = "Cubic"
        # self.name = "Quartic"
        self.name = "Quintic"

        if self.name =="Cubic":
            out_dim = 6
        elif self.name == "Quartic":
            out_dim = 8
        elif self.name == "Quintic":
            out_dim = 10
        elif self.name == "Sextic":
            out_dim = 12

        labels = []
        for y in y_train:
            x_data = np.array([i + 1 for i in range(len(y))])
            y_data = np.array(y)

            init_control_points = bezier_fit(x_data, y_data, self.name)
            size = 200 * 200
            learning_rate = is_close_to_linev2(x_data, y_data, size)
            if self.name == "Sextic":
                x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6 = train(x_data, y_data, init_control_points, learning_rate, self.name)
                label = np.array([y0, y6, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5])
            elif self.name == "Cubic":
                x0, x1, x2, x3, y0, y1, y2, y3 = train(x_data, y_data, init_control_points, learning_rate, self.name)
                label = np.array([y0, y3, x1, x2, y1, y2])
            elif self.name == "Quartic":
                x0, x1, x2, x3, x4, y0, y1, y2, y3, y4 = train(x_data, y_data, init_control_points, learning_rate, self.name)
                label=np.array([y0, y4, x1, x2, x3, y1, y2, y3])
            elif self.name == "Quintic":
                x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5 = train(x_data, y_data, init_control_points, learning_rate, self.name)
                label=np.array([y0, y5, x1, x2, x3, x4, y1, y2, y3, y4])

            labels.append(label)
        
        if self.name == "Quartic":
            param_config = {
                "hidden_dim":238,
                "dropout_p": 0.1,
                "learning_rate": 0.011679111515474187,
                "num_epochs": 160,
                "batch_size": 1024,
            }
            param_config = {
                "hidden_dim":247,
                "dropout_p": 0.0,
                "learning_rate": 0.00033586740875637613,
                "num_epochs": 180,
                "batch_size": 64,
            }

            ### TinyImageNet
            param_config = {
                "hidden_dim":245,
                "dropout_p": 0.1,
                "learning_rate":  0.00019043239853760313,
                "num_epochs": 185,
                "batch_size": 64,
            }

            param_config = {
                "hidden_dim":43,
                "dropout_p": 0.1,
                "learning_rate":  0.020336314941789963,
                "num_epochs": 175,
                "batch_size": 128,
            }

            param_config = {
                "hidden_dim":239,
                "dropout_p": 0.1,
                "learning_rate":  0.0001002330776182924,
                "num_epochs": 150,
                "batch_size": 64,
            }



        elif self.name == "Quintic":
            param_config = {
                "hidden_dim":209,
                "dropout_p": 0.1,
                "learning_rate": 0.00016238575321863175,
                "num_epochs": 145,
                "batch_size": 64,
            }
            param_config = {
                "hidden_dim":242,
                "dropout_p": 0.0,
                "learning_rate": 0.0013133403591351234,
                "num_epochs": 80,
                "batch_size": 256,
            }

            ### TinyImageNet
            param_config = {
                "hidden_dim":175,
                "dropout_p": 0.1,
                "learning_rate": 0.00026741326506786136,
                "num_epochs": 120,
                "batch_size": 64,
            }
        

        self.model = NNSurrogateModel(pX.shape, param_config["hidden_dim"], 10, out_dim, param_config["dropout_p"])
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.KLDivLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=param_config["learning_rate"])


        py = torch.tensor(labels, dtype=torch.float32)
        train_dataset = TensorDataset(pX, py)
        train_dataloader = DataLoader(train_dataset, batch_size=param_config["batch_size"])

        self.model = self.model.cuda()

        self.model.train()
        for epoch in range(param_config["num_epochs"]):
            running_loss = 0.0
            total_kt_loss = 0.0
            total_mse_loss = 0.0

            for i, (x, y) in enumerate(train_dataloader, 0):
                optimizer.zero_grad()
                outputs = self.model(x.cuda())
                
                # eval = utils.evaluate_learning_curve_metrics(y.cpu().detach().numpy(), outputs.cpu().detach().numpy(), prediction_is_first_arg=False)
                # KT_loss = 1 - eval["kendall_tau"]
                mse_loss = criterion(outputs, y.cuda())
                loss = mse_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # total_kt_loss += KT_loss.item()
                total_mse_loss += mse_loss.item()
                if not i % 500:
                    print(f'[{epoch+1}, {i+1}] loss: {running_loss / 1000:.5f} MSE loss: {total_mse_loss / 1000:.5f}' )
                    running_loss = 0.0

        self.model.eval()
        train_pred_final = []
        val_pred_final = []

        with torch.no_grad():
            val_pred =self.model(torch.tensor(X_val, dtype=torch.float32).cuda()).cpu()

        val_pred_50, val_pred_100, val_pred_200 = [], [], []
        y_val_50, y_val_100, y_val_200 = [], [], []

        for i in range(len(X_val)):
            pred = val_pred[i,:]
            if X_val[i][-3] == 1:
                length = 50 
                val_pred_50.append(get_points(pred, length, self.name))
                y_val_50.append(y_val[i])

            if X_val[i][-2] == 1:
                length = 100
                val_pred_100.append(get_points(pred, length, self.name))
                y_val_100.append(y_val[i])

            if X_val[i][-1] == 1:
                length = 200    
                val_pred_200.append(get_points(pred, length, self.name))
                y_val_200.append(y_val[i])

        valid_metrics = utils.evaluate_learning_curve_metrics_diff_epoch([y_val_50, y_val_100, y_val_200], [val_pred_50, val_pred_100, val_pred_200], prediction_is_first_arg=False)
        logging.info('valid metrics: %s', valid_metrics)

        return valid_metrics

    def test(self):
        X_test, y_test, _, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        # with open("/RegNet/checkpoint/sweeps/cifar/GT/sweep.json", "r") as f:
        with open("/RegNet/checkpoint/sweeps/tinyimagenet/GT/sweep.json", "r") as f:
            GT_data = json.load(f)

        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu()

        test_pred_50, test_pred_100, test_pred_200 = [], [], []
        y_test_50, y_test_100, y_test_200 = [], [], []
        y_test_final = y_test

        for i in range(len(X_test)):
            pred = test_pred[i,:]
            if X_test[i][-3] == 1:
                length = 50 
                test_pred_50.append(get_points(pred, length, self.name))
                y_test_50.append(y_test[i])

            if X_test[i][-2] == 1:
                length = 100
                test_pred_100.append(get_points(pred, length, self.name))
                y_test_100.append(y_test[i])

            if X_test[i][-1] == 1:
                length = 200    
                test_pred_200.append(get_points(pred, length, self.name))
                y_test_200.append(y_test[i])
                
        test_pred_final = np.array(test_pred)
        print(len(y_test_50), len(y_test_100), len(y_test_200))
        test_metrics = utils.evaluate_learning_curve_metrics_diff_epoch([y_test_50, y_test_100, y_test_200 ], [test_pred_50, test_pred_100, test_pred_200], prediction_is_first_arg=False)
        logging.info('test metrics %s', test_metrics)

        # data = [d for d in data if  not (((d["cfg"]['OPTIM']['BASE_LR'] == 1.0 or d["cfg"]['OPTIM']['BASE_LR']==0.5) and (d["cfg"]['OPTIM']['OPTIMIZER']=="adam" or d["cfg"]['OPTIM']['OPTIMIZER']=="adamw")) or ((d["cfg"]['OPTIM']['BASE_LR'] == 0.001 or d["cfg"]['OPTIM']['BASE_LR'] == 0.003) and d["cfg"]['OPTIM']['OPTIMIZER']=="sgd"))]
        GT_50, GT_100, GT_200 = [], [], []
        for i in range(len(GT_data)):
            a = np.array(GT_data[i]['test_ema_epoch']['top1_err'])
            if a.shape[0]==50:
                GT_50.append(a)
            if a.shape[0]==100:
                GT_100.append(a)
            if a.shape[0]==200:
                GT_200.append(a)


        print(len(GT_50), len(GT_100), len(GT_200))
        GT_metrics = utils.evaluate_learning_curve_metrics_diff_epoch([y_test_50, y_test_100, y_test_200], [GT_50, GT_100, GT_200], prediction_is_first_arg=False)

       
        logging.info('GT metrics %s', GT_metrics)

        return test_metrics


    def save(self):
        save_list = [self.model]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        model = joblib.load(model_path)
        self.model = model[0]

    def evaluate(self, result_paths):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        self.model.eval()
        with torch.no_grad():
            test_pred = self.ss.inverse_transform(self.model(torch.tensor(X_test, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        return test_metrics, test_pred, y_test

    def query(self, config_dict, search_space='darts', epoch=98, use_noise=False):
        if search_space == 'darts':
            config_space_instance = self.config_loader.query_config_dict(config_dict)
            X = config_space_instance.get_array().reshape(1, -1)
            idx = np.isnan(X)
            X[idx] = -1
            X = X.reshape(1, -1)
        else:
            X = np.array([config_dict])

        self.model.eval()
        with torch.no_grad():
            ypred = self.model(torch.tensor(X, dtype=torch.float32).cuda()).cpu()
        
        if X[0,-3] == 1:
            length = 50 
        elif X[0, -2] == 1:
            length = 100
        elif X[0, -1] == 1:
            length = 200    
        self.name = "Quartic"
        return get_points(ypred[0], length, self.name)
