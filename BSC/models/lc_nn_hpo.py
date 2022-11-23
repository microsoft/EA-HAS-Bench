import logging
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from BSC.utils import utils
from BSC.surrogate_model import SurrogateModel
from hyperopt import fmin, tpe, hp, partial

from BSC.bezier_vis import *
from BSC.utils.curvefunctions import all_models, model_defaults
from BSC.utils.curvemodel import MLCurveModel, MCMCCurveModel
                

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


class MLPBlock(nn.Module):
    def __init__(self, w_in, w_out, act="relu"):
        super().__init__()
        self.linear_1 = nn.Linear(w_in, w_out, bias=True)
        self.af = activation(act)
    def forward(self, x):
        return self.af(self.linear_1(x))


class NNSTARSurrogateModel(nn.Module):

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
        return self.allout(feats)


class LCNNHPOModel(SurrogateModel):
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

    def argsDict_tranform(self, argsDict, isPrint=False):
        argsDict["num_epochs"] = (argsDict["num_epochs"] + 1) * 5
        argsDict["hidden_dim"] = argsDict["hidden_dim"] + 32
        argsDict["learning_rate"] = np.exp(argsDict["learning_rate"])
        
        if isPrint:
            print(argsDict)
        else:
            pass

        return argsDict
    
    def nn_factory(self, argsDict):
        argsDict = self.argsDict_tranform(argsDict, isPrint=True)

        params = {'num_epochs': argsDict['num_epochs'], 
                 'hidden_dim':argsDict['hidden_dim'],
                'learning_rate': argsDict['learning_rate'],  
                'dropout_p': argsDict['dropout_p'],  
                'batch_size': argsDict['batch_size'],
                'bananas_enc': False,
                }

        out_dim = len(model_defaults[self.name].keys())

        model_nn = NNSTARSurrogateModel(self.pX.shape, params["hidden_dim"], 3, out_dim, params["dropout_p"])
        model_nn = model_nn.cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_nn.parameters(), lr=params["learning_rate"])
        
        train_dataset = TensorDataset(self.pX, self.py)
        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'])
        
        model_nn.train()
        for epoch in range(params["num_epochs"]):
            running_loss = 0.0
            for i, (x, y) in enumerate(train_dataloader, 0):
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                outputs = model_nn(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        return self.get_tranformer_score(model_nn)

    def get_tranformer_score(self, model_nn):
        model_nn.eval()
        with torch.no_grad():
            val_pred = model_nn(torch.tensor(self.X_val, dtype=torch.float32).cuda()).cpu().detach().numpy()

        val_pred_50, val_pred_100, val_pred_200 = [], [], []
        y_val_50, y_val_100, y_val_200 = [], [], []
        def func(x, a, b, c):
            return a*np.exp(b/x) + c
            
        for i in range(len(self.X_val)):
            pred = val_pred[i,:]
            func = all_models[self.name]
            if self.X_val[i][-3] == 1:
                length = 50 
                x = np.array([i+1 for i in range(length)])
                val_pred_50.append(func(x, *pred))
                y_val_50.append(self.y_val[i])

            if self.X_val[i][-2] == 1:
                length = 100
                x = np.array([i+1 for i in range(length)])
                val_pred_100.append(func(x, *pred))
                y_val_100.append(self.y_val[i])

            if self.X_val[i][-1] == 1:
                length = 200    
                x = np.array([i+1 for i in range(length)])
                val_pred_200.append(func(x, *pred))
                y_val_200.append(self.y_val[i])  

        valid_metrics = utils.evaluate_learning_curve_metrics_diff_epoch([y_val_50, y_val_100, y_val_200], [val_pred_50, val_pred_100, val_pred_200], prediction_is_first_arg=False)
        print(valid_metrics)
        return 1-valid_metrics["r2"]

    def train(self):
        self.X_train, self.y_train, _, _ = self.load_dataset(dataset_type='train', use_full_lc=True)
        self.X_val, self.y_val, _, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.name = "janoschek"

        labels = []
        for y in self.y_train:
            x_data = np.array([i + 1 for i in range(len(y))])
            y_data = np.array(y)

            m = MLCurveModel(function=all_models[self.name],
                                    default_vals=model_defaults[self.name],
                                    recency_weighting=True)
            successful = m.fit(x_data, y_data)
            if successful:
                label = m.ml_params[:-1]
            else:
                label = m.default_function_param_array()

            labels.append(label)
            
        self.py = torch.tensor(labels, dtype=torch.float32)
        self.pX = torch.tensor(self.X_train, dtype=torch.float32)
         

        # name = "Quartic"
        # name = "Quintic"

        space = {"num_epochs": hp.randint("num_epochs", 39),
                "hidden_dim": hp.randint("hidden_dim", 224),
                'learning_rate': hp.uniform('learning_rate', np.log(0.0001), np.log(0.1)),
                "dropout_p": hp.choice("dropout_p", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                "batch_size": hp.choice("batch_size", [64, 128, 256, 512, 1024]),
        }

        best = fmin(self.nn_factory, space, algo=tpe.suggest, max_evals=500)
        print('best :', best)
        print('best param after transform :')
        self.argsDict_tranform(best,isPrint=True)



    def test(self):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        self.model.eval()
        with torch.no_grad():
            test_pred = self.ss.inverse_transform(self.model(torch.tensor(X_test, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        self.model.eval()
        with torch.no_grad():
            val_pred = self.ss.inverse_transform(self.model(torch.tensor(X_val, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_val_final = y_val
        val_pred_final = np.array(val_pred)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', valid_metrics)

        return valid_metrics

    def save(self):
        save_list = [self.model, self.ss, self.svd_s, self.svd_vh, self.num_components]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        model, ss, svd_s, svd_vh, num_components = joblib.load(model_path)
        self.model = model
        self.ss = ss
        self.svd_s = svd_s
        self.svd_vh = svd_vh
        self.num_components = num_components

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
            ypred = self.ss.inverse_transform(self.model(torch.tensor(X, dtype=torch.float32)))\
                @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]
        return ypred[0]
