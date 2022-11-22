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
from sklearn.multioutput import RegressorChain

from nas_bench_x11.utils import utils
from nas_bench_x11.surrogate_model import SurrogateModel
from hyperopt import fmin, tpe, hp, partial


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
        self.arch_encoding = MLPBlock(self.arch_input_dim, 1) # hidden_dim
        self.reg_encoding = MLPBlock(2, hidden_dim)
        self.combine = MLPBlock(30, 64)
        self.hyp_encoding = MLPBlock(input_shape[1]-arch_input_dim, hidden_dim)

        hidden_dim = 2 * hidden_dim

        self.fclayers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.bnlayers = [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        self.droplayers = [nn.Dropout(dropout_p) for _ in range(num_layers)]
        self.outlayer = nn.Linear(hidden_dim, out_dim)
        self.sig = nn.Sigmoid()

        self.dvn = nn.Sequential(
            # nn.Linear(hidden_dim, int(hidden_dim/2)),
            # nn.ReLU(inplace=True),
            # nn.Linear(int(hidden_dim/2), int(hidden_dim/2)),
            # nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p))

    def forward(self, x):
        arch_input = x[:, :self.arch_input_dim]
        # hyp_input = x[:, self.arch_input_dim:]
        # # combine_input = x[:, 5:35]


        arch_embding = self.arch_encoding(arch_input)
        # # combine_embding = self.combine(combine_input)
        # hyp_embding = self.hyp_encoding(hyp_input)

        # x = torch.cat([arch_embding, hyp_embding], axis=1)

        # for fc, bn, drop in zip(self.fclayers, self.bnlayers, self.droplayers):
        #     x = F.relu(fc(x))
        #     x = drop(x)
        # return self.outlayer(x)
        return self.sig(arch_embding)*10


class NNSModel(SurrogateModel):
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
        argsDict["num_layers"] = argsDict["num_layers"] + 1
        argsDict["learning_rate"] = np.exp(argsDict["learning_rate"])
        
        if isPrint:
            print(argsDict)
        else:
            pass

        return argsDict
    
    def nn_factory(self, argsDict):
        argsDict = self.argsDict_tranform(argsDict)

        params = {'num_epochs': argsDict['num_epochs'], 
                 'hidden_dim':argsDict['hidden_dim'],
                'num_layers': argsDict['num_layers'],  
                'learning_rate': argsDict['learning_rate'],  
                'dropout_p': argsDict['dropout_p'],  
                'batch_size': argsDict['batch_size'],
                'bananas_enc': False,
                }

        # labels = self.svd_u[:, :params["num_components"]].copy()
        # fitted_labels = self.ss.fit_transform(labels)
        pX = torch.tensor(self.X_train, dtype=torch.float32)
        py = torch.tensor(self.E_train, dtype=torch.float32)

        model_nn = NNSurrogateModel(pX.shape, params["hidden_dim"], params["num_layers"], 1, params["dropout_p"])
        model_nn = model_nn.cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_nn.parameters(), lr=params["learning_rate"])
        
        train_dataset = TensorDataset(pX, py)
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

    def get_tranformer_score(self, model):
        model.eval()
        with torch.no_grad():
            val_pred =model(torch.tensor(self.X_val, dtype=torch.float32).cuda()).cpu()

        val_pred_final = np.array(val_pred)
        y_val_final = self.E_val

        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)
        print(valid_metrics)
        return 1-valid_metrics["r2"]  # kendall_tau

    def train(self):

        self.X_train, y_train, self.E_train, self.T_train = self.load_dataset(dataset_type='train', use_full_lc=True)
        self.X_val, y_val, self.E_val, self.T_val = self.load_dataset(dataset_type='val', use_full_lc=True)

        # pX = torch.tensor(X_train, dtype=torch.float32)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        # self.model = NNSurrogateModel(pX.shape, param_config["hidden_dim"], param_config["num_layers"], 1, param_config["dropout_p"])
        # criterion = nn.MSELoss()
        # # criterion = nn.CrossEntropyLoss()
        # # criterion = nn.KLDivLoss()
        # optimizer = optim.Adam(self.model.parameters(), lr=param_config["learning_rate"])

        # py = torch.tensor(E_train, dtype=torch.float32)
        # train_dataset = TensorDataset(pX, py)
        # train_dataloader = DataLoader(train_dataset, batch_size=256)

        # self.model = self.model.cuda()

        # self.model.train()
        # for epoch in range(param_config["num_epochs"]):
        #     running_loss = 0.0
        #     for i, (x, y) in enumerate(train_dataloader, 0):
        #         optimizer.zero_grad()
        #         outputs = self.model(x.cuda())
        #         loss = criterion(outputs, y.cuda())
        #         loss.backward()
        #         optimizer.step()

        #         running_loss += loss.item()
        #         if not i % 500:
        #             print(f'[{epoch+1}, {i+1}] loss: {running_loss / 1000:.8f}')
        #             running_loss = 0.0

        space = {"num_epochs": hp.randint("num_epochs", 39),
                "hidden_dim": hp.randint("hidden_dim", 224),
                "num_layers": hp.randint("num_layers", 19),
                'learning_rate': hp.uniform('learning_rate', np.log(0.0001), np.log(0.1)),
                "dropout_p": hp.choice("dropout_p", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                "batch_size": hp.choice("batch_size", [64, 128, 256]),
        }

        best = fmin(self.nn_factory, space, algo=tpe.suggest, max_evals=500)
        print('best :', best)
        print('best param after transform :')
        self.argsDict_tranform(best,isPrint=True)

        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(torch.tensor(self.X_train, dtype=torch.float32).cuda()).cpu()
            val_pred = self.model(torch.tensor(self.X_val, dtype=torch.float32).cuda()).cpu()

        # metrics for final prediction
        train_pred_final = np.array(train_pred)
        val_pred_final = np.array(val_pred)
        y_train_final = self.E_train
        y_val_final = self.E_val

        train_metrics = utils.evaluate_learning_curve_metrics(y_train_final, train_pred_final, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('train metrics: %s', train_metrics)
        logging.info('valid metrics: %s', valid_metrics)

        return valid_metrics

    def test(self):
        X_test, y_test, E_test, T_test = self.load_dataset(dataset_type='test', use_full_lc=True)

        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu()

        test_pred_final = np.array(test_pred)
        y_test_final = E_test
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)
        # logging.info('GT metrics %s', GT_metrics)

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
        joblib.dump(self.model, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        model = joblib.load(model_path)
        self.model = model

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
