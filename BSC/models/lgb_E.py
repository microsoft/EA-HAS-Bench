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
import lightgbm as lgb

from BSC.utils import utils
from BSC.surrogate_model import SurrogateModel


class LGBEModel(SurrogateModel):
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

        pX = torch.tensor(X_train, dtype=torch.float32)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.model = RegressorChain(lgb.LGBMRegressor(
            num_iterations=param_config["num_rounds"],
            boosting_type=param_config["boosting_type"],
            num_leaves=param_config["num_leaves"],
            max_depth=param_config["max_depth"],
            learning_rate=param_config["learning_rate"],
            min_child_weight=param_config["min_child_weight"],
            reg_alpha=param_config["lambda_l1"],
            reg_lambda=param_config["lambda_l2"],
        ))
        # X_train = [x[:5] for x in X_train]
        # Train a runtime surrogate_model
        self.model.fit(X_train, E_train, verbose=True)

        train_pred = self.model.predict(X_train)
        # X_val = [x[:5] for x in X_val]
        val_pred = self.model.predict(X_val)

        # metrics for final prediction
        train_pred_final = np.array(train_pred)
        val_pred_final = np.array(val_pred)
        y_train_final = E_train
        y_val_final = E_val

        train_metrics = utils.evaluate_learning_curve_metrics(y_train_final, train_pred_final, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('train metrics: %s', train_metrics)
        logging.info('valid metrics: %s', valid_metrics)

        return valid_metrics

    def test(self):
        X_test, y_test, E_test, T_test = self.load_dataset(dataset_type='test', use_full_lc=True)
        # X_test = [x[:5] for x in X_test]
        test_pred = self.model.predict(X_test)

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

        pred = self.model.predict(X)
        pred_final = np.array(pred)
        return pred_final
