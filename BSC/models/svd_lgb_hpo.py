import logging
import os
import joblib
import numpy as np
import lightgbm as lgb

from hyperopt import fmin, tpe, hp, partial
import warnings
warnings.filterwarnings("ignore")

from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from nas_bench_x11.utils import utils
from nas_bench_x11.surrogate_model import SurrogateModel


class SVDLGBHPOModel(SurrogateModel):
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
        argsDict["max_depth"] = argsDict["max_depth"] + 1
        argsDict["num_leaves"] = argsDict["num_leaves"] + 10
        argsDict["learning_rate"] = np.exp(argsDict["learning_rate"])
        argsDict["min_child_weight"] = np.exp(argsDict["min_child_weight"])
        argsDict["l1"] = np.exp(argsDict["l1"])
        argsDict["l2"] = np.exp(argsDict["l2"])
        argsDict["num_components"] = argsDict["num_components"] + 1
        if isPrint:
            print(argsDict)
        else:
            pass

        return argsDict

    def lightgbm_factory(self, argsDict):
        argsDict = self.argsDict_tranform(argsDict, isPrint=True)

        params = {'max_depth': argsDict['max_depth'],  # 最大深度
                 'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
                 'learning_rate':argsDict['learning_rate'],
                'lambda_l1': argsDict['l1'],  
                'lambda_l2': argsDict['l2'],  
                'min_child_weight': argsDict['min_child_weight'],
                'num_components': argsDict["num_components"],
                # 'learning_rate': 0.02182249761978233,  # argsDict['learning_rate'],  # 学习率
                # 'lambda_l1': 0.011489772418204284, #  argsDict['l1'],  
                # 'lambda_l2': 134.50743557010372,       # argsDict['l2'],  
                # 'min_child_weight': 0.582192788716,       #argsDict['min_child_weight'],
                'num_rounds': 5000,
                'boosting_type': "gbdt"
                }
       
        # params = {'max_depth': 9,  # 最大深度
        #          'num_leaves': 84,  # 终点节点最小样本占比的和
        #          'learning_rate': 0.0976572239888193,
        #         'lambda_l1': 0.005588465041599705,  
        #         'lambda_l2': 0.005420874153751947,  
        #         'min_child_weight': 0.462227810054499,
        #         'num_components': 4,
        #         # 'learning_rate': 0.02182249761978233,  # argsDict['learning_rate'],  # 学习率
        #         # 'lambda_l1': 0.011489772418204284, #  argsDict['l1'],  
        #         # 'lambda_l2': 134.50743557010372,       # argsDict['l2'],  
        #         # 'min_child_weight': 0.582192788716,       #argsDict['min_child_weight'],
        #         'num_rounds': 3000,
        #         'boosting_type': "gbdt"
        #         }

        model_lgb = RegressorChain(lgb.LGBMRegressor(
            num_iterations=params["num_rounds"],
            boosting_type=params["boosting_type"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["lambda_l1"],
            reg_lambda=params["lambda_l2"],
        ))
        labels = self.svd_u[:, :params["num_components"]].copy()
        fitted_labels = self.ss.fit_transform(labels)

        # model = model_lgb.fit(self.X_train, fitted_labels, verbose=True)
        self.X_train = [x[:5] for x in self.X_train]
        model = model_lgb.fit(self.X_train, self.T_train, verbose=True)

        return self.get_tranformer_score(model, params["num_components"])

    def get_tranformer_score(self, model, num_components):
        # val_pred = self.ss.inverse_transform(model.predict(self.X_val))\
        #     @ np.diag(self.svd_s[:num_components])@self.svd_vh[:num_components, :]
        self.X_val = [x[:5] for x in self.X_val]
        val_pred = model.predict(self.X_val)
        val_pred_final = np.array(val_pred)

        # valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_learning_curve_metrics(self.T_val, val_pred_final, prediction_is_first_arg=False)
        print(valid_metrics)
        return 1-valid_metrics["r2"]

    def train(self):
        # matrices (e.g. X) are capitalized, vectors (e.g. y) are uncapitalized
        self.X_train, self.y_train, self.E_train, self.T_train = self.load_dataset(dataset_type='train', use_full_lc=True)
        self.X_val, self.y_val, self.E_val, self.T_val = self.load_dataset(dataset_type='val', use_full_lc=True)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.ss = StandardScaler()

        u, s, vh = np.linalg.svd(self.y_train, full_matrices=False)

        self.svd_u = u
        self.svd_s = s
        self.svd_vh = vh

        space = {"max_depth": hp.randint("max_depth", 100), # 24
                "num_leaves": hp.randint("num_leaves", 1000), # 90
                "l1": hp.uniform("l1", np.log(0.001), np.log(1000)),
                "l2": hp.uniform("l2", np.log(0.001), np.log(1000)),
                'learning_rate': hp.uniform('learning_rate', np.log(0.001), np.log(0.1)),
                "min_child_weight": hp.uniform("min_child_weight", np.log(0.001), np.log(10)),
                'num_components': hp.randint("num_components", 19)
        }

        # the labels are the first n components of the SVD on the training data
        
        best = fmin(self.lightgbm_factory, space, algo=tpe.suggest, max_evals=500)
        print('best :', best)
        print('best param after transform :')
        self.argsDict_tranform(best,isPrint=True)

        # self.model.fit(self.X_train, self.fitted_labels, verbose=True)

        # train_pred = self.ss.inverse_transform(self.model.predict(X_train))\
        #     @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        # val_pred = self.ss.inverse_transform(self.model.predict(X_val))\
        #     @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        # residuals = u[:, :self.num_components]@np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :] - np.stack(y_train)
        # self.kernel = gaussian_kde(residuals.T+np.random.randn(*residuals.T.shape)*1e-8)

        # # metrics for final prediction
        # train_pred_final = np.array(train_pred)
        # val_pred_final = np.array(val_pred)
        # y_train_final = y_train
        # y_val_final = y_val

        # train_metrics = utils.evaluate_learning_curve_metrics(y_train_final, train_pred_final, prediction_is_first_arg=False)
        # valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        # logging.info('train metrics: %s', train_metrics)
        # logging.info('valid metrics: %s', valid_metrics)
        # return valid_metrics

    def test(self):
        X_test, y_test, E_test, T_test = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.ss.inverse_transform(self.model.predict(X_test))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val, _ = self.load_dataset(dataset_type='val', use_full_lc=True)

        val_pred = self.ss.inverse_transform(self.model.predict(X_val))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_val_final = y_val
        val_pred_final = np.array(val_pred)
        valid_metrics = utils.evaluate_learning_curve_metrics(y_val_final, val_pred_final, prediction_is_first_arg=False)

        logging.info('test metrics %s', valid_metrics)

        return valid_metrics

    def save(self):
        save_list = [self.model, self.ss, self.svd_s, self.svd_vh, self.num_components, self.kernel]
        joblib.dump(save_list, os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        if len(joblib.load(model_path)) == 5:
            # load without noise model
            model, ss, svd_s, svd_vh, num_components = joblib.load(model_path)
            self.model = model
            self.ss = ss
            self.svd_s = svd_s
            self.svd_vh = svd_vh
            self.num_components = num_components

        else:
            # load with noise model
            logging.info('loading model with noise kernel')
            model, ss, svd_s, svd_vh, num_components, kernel = joblib.load(model_path)
            self.model = model
            self.ss = ss
            self.svd_s = svd_s
            self.svd_vh = svd_vh
            self.num_components = num_components
            self.kernel = kernel

    def evaluate(self):
        X_test, y_test, _ = self.load_dataset(dataset_type='test', use_full_lc=True)

        test_pred = self.ss.inverse_transform(self.model.predict(X_test))\
            @ np.diag(self.svd_s[:self.num_components])@self.svd_vh[:self.num_components, :]

        y_test_final = y_test
        test_pred_final = np.array(test_pred)
        test_metrics = utils.evaluate_learning_curve_metrics(y_test_final, test_pred_final, prediction_is_first_arg=False)

        return test_metrics, test_pred, y_test

    def query(self, config_dict, search_space='darts', use_noise=False, components=False):
        if search_space == 'darts':
            config_space_instance = self.config_loader.query_config_dict(config_dict)
            X = config_space_instance.get_array().reshape(1, -1)
            idx = np.isnan(X)
            X[idx] = -1
            X = X.reshape(1, -1)

        else:
            X = np.array([config_dict])

        comp = self.model.predict(X)        
        if components:
            return self.ss.inverse_transform(comp)

        ypred = self.ss.inverse_transform(comp) @ np.diag(self.svd_s[:self.num_components])\
            @ self.svd_vh[:self.num_components, :]

        if use_noise:
            noise = np.squeeze(self.kernel.resample(1))
            noise = np.clip(noise, -3, 3)  # clip outliers
            return ypred[0] + noise / 2

        return ypred[0]