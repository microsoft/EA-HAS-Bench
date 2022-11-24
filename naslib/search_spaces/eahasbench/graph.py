# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import os
import pickle
from platform import architecture
from tkinter import W
import numpy as np
import random
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric

from naslib.search_spaces.nasbenchgreen.conversions import regnet_sampler, hpo_sampler
from naslib.search_spaces.nasbenchgreen.config import load_sampler
import naslib.search_spaces.nasbenchgreen.random as rand

from naslib.utils.utils import get_project_root


class RegNetSearchSpace(Graph):
    """
    Implementation of the Regnet + HPO search space.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True


    def __init__(self):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, 'NUM_CLASSES') else 10

        self.max_epoch = 199
        self.space_name = 'RegNet'
     
    def get_max_epochs(self):
        # Return the max number of epochs that can be queried
        return 199
    
    
class NasBenchGreenSearchSpace(RegNetSearchSpace):
    """
    Implementation of the nasbench 201 search space.
    It also has an interface to the tabular benchmark of nasbench 201.
    """

    QUERYABLE = True
    
    def __init__(self):
        super().__init__()
        self.space_name = 'nasbenchgreen'
        self.sampler = load_sampler()
        self.config_encoding = []

    def get_type(self):
        return 'nasbenchgreen'


    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture

        
        optim = self.config['OPTIM']
        train = self.config['TRAIN']
        arch = self.config['REGNET']

        neigh_config_list = []
        # keep HPO and adjust the Arch
        for i in range(10):
            neigh_arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}
            neigh_arch_str = regnet_sampler(self.sampler, neigh_arch_str)
            neigh_arch_str['TRAIN'] = train
            neigh_arch_str['OPTIM'] = optim
            nbr = NasBenchGreenSearchSpace()
            nbr.config = neigh_arch_str
            nbr_model = torch.nn.Module()
            nbr_model.arch = nbr
            neigh_config_list.append(nbr_model)

        # keep Arch and adjust the HPO
        for i in range(10):
            neigh_arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}
            neigh_arch_str = hpo_sampler(neigh_arch_str)
            neigh_arch_str['REGNET'] = arch
            nbr = NasBenchGreenSearchSpace()
            nbr.config = neigh_arch_str
            nbr_model = torch.nn.Module()
            nbr_model.arch = nbr
            neigh_config_list.append(nbr_model)

        random.shuffle(neigh_config_list)
        return neigh_config_list


    def mutate(self, parent, dataset_api=None):

        parent_config = parent.config
        self.config = copy.deepcopy(parent_config)

        # randomly chocie one dim form Arch and HPO
        arch_hpo_list = ['REGNET\DEPTH', 'REGNET\W0', 'REGNET\GROUP_W', 'REGNET\WM', 'REGNET\WA', 'OPTIM\BASE_LR', 'OPTIM\OPTIMIZER', 'OPTIM\LR_POLICY']
        arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}

        change = np.random.choice(arch_hpo_list)
        # print(self.config)
        if "REGNET" in change:
            arch_str = regnet_sampler(self.sampler, arch_str)
            self.config['REGNET'][change.split('\\')[-1]] = arch_str['REGNET'][change.split('\\')[-1]]
        else:
            arch_str = hpo_sampler(arch_str)
            self.config[change.split('\\')[0]][change.split('\\')[-1]] = arch_str[change.split('\\')[0]][change.split('\\')[-1]]

    
    def sample_random_architecture(self, dataset_api=None):
        arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}
        arch_str = regnet_sampler(self.sampler, arch_str)
        arch_str = hpo_sampler(arch_str)
        self.config = arch_str


    def sample_random_architecture_encoding(self, dataset_api=None):
        nbgreen_arch_dict = {
            'DEPTH': range(6, 16),
            'W0': range(48, 136, 8),
            'GROUP_W': [1, 2, 4, 8, 16, 24, 32],
            'lr': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0],
            'optim': ['sgd', 'adam', 'adamw'],
            'policy': ['cos', 'exp', 'lin'],
            'aug': [0, 16],
            'epoch': [50, 100, 200]

        }

        arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}
        arch_str = regnet_sampler(self.sampler, arch_str)
        arch_str = hpo_sampler(arch_str)
        self.config = arch_str


    def sample_random_architecture_fidelities(self, dataset_api=None, seed=0):
        # np.random.seed(seed)

        arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}
        arch_str = regnet_sampler(self.sampler, arch_str)
        arch_str = hpo_sampler(arch_str)
        arch_str["OPTIM"]["MAX_EPOCH"] = 200
        self.config = arch_str


    def model_based_sample_architecture(self, dataset_api=None, minimize_me=None, good_kde=None, good_kde_hpo=None, vartypes=None):
        """
        This will perform a model-based architecture sampling and update the edges in the
        naslib object accordingly.
        """
        num_samples = 128  # 128
        random_fraction = 0.33
        best = np.inf
        best_vector = None
        for i in range(num_samples):
            idx = np.random.randint(0, len(good_kde.data))
            datum = good_kde.data[idx]   
            hpo = good_kde_hpo.data[idx] 
            # print(datum)
            # print(good_kde.bw)
            vector = []
            # how to sample the vector
            for m, bw, t in zip(datum, good_kde.bw, vartypes[:5]):
                if np.random.rand() < (1 - bw): # 
                    vector.append(m)
                else:
                    sampler = load_sampler()
                    if t == 11:
                        depth = rand.uniform(*sampler.DEPTH, 1)
                        depth_norm = (depth-6)/(15-6)
                        vector.append(depth_norm)
                    elif t == 12:
                        w0 = rand.log_uniform(*sampler.W0, 8)
                        w0_norm = (w0-48)/(128-48)
                        vector.append(w0_norm)
                    elif t == 13:
                        wa = rand.log_uniform(*sampler.WA, 0.1)
                        wa_norm = (wa-8)/(32-8)
                        vector.append(wa_norm)
                    elif t ==14:
                        wm = rand.log_uniform(*sampler.WM, 0.001)
                        wm_norm = (wm-2.5)/(3-2.5)
                        vector.append(wm_norm)
                    elif t ==15:
                        gw = rand.power2_or_log_uniform(*sampler.GROUP_W, 8)
                        g_w_norm = (gw-1)/(32-1)
                        vector.append(g_w_norm)
            for m, bw, t in zip(hpo, good_kde_hpo.bw, vartypes[5:]):
                if np.random.rand() < (1 - bw):
                    vector.append(m)
                else:
                    vector.append(np.random.randint(t))

            val = minimize_me(vector[:5])
            val_hpo = minimize_me(vector[5:])
            if val + val_hpo < best:
                best = val + val_hpo
                best_vector = vector
        if best_vector is None or np.random.rand() < random_fraction:
            self.sample_random_architecture_fidelities(dataset_api=dataset_api)
        else:
            # print(best_vector)
            ### convert vector to config
            for i in range(len(best_vector)):
                if i >=5:
                    best_vector[i] = int(best_vector[i])
            print("best vector", best_vector)
            arch_str = {"REGNET": {},"OPTIM":{},"TRAIN":{}}
            depth = best_vector[0]*(15-6) + 6
            arch_str["REGNET"]["DEPTH"] = depth

            w0 = best_vector[1]*(128-48) + 48
            arch_str["REGNET"]["W0"] = w0

            wa = best_vector[2]*(32-8) + 8
            arch_str["REGNET"]["WA"] = wa

            wm = best_vector[3]*(3-2.5) + 2.5
            arch_str["REGNET"]["WM"] = wm

            g_w = best_vector[4]*(32-1) + 1
            arch_str["REGNET"]["GROUP_W"] = g_w

            lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
            arch_str["OPTIM"]["BASE_LR"] = lr_list[best_vector[5]]

            optim_list = ['sgd', 'adam', 'adamw']
            arch_str["OPTIM"]["OPTIMIZER"] = optim_list[best_vector[6]]

            policy_list = ['cos', 'exp', 'lin']
            arch_str["OPTIM"]["LR_POLICY"] = policy_list[best_vector[7]]

            aug_list = [0, 16]
            arch_str["TRAIN"]["CUTOUT_LENGTH"] = aug_list[best_vector[8]]

            epoch_list = [50, 100, 200]
            arch_str["OPTIM"]["MAX_EPOCH"] = 200
                
            print('best config:', arch_str)
            self.config = arch_str

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if metric != Metric.RAW and metric != Metric.ALL:
            assert dataset in ['cifar10', 'tiny'], "Only cifar10  and tiny currently supported for EA-HAS-Bench: {}".format(dataset)
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query nasbench211')
                
        metric_to_nbgreen = {
            Metric.TRAIN_LOSS: 'train_losses',
            Metric.VAL_ACCURACY: 'val_accuracies',
            Metric.TRAIN_TIME: 'runtime'
        }

        # return right input
        arch_str = self.config
        
        if metric == Metric.RAW:
            # return all data
            return 0

        if dataset in ['cifar10', 'cifar10-valid']:
            # query_results = dataset_api['full_lc_data'][arch_str]
            # set correct cifar10 dataset
            dataset = 'cifar10-valid'
        elif dataset == 'cifar100':
            query_results = dataset_api['full_lc_data'][arch_str]
        elif dataset == 'ImageNet16-120':
            query_results = dataset_api['full_lc_data'][arch_str]
        elif dataset == 'tiny':
            dataset = 'tiny'
        else:
            raise NotImplementedError('Invalid dataset')

        if metric == Metric.HP:
            # return hyperparameter info
            return query_results[dataset]['cost_info']
        elif metric == Metric.TRAIN_TIME:
            # return query_results[dataset]['cost_info']['train_time'] * epoch
                if epoch == -1:
                    return int(dataset_api['nbgreen_model'][1].predict(config=arch_str, representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False))
                else:
                    # replace the runtime with erengy
                    # return int(dataset_api['nbgreen_model'][2].predict(config=arch_str, 
                    #                             representation="arch_str", 
                    #                             search_space='nbgreen_c10', 
                    #                             with_noise=False)*epoch)

                    # return the train time of given arch+hpo
                    return int(dataset_api['nbgreen_model'][1].predict(config=arch_str, representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False) * epoch)
        elif metric == Metric.TRAIN_COST:
            if dataset == 'cifar10':
                if epoch == -1:
                    return float(dataset_api['nbgreen_model'][2].predict(config=arch_str, 
                                                    representation="arch_str", 
                                                    search_space='nbgreen_c10', 
                                                    with_noise=False)/100)
                else:
                    return float(dataset_api['nbgreen_model'][2].predict(config=arch_str, 
                                                    representation="arch_str", 
                                                    search_space='nbgreen_c10', 
                                                    with_noise=False)*epoch/100)
            elif dataset == 'tiny':
                if epoch == -1:
                    return float(dataset_api['nbgreen_model'][2].predict(config=arch_str, 
                                                    representation="arch_str", 
                                                    search_space='nbgreen_c10', 
                                                    with_noise=False))
                else:
                    return float(dataset_api['nbgreen_model'][2].predict(config=arch_str, 
                                                    representation="arch_str", 
                                                    search_space='nbgreen_c10', 
                                                    with_noise=False)*epoch)
        elif metric == Metric.TEST_COST:
            return float(dataset_api['nbgreen_model'][3].predict(config=arch_str, 
                                                representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False))
        elif metric == Metric.G_ACC:
            lc = 100 - dataset_api['nbgreen_model'][0].predict(config=arch_str, 
                                                representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False)
            return lc[-1]


        lc = dataset_api['nbgreen_model'][0].predict(config=arch_str, 
                                                representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False)
        lc = 100 - lc


        # Find the arch and hpo with the best Green
        Green = False
        if Green:
            ACC_ = 90
            EC_ = 20000000
            
            train_E = dataset_api['nbgreen_model'][2].predict(config=arch_str, 
                                                representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False)
            test_E = dataset_api['nbgreen_model'][3].predict(config=arch_str, 
                                                representation="arch_str", 
                                                search_space='nbgreen_c10', 
                                                with_noise=False)
            
            if full_lc and epoch == -1:
                raise NotImplementedError()

            elif full_lc and epoch != -1:
                G_list = []
                for i in range(epoch):
                    EC =  train_E * i + 5e8*test_E
                    G = (lc[i]-ACC_)*((EC_/EC)**2)[0][0]
                    G_list.append(G)
                return G_list
            else:
                # return the value of the metric only at the specified epoch
                ACC = lc[epoch]
                EC = train_E *epoch + 5e8*test_E
                return (ACC-ACC_)*((EC_/EC)**2)[0][0]
        
        else:
            
            if full_lc and epoch == -1:
                return lc
            elif full_lc and epoch != -1:
                return lc[:epoch]
            else:
                # return the value of the metric only at the specified epoch
                return lc[epoch]
