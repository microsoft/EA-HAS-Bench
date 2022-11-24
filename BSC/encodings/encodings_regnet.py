# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from sklearn.preprocessing import StandardScaler, OneHotEncoder
import math
import numpy as np


def encode_regnet(arch_str, output_onehot=True, dataset="cifar"):
    # print(arch_str["REGNET"])
    # print(arch_str["OPTIM"])
    # print(arch_str["TRAIN"])
    # turn arch_str into encoding
    if type(arch_str) == dict:
    # if arch_str:
        arch_dict = {
            'DEPTH': range(6, 16),
            'W0': range(48, 136, 8),
            'GROUP_W': [1, 2, 4, 8, 16, 24, 32]
        }
        
        arch_encoding = []
        # encoding depth, w0 and group_w to one hot
        # for key in arch_dict.keys():
        #     data = arch_str["REGNET"][key]
        #     data_list = arch_dict[key]
        #     one_hot = [0 for i in range(len(data_list))]
        #     index = data_list.index(data)
        #     one_hot[index] = 1
        #     arch_encoding.extend(one_hot)

        # log and MinMax nomalization
        
        depth = arch_str["REGNET"]["DEPTH"]
        depth_norm = (depth-6)/(15-6)
        arch_encoding.append(depth_norm)

        w0 = arch_str["REGNET"]["W0"]
        w0_norm = (w0-48)/(128-48)
        arch_encoding.append(w0_norm)

        wa = arch_str["REGNET"]["WA"]
        wa_norm = (wa-8)/(32-8)
        arch_encoding.append(wa_norm)

        if dataset == "cifar":
            wm = arch_str["REGNET"]["WM"]
            wm_norm = (wm-2.5)/(3-2.5)
            arch_encoding.append(wm_norm)
        else:
            wm = arch_str["REGNET"]["WM"]
            wm_norm = (wm-1.5)/(3-1.5)
            arch_encoding.append(wm_norm)

        g_w = arch_str["REGNET"]["GROUP_W"]
        g_w_norm = (g_w-1)/(32-1)
        arch_encoding.append(g_w_norm)

        # arch_encoding = [arch_str["REGNET"]["DEPTH"], arch_str["REGNET"]["W0"], arch_str["REGNET"]["WA"], arch_str["REGNET"]["WM"], arch_str["REGNET"]["GROUP_W"]]
        # print(arch_encoding)

        # return arch_encoding

        # print(len(arch_encoding))
        # print(arch_encoding)

        # one hot encoding
        hyperp_encoding = []

        # combine lr with optim
        lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
        # lr_list = [0.3, 0.5, 1.0]
        optim_list = ['sgd', 'adam', 'adamw']
        lr = arch_str["OPTIM"]["BASE_LR"]
        optim = arch_str["OPTIM"]["OPTIMIZER"]

        all_list = []
        for i in range(len(lr_list)):
            for j in range(len(optim_list)):
                lr_optim = str(lr_list[i])+optim_list[j]
                all_list.append(lr_optim)

        # combine_one_hot = [0 for i in range(len(all_list))]
        # index = all_list.index(str(lr)+optim)
        # combine_one_hot[index] = 1
        # hyperp_encoding.extend(combine_one_hot)
        
        lr_one_hot = [0 for i in range(len(lr_list))]
        index = lr_list.index(lr)
        if output_onehot: 
            lr_one_hot[index] = 1
            hyperp_encoding.extend(lr_one_hot)
        else:
            hyperp_encoding.append(index)

        optim_one_hot = [0 for i in range(len(optim_list))]
        index = optim_list.index(optim)
        if output_onehot:
            optim_one_hot[index] = 1
            hyperp_encoding.extend(optim_one_hot)
        else:   
            hyperp_encoding.append(index)


        lr_policy = arch_str["OPTIM"]["LR_POLICY"]
        policy_list = ['cos', 'exp', 'lin']
        policy_one_hot = [0 for i in range(len(policy_list))]
        index = policy_list.index(lr_policy)
        if output_onehot:
            policy_one_hot[index] = 1
            hyperp_encoding.extend(policy_one_hot)
        else:
            hyperp_encoding.append(index)

        augent = arch_str["TRAIN"]["CUTOUT_LENGTH"]
        aug_list = [0, 16]
        aug_one_hot = [0 for i in range(len(aug_list))]
        index = aug_list.index(augent)
        if output_onehot:
            aug_one_hot[index] = 1
            hyperp_encoding.extend(aug_one_hot)
        else:
            hyperp_encoding.append(index)

        if "MAX_EPOCH" in arch_str["OPTIM"].keys():
            max_epoch = arch_str["OPTIM"]["MAX_EPOCH"]
        else:
            max_epoch = 200
        epoch_list = [50, 100, 200] # [4, 12, 36, 108]
        epoch_one_hot = [0 for i in range(len(epoch_list))]
        index = epoch_list.index(max_epoch) 
        if output_onehot:
            epoch_one_hot[index] = 1
            hyperp_encoding.extend(epoch_one_hot)
        else:
            hyperp_encoding.append(index)

        # arch_encoding = [arch_str["OPTIM"]["BASE_LR"], arch_str["OPTIM"]["OPTIMIZER"], arch_str["REGNET"]["WA"], arch_str["REGNET"]["WM"], arch_str["REGNET"]["GROUP_W"]]
        
        # print(hyperp_encoding)
        # print(arch_encoding + hyperp_encoding)
        # assert len(arch_encoding + hyperp_encoding)==10, "the dim of x_encoding is not equal 10"
        # return hyperp_encoding
        return arch_encoding + hyperp_encoding

    else:
        ### round the discontinuous parts
        hyperp_encoding = []
        config = arch_str

        lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
        lr_one_hot = [0 for i in range(len(lr_list))]
        index = int(np.rint(config[5]))
        lr_one_hot[index] = 1
        hyperp_encoding.extend(lr_one_hot)

        optim_list = ['sgd', 'adam', 'adamw']
        optim_one_hot = [0 for i in range(len(optim_list))]
        index = int(np.rint(config[6]))
        optim_one_hot[index] = 1
        hyperp_encoding.extend(optim_one_hot)

        policy_list = ['cos', 'exp', 'lin']
        policy_one_hot = [0 for i in range(len(policy_list))]
        index = int(np.rint(config[7]))
        policy_one_hot[index] = 1
        hyperp_encoding.extend(policy_one_hot)
        
        aug_list = [0, 16]
        aug_one_hot = [0 for i in range(len(aug_list))]
        index = int(np.rint(config[8]))
        aug_one_hot[index] = 1
        hyperp_encoding.extend(aug_one_hot)

        epoch_list = [50, 100, 200] 
        epoch_one_hot = [0 for i in range(len(epoch_list))]
        index = int(np.rint(config[9]))
        epoch_one_hot[index] = 1
        hyperp_encoding.extend(epoch_one_hot)

        return list(config[:5]) + hyperp_encoding
