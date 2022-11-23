import numpy as np

from BSC.encodings.encodings_nb101 import encode_nb101
from BSC.encodings.encodings_nb201 import encode_nb201
from BSC.encodings.encodings_nlp import encode_nlp
from BSC.encodings.encodings_regnet import encode_regnet


def encode(arch_strings, data, search_space, nlp_max_nodes, nb101_api, cost_data=None):
    E_cost = None
    if search_space == 'nb201':
        x_enc = [encode_nb201(arch_str) for arch_str in arch_strings]
        y = [np.array(data[arch_str]['cifar10-valid']['eval_acc1es']) for arch_str in arch_strings]
        
        print(len(x_enc))
        print(x_enc[0])
        print(len(y))
        print(y[0])

    elif search_space == 'regnet':
        x_enc = [encode_regnet(arch_str) for arch_str in arch_strings]

        # print(x_enc[0])
        # print(data[0]['test_ema_epoch']['top1_err'])
        # 'cfg', 'complexity', 'epoch_times', 'err', 'err_file', 'iter_times', 'log_file', 'test_ema_epoch', 'test_epoch', 'train_epoch'
        
        # padding all epoch to same size
       
        # y = [np.array(data[i]['test_ema_epoch']['top1_err']) for i in range(len(arch_strings))]
        y = []
        for i in range(len(arch_strings)):
            a = np.array(data[i]['test_ema_epoch']['top1_err'])
            epoch = data[i]['cfg']["OPTIM"]["MAX_EPOCH"]
            # clip to 50
            # a = np.where(a>50, 50, a)
            epoch = 200
            if a.shape[0]<epoch:
                a = np.pad(a, (0, epoch-a.shape[0]), 'constant', constant_values=a[-1]) 
            if a.shape[0]>epoch:
                a = a[:epoch]
            # a = np.array([(100-i)/100 for i in a])
            # print(a.shape)
            y.append(a)
            # y.append((100-a)/100)
        
        E_cost = [np.array([np.mean(d["cons"]["train_cons"][1:])]) for d in cost_data]  # 
        # E_cost = [np.array([np.mean(d["cons"]["test_cons"])]) for d in cost_data]  # 
        train_time = [np.array([np.mean(d['train_epoch']['time_epoch'][1:])]) for d in cost_data]
    
        # print(y[0].shape)
        # print(data[0]['test_ema_epoch'])
        # y = []
        # for i in range(len(arch_strings)):
        #     if data[i]['test_ema_epoch']['epoch_max'][0]!=108:
        #         print(data[i]['test_ema_epoch']['epoch_max'][0])
        #         expand_list = [data[i]['test_ema_epoch']['top1_err'][-1] for k in range(int(108 - data[i]['test_ema_epoch']['epoch_max'][0]))]
        #         print(expand_list)
        #         y.append(np.array(data[i]['test_ema_epoch']['top1_err'].extend(expand_list)))
        #     else:
        #         y.append(np.array(data[i]['test_ema_epoch']['top1_err']))

        print(len(x_enc))
        print(x_enc[0])
        # print(x_enc[998])
        # print(x_enc[999])
        # print(x_enc[1000])
        # for i in range(len(y)):
        #     print(y[i].shape)
        #     print(i)

    elif search_space == 'nlp':
        x_enc = []
        epoch = 3
        for arch_str in arch_strings:
            lc_acc = np.array([100.0 - loss for loss in data[arch_str]['val_losses']])
            accs = lc_acc[:epoch]
            enc = encode_nlp(compact=arch_str, max_nodes=nlp_max_nodes, accs=accs, one_hot=False, lc_feature=True,
                             only_accs=False)
            x_enc.append(enc)
        y = []
        for arch_str in arch_strings:
            lc_acc = np.array([100.0 - loss for loss in data[arch_str]['val_losses']])
            y.append(lc_acc)
            
    elif search_space == 'nb101':
        x_enc = [encode_nb101(nb101_api, arch_str=arch_str, lc_feature=True, only_accs=False) for arch_str in arch_strings]
        y = [np.array(data[arch_str]) for arch_str in arch_strings]

    return x_enc, y, E_cost, train_time
