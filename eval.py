import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SSL.models.ssl_proto import SSL_boost 
from SSL.dataloader.samplers import CategoriesSampler
from SSL.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval
from tensorboardX import SummaryWriter
from collections import OrderedDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5)    
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embed_size', type=int, default=512)


    parser.add_argument('--test_episodes', type=int, default=2000)
    parser.add_argument('--model_type', type=str, default='ResNet12', choices=['ConvNet', 'ResNet12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet'])    
    parser.add_argument('--model_path', type=str) # need 

    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--head', type=int, default=10)   

    args = parser.parse_args()
    pprint(vars(args))
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    set_gpu(args.gpu)
    if args.dataset == 'MiniImageNet':
        from SSL.dataloader.mini_imagenet import MiniImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')
    
    model = SSL_boost(args, dropout = args.dropout)    

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        model = model.cuda()

    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episodes, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((args.test_episodes,))

    model.load_state_dict(torch.load(args.model_path)['params'])
    model.eval()

    ave_acc = Averager()
    tasks_acc = [Averager() for i in range(model.trans_num)]
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
            
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
            rot_loss, MI_loss, fsl_loss, final_loss, acc_list = model(data_shot, data_query)

            ave_acc.add(acc_list[-1])
            for j in range(model.trans_num):
                tasks_acc[j].add(acc_list[j])

            test_acc_record[i-1] = acc_list[-1]
            #print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc_list[-1] * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    for i in range(model.trans_num):
        print('Rotation {} acc is {:.4f}'.format(90*i, tasks_acc[i].item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

    ave_acc = np.array(test_acc_record).mean() * 100 
    acc_std = np.array(test_acc_record).std() * 100
    ci95 = 1.96 * np.array(test_acc_record).std() / np.sqrt(float(len(np.array(test_acc_record)))) * 100

    print('evaluation: accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%'%(ave_acc, acc_std, ci95))
