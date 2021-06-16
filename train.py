import argparse
import os.path as osp
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from SSL.models.ssl_proto import SSL_boost 
from SSL.dataloader.samplers import CategoriesSampler
from SSL.utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval, cfg, cfg_from_yaml_file
from tensorboardX import SummaryWriter
from collections import OrderedDict

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5)       
    parser.add_argument('--shot', type=int, default=1)   
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cfg_file', type=str, default='./config/mini-imagenet/conv64.yaml')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    
    cfg.way = args.way
    cfg.shot = args.shot
    cfg.gpu = args.gpu

    args = cfg

    pprint(vars(args))

    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type, str(args.embed_size),'SSL', str(args.shot), str(args.way)])
    save_path2 = '_'.join([str(args.step_size), str(args.gamma), str(args.lr), str(args.temperature)])
    training_params = '_'.join(['mom', str(args.momentum), 'wd', str(args.weight_decay), 'bsz' ,str(args.back_ward_step)])
    args.save_path = osp.join(args.checkpoint_dir, save_path1, save_path2 + training_params)

    print(args.save_path)

    ensure_path(args.save_path, remove=False)

    if args.dataset == 'MiniImageNet':
        from SSL.dataloader.mini_imagenet import MiniImageNet as Dataset        
    else:
        raise ValueError('Non-supported Dataset.')
    
    # preparing datasets
    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, args.num_episodes_epoch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.num_eval_episodes, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    # preparing model
    model = SSL_boost(args, dropout = args.dropout)    
    param_list = [{'params': model.encoder.parameters(), 'lr': args.lr},
                {'params': model.slf_attn.parameters(), 'lr': args.lr * args.lr_mul},
                {'params': model.Rotation_classifier.parameters(), 'lr': args.lr}]

    # optimizer and lr_scheduler
    optimizer = torch.optim.SGD(param_list, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay) # 0.9 True
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        model = model.cuda()

    # load pre-trained model (no FC weights)
    pth = torch.load(args.init_weights)['params'] # original state_dict()
    pretrained_dict = OrderedDict()
    pretrained_dict = {k:v for k,v in pth.items() if 'fc' not in k}
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    #print(model_dict.keys())
    #print('----------------------------------------------------------------------------')
    #print(pretrained_dict.keys())

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if args.model_type == 'ResNet12': # gpu
        model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))

    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0
    
    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=args.save_path) # should be changed to logdir in the latest version
        
    label = torch.arange(args.way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
            
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()
        model.train()
        tl = Averager()
        ta = Averager()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, index_label = batch[0].cuda(), batch[1].cuda()
            else:
                data, index_label = batch[0], batch[1]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]

            rot_loss, MI_loss, fsl_loss, final_loss, acc_list = model(data_shot, data_query, 'train')
            
            ### update
            total_loss = args.ave_weight * (fsl_loss + rot_loss + MI_loss) + args.final_weight * final_loss
            total_loss = total_loss / args.back_ward_step
            total_loss.backward()

            writer.add_scalar('data/rot_loss', float(rot_loss), global_count)
            writer.add_scalar('data/MI_loss', float(MI_loss), global_count)
            writer.add_scalar('data/fsl_loss', float(fsl_loss), global_count)
            writer.add_scalar('data/final_loss', float(final_loss), global_count)
            writer.add_scalar('data/acc', float(acc_list[-1]), global_count)

            writer.add_scalar('data/total_loss', float(total_loss), global_count)

            print('epoch {}, train {}/{}, total_loss={:.4f}, final_loss={:.4f} acc={:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                  .format(epoch, i, len(train_loader), total_loss.item(), final_loss.item(), acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4]))
            
            tl.add(total_loss.item())
            ta.add(acc_list[-1])

            if (i+1) % args.back_ward_step == 0:
                optimizer.step()
                optimizer.zero_grad()


        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()
            
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
                rot_loss, MI_loss, fsl_loss, final_loss, acc_list = model(data_shot, data_query)
                total_loss = args.ave_weight * (rot_loss + MI_loss + fsl_loss) + args.final_weight * final_loss

                vl.add(total_loss.item())
                va.add(acc_list[-1])

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)             
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')          
                
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    
    writer.close()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 600, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((600,))

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
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
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc_list[-1] * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
    for i in range(model.trans_num):
        print('Rotation {} acc is {:.4f}'.format(90*i, tasks_acc[i].item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    
    ave_acc = np.array(test_acc_record).mean() * 100 
    acc_std = np.array(test_acc_record).std() * 100
    ci95 = 1.96 * np.array(test_acc_record).std() / np.sqrt(float(len(np.array(test_acc_record)))) * 100

    print('evaluation: accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%'%(ave_acc, acc_std, ci95))
