import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SSL.utils import euclidean_metric, count_acc
from scipy.io import loadmat
from SSL.models.attention import *
import os


class SSL_boost(nn.Module):

    def __init__(self, args, dropout=0.2):
        super().__init__()
        self.args = args

        if args.model_type == 'ConvNet':
            from SSL.networks.convnet import ConvNet
            cnn_dim = args.embed_size
            self.encoder = ConvNet(args, z_dim=cnn_dim)

        elif args.model_type == 'ResNet12':
            from SSL.networks.ResNet12_embedding import resnet12
            self.encoder = resnet12()
            cnn_dim = args.embed_size

        else:
            raise ValueError('')

        z_dim = cnn_dim
        
        self.slf_attn = MultiHeadAttention(args, args.head, z_dim, z_dim, z_dim, dropout=dropout, do_activation=True) 


        self.KDloss = nn.KLDivLoss(reduce=True)  # logitis_aug = F.log_softmax(logitis_aug, -1)
        self.MSELoss = nn.MSELoss()

        self.trans_num = 4
            
        self.Rotation_classifier = nn.Sequential(nn.Linear(z_dim, self.trans_num),
                                                nn.ReLU()) # 4 angle classifier
        
    def expend_tasks(self, support_tasks, query_tasks): # create task list
        # support_tasks   <K*N, 4, 3, width, height>
        # query_tasks     <Q*N, 4, 3, width, height>

        support_s = list(torch.split(support_tasks, 1, dim=1)) # [<K*N, 3, width, height> * 4]
        support_s = [support.squeeze(1) for support in support_s]
        query_s = list(torch.split(query_tasks, 1, dim=1)) # [<Q*N, 3, width, height> * 4]
        query_s = [query.squeeze(1) for query in query_s]
        
        return support_s, query_s

    def fsl_module_per_task(self, support, query): # fsl for a single task
        # input: images tensor

        N, K, Q = self.args.way, self.args.shot, self.args.query
        input_tensor = torch.cat([support, query], 0)
        output = self.encoder(input_tensor)
        support = output[:support.size(0)]
        query = output[support.size(0):]

        
        proto = support.reshape(K, -1, support.shape[-1]).mean(dim=0) # N x d
        logitis = euclidean_metric(query, proto)
        logitis = logitis / self.args.temperature
        #print(logitis)


        # <K*N, dim>, <Q*N, dim>, <N, d>, <Q*N, N>
        # return  feature, prototype, logits(classification distribution)
        return support, query, proto, logitis

    
    def forward(self, support, query, mode = 'test'):

        N, K, Q = self.args.way, self.args.shot, self.args.query
        support_s, query_s = self.expend_tasks(support, query)
        ## prepare label for these tasks
        ## 1st:  rotation label
        rot_label = torch.arange(self.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, Q*N+K*N).type(torch.LongTensor)
        rot_label = rot_label.view(-1).cuda()
        ## 2nd:  fsl label
        fsl_label = torch.arange(N, dtype=torch.int8).repeat(Q).type(torch.LongTensor).cuda()

        ## perform fsl method on each task copy
        sup_feats, que_feats, protos, class_dist = [], [], [], []
        rotation_samples = []
        for (support_ang, query_ang) in zip(support_s, query_s):
            #print(support_ang.size())
            support, query, proto, logitis = self.fsl_module_per_task(support_ang, query_ang)
            sup_feats.append(support)
            que_feats.append(query)
            protos.append(proto)
            class_dist.append(logitis)
            rotation_samples.append(torch.cat([support, query], 0))

        # rotation loss
        rotation_samples = torch.cat(rotation_samples, 0)
        rot_pred = self.Rotation_classifier(rotation_samples)
        rot_loss = F.cross_entropy(rot_pred, rot_label)

        # MI mutual information:  KD loss
        raw_logits = sum(class_dist) / len(class_dist)
        raw_logits = F.log_softmax(raw_logits, -1)
        MI_losses = [F.kl_div(raw_logits, F.softmax(logits, -1), size_average=True) for logits in class_dist]
        MI_loss = sum(MI_losses) / len(MI_losses)

        # fsl loss for all the tasks copy
        fsl_losses = [F.cross_entropy(logits, fsl_label) for logits in class_dist]
        fsl_loss = sum(fsl_losses) / len(fsl_losses)
        #fsl_loss = F.cross_entropy(raw_logits, fsl_label)
        
        ### transform these tasks copy to final fsl task:
        trans_support = torch.stack(sup_feats, 1) # <K*N, 4, dim>
        trans_query = torch.stack(que_feats, 1) # <Q*N, 4, dim>
        
        ### transformer
        trans_support = self.slf_attn(trans_support, trans_support, trans_support)
        trans_query = self.slf_attn(trans_query, trans_query, trans_query)
        
        # cat
        trans_support = trans_support.view(K*N, -1) # version 2  <K*N, trans_num*dim>
        trans_query = trans_query.view(Q*N, -1) # version 2
        
        proto = trans_support.reshape(K, -1, trans_support.shape[-1]).mean(dim=0) # N x d
        logitis = euclidean_metric(trans_query, proto) 
        logitis = logitis / self.args.temperature 

        final_loss = F.cross_entropy(logitis, fsl_label)
        
        acc_list = [count_acc(logits, fsl_label) for logits in class_dist] # for 4 single angles tasks
        acc_list.append(count_acc(logitis, fsl_label)) # the final task

        if mode == 'extract':
            proto, feats = [], []
            for i in range(self.trans_num):
                proto.append(sup_feats[i].view(K, N, -1).mean(dim=0))
                feats.append(torch.cat([sup_feats[i].view(K, N, -1), que_feats[i].view(Q, N, -1)], 0).view((K+Q)*N, -1))

            trans_proto = trans_support.view(K, N, -1).mean(dim=0)
            trans_feats = torch.cat([trans_support.view(K, N, -1), trans_query.view(Q, N, -1)], 0).view((K+Q)*N, -1)
            
            proto = torch.stack(proto, 1)
            feats = torch.stack(feats, 1)
            return acc_list, proto, feats, trans_proto, trans_feats
            # <number> * 5, <N, 4, dim>, <(K+Q)*N, 4, dim>, <N, dim*4>, <(K+Q)*N, dim*4>

        return rot_loss, MI_loss, fsl_loss, final_loss, acc_list
    