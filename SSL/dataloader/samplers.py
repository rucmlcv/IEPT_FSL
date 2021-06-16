import torch
import numpy as np


class CategoriesSamplerBak():

    def __init__(self, label, n_batch, n_cls, n_per): #n_batch 为 一个epoch的episode数亩
        
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.n_step = 0
        self.mark = {}
        self.r_clses = None

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            if self.r_clses is None:
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]
                self.r_clses = classes
            else:
                classes = self.r_clses
                self.r_clses = None

            for c in classes:
                l = self.m_ind[c]
                self.mark[l] = True

                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

    def getmark(self):
        count = 0
        for c in self.m_ind:
            if c not in self.mark:
                count += 1
        print(count)


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per): #n_batch 为 一个epoch的episode数亩
        
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.n_step = 0

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []

            classes = torch.randperm(len(self.m_ind))[:self.n_cls]

            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


