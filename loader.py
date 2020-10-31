import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import collections
from collections import deque
import random

# sphere mesh size at different levels
nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]

class S2D3DSegLoader(Dataset):
    """Data loader for 2D3DS dataset."""

    def __init__(self, data_dir, partition, fold, sp_level, classes, in_ch, seed, deg, kcv, hemi):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
            fold: 1 to 5 (for 5-fold cross-validation)
            sp_level: sphere mesh level. integer between 0 and 7.
            
        """
        assert(partition in ["train", "test", "val"])
        self.in_ch = len(in_ch) - 1
        self.nv = nv_sphere[sp_level]
        self.partition = partition
        total_fold = kcv

        feature_type = [feat.split('/')[-1] for feat in glob(data_dir)]
        flist = []
        file_format1 = data_dir + '/features/*.' + hemi + '.*.dat'
        flist += sorted(glob(file_format1))
        file_format1 = data_dir + '/labels/*.' + hemi + '.*.dat'
        flist += sorted(glob(file_format1))

        # dict construction
        data = dict()
        for i in flist:
            key = '.'.join(i.split('.')[0:2]).split('/')[-1]
            cat = i.split('.')[0].split('/')[-3]
            if not key in data:
                data[key] = {'subject': key}
            data[key].setdefault(i.split('.')[2]+cat, []).append(i)

        for key in data:
            for i in data[key]:
                if i != 'subject':
                    d = data[key][i]
                    data[key][i] = [d[_x] for _x in [[x.split('.')[-2] for x in d].index(a) for a in in_ch]]

        # subject list
        subj = [entry for entry in data]
        subj = sorted(subj)
        random.seed(seed)
        random.shuffle(subj)
        
        total_subj = len(subj)
        fold_batch = int(total_subj / total_fold);
        fold_batch_im = total_subj - fold_batch * (total_fold - 1);
        
        subj = deque(subj)
        subj.rotate(fold_batch * (fold - 1))
        subj = list(subj)
        if fold >= total_fold - 1:
            test = subj[total_subj-fold_batch-fold_batch_im:total_subj]
        else:
            test = subj[total_subj-2*fold_batch:total_subj]
        train = [item for item in subj if item not in set(test)]
        if fold == total_fold - 1:
            val = test[0:fold_batch_im]
            test = test[fold_batch_im:fold_batch_im+fold_batch]
        elif fold == total_fold:
            val = test[0:fold_batch]
            test = test[fold_batch:fold_batch+fold_batch_im]
        else:
            val = test[0:fold_batch]
            test = test[fold_batch:2*fold_batch]
        
        self.flist = []

        # final list
        if partition == "train":
            flist_train = []
            for i in train:
                for feat in feature_type:
                    for aug in range(0, deg+1):
                        flist_train.append(data[i]['deg' + str(aug) + feat])
            self.flist = flist_train

        if partition == "val":
            flist_test = []
            for i in val:
                for feat in feature_type:
                    flist_test.append(data[i]['deg0curv'])
            self.flist = flist_test

        if partition == "test":
            flist_test = []
            for i in test:
                for feat in feature_type:
                    flist_test.append(data[i]['deg0curv'])
            self.flist = flist_test

        # label dictionary
        lut = collections.defaultdict(lambda : 0) 
        for i, label in enumerate(classes):
            lut[label] = i
        self.lut = lut

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # load files
        subj = self.flist[idx]
        data = np.array([])
        for feat in subj[:-1]:
            T = np.fromfile(feat,count=self.nv,dtype=np.double)
            data = np.append(data, T)
        
        T = np.fromfile(subj[-1],count=self.nv,dtype=np.int16)
        data = np.append(data, T)

        data = np.reshape(data, (-1, self.nv))
        labels = data[self.in_ch, :self.nv]
        data = data[:self.in_ch, :self.nv].astype(np.float32)

        labels = [self.lut[label] for label in labels]
        labels = np.asarray(labels).astype(np.int)
        return data, labels
