import numpy as np
from glob import glob
from torch.utils.data import Dataset

# sphere mesh size at different levels
nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
           
class S2D3DSegLoaderSingle(Dataset):
    """Data loader for 2D3DS dataset."""

    def __init__(self, data_dir, sp_level, in_ch, hemi):
        self.in_ch = len(in_ch) - 1
        self.nv = nv_sphere[sp_level]

        flist = []
        file_format = data_dir + '/*.' + hemi + '.*.dat'
        flist += sorted(glob(file_format))

        flist = [flist[_x] for _x in [[x.split('.')[-2] for x in flist].index(a) for a in in_ch]]

        # final list
        self.flist = [flist]

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # load files
        subj = self.flist[idx]

        # single-thread loader
        data = np.array([])
        for feat in subj:
            data = np.fromfile(feat,count=self.nv,dtype=np.double)
        data = np.reshape(data, (-1, self.nv))
        data = data[:self.in_ch, :self.nv].astype(np.float32)

        return data
