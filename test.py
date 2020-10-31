import argparse
import sys
sys.path.append("./meshcnn")
import os
from glob import glob

from loader_individual import S2D3DSegLoaderSingle
from model import SphericalUNet

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import scipy.io

from cpuModel import cpuModel

def export(args, model, test_loader, use_cuda):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            if use_cuda:
                data, reg = data.cuda(), target.cuda()
            output = model(data)
            if use_cuda:
                output = output.cuda().detach().cpu()
            else:
                output = output.numpy()
            break
    output = np.squeeze(output)
    output = output.astype(np.float32)

    file_format = args.data_folder + '/*.' + args.hemi + '.*.dat'
    fn = glob(file_format)[0]
    fn = '.'.join(fn.split('.')[0:3]) + '.' + args.export_file

    if args.fmt == 'mat':
        scipy.io.savemat(fn + '.mat', {'prob': output})
    elif args.fmt in ['txt', 'dat']:
        for i in range(0, len(output)):
            if args.fmt == 'txt':
                prob = '%s%d.txt' % (fn, i+1)
                np.savetxt(prob, output[i,:], fmt='%f')
            else:
                prob = '%s%d.dat' % (fn, i+1)
                binwrite=open(prob,'wb')
                output[i,:].tofile(binwrite)
                binwrite.close()
    else:
        print("Unknown format: " + fmt)
        exit(1)
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Climate Segmentation Example')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="./mesh_files",
                        help='path to mesh folder (default: ./mesh_files)')
    parser.add_argument('--ckpt', type=str, default="checkpoint_latest.pth.tar_SUNet_best.pth")
    parser.add_argument('--data_folder', type=str, default="data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--max_level', type=int, default=5, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--feat', type=int, default=32, help='filter dimensions')
    parser.add_argument('--export_file', type=str, default='out.mat', help='file name for exporting samples', required=True)
    parser.add_argument('--in_ch', type=str, nargs='+', help="input channels (list of features)")
    parser.add_argument('--hemi', type=str, default="lh", choices=["lh", "rh"])
    parser.add_argument('--fmt', type=str, default="txt", choices=["txt", "dat", "mat"])

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # load checkpoint
    assert(os.path.isfile(args.ckpt))
    print("=> loading checkpoint '{}'".format(args.ckpt))
    if use_cuda:
        resume_dict = torch.load(args.ckpt)
    else:
        resume_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    out_ch = len(resume_dict['state_dict']['module.out_conv.coeffs'])
    model = SphericalUNet(mesh_folder=args.mesh_folder, in_ch=len(args.in_ch), out_ch=out_ch, max_level=args.max_level, min_level=args.min_level, fdim=args.feat)
    if use_cuda:
        model = nn.DataParallel(model)
    else:
        model = cpuModel(model)
    model.to(device)

    def load_my_state_dict(self, state_dict, exclude='none'):
        from torch.nn.parameter import Parameter
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if exclude in name:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    load_my_state_dict(model, resume_dict['state_dict'])  
    testset = S2D3DSegLoaderSingle(args.data_folder, sp_level=args.max_level, in_ch=args.in_ch, hemi=args.hemi)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    export(args, model, test_loader, use_cuda)
        
if __name__ == "__main__":
    main()
