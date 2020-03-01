import os
import sys
from shutil import copyfile
import h5py
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataset import UnlabeledModelNet40Dataset
from model.rscnn_ssn_cls import RSCNN_SSN
from utils import data_utils


test_transforms = transforms.Compose([
    data_utils.PointcloudToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = UnlabeledModelNet40Dataset(1024, 'data/', test_transforms, 'unlabeled')

    # model_loc = './PointNet/final'
    model_loc = './epoch_save_model/RSCNN_SSN_ckpt_121'
    model = torch.load(model_loc + '.pt')
    model = model.cuda()
    model.eval()

    label = []

    for i in range(2048):
        X = ds_test.__getitem__(i)
        X = X.view(1, 3, 2048).cuda()
        pred, _, _ = model(X)
        label.append(int((pred == pred.max()).nonzero().flatten()[-1]))

    label = np.array(label)
    label = np.expand_dims(label, axis=1)
    label = label.astype(np.uint8)
    print('LABELING DONE...')
    print('MAKING INTO HDF FILE...')

    f = h5py.File('data/modelnet40/ply_data_unlabeled0.h5')
    data  = f['data'][:]
    
    final_label = h5py.File('data/modelnet40/ply_data_labeled0.h5', 'w')
    final_label['data'] = data
    final_label['label'] = label

    copyfile('data/modelnet40/ply_data_unlabeled_0_id2file.json', 'data/modelnet40/ply_data_labeled_0_id2file.json')
