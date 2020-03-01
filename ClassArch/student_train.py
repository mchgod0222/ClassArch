import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim.lr_scheduler as lr_sched
from dataset import ModelNet40Dataset
from model.rscnn_ssn_cls import RSCNN_SSN
from rscnn_trainer import RSCNNTrainer
from utils import data_utils
import utils.pytorch_utils as pt_utils
from accuracy_metric import accuracy


def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Creating Directory." + dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=32
)
parser.add_argument(
    '--epoch', type=int, default=400
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./save_model/'
)

parser.add_argument(
    '--base_model', type=str, default='rscnn'
)
parser.add_argument(
    '--lr_clip', type=float, default=0.00001
)
parser.add_argument(
    '--lr_decay', type=float, default=0.7
)
parser.add_argument(
    '--decay_step', type=int, default=21
)
parser.add_argument(
    '--bn_momentum', type=float, default=0.9
)
parser.add_argument(
    '--bnm_clip', type=float, default=0.01
)
parser.add_argument(
    '--bn_decay', type=float, default=0.5
)

cfg = parser.parse_args()
print(cfg)

model_dict = {'rscnn': (RSCNN_SSN(num_classes=40), nn.CrossEntropyLoss(), accuracy, torch.optim.Adam, RSCNNTrainer)}

train_transforms = transforms.Compose([
    data_utils.PointcloudToTensor()
])

test_transforms = transforms.Compose([
    data_utils.PointcloudToTensor()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    ds_train = ModelNet40Dataset(num_points=1024, root=cfg.dataset, transforms=train_transforms)
    ds_test = ModelNet40Dataset(num_points=1024, root=cfg.dataset, split='test', transforms=test_transforms)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    print("DATA LOADED")

    model, criterion, metric, optimizer, trainer = model_dict[cfg.base_model]
    optimizer = optimizer(model.parameters(), lr=cfg.lr)

    lr_lbmd = lambda e: max(cfg.lr_decay**(e // cfg.decay_step), cfg.lr_clip / cfg.lr)
    bnm_lmbd = lambda e: max(cfg.bn_momentum * cfg.bn_decay**(e // cfg.decay_step), cfg.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    print(model, criterion, optimizer)

    trainer = trainer(model, criterion, optimizer, lr_scheduler, bnm_scheduler, metric, device, None)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints=cfg.save_model+model.__class__.__name__+'.pt')

    create_folder(model.__class__.__name__)
    torch.save(model.state_dict(), os.path.join(model.__class__.__name__, 'final_state_dict_student.pt'))
    torch.save(model, os.path.join(model.__class__.__name__, 'final_student.pt'))
