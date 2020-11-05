import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter
import random

from data_loader import *
from nets_unet import *
from util import *

experiment_csv = "datasets/split/60.csv"
experiment_pre = "datasets/split/preselection_top1_60.npy"
mode = 'wo_13'
seed = 1
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


with_msk_channel = False
num_epochs = 30
batch_size = 64
restore = True
num_labels = 5
checkpoint_path = 'checkpoints/checkpoint_skipconnection_trainval_' + mode + '_seed_' + str(seed) +'.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


# Define dataloaders
road_set = PriorDataset()
road_loader = DataLoader(road_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
road_iter = iter(road_loader)
train_full_set = PartialDataset(experiment_csv, experiment_pre)
train_size = int(0.8 * len(train_full_set)) #we have custom training sizes
val_size = len(train_full_set) - train_size #and val sizes. Apply this later.
train_set, val_set = torch.utils.data.random_split(train_full_set, [train_size, val_size])
train_loader = DataLoader(train_full_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
dataloaders = {'train': train_loader, 'val': val_loader}

G = vae_road_layout(with_msk_channel=with_msk_channel).to(device)
bce_loss = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=0.0001, weight_decay=0.0001)
schedulerG = lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.1)


if restore:
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        epoch = state['epoch']
        G.load_state_dict(state['state_dict_G'])
        optimizerG.load_state_dict(state['optimizer_G'])
        schedulerG.load_state_dict(state['scheduler_G'])
    else:
        epoch = 0
else:
    epoch = 0


while epoch < num_epochs:
    print(' ')
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            schedulerG.step()
            G.train()  # Set model to training mode
        else:
            G.eval()  # Set model to evaluate mode

        running_loss_0 = 0.0
        running_loss_1 = 0.0
        running_loss_2 = 0.0

        # Iterate over data.

        for i, temp_batch in enumerate(dataloaders[phase]):
            temp_map_input = temp_batch['input'].float().to(device)

            temp_style_tgt = temp_batch['pre_sele'].float().to(device)

            try:
                temp_road = next(road_iter)['prior'].float().to(device)
            except StopIteration:
                road_iter = iter(road_loader)
                temp_road = next(road_iter)['prior'].float().to(device)

            temp_road_input = temp_road.detach().clone()
            temp_road_input[:, :, 21:43, 21:43] = 0.5

            with torch.set_grad_enabled(phase == 'train'):
                optimizerG.zero_grad()
                if with_msk_channel:
                    pass
                else:
                    pred_road = G(temp_map_input.clone().detach(), phase == 'train')
                    pred_road_1 = G(temp_road_input, phase == 'train')

                loss_road_1 = loss_function_road_pred(pred_road, temp_map_input)
                loss_road_2 = loss_function_pre_selection(pred_road, temp_style_tgt)
                loss_road_3 = loss_function_road_layout(pred_road_1, temp_road)

                # torch_img_visualization(2, (temp_road_input, temp_road))

                if mode == 'full':
                    loss_all = 0.5 * loss_road_1 + 0.25 * loss_road_2 + 0.25 * loss_road_3
                elif mode == 'wo_1':
                    loss_all = 0.5 * loss_road_2 + 0.5 * loss_road_3
                elif mode == 'wo_2':
                    loss_all = 2/3 * loss_road_1 + 1/3 * loss_road_3
                elif mode == 'wo_3':
                    loss_all = 2/3 * loss_road_1 + 1/3 * loss_road_2
                elif mode == 'wo_12':
                    loss_all = loss_road_3
                elif mode == 'wo_13':
                    loss_all = loss_road_2
                elif mode == 'wo_23':
                    loss_all = loss_road_1


                if phase == 'train':
                    loss_all.backward()
                    optimizerG.step()

            running_loss_0 += loss_road_1.item()
            running_loss_1 += loss_road_2.item()
            running_loss_2 += loss_road_3.item()

            # tensorboardX logging
            if phase == 'train':
                writer.add_scalar(phase+'_loss_road_0', loss_road_1.item(), epoch * len(train_set) / batch_size + i)
                writer.add_scalar(phase+'_loss_road_1', loss_road_2.item(), epoch * len(train_set) / batch_size + i)
                writer.add_scalar(phase+'_loss_road_2', loss_road_3.item(), epoch * len(train_set) / batch_size + i)

            # statistics
        if phase == 'train':
            running_loss_0 = running_loss_0 / len(train_set)
            running_loss_1 = running_loss_1 / len(train_set)
            running_loss_2 = running_loss_2 / len(train_set)
        else:
            running_loss_0 = running_loss_0 / len(val_set)
            running_loss_1 = running_loss_1 / len(val_set)
            running_loss_2 = running_loss_2 / len(val_set)

        print(phase, running_loss_0, running_loss_1, running_loss_2)
        if phase == 'val':
            writer.add_scalar(phase+'_loss_road_0', loss_road_1.item(), (epoch+1) * len(train_set) / batch_size)
            writer.add_scalar(phase+'_loss_road_1', loss_road_2.item(), (epoch+1) * len(train_set) / batch_size)
            writer.add_scalar(phase+'_loss_road_2', loss_road_3.item(), (epoch+1) * len(train_set) / batch_size)

    # save model per epoch
    torch.save({
        'epoch': epoch + 1,
        'state_dict_G': G.state_dict(),
        'optimizer_G': optimizerG.state_dict(),
        'scheduler_G': schedulerG.state_dict(),
        }, checkpoint_path)
    print('model after %d epoch saved...' % (epoch+1))
    epoch += 1

writer.close()

