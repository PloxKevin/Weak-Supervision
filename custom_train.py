import os
import random

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from data_loader import *
from nets_unet import *
from util import *

train_csv = "datasets/split/60.csv"
train_pre = "datasets/split/preselection_top1_60.npy"
validation_csv = "datasets/split/30.csv"
validation_pre = "datasets/split/preselection_top1_30.npy"
test_csv = "datasets/split/10.csv"
test_pre = "datasets/split/preselection_top1_10.npy"

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
checkpoint_path = 'checkpoints/checkpoint_skipconnection_trainval_seed_' + str(seed) +'.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


# Define dataloaders
road_set = PriorDataset()
road_loader = DataLoader(road_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
road_iter = iter(road_loader)




###EXPERIMENTAL
train_full_set = PartialDataset(train_csv, train_pre)
val_full_set = PartialDataset(validation_csv, validation_pre)
train_size = 3000 #we have custom training sizes
val_size = 1500 #and val sizes. Apply this later.
train_set = train_full_set
val_set = val_full_set
###EXPERIMENTAL



#train_set, val_set = torch.utils.data.random_split(train_full_set, [train_size, val_size])
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
                loss_all = loss_road_1


                if phase == 'train':
                    loss_all.backward()
                    optimizerG.step()

            running_loss_0 += loss_road_1.item()


            # tensorboardX logging
            if phase == 'train':
                writer.add_scalar(phase+'_loss_road_0', loss_road_1.item(), epoch * len(train_set) / batch_size + i)


            # statistics
        if phase == 'train':
            running_loss_0 = running_loss_0 / len(train_set)

        else:
            running_loss_0 = running_loss_0 / len(val_set)


        print(phase, running_loss_0)
        if phase == 'val':
            writer.add_scalar(phase+'_loss_road_0', loss_road_1.item(), (epoch+1) * len(train_set) / batch_size)


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

