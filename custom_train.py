import os
import random

import torch.optim as optim
from torch.optim import lr_scheduler

from data_loader import *
from nets_unet import *
from util import *
import re

train_csv = "datasets/split/60.csv"
train_pre = "datasets/split/preselection_top32_60.npy"
validation_csv = "datasets/split/30.csv"
validation_pre = "datasets/split/preselection_top32_30.npy"
test_csv = "datasets/split/10.csv"
test_pre = "datasets/split/preselection_top32_10.npy"

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
result = re.search('preselection_(.*)_', test_pre)
checkpoint_path = 'checkpoints/checkpoint_'+  result.group(1) +'_seed_'+ str(seed) +'.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Define dataloaders
mnist_set = PriorDataset()
mnist_loader = DataLoader(mnist_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
mnist_iter = iter(mnist_loader)


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

G = vae_road_layout(with_msk_channel=with_msk_channel).to(device) #G is the network. vae = variational autoencoder. It's just a type of network.
#bce_loss = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=0.0001, weight_decay=0.0001) #adam algo
schedulerG = lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.1) #learning with decaying rate.


if restore: # in case the learning shuts down (unexpectedly)
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
            G.train()  # Set model to training mode
        else:
            G.eval()  # Set model to evaluate mode

        running_loss_0 = 0.0

        # Iterate over data.
        for i, temp_batch in enumerate(dataloaders[phase]):
            temp_input_img = temp_batch['input'].float().to(device)
            temp_ground_truth = temp_batch['pre_sele'].float().to(device) #pre is GT

            try:
                temp_mnist = next(mnist_iter)['prior'].float().to(device)
            except StopIteration:
                mnist_iter = iter(mnist_loader)
                temp_mnist = next(mnist_iter)['prior'].float().to(device)

            with torch.set_grad_enabled(phase == 'train'):
                optimizerG.zero_grad() #we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
                if with_msk_channel:
                    pass
                else:
                    pred_mnist = G(temp_input_img.clone().detach(), phase == 'train')

                loss_mnist_1 = F.mse_loss(temp_ground_truth, pred_mnist) #loss is MSE loss between GT (pre select) and prediction
                loss_all = loss_mnist_1

                if phase == 'train':
                    loss_all.backward() #training routine
                    optimizerG.step()#training routine


            running_loss_0 += loss_mnist_1.item()
            # statistics
        if phase == 'train':
            running_loss_0 = running_loss_0 / len(train_set)
        else:
            running_loss_0 = running_loss_0 / len(val_set)

        print(phase, running_loss_0)

    if phase == 'train':
        schedulerG.step()

    # save model per epoch
    torch.save({
        'epoch': epoch + 1,
        'state_dict_G': G.state_dict(),
        'optimizer_G': optimizerG.state_dict(),
        'scheduler_G': schedulerG.state_dict(),
        }, checkpoint_path)
    print('model after %d epoch saved...' % (epoch+1))
    epoch += 1


