from torchvision import datasets, models, transforms
import time
import os
import copy
import glob
# import cv2
from skimage import io, transform
import random
# import visualization as vis
from data_loader import *
from nets_unet import *
# from util import *

mode = 'wo_13'
seed = 1
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

with_msk_channel = False
num_labels = 5
checkpoint_path = 'checkpoints/checkpoint_skipconnection_trainval_' + mode + '_seed_' + str(seed) + '.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define dataloaders
test_set = PartialDataset('datasets/mnist_train.csv', 'datasets/preselection_top4.npy')
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


G = vae_road_layout(with_msk_channel=with_msk_channel).to(device)


if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    G.load_state_dict(state['state_dict_G'])
    print('trained model loaded...')
else:
    print('cannot load trained model...')


G.eval()  # Set model to evaluate mode

# Iterate over data.

for i, temp_batch in enumerate(test_loader):
    if i % 100 == 0:
        print('example no. ', i)
    temp_map_input = temp_batch['input'].float().to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        if with_msk_channel:
            pred_map = G(torch.cat([temp_map_input, temp_map[:, 4, :, :].unsqueeze(1)], dim=1), False)
        else:
            pred_map = G(temp_map_input, False)
        

        # torch_img_visualization(2, [pred_map.detach(), temp_map_input.detach()])

        io.imsave('datasets/predictions/{0:06d}_input.png'.format(i), (temp_map_input.cpu().numpy()*255).astype(np.uint8).reshape((64, 64)))
        io.imsave('datasets/predictions/{0:06d}_pred_wo_13.png'.format(i), (pred_map.cpu().numpy()*255).astype(np.uint8).reshape((64, 64)))
