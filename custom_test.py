import os
# import cv2
import random

# import visualization as vis
from data_loader import *
from nets_unet import *

# from util import *

seed = 2
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

test_csv = "datasets/split/10.csv"
test_pre = "datasets/split/preselection_top1_10.npy"

with_msk_channel = False
num_labels = 5
checkpoint_path = 'checkpoints/checkpoint_skipconnection_trainval_seed_' + str(seed) + '.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define dataloaders
test_set = PartialDataset(test_csv, test_pre)
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
            pred_map = G(torch.cat([temp_map_input, temp_map_input[:, 4, :, :].unsqueeze(1)], dim=1), False)
        else:
            pred_map = G(temp_map_input, False)

        # torch_img_visualization(2, [pred_map.detach(), temp_map_input.detach()])

        io.imsave('datasets/predictions/{0:06d}_input.png'.format(i),
                  (temp_map_input.cpu().numpy() * 255).astype(np.uint8).reshape((64, 64)))
        io.imsave('datasets/predictions/{0:06d}_pred_wo_13.png'.format(i),
                  (pred_map.cpu().numpy() * 255).astype(np.uint8).reshape((64, 64)))
