import os
# import cv2
import random

# import visualization as vis
from data_loader import *
from nets_unet import *
import re
# from util import *

seed = 1
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#top32: 0.0026415655612945558
#top16: 0.0038455567359924315
#top8: 0.005967601299285889
#top4: 0.012162957191467285
#top2: 0.018501659393310547
#top1: 0.035490142822265626

test_csv = "datasets/split/10.csv"
test_pre = "datasets/split/preselection_top32_10.npy"

with_msk_channel = False
result = re.search('preselection_(.*)_', test_pre)
checkpoint_path = 'checkpoints/checkpoint_'+  result.group(1) +'_seed_'+ str(seed) +'.pth.tar'
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

loss_input_pred = 0
loss_gt_pred = 0
for i, temp_batch in enumerate(test_loader):
    if i % 100 == 0:
        print('example no. ', i)
    temp_input_img = temp_batch['input'].float().to(device)
    unmasked = temp_batch['unmasked'].float().to(device)
    temp_ground_truth = temp_batch['pre_sele'].float().to(device)
    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        if with_msk_channel:
            pred_map = G(torch.cat([temp_input_img, temp_input_img[:, 4, :, :].unsqueeze(1)], dim=1), False)
        else:
            pred_map = G(temp_input_img, False)

        # torch_img_visualization(2, [pred_map.detach(), temp_map_input.detach()])
        io.imsave('datasets/predictions/'+result.group(1)+'/{0:06d}_um.png'.format(i),
                  (unmasked.cpu().numpy() * 255).astype(np.uint8).reshape((64, 64)))
        io.imsave('datasets/predictions/'+result.group(1)+'/{0:06d}_gt.png'.format(i),
                  (temp_ground_truth.cpu().numpy() * 255).astype(np.uint8).reshape((64, 64)))
        io.imsave('datasets/predictions/'+result.group(1)+'/{0:06d}_input.png'.format(i),
                  (temp_input_img.cpu().numpy() * 255).astype(np.uint8).reshape((64, 64)))
        io.imsave('datasets/predictions/'+result.group(1)+'/{0:06d}_pred.png'.format(i),
                 (pred_map.cpu().numpy() * 255).astype(np.uint8).reshape((64, 64)))
        loss_input_pred += F.mse_loss(temp_input_img, pred_map)
        loss_gt_pred += F.mse_loss(unmasked, pred_map)
print(loss_gt_pred.item()/len(test_set))

#for this test show the ground truth for report
#show 3 or 4 samples, input, prediction and gt
#alongside we need a metric of loss (L2-metric)
#loss_mnist_1 = F.mse_loss(temp_ground_truth, pred_mnist)

#between unmasked input (modify dataloader to )
#io.imsave the unmasked as well for clarification in the paper

#1 intro
#2 Related work
#3 methodology (possible split)
#4 experiment
#5 conclusion

