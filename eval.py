import os
import glob
import numpy as np
from skimage import io
from tqdm import tqdm


mode = 'wo_12'
prediction_dir = 'datasets/predictions/'
gt_list = sorted(glob.glob(prediction_dir + '*_gt.png'))
pred_list = sorted(glob.glob(prediction_dir + '*_pred_' + mode + '.png'))

gt = np.zeros([10000, 64, 64])
pred = np.zeros([10000, 64, 64])
for i in tqdm(range(len(gt_list))):
    gt[i, :, :] = io.imread(gt_list[i]) / 255.

for i in tqdm(range(len(pred_list))):
    pred[i, :, :] = io.imread(pred_list[i]) / 255.

l2 = (gt - pred) * (gt- pred)
print(l2.shape)
print(np.min(l2))
l2_avg = np.mean(l2)
print(mode, l2_avg)
