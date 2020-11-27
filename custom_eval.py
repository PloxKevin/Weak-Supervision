import os
import glob
import numpy as np
from skimage import io
from tqdm import tqdm
import re
prediction_dir = 'datasets/predictions/top16/'
gt_list = sorted(glob.glob(prediction_dir + '*_gt.png'))
pred_list = sorted(glob.glob(prediction_dir + '*_pred.png'))

gt = np.zeros([500, 64, 64])
pred = np.zeros([500, 64, 64])
for i in tqdm(range(len(gt_list))):
    gt[i, :, :] = io.imread(gt_list[i]) / 255.

for i in tqdm(range(len(pred_list))):
    pred[i, :, :] = io.imread(pred_list[i]) / 255.

l2 = (gt - pred) * (gt - pred)
print(l2.shape)
print(np.min(l2))
l2_avg = np.mean(l2)
print(l2_avg)
