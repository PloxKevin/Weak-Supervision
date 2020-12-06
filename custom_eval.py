import os
import glob
import numpy as np
from skimage import io
from tqdm import tqdm
import re

top_ranges = [1,2,4,8,16,32]

for top in top_ranges:

    prediction_dir = 'datasets/predictions/top'+ str(top) +'/'
    gt_list = sorted(glob.glob(prediction_dir + '*_um.png'))
    pred_list = sorted(glob.glob(prediction_dir + '*_pred.png'))

    gt = np.zeros([500, 64, 64])
    pred = np.zeros([500, 64, 64])
    for i in tqdm(range(len(gt_list))):
        gt[i, :, :] = io.imread(gt_list[i]) / 255.

    for i in tqdm(range(len(pred_list))):
        pred[i, :, :] = io.imread(pred_list[i]) / 255.

    l2 = (gt - pred) * (gt - pred)
    l2_avg = np.mean(l2)
    print('top'+str(top), l2_avg)

