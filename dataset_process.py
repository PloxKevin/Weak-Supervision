import numpy as np
import matplotlib.pyplot as plt
from py_img_seg_eval import eval_segm
from tqdm import tqdm


amount_images = 50
top_ranges = [1, 2, 4, 8, 16, 32]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "datasets/"
train_data = np.loadtxt(data_path + "mnist_combined.csv",
                        delimiter=",")


mask = np.ones([28, 28])
mask[9:19, 9:19] = 0
mask = mask > 0.5
# msk  = msk.reshape(-1)

#np.random.shuffle(train_data)
train_partial = train_data[:amount_images]
train_prior = train_data[amount_images:2*amount_images]
# augment priors


intersectionoverunion = np.zeros([train_partial.shape[0], train_prior.shape[0]])
targets = np.zeros([len(top_ranges), train_partial.shape[0], 28, 28])

label_partial = train_partial[:, 0]
img_masked = train_partial[:, 1:].reshape(-1, 28, 28) > 128
label_prior = train_prior[:, 0]
img_prior = train_prior[:, 1:].reshape(-1, 28, 28) > 128


for i in tqdm(range(img_masked.shape[0])):
    # print('label is', label_partial[i])
    for j in range(img_prior.shape[0]):
        intersectionoverunion[i, j] = eval_segm.mean_IU(img_prior[j].copy().astype(np.uint8)[mask].reshape(1, -1), img_masked[i].copy().astype(np.uint8)[mask].reshape(1, -1))
        #print((metrics[i,j]))
        #plt.figure()
        #plt.imshow(img_prior[j])
        #plt.show()
    for top_range in top_ranges:
        top_similar = np.argpartition(intersectionoverunion[i, :].copy(), -top_range)[-top_range:]
        target = img_prior[top_similar, :, :]
        target = np.mean(target, axis=0).astype(np.float32)
        plt.figure()
        plt.imshow(target)
        plt.show()
        targets[top_ranges.index(top_range), i, :, :] = target
    # print(i)

       
for top_range in top_ranges:
    np.save('datasets/preselection_top%d.npy'%top_range, targets[top_ranges.index(top_range)])
