import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

experiment = "datasets/split/60.csv"

class PartialDataset(Dataset):
    def __init__(self, mnist_dir, pre_selection_dir):
        self.examples = np.loadtxt(mnist_dir, delimiter=",")[:5000]
        self.pre_selections = np.load(pre_selection_dir)
        print(self.pre_selections.shape)
    def __len__(self):
        assert self.examples.shape[0] == self.pre_selections.shape[0]
        return self.examples.shape[0]
    def __getitem__(self, item): #item = index
        input_img = self.examples[item, 1:].reshape(28, 28) / 255.
        unmasked = np.copy(input_img)
        input_img[9:19, 9:19] = 0.5
        pre_sele_img = self.pre_selections[item]
        #plt.figure()
        #plt.imshow(np.squeeze(pre_sele_img))
        #plt.show()
        unmasked = transform.resize(unmasked, (64, 64), preserve_range=True)
        input_img =  transform.resize(input_img, (64, 64), preserve_range=True)
        pre_sele_img =  transform.resize(pre_sele_img, (64, 64), preserve_range=True)
        unmasked = torch.from_numpy(np.expand_dims(unmasked, 0))
        input_img = torch.from_numpy(np.expand_dims(input_img, 0))
        pre_sele_img = torch.from_numpy(np.expand_dims(pre_sele_img, 0))

        example = {
                   'input': input_img, 
                   'pre_sele': pre_sele_img,
                    'unmasked': unmasked
                  }

        return example




if __name__ == '__main__':
    dataset = PartialDataset('datasets/split/preselection_top8_60.npy')

    print(len(dataset))
    print(dataset[0]['input'])
    print(dataset[0]['pre_sele'])
