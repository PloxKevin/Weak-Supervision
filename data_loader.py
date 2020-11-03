import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader

experiment = "datasets/split/60.csv"

class PartialDataset(Dataset):
    def __init__(self, mnist_dir):
        self.examples = np.loadtxt(experiment, delimiter=",")
        self.pre_selections = np.load(mnist_dir)
        print(self.pre_selections.shape)
    def __len__(self):
        assert self.examples.shape[0] == self.pre_selections.shape[0]
        return self.examples.shape[0]
    def __getitem__(self, item): #item = index
        input_img = self.examples[item, 1:].reshape(1, 28, 28) / 255.
        #(1,28,28) here or in the training
        input_img[9:19, 9:19] = 0.5
        pre_sele_img = self.pre_selections[item]

        input_img =  transform.resize(input_img, (64, 64), preserve_range=True)
        pre_sele_img =  transform.resize(pre_sele_img, (64, 64), preserve_range=True)

        input_img = torch.from_numpy(np.expand_dims(input_img, 0))
        pre_sele_img = torch.from_numpy(np.expand_dims(pre_sele_img, 0))

        example = {
                   'input': input_img, 
                   'pre_sele': pre_sele_img
                  }

        return example

class PriorDataset(Dataset):

    def __init__(self):
        self.examples = np.loadtxt(experiment, delimiter=",")

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, item):
        input_img = self.examples[item, 1:].reshape(1,28, 28) / 255.
        input_img[9:19, 9:19] = 0.5

        input_img =  transform.resize(input_img, (64, 64), preserve_range=True)
        input_img = torch.from_numpy(np.expand_dims(input_img, 0))

        example = {
                   'prior': input_img,
                 }

        return example



if __name__ == '__main__':
    #dataset = PartialDataset('datasets/split/preselection_top1_60.npy')

    #print(len(dataset))
    #print(dataset[0]['input'])
    #print(dataset[0]['pre_sele'])

    dataset = PriorDataset()
    print(len(dataset))
    print(dataset[0]['prior'])
