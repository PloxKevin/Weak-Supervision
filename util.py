import torch
import matplotlib.pyplot as plt


def torch_img_visualization(num_tensor, tensor_list):
    for i in range(num_tensor):
        tensor = tensor_list[i]
        tensor = tensor.cpu().numpy()
        tensor = tensor.transpose((0, 2, 3, 1))
        plt.figure()
        plt.imshow(tensor[0, :, :, 0].squeeze())
    plt.show()


