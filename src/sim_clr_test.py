
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from sim_clr import SimCLRModel, generate_embeddings

path_to_weights = '../models/sim_clr_res18_e(100)_mango.pth'
path_to_data = "../fruit/lemon-dataset/images/"

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    try:
        save_dir = max([int(d) for d in os.listdir('../results/') if os.path.isdir(f'../results/{d}')]) + 1 
    except:
        save_dir = 0
    os.makedirs(f'../results/{save_dir}')
    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    annotations = {}
    annotations["model"] = path_to_weights
    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        files = []
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(path_to_data, filenames[neighbor_idx])
            files.append(fname)
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")
        plt.savefig(f'../results/{save_dir}/knn_{idx}.png')
        annotations[str(idx)] = files
    # Dump annotations
    data = json.dumps(annotations,indent=4, sort_keys=True)
    with open(f'../results/{save_dir}/annotations.json', 'w') as f:
        f.write(data)


num_workers = 4
batch_size = 256
seed = 1
input_size = 128
num_ftrs = 32
# We create a torchvision transformation for embedding the dataset after
# training
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)
dataset_test = LightlyDataset(input_dir=path_to_data, transform=test_transform)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
model = SimCLRModel()
model.load_state_dict(torch.load(path_to_weights))
model.eval()
embeddings, filenames = generate_embeddings(model, dataloader_test)

plot_knn_examples(embeddings, filenames)