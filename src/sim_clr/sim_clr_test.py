from collections import defaultdict
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from lightly.data import LightlyDataset
from lightly.transforms import utils


from sim_clr.sim_clr_model import SimCLRModel, generate_embeddings

# path_to_weights = '../../models/sim_clr_res18_e(100)_mango.pth'
# path_to_data = "../../fruit/lemon-dataset/images/"


def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)

def get_knn_accuracy(
    embeddings,
    filenames,
    results_dir,
    path_to_weights,
    path_to_data,
    n_neighbors=5,
    num_examples=6,
):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    neigh = nbrs.kneighbors(embeddings)
    try:
        save_dir = (
            max(
                [
                    int(d)
                    for d in os.listdir(results_dir)
                    if os.path.isdir(results_dir / str(d))
                ]
            )
            + 1
        )
    except:
        save_dir = 0
    res_folder = f"../results/{save_dir}"
    predictions = []
    for ns in neigh:
        _, true_idx = ns[0]
        ground_truth = pathlib.Path(filenames[true_idx]).parent
        pred_i = defaultdict(int)
        pred_d = defaultdict(int)
        for d, i in ns[1:]:
            p = pathlib.Path(filenames[i]).parent  # noqa: F821
            pred_d[p] += 1/(d**2)
            pred_i[p] += 1
        i_prediction = max(pred_i, key=pred_i.get)
        d_w_prediction = max(pred_d, key=pred_d.get)
        predictions.append(
            {
                "neighbors":pred_i,
                "distance_weighted":pred_d,
                "i_prediction":max(pred_i, key=pred_i.get),
                "d_w_prediction":max(pred_d, key=pred_d.get),
                "correct_i":i_prediction==ground_truth,
                "correct_d":d_w_prediction==ground_truth,
                
                }
            ) 
    acc_d = sum([p["correct_d"] for p in predictions])/len(predictions)   
    acc_i = sum([p["correct_i"] for p in predictions])/len(predictions)   
    data = json.dumps(predictions, indent=4)
    with open(f"../results/{save_dir}/predictions_{round(acc_d*100)}_{round(acc_i*100)}.json", "w") as f:
        f.write(data)
    
def plot_knn_examples(
    embeddings,
    filenames,
    results_dir,
    path_to_weights,
    path_to_data,
    n_neighbors=5,
    num_examples=6,
):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    try:
        save_dir = (
            max(
                [
                    int(d)
                    for d in os.listdir(results_dir)
                    if os.path.isdir(results_dir / str(d))
                ]
            )
            + 1
        )
    except:
        save_dir = 0
    os.makedirs(f"../results/{save_dir}")
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
        plt.savefig(f"../results/{save_dir}/knn_{idx}.png")
        annotations[str(idx)] = files
    # Dump annotations
    data = json.dumps(annotations, indent=4, sort_keys=True)
    with open(f"../results/{save_dir}/annotations.json", "w") as f:
        f.write(data)


def run_self_supervised_testing(args):
    print(
        f"Testing mode enabled. Using dataset: {args.dataset_test}, model: {args.model}, output directory: {args.output}"
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],
            ),
        ]
    )

    dataset_test = LightlyDataset(input_dir=args.dataset_test, transform=test_transform)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = SimCLRModel()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    embeddings, filenames = generate_embeddings(model, dataloader_test)

    plot_knn_examples(
        embeddings, filenames, args.output, args.weights, args.dataset_test
    )
    get_knn_accuracy(
        embeddings, filenames, args.output, args.weights, args.dataset_test
    )

