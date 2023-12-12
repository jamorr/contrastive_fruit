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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from lightly.data import LightlyDataset
from lightly.transforms import utils


from sim_clr.sim_clr_model import SimCLRModel, generate_embeddings



def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def get_knn_accuracy(
    embeddings:np.ndarray,
    filenames:list,
    res_folder:str|pathlib.Path,
    n_neighbors=6,
):
    """Collects predictions made by KNN into a json file"""
    # make model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    # get predictions ids
    dist, idxs = nbrs.kneighbors(embeddings)

    # list to store prediciton data
    predictions = []

    for ns in zip(dist, idxs):
        # get pairs of distance and index
        all_ns = list(zip(*ns))

        # closest point should be itself (embedding is deterministic)
        _, true_idx = all_ns[0]

        # get label from image path for ground truth
        ground_truth = pathlib.Path(filenames[int(true_idx)]).parent.as_posix()

        # dicts to store values for each label
        pred_i = defaultdict(int)
        pred_d = defaultdict(int)

        # iterate over neighbors and get associated labels
        # use labels to create predictions
        for d, i in all_ns[1:]:
            p = pathlib.Path(filenames[int(i)]).parent.as_posix()

            pred_d[p] += 1 / (d**2)
            pred_i[p] += 1

        # find final prediction from dict
        i_prediction = max(pred_i, key=pred_i.get)
        d_w_prediction = max(pred_d, key=pred_d.get)

        # add prediction for point to list
        predictions.append(
            {
                "neighbors": pred_i,
                "distance_weighted": pred_d,
                "gt": ground_truth,
                "i_prediction": max(pred_i, key=pred_i.get),
                "d_w_prediction": max(pred_d, key=pred_d.get),
                "correct_i": i_prediction == ground_truth,
                "correct_d": d_w_prediction == ground_truth,
            }
        )

    # calculate the accuracy of each model over all predictions
    acc_d = sum([p["correct_d"] for p in predictions]) / len(predictions)
    acc_i = sum([p["correct_i"] for p in predictions]) / len(predictions)

    # create json dump for predictions
    data = json.dumps(predictions, indent=4)
    with open(
        res_folder / f"predictions_{round(acc_d*100)}_{round(acc_i*100)}.json",
        "w"
    ) as f:
        f.write(data)

    # list of ground truth and predictions for each image
    y_true = [p["gt"] for p in predictions]
    y_pred_d = [p["d_w_prediction"] for p in predictions]
    y_pred_i = [p["i_prediction"] for p in predictions]

    # Make confusion matrices for both and save
    cm_d = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred_d,
        normalize="true",
        cmap="Blues",
        colorbar=False
    )
    cm_d.figure_.savefig(res_folder / "distance_confusion.png", bbox_inches="tight")
    cm_i = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred_i,
        normalize="true",
        cmap="Blues",
        colorbar=False
    )
    cm_i.figure_.savefig(res_folder / "voting_confusion.png", bbox_inches="tight")

    # print prediction results
    print(
        f"""
KNN # neighbors
    {n_neighbors}
Distance weighted
    {classification_report(y_true, y_pred_d)}
Unweighted
    {classification_report(y_true, y_pred_i)}
"""
    )


def plot_knn_examples(
    embeddings,
    filenames,
    results_dir,
    path_to_weights,
    path_to_data,
    n_neighbors=6,
    num_examples=6,
):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

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
        plt.savefig(results_dir / f"knn_{idx}.png", bbox_inches="tight")
        annotations[str(idx)] = files
    # Dump annotations
    data = json.dumps(annotations, indent=4, sort_keys=True)
    with open(results_dir / "annotations.json", "w") as f:
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
    try:
        model = SimCLRModel()
        model.load_state_dict(torch.load(args.weights))
    except RuntimeError:
        model = SimCLRModel.load_from_checkpoint(args.weights, map_location=torch.device("cpu"))

    model.eval()

    embeddings, filenames = generate_embeddings(model, dataloader_test)
    results_dir = args.output
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

    res_path = results_dir / str(save_dir)
    os.makedirs(res_path)


    plot_knn_examples(
        embeddings, filenames, res_path, args.weights, args.dataset_test
    )
    get_knn_accuracy(
        embeddings, filenames, res_path
    )
