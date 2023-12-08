from sim_clr.sim_clr_model import SimCLRModel

import pytorch_lightning as pl
import torch
import torchvision

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils



def train_sim_clr(num_workers, batch_size, path_to_weights, path_to_data, seed, max_epochs, input_size):
    pl.seed_everything(seed)



    transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5)

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

    dataset_train_simclr = LightlyDataset(input_dir=path_to_data, transform=transform)

    dataset_test = LightlyDataset(input_dir=path_to_data, transform=test_transform)

    dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

    dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
    model = SimCLRModel()
    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
    trainer.fit(model, dataloader_train_simclr)
    torch.save(model.state_dict(), path_to_weights)
