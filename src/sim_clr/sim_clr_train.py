import os
from sim_clr.sim_clr_model import LossLoggingCallback, SimCLRModel

import pytorch_lightning as pl
import torch
import torchvision

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
from torch.utils.data import random_split

import matplotlib.pyplot as plt

import pandas as pd   

def plot_validation_loss(logs_directory):
    path_to_metrics = os.path.join(logs_directory, 'metrics.csv')
    df = pd.read_csv(path_to_metrics)
    df.ffill(inplace=True)
    df.drop_duplicates("epoch", keep="last",inplace=True)
    df.set_index("epoch", inplace=True)
    df.drop("step", axis="columns",inplace=True)
    df.plot()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(['Training', 'Validation'])
    plt.savefig(os.path.join(logs_directory, 'loss.png'))



def train_sim_clr(
    num_workers,
    batch_size,
    path_to_weights,
    path_to_data,
    seed,
    max_epochs,
    input_size,
    validate
):
    pl.seed_everything(seed)

    transform = SimCLRTransform(input_size=input_size, vf_prob=0.5,hf_prob=0.5, rr_prob=0.5)

    # We create a torchvision transformation for embedding the dataset after
    # training


    train_set = LightlyDataset(input_dir=path_to_data, transform=transform)
    # use 20% of training data for validation
   

    
    model = SimCLRModel(max_epochs)
    if validate:
        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)


        dataloader_validation = torch.utils.data.DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        dataloader_train_simclr = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        # callback = LossLoggingCallback()
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            # callbacks=[callback],
            devices=1,
            accelerator="auto",
            log_every_n_steps=12,
        )
        trainer.fit(model, dataloader_train_simclr, dataloader_validation)
        plot_validation_loss(trainer.logger.log_dir)
    else:
        dataloader_train_simclr = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=1,
            accelerator="auto",
        )
        trainer.fit(model, dataloader_train_simclr)

    torch.save(model.state_dict(), path_to_weights)
