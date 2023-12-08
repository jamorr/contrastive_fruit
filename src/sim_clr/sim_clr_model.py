import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from sklearn.preprocessing import normalize

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class LossLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, *args):
        print(*args)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_losses.append(outputs.item())

    def on_epoch_end(self, trainer, pl_module):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0.0
        avg_val_loss = sum(self.val_losses) / len(self.val_losses) if self.val_losses else 0.0
        trainer.logger.experiment.add_scalars("loss", {"train": avg_train_loss, "val": avg_val_loss}, trainer.current_epoch)
        self.train_losses = []  # Reset for the next epoch
        self.val_losses = []

class SimCLRModel(pl.LightningModule):
    def __init__(self, max_epochs:int=100):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        self.max_epochs = max_epochs
        self.criterion = NTXentLoss()
        self.validation_loss = 0.0
        self.validation_set_size = 0

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.validation_loss += loss.item()
        self.validation_set_size += 1
        return loss

    def on_validation_epoch_end(self):
        avg_validation_loss = self.validation_loss / self.validation_set_size
        self.log("val_loss_ssl", avg_validation_loss)
        self.validation_set_size = 0
        

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
    


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames



