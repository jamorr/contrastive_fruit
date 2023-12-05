import os
from lightly.data import LightlyDataset
from lightly.embedding import SelfSupervisedEmbedding
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Set the path to your dataset
data_path = '/path/to/your/dataset'

# Define the transformation applied to your images
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
])
