import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, grayscale=True):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.filenames = sorted(os.listdir(noisy_dir))

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]

        noisy_path = os.path.join(self.noisy_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        noisy_img = self.transform(noisy_img)
        clean_img = self.transform(clean_img)

        # The dataset has images of two sizes -
        # torch.Size([1, 321, 481]) - 53
        # torch.Size([1, 481, 321]) - 15
        # We need to ensure that the images are of the same size
        if noisy_img.shape[1] != 321:  # Check height
            noisy_img = noisy_img.permute(0, 2, 1)  # Swap height and width
            clean_img = clean_img.permute(0, 2, 1)

        return noisy_img, clean_img


def load_datasets(
    noisy_dir="CBSD68-dataset\CBSD68\\noisy35",
    clean_dir="CBSD68-dataset\CBSD68\original_png",
    grayscale=True,
    test_size=0.2,
):
    dataset = DenoisingDataset(noisy_dir, clean_dir, grayscale)

    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


def load_dataloaders(
    train_dataset=None,
    test_dataset=None,
    batch_size=8,
    shuffle=True,
):
    if train_dataset is None or test_dataset is None:
        train_dataset, test_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def show(x, title=None, cbar=False, figsize=None):
    """Displays a single image with optional title and colorbar.

    Parameters:
        x (Tensor or ndarray): Image data (PyTorch Tensor or NumPy array).
        title (str, optional): Title for the image.
        cbar (bool, optional): Whether to display a colorbar.
        figsize (tuple, optional): Figure size.
    """
    if torch.is_tensor(x):  # Check if x is a PyTorch Tensor
        x = x.detach().cpu().numpy()  # Convert to NumPy

    if x.ndim == 3 and x.shape[0] in [1, 3]:  # If channel-first (C, H, W), convert to (H, W, C)
        x = np.transpose(x, (1, 2, 0))  # Convert (C, H, W) -> (H, W, C) for RGB

    plt.figure(figsize=figsize)
    cmap = "gray" if x.ndim == 2 or x.shape[2] == 1 else None  # Ensure correct colormap
    plt.imshow(x, interpolation="nearest", cmap=cmap)

    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.axis("off")  # Remove axis for better visualization
    plt.show()
