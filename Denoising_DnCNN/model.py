import torch.nn as nn
import torch.nn.init as init


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_filters=64, kernel_size=3, in_channels=1):
        """Pytorch implementation of DnCNN.
        https://github.com/SaoYan/DnCNN-PyTorch
        https://github.com/cszn/DnCNN
        https://github.com/yjn870/DnCNN-pytorch/

        # For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
        # The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.

        Parameters
        ----------
        depth : int
            Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17 for non-
            blind denoising and depth=20 for blind denoising.
        n_filters : int
            Number of filters on each convolutional layer.
        kernel_size : int tuple
            2D Tuple specifying the size of the kernel window used to compute activations.
        n_channels : int
            Number of image channels that the network processes (1 for grayscale, 3 for RGB)

        """
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, n_filters, kernel_size=kernel_size, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(n_filters, n_filters, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(n_filters, momentum=0.95),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(n_filters, in_channels, kernel_size, padding=1, bias=False))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        noise = self.layers(x)
        return x - noise
