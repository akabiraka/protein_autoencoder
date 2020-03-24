import torch
import torch.nn as nn


class SimpleLinearAutoencoder(nn.Module):
    """docstring for Autoencoder."""

    def __init__(self, input_size):
        super(SimpleLinearAutoencoder, self).__init__()
        dims = [input_size, 784, 256, 128, 64, 32, 16]

        encoders = []
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            encoders += [layer, nn.ReLU(inplace=True)]

        decoders = []
        for i in range(len(dims) - 1, 0, -1):
            layer = nn.Linear(dims[i], dims[i - 1])
            decoders += [layer, nn.ReLU(inplace=True)]

        self.encoder = nn.Sequential(*encoders)
        self.decoder = nn.Sequential(*decoders)

    def forward(self, x):
        latent_vector = self.encoder(x)
        out = self.decoder(latent_vector)
        return out, latent_vector
