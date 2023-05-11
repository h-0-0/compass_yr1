import torch.nn as nn
from torch import optim, nn, sigmoid, flatten, zeros, sum, exp, randn_like
import torch.nn.functional as F
import lightning as L

# Class for an encoderfrom torch import optim, nn, sigmoid, flatten, zeros, sum, exp, randn_like
import torch.nn.functional as F
import lightning as L

class VGG(nn.Module):
    def __init__(self, data_name):
        # We create the network architecture depending on the dataset
        if(data_name == "CIFAR10"):
            # We customize the input size depending on the encoder used,
            super(VGG, self).__init__()
            # Based on VGG16 architecture
            self.encoder = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Block 2
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Block 3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Block 4
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Block 5
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 10),
            )
            self.flatten = nn.Flatten()
            self.device = None

        elif(data_name == "CIFAR100"):
            super(VGG, self).__init__()
            # Based on VGG16 architecture
            self.encoder = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5),

                # Block 2
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5),

                # Block 3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5),

                # Block 4
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5),

                # Block 5
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5)
            )
            self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 100),
            )
            self.flatten = nn.Flatten()
            self.device = None
        else:
            raise Exception("Not given valid dataset name must be: CIFAR10 or CIFAR100")

    # Compute forward pass
    def forward(self, x):
        out = self.encoder(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
# Class for an encoder
class Encoder(L.LightningModule):
    def __init__(self, num_input_channels: int, latent_dim: int, act_fn: object = nn.ReLU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, 16, kernel_size=3, padding=1, stride=1),
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), 
            nn.Linear(64 *4 *4 , latent_dim)
        )

    def forward(self, x):
        return self.net(x)


# Class or a decoder
class Decoder(L.LightningModule):
    def __init__(self, num_input_channels: int, latent_dim: int, act_fn: object = nn.ReLU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(latent_dim, 64*4*4), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    
# Class for an auteencoder, it uses the lightning module super class
class AutoEncoder(L.LightningModule):
    """
    The AutoEncoder class is a wrapper for the encoder and decoder classes. It employs the LightningModule super class.
    Args:
        latent_dim : Dimensionality of latent representation z
        encoder_class : Class of the encoder to use
        decoder_class : Class of the decoder to use
        num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
        width : Width of the input image
        height : Height of the input image
    """
    def __init__(
        self,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, latent_dim)
        self.decoder = decoder_class(num_input_channels, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        """Configures the optimizer, we use Adam with a learning rate of 1e-3."""
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Given a batch of images, this function returns the training loss."""
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Given a batch of images, this function returns the validation loss."""
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Given a batch of images, this function returns the test loss."""
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_epoch=True)