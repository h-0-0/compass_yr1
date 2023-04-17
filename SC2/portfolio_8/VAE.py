from torch import optim, nn, sigmoid, flatten, zeros, sum, exp, randn_like
import torch.nn.functional as F
import lightning as L

# Class for an encoderfrom torch import optim, nn, sigmoid, flatten, zeros, sum, exp, randn_like
import torch.nn.functional as F
import lightning as L

# Class for an encoder
class Encoder(L.LightningModule):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        # self.net = nn.Sequential(
        #     nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
        #     act_fn(),
        #     nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
        #     act_fn(),
        #     nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
        #     act_fn(),
        #     nn.Flatten(),  # Image grid to single feature vector
        #     nn.Linear(2 * 16 * c_hid, latent_dim),
        # )

        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2), 
            act_fn(),
            nn.Flatten(), 
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

# Class for a variational encoder
class VarEncoder(L.LightningModule):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn()
        )
        self.flatten = nn.Flatten()  # Image grid to single feature vector
        self.fc_mu = nn.Linear(c_hid*8*2*2, latent_dim)
        self.fc_logvar = nn.Linear(c_hid*8*2*2, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Class or a decoder
class Decoder(L.LightningModule):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        # self.net = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
        #     ),  # 4x4 => 8x8
        #     act_fn(),
        #     nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
        #     act_fn(),
        #     nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.ConvTranspose2d(
        #         c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
        #     ),  # 16x16 => 32x32
        #     nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        # )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                4 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2), 
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2),  
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, padding=1, stride=2), 
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
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
        base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
        latent_dim : Dimensionality of latent representation z
        encoder_class : Class of the encoder to use
        decoder_class : Class of the decoder to use
        num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
        width : Width of the input image
        height : Height of the input image
    """
    def __init__(
        self,
        base_channel_size: int,
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
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
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

# Class for variational autoencoder
class VarAutoEncoder(L.LightningModule):
    """The AutoEncoder class is a wrapper for the VarEncoder and Decoder classes. It employs the LightningModule super class. It is a variational autoencoder.
    Args:
        base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
        latent_dim : Dimensionality of latent representation z
        encoder_class : Class of encoder to use
        decoder_class : Class of decoder to use
        num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
        width : Width of the image to reconstruct
        height : Height of the image to reconstruct
    """
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = VarEncoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = exp(0.5 * logvar)
        eps = randn_like(std)
        return eps * std + mu
    
    def _get_total_loss(self, batch):
        """Given a batch of images, this function returns the total loss (MSE in our case) which is the reconstruction loss + the KL divergence loss, aswell as the reconstruction loss and the KL divergence loss separately."""
        x, _ = batch  # We do not need the labels
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        reconstruction_loss = F.mse_loss(x, x_hat, reduction="none")
        reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        kl_div_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div_loss /= x.shape[0]
        total_loss = reconstruction_loss + self.hparams.beta * kl_div_loss
        return reconstruction_loss, kl_div_loss, total_loss
    
    def training_step(self, batch, batch_idx):
        """Given a batch of images, this function returns the training loss."""
        reconstruction_loss, kl_div_loss, total_loss = self._get_total_loss(batch)

        self.log("train_loss", total_loss)
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("kl_div_loss", kl_div_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Given a batch of images, this function returns the validation loss."""
        reconstruction_loss, kl_div_loss, total_loss = self._get_total_loss(batch)

        self.log("val_loss", total_loss)
        self.log("val_reconstruction_loss", reconstruction_loss)
        self.log("val_kl_div_loss", kl_div_loss)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Given a batch of images, this function returns the test loss."""
        reconstruction_loss, kl_div_loss, total_loss = self._get_total_loss(batch)

        self.log("test_loss", total_loss)
        self.log("test_reconstruction_loss", reconstruction_loss)
        self.log("test_kl_div_loss", kl_div_loss)

        return total_loss

    def configure_optimizers(self):
        """Configures the optimizer, we use Adam with a learning rate of 1e-3."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
