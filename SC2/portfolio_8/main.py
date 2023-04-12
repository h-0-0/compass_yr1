import os
from torch import optim, nn, utils, Tensor, cuda
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import VAE as vae
import data
from argparse import ArgumentParser
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal

def main(args):
    # Set the latent dimension
    l_dim = args.l_dim

    # Setup device
    device = "cuda" if cuda.is_available() else "cpu"
    print("Using device: ", device)

    # Setup model
    if args.ae:
        ED = vae.AutoEncoder(32, l_dim, num_input_channels=3)
    elif args.vae:
        ED = vae.VarAutoEncoder(32, l_dim, num_input_channels=3)

    # Setup data
    training_loader, test_loader = data.get_data_loader("CIFAR10", 64, device, num_workers=0)

    # Train the model 
    trainer = pl.Trainer(accelerator="auto", devices=4, num_nodes=1, strategy="ddp", max_epochs=500, plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)])
    trainer.fit(model=ED, train_dataloaders=training_loader)

    # Save the model
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    if args.ae:
        trainer.save_checkpoint("saved_models/autoencoder_"+ str(l_dim) + ".ckpt")
    elif args.vae:
        trainer.save_checkpoint("saved_models/varautoencoder_"+ str(l_dim) + ".ckpt")

    # Test the model 
    trainer.test(model=ED, dataloaders=test_loader)

    # choose your trained nn.Module
    encoder = ED.encoder
    encoder.eval()

# TODO: check image reconstruction ability (Notebook)
# TODO: plot latent space (Notebook)
# TODO: check performance for different latent dimensions (Notebook)
# TODO: get working on different datasets
# TODO: export encoder? / encode and save data as hdf5 file

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--l_dim", type=int, default=16)
    parser.add_argument("--vae", action='store_true')
    parser.set_defaults(vae=False)
    parser.add_argument("--ae", action='store_true')
    parser.set_defaults(ae=False)
    args = parser.parse_args()

    if args.ae == args.vae:
        raise ValueError("Please select either AE or VAE")
    # TRAIN
    main(args)