import data
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, test_loader = data.get_data_loader_encoder("CIFAR100",  64, "cpu")
print(train_loader)