import data
import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
# scenario, scenario_test = data.get_data_loader_encoder("CIFAR100", "RN50_clip", batch_size, device)

from sklearn.manifold import TSNE

# for task_id, train_taskset in enumerate(scenario):
#     train_dataloader = DataLoader(train_taskset, batch_size=batch_size, shuffle=True)
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')  
#     for batch, (X, y, t) in enumerate(train_dataloader):
#         X_tsne = TSNE(n_components=3).fit_transform(X)
#         # plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)
#         # Plot 3d scatter plot
#         ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=y)
#         if(batch>2):
#             break
#     plt.show()

# for task_id, (train_taskset, test_taskset) in enumerate(zip(scenario, scenario_test)):
#     train_dataloader = DataLoader(train_taskset, batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_taskset, batch_size=batch_size, shuffle=True)
#     x,y,t = train_taskset[0]
#     print(train_taskset.shape)
#     # print(f"task {task_id}, x in {x.shape}, y {y}")  
#     # print(x)
#     # Unflatten and plot image
#     # x = torch.reshape(x, (32, 32))
#     # Plot image
#     # plt.imshow(x, cmap='gray')
#     # plt.show()

#     x_tsne = TSNE(n_components=2).fit_transform(x.reshape(1,-1))
#     print(x_tsne)
#     plt.scatter(x_tsne[:,0], x_tsne[:,1], c=y)
#     plt.show()


# Code for printing dimensions of encoded data
encoder_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'RN50_clip', 'RN101_clip', 'RN50x4_clip', 'RN50x16_clip', 'RN50x64_clip', 'ViT-B/16', 'ViT-B/32', 'BiT-S-R50x1', 'BiT-S-R50x3', 'BiT-S-R101x1', 'BiT-S-R101x3', 'BiT-S-R152x2', 'BiT-S-R152x4', 'BiT-M-R50x1', 'BiT-M-R50x3', 'BiT-M-R101x1', 'BiT-M-R101x3', 'BiT-M-R152x2', 'BiT-M-R152x4', 'ViT-L/14_clip', 'ViT-B/16_clip', 'ViT-B/32_clip']
# we will cycle through each encoder in list above and print its dimensions
encoder_names = ['BiT-S-R50x1', 'BiT-S-R50x3', 'BiT-S-R101x1', 'BiT-S-R101x3', 'BiT-S-R152x2', 'BiT-S-R152x4', 'BiT-M-R50x1', 'BiT-M-R50x3', 'BiT-M-R101x1', 'BiT-M-R101x3', 'BiT-M-R152x2', 'BiT-M-R152x4', 'ViT-L/14_clip', 'ViT-B/16_clip', 'ViT-B/32_clip']
for name in encoder_names:
    torch.cuda.empty_cache()
    gc.collect()
    scenario, scenario_test = data.get_data_loader_encoder("CIFAR100", name,  batch_size, device)
    train_taskset = scenario[0]
    train_dataloader = DataLoader(train_taskset, batch_size=batch_size, shuffle=True)
    for batch, (X, y, t) in enumerate(train_dataloader):
        # write to a file the dimensions of the encoded data
        with open("encoded_data_dimensions.txt", "a") as f:
            f.write(f"Encoder: {name}, X shape: {X.shape}, y shape: {y.shape}, t shape: {t.shape}\n")
        break