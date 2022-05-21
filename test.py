import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchvision.utils import save_image
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
train_set = datasets.MNIST(root='./res/data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./res/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

#分析sigmoid和relu
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.model = nn.Sequential(
            # nn.Linear(784, 784),
            #nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

model = D()

for batch in test_loader:
    img, _ = batch
    img = img.view(img.size(0), -1)
    print(img)
    outputs = model(img)
    print(outputs)
    outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
    k = torch.Tensor(np.array([[0.2000,0.5000],
                               [0.7000,0.9000]]))
    # plt.imshow(outputs.squeeze(), cmap="gray")
    plt.imshow(k, cmap="gray")
    plt.show()
    break