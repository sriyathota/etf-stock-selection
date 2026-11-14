import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matryoshka_utils import Matryoshka_Linear_Layer, MatryoshkaLoss
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np

# 1. Define the transformation for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. Create the train_loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print("train_loader defined successfully!")

# -------------------------------------------------------------------

class MNIST_Matryoshka_Model(nn.Module):
    def __init__(self, nesting_list):
        super(MNIST_Matryoshka_Model, self).__init__()
        # Backbone remains 1024-dim
        self.backbone = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        # FIXED: Arguments now match matryoshka_utils.py exactly
        self.mrl_layer = Matryoshka_Linear_Layer(
            input_dim=1024,
            output_dim=10,
            nesting_list=nesting_list,
            efficient=True
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.backbone(x)
        return self.mrl_layer(z)

# Initialize
my_nesting_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
model = MNIST_Matryoshka_Model(my_nesting_list)

print("Model created and ready for training!")

# -------------------------------------------------------------------

# 1. Setup loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting Training...")

# 2. The Loop
for epoch in range(1): # We will start with 1 'Epoch' (one full pass through the data)
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: This gives us a TUPLE of 5 predictions
        nesting_logits = model(images)

        # Calculate TOTAL Matryoshka Loss
        # We add up the loss for the 8-dim version, 16-dim version, etc.
        total_loss = 0
        for logits in nesting_logits:
            total_loss += criterion(logits, labels)

        # Backward pass: The model learns from all 5 sizes at once!
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # Print progress every 200 batches
        if i % 200 == 199:
            print(f"Batch {i+1}: Average Loss: {running_loss / 200:.3f}")
            running_loss = 0.0
print("Training Complete!")

# -------------------------------------------------------------------

def visualize_latent_space(model, loader, dimensions_to_show=[1, 2, 4, 8, 16, 1024]):
    model.eval()
    images, labels = next(iter(loader))

    # Get the 128-dimensional latent vector (z) from the backbone
    with torch.no_grad():
        x = images.view(images.size(0), -1)
        z = model.backbone(x)

    plt.figure(figsize=(18, 5))

    for i, dim in enumerate(dimensions_to_show):
        # Slice the latent space to the current Matryoshka dimension
        z_slice = z[:, :dim].cpu().numpy()

        if dim == 1:
            # Create 2D coords: X = the 1D value, Y = 0
            z_2d = np.zeros((z_slice.shape[0], 2))
            z_2d[:, 0] = z_slice.reshape(-1)
        elif dim == 2:
            z_2d = z_slice
        else:
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_slice)

        # 3. Plot the clusters
        plt.subplot(1, len(dimensions_to_show), i + 1)
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.title(f"Latent Space (Dim: {dim})")
        plt.colorbar(scatter, label="Digit Class")

    plt.show()

# Run the visualization
visualize_latent_space(model, train_loader)
