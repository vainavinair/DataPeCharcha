
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torch
from skimage.transform import resize

# Load MNIST
mnist = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

# Pick an image that is a '7'
for img, label in mnist:
    if label == 7:
        image = img
        break

# Convert to numpy (28x28)
img = image.numpy().squeeze()

# Shift the image right by 5 pixels
shifted = np.zeros_like(img)
shifted[:, 5:] = img[:, :-5]

# Downsample both images to 16x16
img_resized = resize(img, (16, 16), anti_aliasing=True)
shifted_resized = resize(shifted, (16, 16), anti_aliasing=True)

# Plot Original and Shifted with numeric overlay
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, data, title in zip(axes, [img_resized, shifted_resized], ["Original '7' (16x16)", "Shifted '7' (16x16)"]):
    ax.imshow(data, cmap="gray", extent=[0, 16, 16, 0])
    for i in range(16):
        for j in range(16):
            ax.text(j + 0.5, i + 0.5, f"{data[i, j]:.2f}", ha="center", va="center", color="red", fontsize=8)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

# Print the 16x16 matrices
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print("Original '7' Matrix (16x16):\n", img_resized, "\n")
print("Shifted '7' Matrix (16x16):\n", shifted_resized, "\n")

# Flatten into tensors
flat_original = torch.tensor(img_resized).flatten()
flat_shifted = torch.tensor(shifted_resized).flatten()

print("Original '7' Flattened Tensor (256 values):")
print(flat_original.numpy(), "\n")

print("Shifted '7' Flattened Tensor (256 values):")
print(flat_shifted.numpy())
