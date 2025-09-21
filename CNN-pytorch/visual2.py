import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from torchvision import datasets, transforms

# --- Load one MNIST digit ---
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
image, label = mnist[0]   # take the first sample
image = image.squeeze().numpy()  # convert 1x28x28 tensor -> 28x28 numpy

print("Label:", label)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(5, 5))

# Show the MNIST digit (28x28 grayscale)
im = ax.imshow(image, cmap="gray", vmin=0, vmax=1,
               origin="upper", interpolation="nearest")

# Gridlines
ax.set_xticks(np.arange(29))
ax.set_yticks(np.arange(29))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, color="lightgray", linewidth=0.5)

# Lock axes to pixel boundaries
ax.set_xlim(-0.5, 27.5)
ax.set_ylim(27.5, -0.5)
ax.set_aspect("equal")

# Create the 3x3 filter rectangle
rect = patches.Rectangle((-0.5, -0.5), 3, 3, linewidth=2,
                         edgecolor="red", facecolor="none")
ax.add_patch(rect)

# Valid top-left positions for 3x3 filter on 28x28 image
positions = [(r, c) for r in range(28 - 3 + 1) for c in range(28 - 3 + 1)]

def update(frame):
    r, c = positions[frame]
    rect.set_xy((c - 0.5, r - 0.5))  # align rectangle with pixels
    return (rect,)

ani = FuncAnimation(fig, update, frames=len(positions), interval=20, blit=True)

plt.show()
ani.save("mnist_filter.gif", writer="pillow", fps=10)