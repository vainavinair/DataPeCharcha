import matplotlib.pyplot as plt
import numpy as np

# Simulated accuracy data
epochs = np.arange(1, 21)

# Accuracy for training from scratch (flat low performance)
scratch_acc = 0.2 + 0.02 * np.log(epochs)  # slow growth

# Accuracy for transfer learning (quick improvement)
transfer_acc = 0.2 + 0.6 * (1 - np.exp(-0.3 * epochs))  # fast growth to high accuracy

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot: Training from scratch ---
axes[0].plot(epochs, scratch_acc, marker='o', color='red', linewidth=2)
axes[0].set_title("Training from Scratch", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Epochs", fontsize=12)
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].set_ylim(0, 1)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Annotate final accuracy
axes[0].annotate(f"{scratch_acc[-1]*100:.1f}%", 
                 xy=(epochs[-1], scratch_acc[-1]),
                 xytext=(epochs[-1]-3, scratch_acc[-1]+0.05),
                 arrowprops=dict(facecolor='black', arrowstyle="->"))

# --- Plot: Transfer Learning ---
axes[1].plot(epochs, transfer_acc, marker='o', color='green', linewidth=2)
axes[1].set_title("Transfer Learning", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Epochs", fontsize=12)
axes[1].set_ylabel("Accuracy", fontsize=12)
axes[1].set_ylim(0, 1)
axes[1].grid(True, linestyle='--', alpha=0.5)

# Annotate final accuracy
axes[1].annotate(f"{transfer_acc[-1]*100:.1f}%", 
                 xy=(epochs[-1], transfer_acc[-1]),
                 xytext=(epochs[-1]-3, transfer_acc[-1]+0.05),
                 arrowprops=dict(facecolor='black', arrowstyle="->"))

# Add a main title
fig.suptitle("The Cold Start Problem: Training from Scratch vs Transfer Learning", fontsize=16, fontweight='bold')

# Tight layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
plt.savefig("TransferLearning-pytorch/cold_start_transfer_learning.png", dpi=300)

plt.show()
