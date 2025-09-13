import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define loss function and gradient
def loss(x):
    return np.sin(3 * x) + (x - 2)**2

def gradient(x):
    return 3 * np.cos(3 * x) + 2 * (x - 2)

# Parameters
theta_momentum = 0.0
theta_nag = 0.0
momentum = 0.9
velocity_m = 0.0
velocity_n = 0.0
learning_rate = 0.09
steps = 40

# History storage
history_m = [(theta_momentum, loss(theta_momentum))]
history_n = [(theta_nag, loss(theta_nag))]
lookahead_history = []

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.linspace(-1.5, 4.5, 600)
y = loss(x)

for ax, title in zip(axes, ["Standard Momentum", "Nesterov Accelerated Gradient"]):
    ax.plot(x, y, label="Loss function")
    ax.set_xlabel("θ")
    ax.set_ylabel("J(θ)")
    ax.set_title(title)
    ax.grid(True)

# Scatter points and lookahead marker
scatter_m = axes[0].scatter([], [], color='blue')
arrow_m = axes[0].annotate('', xy=(0,0), xytext=(0,0),
                            arrowprops=dict(facecolor='red', arrowstyle='->'))

scatter_n = axes[1].scatter([], [], color='blue')
arrow_n = axes[1].annotate('', xy=(0,0), xytext=(0,0),
                            arrowprops=dict(facecolor='red', arrowstyle='->'))
lookahead_marker = axes[1].scatter([], [], color='orange', s=80, marker='x', label='Look-ahead')

axes[1].legend()

# Update function
def update(frame):
    global theta_momentum, velocity_m, theta_nag, velocity_n

    # Momentum update
    grad_m = gradient(theta_momentum)
    velocity_m = momentum * velocity_m - learning_rate * grad_m
    new_theta_m = theta_momentum + velocity_m
    history_m.append((new_theta_m, loss(new_theta_m)))

    # NAG update
    lookahead = theta_nag + momentum * velocity_n
    lookahead_history.append((lookahead, loss(lookahead)))
    grad_n = gradient(lookahead)
    velocity_n = momentum * velocity_n - learning_rate * grad_n
    new_theta_n = theta_nag + velocity_n
    history_n.append((new_theta_n, loss(new_theta_n)))

    # Update Momentum scatter and arrow
    scatter_m.set_offsets(history_m[-1])
    start_m = history_m[-2]
    end_m = history_m[-1]
    arrow_m.xy = end_m
    arrow_m.set_position(start_m)

    # Update NAG scatter and arrow
    scatter_n.set_offsets(history_n[-1])
    start_n = history_n[-2]
    end_n = history_n[-1]
    arrow_n.xy = end_n
    arrow_n.set_position(start_n)

    # Update look-ahead marker
    lookahead_marker.set_offsets(lookahead_history[-1])

    # Update positions
    theta_momentum = new_theta_m
    theta_nag = new_theta_n

    return scatter_m, arrow_m, scatter_n, arrow_n, lookahead_marker

# Animate
ani = FuncAnimation(fig, update, frames=steps, interval=500, blit=False, repeat=False)
plt.tight_layout()
plt.show()
