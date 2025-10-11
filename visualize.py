import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Read the data
data = pd.read_csv('simulation_data.csv')

# Extract positions
body1 = data[['body1_x', 'body1_y', 'body1_z']].values
body2 = data[['body2_x', 'body2_y', 'body2_z']].values
body3 = data[['body3_x', 'body3_y', 'body3_z']].values

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
ax.plot(body1[:, 0], body1[:, 1], body1[:, 2], 'r-', label='Body 1', linewidth=1)
ax.plot(body2[:, 0], body2[:, 1], body2[:, 2], 'b-', label='Body 2', linewidth=1)
ax.plot(body3[:, 0], body3[:, 1], body3[:, 2], 'g-', label='Body 3', linewidth=1)

# Plot starting positions
ax.scatter(body1[0, 0], body1[0, 1], body1[0, 2], c='red', s=100, marker='o')
ax.scatter(body2[0, 0], body2[0, 1], body2[0, 2], c='blue', s=100, marker='o')
ax.scatter(body3[0, 0], body3[0, 1], body3[0, 2], c='green', s=100, marker='o')

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Three-Body Problem Simulation')
ax.legend()

plt.savefig('three_body_simulation.png', dpi=300)
plt.show()

# Create animated version
from matplotlib.animation import FuncAnimation

fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111, projection='3d')

def animate(frame):
    ax2.clear()
    
    # Plot trajectories up to current frame
    ax2.plot(body1[:frame, 0], body1[:frame, 1], body1[:frame, 2], 'r-', alpha=0.5, linewidth=1)
    ax2.plot(body2[:frame, 0], body2[:frame, 1], body2[:frame, 2], 'b-', alpha=0.5, linewidth=1)
    ax2.plot(body3[:frame, 0], body3[:frame, 1], body3[:frame, 2], 'g-', alpha=0.5, linewidth=1)
    
    # Plot current positions
    ax2.scatter(body1[frame, 0], body1[frame, 1], body1[frame, 2], c='red', s=200, marker='o')
    ax2.scatter(body2[frame, 0], body2[frame, 1], body2[frame, 2], c='blue', s=200, marker='o')
    ax2.scatter(body3[frame, 0], body3[frame, 1], body3[frame, 2], c='green', s=200, marker='o')
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Z Position (m)')
    ax2.set_title(f'Three-Body Problem - Frame {frame}')
    ax2.legend(['Body 1', 'Body 2', 'Body 3'])

# Animate every 10th frame for speed
anim = FuncAnimation(fig2, animate, frames=range(0, len(data), 10), interval=50)
anim.save('three_body_animation.gif', writer='pillow', fps=20)
plt.show()

print("Visualization complete!")
