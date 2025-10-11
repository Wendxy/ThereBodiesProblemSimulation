import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# Read the data
data = pd.read_csv('simulation_data.csv')

# Extract positions
body1 = data[['body1_x', 'body1_y', 'body1_z']].values
body2 = data[['body2_x', 'body2_y', 'body2_z']].values
body3 = data[['body3_x', 'body3_y', 'body3_z']].values

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initial frame
frame = 0
trail_length = 100

def update_plot(frame):
    ax.clear()
    
    # Determine trail range
    start_idx = max(0, frame - trail_length)
    
    # Plot trajectories with trailing effect
    ax.plot(body1[start_idx:frame+1, 0], 
            body1[start_idx:frame+1, 1], 
            body1[start_idx:frame+1, 2], 
            'r-', linewidth=1, alpha=0.6, label='Body 1')
    ax.plot(body2[start_idx:frame+1, 0], 
            body2[start_idx:frame+1, 1], 
            body2[start_idx:frame+1, 2], 
            'b-', linewidth=1, alpha=0.6, label='Body 2')
    ax.plot(body3[start_idx:frame+1, 0], 
            body3[start_idx:frame+1, 1], 
            body3[start_idx:frame+1, 2], 
            'g-', linewidth=1, alpha=0.6, label='Body 3')
    
    # Plot current positions
    ax.scatter(body1[frame, 0], body1[frame, 1], body1[frame, 2], 
               c='red', s=200, marker='o', edgecolors='darkred', linewidths=2)
    ax.scatter(body2[frame, 0], body2[frame, 1], body2[frame, 2], 
               c='blue', s=200, marker='o', edgecolors='darkblue', linewidths=2)
    ax.scatter(body3[frame, 0], body3[frame, 1], body3[frame, 2], 
               c='green', s=200, marker='o', edgecolors='darkgreen', linewidths=2)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'Three-Body Problem - Frame {frame}/{len(data)-1}')
    ax.legend()
    
    # Set consistent axis limits
    all_positions = np.vstack([body1, body2, body3])
    margin = 0.1 * (all_positions.max() - all_positions.min())
    ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
    ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
    ax.set_zlim(all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin)

# Create slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(data)-1, valinit=0, valstep=1)

def on_slider_change(val):
    frame = int(slider.val)
    update_plot(frame)
    plt.draw()

slider.on_changed(on_slider_change)

# Create play/pause buttons
ax_play = plt.axes([0.3, 0.05, 0.1, 0.04])
ax_pause = plt.axes([0.6, 0.05, 0.1, 0.04])
btn_play = Button(ax_play, 'Play')
btn_pause = Button(ax_pause, 'Pause')

# Animation variables
anim = None
is_playing = False

def play(event):
    global anim, is_playing
    if not is_playing:
        is_playing = True
        anim = FuncAnimation(fig, lambda i: slider.set_val((slider.val + 1) % len(data)),
                            frames=len(data), interval=50, repeat=True)
        plt.draw()

def pause(event):
    global anim, is_playing
    if is_playing and anim:
        is_playing = False
        anim.event_source.stop()

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)

# Initial plot
update_plot(0)
plt.show()
