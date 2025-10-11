import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the data
data = pd.read_csv('simulation_data.csv')

# Extract positions
body1 = data[['body1_x', 'body1_y', 'body1_z']].values
body2 = data[['body2_x', 'body2_y', 'body2_z']].values
body3 = data[['body3_x', 'body3_y', 'body3_z']].values
time = data['time'].values

# Downsample for smoother animation (every 10th frame)
skip = 10
frames_data = []

for i in range(0, len(data), skip):
    frame = go.Frame(
        data=[
            # Trajectories up to this point
            go.Scatter3d(
                x=body1[:i+1, 0], y=body1[:i+1, 1], z=body1[:i+1, 2],
                mode='lines',
                name='Body 1 Trail',
                line=dict(color='red', width=2),
                opacity=0.6
            ),
            go.Scatter3d(
                x=body2[:i+1, 0], y=body2[:i+1, 1], z=body2[:i+1, 2],
                mode='lines',
                name='Body 2 Trail',
                line=dict(color='blue', width=2),
                opacity=0.6
            ),
            go.Scatter3d(
                x=body3[:i+1, 0], y=body3[:i+1, 1], z=body3[:i+1, 2],
                mode='lines',
                name='Body 3 Trail',
                line=dict(color='green', width=2),
                opacity=0.6
            ),
            # Current positions
            go.Scatter3d(
                x=[body1[i, 0]], y=[body1[i, 1]], z=[body1[i, 2]],
                mode='markers',
                name='Body 1',
                marker=dict(size=15, color='red', symbol='circle')
            ),
            go.Scatter3d(
                x=[body2[i, 0]], y=[body2[i, 1]], z=[body2[i, 2]],
                mode='markers',
                name='Body 2',
                marker=dict(size=15, color='blue', symbol='circle')
            ),
            go.Scatter3d(
                x=[body3[i, 0]], y=[body3[i, 1]], z=[body3[i, 2]],
                mode='markers',
                name='Body 3',
                marker=dict(size=15, color='green', symbol='circle')
            )
        ],
        name=f'frame_{i}',
        layout=go.Layout(title_text=f"Time: {time[i]:.2f} s")
    )
    frames_data.append(frame)

# Create initial figure
fig = go.Figure(
    data=frames_data[0].data,
    layout=go.Layout(
        title="Interactive Three-Body Simulation",
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)',
            aspectmode='data'
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=1.15,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 50, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 0}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}])
                ]
            )
        ],
        sliders=[dict(
            active=0,
            yanchor="top",
            y=0.1,
            xanchor="left",
            x=0.1,
            currentvalue=dict(prefix="Frame: ", visible=True, xanchor="right"),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=[dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                       method="animate",
                       label=str(k)) for k, f in enumerate(frames_data)]
        )]
    ),
    frames=frames_data
)

fig.write_html('three_body_animated.html')
print("Animated interactive visualization saved to three_body_animated.html")
fig.show()
