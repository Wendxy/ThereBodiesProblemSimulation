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

# Create figure
fig = go.Figure()

# Add trajectory traces
fig.add_trace(go.Scatter3d(
    x=body1[:, 0], y=body1[:, 1], z=body1[:, 2],
    mode='lines',
    name='Body 1 Trail',
    line=dict(color='red', width=2),
    opacity=0.6
))

fig.add_trace(go.Scatter3d(
    x=body2[:, 0], y=body2[:, 1], z=body2[:, 2],
    mode='lines',
    name='Body 2 Trail',
    line=dict(color='blue', width=2),
    opacity=0.6
))

fig.add_trace(go.Scatter3d(
    x=body3[:, 0], y=body3[:, 1], z=body3[:, 2],
    mode='lines',
    name='Body 3 Trail',
    line=dict(color='green', width=2),
    opacity=0.6
))

# Add current position markers
fig.add_trace(go.Scatter3d(
    x=[body1[-1, 0]], y=[body1[-1, 1]], z=[body1[-1, 2]],
    mode='markers',
    name='Body 1',
    marker=dict(size=10, color='red', symbol='circle')
))

fig.add_trace(go.Scatter3d(
    x=[body2[-1, 0]], y=[body2[-1, 1]], z=[body2[-1, 2]],
    mode='markers',
    name='Body 2',
    marker=dict(size=10, color='blue', symbol='circle')
))

fig.add_trace(go.Scatter3d(
    x=[body3[-1, 0]], y=[body3[-1, 1]], z=[body3[-1, 2]],
    mode='markers',
    name='Body 3',
    marker=dict(size=10, color='green', symbol='circle')
))

# Update layout
fig.update_layout(
    title='Interactive Three-Body Problem Simulation',
    scene=dict(
        xaxis_title='X Position (m)',
        yaxis_title='Y Position (m)',
        zaxis_title='Z Position (m)',
        aspectmode='data'
    ),
    showlegend=True,
    hovermode='closest'
)

# Save as interactive HTML
fig.write_html('three_body_interactive.html')
print("Interactive visualization saved to three_body_interactive.html")

# Show in browser
fig.show()
