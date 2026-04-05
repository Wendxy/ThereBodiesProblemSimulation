# Three-Body Problems

This project simulates the gravitational three-body problem and uses reinforcement learning to search for initial conditions that remain compact, non-colliding, and numerically well-behaved over a finite time horizon.

The repository has three main layers:

- A C++ physics engine in `planet.cpp` / `planet.hpp`
- A C++ OpenGL simulator in `threebody_opengl.cpp`
- A Python ML pipeline in `ml/` for dataset generation, next-step model training, and RL search

## What The Project Does

The project combines direct simulation with search:

- Simulate three gravitating bodies with configurable masses, positions, and velocities
- Visualize the trajectories in an interactive OpenGL viewer
- Export headless simulation runs to CSV
- Generate trajectory datasets for ML experiments
- Train a physics-informed next-step predictor in PyTorch
- Run an RL loop that proposes initial conditions, evaluates them with the simulator, and keeps the best candidate

This is a search-and-simulation project, not a closed-form solver for the three-body problem.

## Repository Layout

- `threebody_opengl.cpp`: main visual simulator and headless CSV generator
- `threebodiessimulation.cpp`: alternate CLI simulator
- `planet.cpp`, `planet.hpp`: vector math, body state, force calculation, and time integration
- `visualize.py`: static Matplotlib plot plus GIF output
- `visualize_interactive.py`: Plotly 3D HTML visualization
- `visualize_animation_interactive.py`: animated Plotly HTML visualization
- `visualize_widgets.py`: Matplotlib widget-based visualization
- `ml/generate_dataset.py`: generate datasets by repeatedly calling the simulator
- `ml/train_pinn.py`: train a physics-informed next-step model in PyTorch
- `ml/rl_initial_conditions.py`: RL search over initial positions and velocities
- `ml/utils.py`: CSV loading, energy calculation, and stability metrics
- `run_all.sh`: end-to-end workflow

## Requirements

### C++

- `g++` with C++17 support
- GLFW
- OpenGL libraries

On macOS with Homebrew:

```bash
brew install glfw
```

### Python

- Python 3.9+
- Packages from `requirements.txt`

Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

## Build

Build the OpenGL simulator:

```bash
g++ -std=c++17 threebody_opengl.cpp planet.cpp -o threebody_opengl \
  -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
```

## Quick Start

Run the full workflow:

```bash
bash run_all.sh
```

The script currently does this:

1. Builds `threebody_opengl`
2. Prompts once for mass ratios such as `1.0,0.5,0.3`
3. Generates a small dataset in `ml/data`
4. Trains `ml/model.pt` if `torch` imports successfully
5. Runs the RL search with the chosen mass ratios if `torch` imports successfully
6. Launches a visual simulation of the best RL candidate

If `torch` is unavailable, the script skips training and RL, then falls back to a normal visual simulation using the chosen mass ratios.

## Running The Simulator

### Visual Mode

Launch the OpenGL simulator:

```bash
./threebody_opengl --scale 2e-8
```

If you do not pass explicit masses, the simulator prompts for three mass ratios such as:

```text
1.0,0.5,0.3
```

Those ratios are multiplied by the simulator base mass. You can also pass them directly:

```bash
./threebody_opengl --scale 2e-8 --mass-ratios 1.0,0.5,0.3
```

You can control the base mass with:

```bash
./threebody_opengl --scale 2e-8 --mass-ratios 1.0,0.5,0.3 --mass-scale 1e26
```

If you want exact masses instead of ratios, use:

```bash
./threebody_opengl --scale 2e-8 --m1 1e30 --m2 5e29 --m3 3e29
```

### Headless Mode

Write a CSV without opening a window:

```bash
./threebody_opengl --headless --steps 2000 --dt 50 --out simulation_data.csv
```

See all options:

```bash
./threebody_opengl --help
```

## Camera And Interaction Controls

The OpenGL viewer now uses a movable camera target instead of being permanently locked to the origin.

- Mouse drag: orbit around the current camera target
- Mouse scroll: zoom in and out
- `W/A/S/D`: pan the camera target in the view plane
- `Q/E`: move the camera target up and down
- `F`: toggle follow-center-of-mass mode
- `C`: reset the camera target to the origin
- `SPACE`: pause or resume simulation
- `R`: clear trails and recorded replay frames
- `T`: toggle unlimited trails
- `P`: play or stop replay mode
- `+` / `-`: change simulation speed
- `ESC`: exit

## Visualization Scripts

The Python visualization scripts read `simulation_data.csv` unless you edit them:

```bash
python3 visualize.py
python3 visualize_interactive.py
python3 visualize_animation_interactive.py
python3 visualize_widgets.py
```

Typical outputs include:

- `three_body_simulation.png`
- `three_body_animation.gif`
- `three_body_interactive.html`
- `three_body_animated.html`

## ML Workflow

### 1. Generate A Dataset

```bash
python3 ml/generate_dataset.py --num-trajectories 50 --steps 2000 --dt 200
```

This repeatedly launches the simulator in headless mode with randomly sampled initial conditions and stores the generated trajectories in `ml/data`.

### 2. Train The Physics-Informed Model

```bash
python3 ml/train_pinn.py --data-dir ml/data --epochs 30 --out ml/model.pt
```

This trains a PyTorch next-step predictor from the generated trajectories. The model is an auxiliary ML experiment and is not currently used inside the RL scoring loop.

### 3. Run The RL Search

```bash
python3 ml/rl_initial_conditions.py \
  --episodes 200 \
  --batch 8 \
  --mass-ratios 1.0,0.5,0.3 \
  --out-dir ml/rl_runs
```

With visualization of the best candidate:

```bash
python3 ml/rl_initial_conditions.py \
  --episodes 200 \
  --batch 8 \
  --mass-ratios 1.0,0.5,0.3 \
  --out-dir ml/rl_runs \
  --visualize-best \
  --visual-scale 2e-11 \
  --visual-out ml/rl_runs/best_visual.csv
```

### What The RL Policy Learns

The RL policy does not control the system step by step. Instead, it learns a distribution over initial conditions:

- 9 values for the three initial positions
- 9 values for the three initial velocities

So the policy output is an 18-dimensional sampled action that becomes:

- `p1`, `p2`, `p3`
- `v1`, `v2`, `v3`

The mass values are not learned by the policy. They are fixed for a given run from:

- `mass_scale * mass_ratios`

The script then recenters the sampled initial state so the center-of-mass position and center-of-mass velocity are both zero before simulation.

## Reward Function

The reward is defined in `ml/rl_initial_conditions.py`, and the trajectory metrics are computed in `ml/utils.py`.

The score is evaluated on the simulated trajectory, not on a single frame. It is a finite-horizon heuristic designed to prefer trajectories that:

- stay inside a configurable radius
- avoid close approaches and collisions
- remain compact instead of slowly drifting outward
- stay numerically well-behaved
- look energetically bound rather than like a flyby

### Metrics Used By The Reward

For each simulated trajectory, the code measures:

- `max_radius`: the largest distance any body reaches from the center of mass at any time
- `final_max_radius`: the largest center-of-mass-relative radius at the final frame
- `min_separation`: the smallest pairwise distance across the whole run
- `energy_drift`: the maximum relative change in total energy compared with the start
- `bound_fraction`: the fraction of frames where total energy is negative
- `survival_fraction`: the fraction of frames where the system stays within the radius limit and above the minimum separation limit
- `radius_growth`: the average radius in the last 10% of the run divided by the average radius in the first 10%
- `stable`: `1.0` only if `survival_fraction == 1.0`

### Reward Formula

The reward is currently:

```text
compact_bonus = max(0, 1 - final_max_radius / max_radius_limit)

reward =
  survival_fraction
  + 0.5 * bound_fraction
  + 0.25 * compact_bonus

penalty =
  radius_penalty
  + separation_penalty
  + drift_penalty
  + 1.5 * survival_penalty
  + 0.75 * unbound_penalty
  + 0.5 * growth_penalty

reward -= penalty

if stable == 0:
  reward -= 1.0
```

Where:

- `radius_penalty = max(0, max_radius - max_radius_limit) / max_radius_limit`
- `separation_penalty = max(0, min_separation_limit - min_separation) / min_separation_limit`
- `drift_penalty = energy_drift`
- `survival_penalty = 1 - survival_fraction`
- `unbound_penalty = 1 - bound_fraction`
- `growth_penalty = max(0, radius_growth - 1)`

### Why Each Component Exists

#### `survival_fraction`

This is the main positive term. It rewards trajectories that remain inside the allowed radius and avoid dangerous close approaches for the whole run.

Why it exists:

- A single final-state check is too weak
- Partial survival is still informative during learning
- It encourages persistent rather than momentary stability

#### `radius_penalty`

This directly punishes escape-like behavior when one body travels too far from the center of mass.

Why it exists:

- Without it, the policy can exploit trajectories that briefly look structured but are actually dispersing
- It makes the radius limit a hard physical preference rather than a soft visual preference

#### `min_separation` / `separation_penalty`

This punishes near-collisions and close passes below the configured threshold.

Why it exists:

- A compact trajectory is not useful if it achieves compactness by collapsing bodies together
- It prevents obviously non-physical or collision-like solutions from scoring well

#### `energy_drift`

This penalizes large changes in total energy relative to the start.

Why it exists:

- Large drift is often a sign of a numerically poor trajectory
- It biases the search toward solutions that are more trustworthy under the simulator

#### `bound_fraction`

This measures how often the total energy is negative during the run.

Why it exists:

- Negative total energy is used as a proxy for gravitationally bound behavior
- It helps separate compact flyby trajectories from trajectories that look more genuinely trapped

This is still only a proxy. It does not prove long-term stability on its own.

#### `final_max_radius` / `compact_bonus`

This rewards ending the run in a compact configuration rather than merely staying inside the radius limit.

Why it exists:

- Two trajectories can both satisfy the radius limit, but one can be much more tightly contained at the end
- It nudges the search toward visually and physically tighter solutions

#### `radius_growth`

This compares the average radius near the end of the run to the average radius near the start.

Why it exists:

- A trajectory can survive the whole horizon and still be steadily expanding
- This catches slow outward drift that would not necessarily trigger the hard radius limit yet

#### Final instability penalty

If the trajectory is not fully stable by the code's definition, the reward gets an extra `-1.0`.

Why it exists:

- It gives the optimizer a clear boundary between fully surviving trajectories and partial failures
- It makes unstable candidates distinctly less attractive

### What The Reward Does Well

- It prefers compact, persistent trajectories over obvious escape trajectories
- It discourages collisions and close approaches
- It discourages numerically noisy solutions
- It distinguishes "survived briefly" from "survived the whole evaluation window"

### Important Limitations

- The reward is still a heuristic, not a proof of true long-term stability
- It only evaluates a finite trajectory horizon
- `bound_fraction` uses negative total energy as a practical proxy, not a mathematically complete criterion
- Good reward values mean "good under this simulator and this horizon", not "stable forever"

## Notes On The Physics Engine

The simulator currently uses a velocity-Verlet style integrator in `planet.cpp` rather than a simple Euler step. That improves long-run energy behavior compared with the earlier version of the project and makes the RL reward less dominated by integrator drift.
