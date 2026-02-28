# Three-Body Problems

This projects aims to use RL to search for good initial conditions to find a stable thee bodies system. It simulates and visualises the gravitational three-body problem with:

- A C++ physics engine (`planet.cpp`, `planet.hpp`)
- A C++ OpenGL visual simulator (`threebody_opengl.cpp`)
- A headless CSV simulator mode for dataset generation
- Python tooling for static/interactive visualization
- An ML pipeline (`ml/`) for:
  - dataset generation from the simulator
  - training a physics-informed next-step model (PyTorch)
  - RL search for stable initial conditions

## Project Structure

- `threebody_opengl.cpp`: main simulator with visual and `--headless` modes
- `threebodiessimulation.cpp`: alternate CLI simulator
- `planet.cpp`, `planet.hpp`: core math/physics classes
- `visualize*.py`: plotting and HTML visualization scripts
- `ml/generate_dataset.py`: generate trajectory CSV datasets
- `ml/train_pinn.py`: train PyTorch physics-informed model
- `ml/rl_initial_conditions.py`: RL-based initial condition search
- `run_all.sh`: end-to-end workflow (build -> data -> train -> RL -> visual run)
- `requirements.txt`: Python dependencies

## Prerequisites

### C++

- `g++` (C++17)
- GLFW
- OpenGL libraries

On macOS (Homebrew):

```bash
brew install glfw
```

### Python

- Python 3.9+ recommended
- Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Build

Build the OpenGL simulator:

```bash
g++ -std=c++17 threebody_opengl.cpp planet.cpp -o threebody_opengl \
  -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
```

## Quick Start (One Command)

Run the full pipeline:

```bash
bash run_all.sh
```

This will:

1. Build `threebody_opengl`
2. Generate ML dataset in `ml/data`
3. Train model to `ml/model.pt` (if `torch` is available)
4. Run RL episodes in `ml/rl_runs` (if `torch` is available)
5. Launch visual simulation window

## Run Simulator Only

Visual mode:

```bash
./threebody_opengl --scale 2e-8
```

Headless mode (write CSV):

```bash
./threebody_opengl --headless --steps 2000 --dt 50 --out simulation_data.csv
```

See all options:

```bash
./threebody_opengl --help
```

## Visualization Scripts

All scripts read `simulation_data.csv` unless edited.

```bash
python3 visualize.py
python3 visualize_interactive.py
python3 visualize_animation_interactive.py
python3 visualize_widgets.py
```

Outputs include:

- `three_body_simulation.png`
- `three_body_animation.gif`
- `three_body_interactive.html`
- `three_body_animated.html`

## ML Workflow

The reward function is theorised based on:
- Radius violation penalty: how much the trajectory exceeds the allowed maximum system radius.
- Separation violation penalty: how much the minimum body-to-body distance falls below the allowed threshold.
- Energy drift penalty: how much total mechanical energy changes during the trajectory.
- Stability failure penalty: an extra fixed negative penalty if the run is classified as unstable.

1) Generate dataset:

```bash
python3 ml/generate_dataset.py --num-trajectories 50 --steps 2000 --dt 200
```

2) Train physics-informed model:

```bash
python3 ml/train_pinn.py --data-dir ml/data --epochs 30 --out ml/model.pt
```

3) Search stable initial conditions with RL:

```bash
python3 ml/rl_initial_conditions.py --episodes 200 --batch 8 --out-dir ml/rl_runs
```

Optional best-candidate visual playback:

```bash
python3 ml/rl_initial_conditions.py --episodes 200 --batch 8 \
  --out-dir ml/rl_runs --visualize-best --visual-out ml/rl_runs/best_visual.csv
```

