#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SIM_BIN="./threebody_opengl"
DATA_DIR="ml/data"
MODEL_PATH="ml/model.pt"
RL_DIR="ml/rl_runs"
RL_VISUALIZED=0
MASS_RATIOS="${MASS_RATIOS:-1.0,0.6,0.3}"

echo "== Build =="
if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found. Install Xcode Command Line Tools."
  exit 1
fi

GLFW_PREFIX="$(brew --prefix glfw 2>/dev/null || true)"
if [[ -n "${GLFW_PREFIX}" && -d "${GLFW_PREFIX}/include" ]]; then
  g++ -std=c++17 threebody_opengl.cpp planet.cpp -o "$SIM_BIN" \
    -I"$GLFW_PREFIX/include" -L"$GLFW_PREFIX/lib" -lglfw \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
else
  g++ -std=c++17 threebody_opengl.cpp planet.cpp -o "$SIM_BIN" \
    -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
fi

if [[ -t 0 ]]; then
  read -r -p "Enter mass ratios for RL/simulation (e.g. 1.0,0.5,0.3; press Enter for ${MASS_RATIOS}): " INPUT_MASS_RATIOS
  if [[ -n "${INPUT_MASS_RATIOS}" ]]; then
    MASS_RATIOS="${INPUT_MASS_RATIOS}"
  fi
fi

echo "== Mass Ratios =="
echo "Using mass ratios: ${MASS_RATIOS}"

echo "== Dataset =="
python3 ml/generate_dataset.py --num-trajectories 10 --steps 500 --dt 200.0

echo "== Train (PINN) =="
if python3 - <<'PY'
import torch
print(torch.__version__)
PY
then
  python3 ml/train_pinn.py --data-dir "$DATA_DIR" --epochs 5 --out "$MODEL_PATH"
else
  echo "Skipping training: torch import failed."
fi

echo "== RL (headless) =="
if python3 - <<'PY'
import torch
print(torch.__version__)
PY
then
  python3 ml/rl_initial_conditions.py --episodes 20 --batch 4 --steps 2000 --dt 200 \
    --mass-ratios "$MASS_RATIOS" \
    --out-dir "$RL_DIR" --visualize-best --visual-scale 2e-11 --visual-out "$RL_DIR/best_visual.csv"
  RL_VISUALIZED=1
else
  echo "Skipping RL: torch import failed."
fi

if [[ "$RL_VISUALIZED" -eq 0 ]]; then
  echo "== Visual Run =="
  echo "Launching default visual mode. Close the window to finish."
  "$SIM_BIN" --scale 2e-8 --mass-ratios "$MASS_RATIOS" --record "$RL_DIR/visual.csv"
fi

echo "Done."
