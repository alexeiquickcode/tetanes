# tetanes-py

Python bindings for TetaNES - a NES emulator written in Rust.

## Install

```bash
pip install tetanes-py
```

Or from source:
```bash
pip install git+https://github.com/alexeiquickcode/tetanes.git#subdirectory=tetanes-python
```

## Usage

Basic emulator usage:

```python
import tetanes_py

# Create emulator
env = tetanes_py.NesEnv(headless=True)

# Load ROM from file
with open("game.nes", "rb") as f:
    rom_data = f.read()
env.load_rom("game.nes", rom_data)

# Step through frames
obs = env.step(0)  # 0 = no button pressed
```

### Gymnasium Environment

Works with OpenAI Gym/Gymnasium for RL:

```python
import gymnasium as gym
import tetanes_py

env = gym.make("ExampleGame-v0", rom_path="game.nes")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Requirements

- Python 3.9+
- NumPy
- Gymnasium (optional, for RL environment)

## License

MIT OR Apache-2.0
