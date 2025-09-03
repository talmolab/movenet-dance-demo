# Real-time pose matching demo

A real-time demo running on webcam feed that guides subject to move to match pre-defined target poses.

![Match the target pose!](screenshot.png "Screenshot of live demo.")

*Move your body to match each of the target poses on the right!*

## Setup

### Option 1: Using uv (recommended)

Install and run directly from GitHub:
```bash
# CPU-only (works on all platforms)
uvx --from git+https://github.com/talmolab/movenet-dance-demo live-demo

# With GPU acceleration
uvx --from git+https://github.com/talmolab/movenet-dance-demo --with movenet-dance-demo[cuda] live-demo      # Linux CUDA
uvx --from git+https://github.com/talmolab/movenet-dance-demo --with movenet-dance-demo[apple-gpu] live-demo # Apple Silicon
```

Or install locally:
```bash
git clone https://github.com/talmolab/movenet-dance-demo
cd movenet-dance-demo
uv sync
uv run live-demo
```

**For GPU acceleration:**
- Apple Silicon Mac: `uv sync --extra apple-gpu`
- Linux with CUDA: `uv sync --extra cuda`

### Option 2: Using conda

```bash
conda env create -f environment.yml -n movenet-dance-demo
conda activate movenet-dance-demo
```

To change the target poses, see the notebook: [`make_target_poses.ipynb`](make_target_poses.ipynb)

For an example of standalone inference, see the notebook: [`inference_demo.ipynb`](inference_demo.ipynb)


## Usage

### Using uv

To run the demo:
```bash
uv run live-demo
# or if installed via uvx:
uvx --from git+https://github.com/talmolab/movenet-dance-demo live-demo
```

To use a different camera:
```bash
uv run live-demo -c 1
```

To make it easier, increase the target tolerance:
```bash
uv run live-demo --tolerance 0.7
```

### Using conda/python directly

To run the demo:
```bash
python live_demo.py
```

To use a different camera:
```bash
python live_demo.py -c 1
```

To make it easier, increase the target tolerance:
```bash
python live_demo.py --tolerance 0.7
```

**Hotkeys:**

<kbd>1</kbd> - <kbd>9</kbd>: Switch between target poses.

<kbd>Tab</kbd>: Cycle through target poses.

<kbd>R</kbd>: Reset completed poses.

<kbd>Q</kbd> or <kbd>Esc</kbd>: Quit the demo.


## Credits
This demo was written by [Talmo Pereira](https://talmopereira.com) at the Salk Institute for Biological Studies to be presented at the [SICB 2022 Annual Meeting](http://burkclients.com/sicb/meetings/2022/site/workshops.html).

This code uses [MoveNet](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) for pose estimation by Ronny Votel, Na Li and other contributors at Google.

The target pose data is courtesy of the [danceTactics](http://www.dancetactics.org/about) crew directed by [Keith A. Thompson](https://musicdancetheatre.asu.edu/profile/keith-thompson) at Arizona State University.
