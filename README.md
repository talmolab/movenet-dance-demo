# Real-time pose matching demo

Real-time demo running on webcam feed that guides subject to move to match pre-defined poses.

## Setup
```
conda env create -f environment.yml -n movenet-dance-demo
conda activate movenet-dance-demo
```

To change the target poses, see the notebook: `make_target_poses.ipynb`

## Usage
```
python live_demo.py
```
