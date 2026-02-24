# sam3-scripts

Small, public demo scripts for **Segment Anything Model 3 (SAM3)**:
- Image segmentation (text prompt and/or box prompt)
- Video segmentation/tracking (text/box prompts OR SAM2-style point tracking via the SAM3 tracker)

> Checkpoints are gated: request access on Hugging Face and login (`hf auth login`).  
> (See official SAM3 instructions.)  

## Prereqs
The official SAM3 repo recommends:
- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+ GPU for practical use
(See upstream README.)  

## Setup (sibling clone workflow)
From a parent folder:

```bash
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/pagarcia/sam3-scripts.git
````

Create a venv, install torch (pick the right CUDA wheel for your system):

```powershell
# Create + activate venv
cd sam3-scripts
py -3.12 -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1

# Tooling (SAM3 currently needs setuptools<82)
python -m pip install -U pip wheel "setuptools<82"

# PyTorch (CUDA 12.6, matches upstream recommendation)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

```bash
pip install -e ./sam3
pip install -r ./sam3-scripts/requirements.txt
```

### Checkpoints (required)

Request access to `facebook/sam3` on HF, then:

```bash
hf auth login
```

Optionally set a local checkpoint:

```bash
export SAM3_CHECKPOINT=/path/to/sam3.pt
```

## Image demo

Text prompt:

```bash
python sam3-scripts/scripts/sam3_image_demo.py --image ./my.jpg --text "person" --out out.png --show
```

Box prompt (draw interactively):

```bash
python sam3-scripts/scripts/sam3_image_demo.py --image ./my.jpg --interactive-box --out out.png --show
```

## Video demo

Concept tracking with text prompt:

```bash
python sam3-scripts/scripts/sam3_video_demo.py --video ./my.mp4 --prompt text --text "person" --out out.mp4
```

Point-based tracking (SAM2-style) on first frame:

```bash
python sam3-scripts/scripts/sam3_video_demo.py --video ./my.mp4 --prompt points --out out.mp4
# L-click = positive, R-click = negative, M-click = reset, Enter = run
```

Box prompt (draw interactively on frame 0):

```bash
python sam3-scripts/scripts/sam3_video_demo.py --video ./my.mp4 --prompt box --interactive-box --out out.mp4
```

````