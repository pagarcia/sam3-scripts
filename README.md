# sam3-scripts

Small, public demo scripts for **Segment Anything Model 3 (SAM3)**.

This repo supports two backends:

- **Windows (CUDA)**: uses the upstream `facebookresearch/sam3` repo (local clone) + CUDA.
  - Scripts: `scripts/sam3_image_demo.py`, `scripts/sam3_video_demo.py`
- **macOS (Apple Silicon)**: uses the **Transformers** SAM3 implementation + MPS/CPU.
  - Scripts: `scripts/sam3_image_demo_hf.py`, `scripts/sam3_video_demo_hf.py`

> **Checkpoints / weights are gated**: request access on Hugging Face and login with `hf auth login` before first run.

---

## Windows setup (CUDA, upstream SAM3)

### 1) Clone (sibling repos)
From a parent folder:

```powershell
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/pagarcia/sam3-scripts.git
cd sam3-scripts
````

### 2) Create + activate venv

```powershell
py -3.12 -m venv sam3_env
.\sam3_env\Scripts\Activate.ps1
python -m pip install -U pip wheel "setuptools<82"
```

### 3) Install PyTorch (CUDA 12.6)

```powershell
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 4) Install upstream SAM3 + demo deps

```powershell
pip install -e ..\sam3
pip install -r .\requirements_win.txt
```

### 5) Hugging Face auth (for gated weights)

```powershell
hf auth login
```

### 6) Run demos

**Image (interactive points, no output file):**

```powershell
python .\scripts\sam3_image_demo.py
```

**Video (interactive frame 0 → headless processing → saves overlay video next to input):**

```powershell
python .\scripts\sam3_video_demo.py
```

Options:

```powershell
python .\scripts\sam3_video_demo.py --max-frames 200
python .\scripts\sam3_video_demo.py --out C:\path\to\out.mp4
python .\scripts\sam3_video_demo.py --checkpoint C:\path\to\sam3.pt
```

---

## macOS setup (Transformers + MPS)

> Recommended: **Python 3.12/3.13** for best wheel availability.

### 1) Clone

```bash
git clone https://github.com/pagarcia/sam3-scripts.git
cd sam3-scripts
```

### 2) Create + activate venv

```bash
python3 -m venv sam3_env
source sam3_env/bin/activate
python -m pip install -U pip
```

### 3) Install PyTorch (Apple build)

```bash
pip install torch torchvision
```

### 4) Install mac deps (Transformers backend + PyQt5 dialogs)

```bash
pip install -r requirements_mac.txt
```

### 5) Hugging Face auth (gated weights)

```bash
hf auth login
```

### 6) Run demos

**Image (interactive points):**

```bash
python scripts/sam3_image_demo_hf.py
```

**Video (interactive frame 0 → headless processing → saves overlay video next to input):**

```bash
python scripts/sam3_video_demo_hf.py
```

Options:

```bash
python scripts/sam3_video_demo_hf.py --max-frames 200
python scripts/sam3_video_demo_hf.py --out /path/to/out.mp4
python scripts/sam3_image_demo_hf.py --model-id facebook/sam3
```

---

## Notes

* **First run downloads weights** (can be several GB). Downloads are cached under `~/.cache/huggingface/` (macOS/Linux) or the Hugging Face cache directory on Windows (same default path under your user profile).
* **macOS uses the Transformers backend** because the upstream `facebookresearch/sam3` repo is CUDA/Triton-oriented and is not mac-friendly in practice.
* **Windows has two working options**:
  * **Upstream backend (`sam3_image_demo.py`, `sam3_video_demo.py`)**: fastest on NVIDIA GPUs and closest to Meta’s reference implementation, but requires more platform-specific deps (e.g., Triton on Windows).
  * **Transformers backend (`*_hf.py`)**: also works on Windows (CUDA or CPU) and is usually simpler to install, but the codepath is a reimplementation and may differ slightly in behavior/performance compared to upstream.

```