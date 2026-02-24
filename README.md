# sam3-scripts

Small, public demo scripts for **Segment Anything Model 3 (SAM3)**.

This repo supports two backends:

- **Windows (CUDA / upstream SAM3)**: uses the upstream `facebookresearch/sam3` repo (local sibling clone) + CUDA.
  - Scripts: `scripts/sam3_image_demo.py`, `scripts/sam3_video_demo.py`
  - Deps: `requirements_win.txt`
- **macOS (Apple Silicon / Transformers)**: uses the **Transformers** SAM3 implementation + MPS/CPU.
  - Scripts: `scripts/sam3_image_demo_hf.py`, `scripts/sam3_video_demo_hf.py`
  - Deps: `requirements_mac.txt`

---

## Hugging Face access + login (gated weights)

SAM3 weights on Hugging Face are **gated**.

1) **Request access** (one-time, per Hugging Face account):
   - Go to the model page on Hugging Face (e.g. `facebook/sam3`)
   - Click **Request access** and accept the terms (approval can take time).

2) **Log in locally** (per machine / per environment):

```bash
hf auth login
````

This stores a token under your user profile (so future runs won’t prompt again) and allows `transformers` / `huggingface_hub` (and the upstream scripts if they download from HF) to download the weights.

If the `hf` command is not found, install/upgrade the hub package:

```bash
pip install -U huggingface_hub
```

3. **Caching**
   Once downloaded, weights are cached locally and won’t re-download unless you clear the cache. Default cache location is typically:

* macOS/Linux: `~/.cache/huggingface/`
* Windows: `%USERPROFILE%\.cache\huggingface\`

Tip: if you’re on a machine without internet access, download once on a connected machine and copy the cache folder.

---

## Windows setup (CUDA, upstream SAM3)

### 1) Clone (sibling repos)

From a parent folder:

```powershell
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/pagarcia/sam3-scripts.git
cd sam3-scripts
```

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

### 5) Login to Hugging Face (gated weights)

```powershell
hf auth login
```

### 6) Run demos

**Image (interactive points; file picker opens; no output file written):**

```powershell
python .\scripts\sam3_image_demo.py
```

**Video (interactive frame 0 → headless processing → writes overlay video next to input):**

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

## macOS setup (Transformers + MPS/CPU)

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

### 5) Login to Hugging Face (gated weights)

```bash
hf auth login
```

### 6) Run demos

**Image (interactive points; file picker opens):**

```bash
python scripts/sam3_image_demo_hf.py
```

**Video (interactive frame 0 → headless processing → writes overlay video next to input):**

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

## Mask output format (for downstream use)

All demos compute a **binary segmentation mask** in the original image/video resolution:

- `mask_hw`: NumPy `bool` array of shape `(H, W)`
  - `True` = foreground
  - `False` = background

The “green overlay” video/image is just visualization. If you need the mask for your app:
- Convert to uint8: `mask_u8 = mask_hw.astype(np.uint8) * 255`
- Save as PNG, compute contours, apply as alpha, or bit-pack for compact storage.

In the scripts:
- Image demos choose the best mask from multiple candidates (multimask) using the model’s score.
- Video demos produce one mask per frame (and currently save an overlay MP4). You can also save per-frame masks by writing `mask_u8` inside the loop.

## Notes

* **First run downloads weights** (can be several GB). Downloads are cached under the Hugging Face cache directory (see section above).
* **macOS uses the Transformers backend** because upstream `facebookresearch/sam3` is CUDA/Triton-oriented and is not mac-friendly in practice.
* **Windows has two working options**:

  * **Upstream backend (`sam3_image_demo.py`, `sam3_video_demo.py`)**: fastest on NVIDIA GPUs and closest to Meta’s reference implementation, but requires more platform-specific deps (e.g. Triton on Windows).
  * **Transformers backend (`*_hf.py`)**: also works on Windows (CUDA or CPU) and is usually simpler to install, but the codepath is a reimplementation and may differ slightly in behavior/performance compared to upstream.

```