# scripts/sam3_image_demo.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch


def _ensure_sam3_on_path() -> None:
    try:
        import sam3  # noqa: F401
        return
    except Exception:
        repo_root = Path(__file__).resolve().parent.parent  # sam3-scripts/
        for cand in [
            repo_root.parent / "sam3",          # ../sam3
            repo_root.parent / "sam3_repo",     # optional alt name
            repo_root / "third_party" / "sam3", # optional submodule style
        ]:
            if (cand / "sam3").exists():
                sys.path.insert(0, str(cand))
                return
        raise


def pick_image_file() -> Optional[str]:
    """Native file picker (Tkinter). Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def compute_display_base(img_bgr: np.ndarray, max_side: int = 1200) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        disp = img_bgr.copy()
    return disp, scale


def green_overlay(bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    fg = mask_bool.astype(bool)
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 interactive image demo (positive/negative points)")
    ap.add_argument("--image", default=None, help="Optional image path. If omitted, a file dialog opens.")
    ap.add_argument("--checkpoint", default=os.getenv("SAM3_CHECKPOINT", ""),
                    help="Optional local sam3.pt path (avoids HF download).")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cuda", "cpu"])
    ap.add_argument("--max-side", type=int, default=1200, help="Max display size for interactive window.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    _ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Pick image
    img_path = args.image or pick_image_file()
    if not img_path:
        raise SystemExit("No image selected.")

    # Load with PIL for SAM3 (avoid ndarray shape ambiguity in Sam3Processor.set_image)
    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil)  # HWC RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    ckpt = args.checkpoint.strip() or None

    # Build model with instance interactivity enabled (required for predict_inst)
    model = build_sam3_image_model(
        checkpoint_path=ckpt,
        load_from_HF=(ckpt is None),
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )

    # Build inference state once (backbone_out cached here)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.5)
    state = processor.set_image(img_pil, state={})

    # Ensure sam2_backbone_out exists (predict_inst uses it)
    bb = state.get("backbone_out", {})
    if "sam2_backbone_out" not in bb:
        raise SystemExit(
            "backbone_out did not contain 'sam2_backbone_out'. "
            "This interactive point path requires enable_inst_interactivity=True."
        )

    disp_base, scale = compute_display_base(img_bgr, max_side=args.max_side)

    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    last_mask_disp: Optional[np.ndarray] = None

    def redraw():
        vis = disp_base.copy()
        if last_mask_disp is not None:
            vis = green_overlay(vis, last_mask_disp, alpha=0.5)

        for (px, py), lab in zip(points, labels):
            x = int(px * scale)
            y = int(py * scale)
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)  # red=pos, blue=neg
            cv2.circle(vis, (x, y), 6, col, -1)

        cv2.imshow("SAM3 Points (L=pos, R=neg, M/r=reset, u=undo, ESC/q=quit)", vis)

    def run_predict():
        nonlocal last_mask_disp
        if not points:
            last_mask_disp = None
            redraw()
            return

        pc = np.asarray(points, dtype=np.float32)     # (N,2) in pixels (x,y)
        pl = np.asarray(labels, dtype=np.int32)       # (N,) 0/1

        # This uses Sam3Image.predict_inst which reuses backbone_out["sam2_backbone_out"]
        # and calls inst_interactive_predictor.predict under the hood. :contentReference[oaicite:2]{index=2}
        masks, scores, _lowres = model.predict_inst(
            state,
            point_coords=pc,
            point_labels=pl,
            box=None,
            mask_input=None,
            multimask_output=True,
            return_logits=False,
            normalize_coords=True,   # pixels -> normalized via SAM2Transforms :contentReference[oaicite:3]{index=3}
        )

        if masks is None or len(masks) == 0:
            last_mask_disp = None
            redraw()
            return

        best = int(np.argmax(scores)) if scores is not None and len(scores) else 0
        mask_hw = masks[best].astype(bool)  # HxW at original resolution

        if scale != 1.0:
            mask_disp = cv2.resize(mask_hw.astype(np.uint8), (disp_base.shape[1], disp_base.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_disp = mask_hw

        last_mask_disp = mask_disp
        redraw()

    def reset_all():
        nonlocal last_mask_disp
        points.clear()
        labels.clear()
        last_mask_disp = None
        redraw()

    def undo_last():
        if points:
            points.pop()
            labels.pop()
        run_predict()

    def mouse_cb(event, x, y, flags, param):
        px = int(x / scale)
        py = int(y / scale)
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        if event == cv2.EVENT_MBUTTONDOWN:
            reset_all()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((px, py))
            labels.append(1)
            run_predict()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            points.append((px, py))
            labels.append(0)
            run_predict()
            return

    cv2.namedWindow("SAM3 Points (L=pos, R=neg, M/r=reset, u=undo, ESC/q=quit)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 Points (L=pos, R=neg, M/r=reset, u=undo, ESC/q=quit)", mouse_cb)

    print("[INFO] Controls:")
    print("  L-click = positive point")
    print("  R-click = negative point")
    print("  M-click = reset")
    print("  r      = reset")
    print("  u      = undo last point")
    print("  ESC/q  = quit")

    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            reset_all()
        if k == ord("u"):
            undo_last()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())