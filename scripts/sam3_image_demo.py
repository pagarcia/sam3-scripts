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
    disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr.copy()
    return disp, scale


def green_overlay(bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    fg = mask_bool.astype(bool)
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 interactive image demo (points)")
    ap.add_argument("--image", default=None, help="Optional image path. If omitted, a file dialog opens.")
    ap.add_argument("--checkpoint", default=os.getenv("SAM3_CHECKPOINT", ""),
                    help="Optional local sam3.pt path (avoids HF download).")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--max-side", type=int, default=1200, help="Max display size for interactive window.")
    args = ap.parse_args()

    if args.device != "cuda":
        raise SystemExit(
            "This interactive point demo uses SAM3's instance-interactivity predictor, "
            "which is CUDA-oriented in the upstream stack. Use --device cuda."
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This interactive predictor path expects CUDA.")

    _ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_image_model  # upstream builder

    # Pick image
    img_path = args.image or pick_image_file()
    if not img_path:
        raise SystemExit("No image selected.")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ckpt = args.checkpoint.strip() or None

    # Build SAM3 with instance interactivity enabled (gives model.inst_interactive_predictor)
    model = build_sam3_image_model(
        checkpoint_path=ckpt,
        load_from_HF=(ckpt is None),
        device="cuda",
        eval_mode=True,
        enable_inst_interactivity=True,
        enable_segmentation=False,  # not needed for point-based interactive masks
    )

    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None:
        raise SystemExit("Model was built without inst_interactive_predictor. (enable_inst_interactivity=True required)")

    # Set image embeddings once
    predictor.set_image(img_rgb)

    disp_base, scale = compute_display_base(img_bgr, max_side=args.max_side)

    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    last_mask_disp: Optional[np.ndarray] = None

    def redraw():
        vis = disp_base.copy()

        if last_mask_disp is not None:
            vis = green_overlay(vis, last_mask_disp, alpha=0.5)

        # draw points
        for (px, py), lab in zip(points, labels):
            x = int(px * scale)
            y = int(py * scale)
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)  # red=pos, blue=neg
            cv2.circle(vis, (x, y), 6, col, -1)

        cv2.imshow("SAM3 Interactive (L=pos, R=neg, M/reset, u=undo, r=reset, ESC=quit)", vis)

    def run_predict():
        nonlocal last_mask_disp
        if not points:
            last_mask_disp = None
            redraw()
            return

        pc = np.asarray(points, dtype=np.float32)  # pixel coords
        pl = np.asarray(labels, dtype=np.int32)

        masks, scores, _lowres = predictor.predict(
            point_coords=pc,
            point_labels=pl,
            box=None,
            mask_input=None,
            multimask_output=True,
            return_logits=False,
            normalize_coords=True,  # True means input coords are absolute pixels; SAM2Transforms will normalize.
        )

        # masks: [C,H,W] boolean (return_logits=False); scores: [C]
        if masks is None or len(masks) == 0:
            last_mask_disp = None
            redraw()
            return

        best = int(np.argmax(scores)) if scores is not None and len(scores) else 0
        mask_hw = masks[best].astype(bool)

        # resize mask to display
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
        # map display -> original pixel coords
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

    cv2.namedWindow("SAM3 Interactive (L=pos, R=neg, M/reset, u=undo, r=reset, ESC=quit)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 Interactive (L=pos, R=neg, M/reset, u=undo, r=reset, ESC=quit)", mouse_cb)

    print("[INFO] Controls:")
    print("  L-click = positive point")
    print("  R-click = negative point")
    print("  M-click = reset")
    print("  u      = undo last point")
    print("  r      = reset")
    print("  ESC/q  = quit")

    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):  # ESC or q
            break
        if k == ord("r"):
            reset_all()
        if k == ord("u"):
            undo_last()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())