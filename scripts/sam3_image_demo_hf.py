# scripts/sam3_image_demo_hf.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import Sam3TrackerProcessor, Sam3TrackerModel  # HF backend

from qt_dialogs import pick_file

def pick_image_file() -> Optional[str]:
    return pick_file(
        title="Select an image",
        file_filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
    )


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


def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 (HF Transformers) interactive image demo (points)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--max-side", type=int, default=1200)
    args = ap.parse_args()

    device = auto_device()
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32

    img_path = args.image or pick_image_file()
    if not img_path:
        raise SystemExit("No image selected.")

    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    model = Sam3TrackerModel.from_pretrained(args.model_id).to(device, dtype=dtype)
    processor = Sam3TrackerProcessor.from_pretrained(args.model_id)

    disp_base, scale = compute_display_base(img_bgr, max_side=args.max_side)

    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    last_mask_disp: Optional[np.ndarray] = None

    def redraw():
        vis = disp_base.copy()
        if last_mask_disp is not None:
            vis = green_overlay(vis, last_mask_disp, 0.5)
        for (px, py), lab in zip(points, labels):
            x = int(px * scale)
            y = int(py * scale)
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)
            cv2.circle(vis, (x, y), 6, col, -1)
        cv2.imshow("SAM3 HF (L=pos R=neg u=undo r=reset ESC/q=quit)", vis)

    @torch.inference_mode()
    def run_predict():
        nonlocal last_mask_disp
        if not points:
            last_mask_disp = None
            redraw()
            return

        # HF expects 4D points + 3D labels (see docs). :contentReference[oaicite:4]{index=4}
        pts = [[[[float(x), float(y)] for (x, y) in points]]]
        labs = [[[int(v) for v in labels]]]

        inputs = processor(images=img_pil, input_points=pts, input_labels=labs, return_tensors="pt")
        inputs = inputs.to(device)

        outputs = model(**inputs)

        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        # pick best mask by IoU score (docs: outputs.iou_scores exists). :contentReference[oaicite:5]{index=5}
        best = int(outputs.iou_scores[0, 0].argmax().item())
        mask_hw = (masks[0, best] > 0).numpy()

        if scale != 1.0:
            last_mask_disp = cv2.resize(mask_hw.astype(np.uint8), (disp_base.shape[1], disp_base.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            last_mask_disp = mask_hw.astype(bool)
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
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((px, py)); labels.append(1); run_predict()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((px, py)); labels.append(0); run_predict()
        elif event == cv2.EVENT_MBUTTONDOWN:
            reset_all()

    cv2.namedWindow("SAM3 HF (L=pos R=neg u=undo r=reset ESC/q=quit)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 HF (L=pos R=neg u=undo r=reset ESC/q=quit)", mouse_cb)
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