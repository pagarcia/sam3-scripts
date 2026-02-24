# scripts/sam3_video_demo_hf.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from qt_dialogs import pick_file

def pick_video_file() -> Optional[str]:
    return pick_file(
        title="Select a video",
        file_filter="Videos (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*)",
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


def default_out_path(video_path: str) -> str:
    p = Path(video_path)
    base = p.with_name(p.stem + "_sam3_overlay.mp4")
    if not base.exists():
        return str(base)
    i = 2
    while True:
        cand = p.with_name(f"{p.stem}_sam3_overlay_{i}.mp4")
        if not cand.exists():
            return str(cand)
        i += 1


def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 HF video demo (frame0 points -> headless overlay mp4)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=1200)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = auto_device()
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32

    video_path = args.video or pick_video_file()
    if not video_path:
        raise SystemExit("No video selected.")

    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap0.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_count = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first = cap0.read()
    cap0.release()
    if not ok or first is None:
        raise SystemExit("Cannot read first frame.")
    H0, W0 = first.shape[:2]

    model = Sam3TrackerVideoModel.from_pretrained(args.model_id).to(device, dtype=dtype)
    processor = Sam3TrackerVideoProcessor.from_pretrained(args.model_id)

    # --- Interactive first frame (preview mask updates on click) ---
    base, scale = compute_display_base(first, max_side=args.max_side)
    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    last_mask_disp: Optional[np.ndarray] = None

    # Prepare frame0 inputs once (pixel_values + original size)
    frame0_pil = Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    inputs0 = processor(images=frame0_pil, return_tensors="pt")
    pixel0 = inputs0["pixel_values"].to(device)[0]  # (C,H,W)
    orig0 = inputs0["original_sizes"][0]
    if torch.is_tensor(orig0):
        orig0 = tuple(int(x) for x in orig0.tolist())

    # Streaming session (docs show init without video). :contentReference[oaicite:8]{index=8}
    inference_session = processor.init_video_session(inference_device=device, dtype=dtype)

    @torch.inference_mode()
    def run_preview():
        nonlocal last_mask_disp
        if not points:
            last_mask_disp = None
            return

        pts = [[[[float(x), float(y)] for (x, y) in points]]]
        labs = [[[int(v) for v in labels]]]

        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=int(args.obj_id),
            input_points=pts,
            input_labels=labs,
            original_size=orig0,      # required in streaming mode :contentReference[oaicite:9]{index=9}
            clear_old_inputs=True,
        )

        out0 = model(inference_session=inference_session, frame=pixel0)

        # Do postprocess on CPU to avoid MPS pin_memory/device gotchas.
        masks0 = processor.post_process_masks(
            [out0.pred_masks.cpu()],
            original_sizes=[orig0],
            binarize=True,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )[0]  # expected [num_obj, 1, H, W] :contentReference[oaicite:10]{index=10}

        mask_hw = (masks0[0, 0] > 0).numpy()
        if scale != 1.0:
            last_mask_disp = cv2.resize(mask_hw.astype(np.uint8), (base.shape[1], base.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            last_mask_disp = mask_hw.astype(bool)

    def redraw():
        vis = base.copy()
        if last_mask_disp is not None:
            vis = green_overlay(vis, last_mask_disp, 0.5)
        for (px, py), lab in zip(points, labels):
            x = int(px * scale)
            y = int(py * scale)
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)
            cv2.circle(vis, (x, y), 6, col, -1)
        cv2.imshow("SAM3 HF Video (L=pos R=neg u=undo r=reset Enter=run ESC=quit)", vis)

    def reset_all():
        nonlocal last_mask_disp
        points.clear()
        labels.clear()
        last_mask_disp = None
        inference_session.reset_inference_session()
        redraw()

    def undo_last():
        if points:
            points.pop()
            labels.pop()
        run_preview()
        redraw()

    def mouse_cb(event, x, y, flags, param):
        px = int(x / scale)
        py = int(y / scale)
        px = max(0, min(W0 - 1, px))
        py = max(0, min(H0 - 1, py))
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((px, py)); labels.append(1); run_preview(); redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((px, py)); labels.append(0); run_preview(); redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            reset_all()

    cv2.namedWindow("SAM3 HF Video (L=pos R=neg u=undo r=reset Enter=run ESC=quit)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 HF Video (L=pos R=neg u=undo r=reset Enter=run ESC=quit)", mouse_cb)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            return 0
        if k == ord("r"):
            reset_all()
        if k == ord("u"):
            undo_last()
        if k in (13, 10):  # Enter
            if not points:
                print("[WARN] Add at least one positive point.")
                continue
            cv2.destroyAllWindows()
            break

    # --- Headless pass: decode frames with OpenCV, stream into session, write overlay mp4 ---
    out_path = args.out.strip() or default_out_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W0, H0))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open VideoWriter: {out_path}")

    max_frames = None if int(args.max_frames) <= 0 else int(args.max_frames)
    total = frame_count if frame_count > 0 else None
    if total is not None and max_frames is not None:
        total = min(total, max_frames)

    # fresh session for full run
    inference_session = processor.init_video_session(inference_device=device, dtype=dtype)

    pbar = tqdm(total=total, desc="Tracking", unit="frame")
    try:
        fi = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if max_frames is not None and fi >= max_frames:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            inputs = processor(images=frame_pil, return_tensors="pt")
            pixel = inputs["pixel_values"].to(device)[0]
            orig = inputs["original_sizes"][0]
            if torch.is_tensor(orig):
                orig = tuple(int(x) for x in orig.tolist())

            if fi == 0:
                pts = [[[[float(x), float(y)] for (x, y) in points]]]
                labs = [[[int(v) for v in labels]]]
                processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=0,
                    obj_ids=int(args.obj_id),
                    input_points=pts,
                    input_labels=labs,
                    original_size=orig,
                    clear_old_inputs=True,
                )

            out = model(inference_session=inference_session, frame=pixel)

            masks = processor.post_process_masks(
                [out.pred_masks.cpu()],
                original_sizes=[orig],
                binarize=True,
                max_hole_area=0.0,
                max_sprinkle_area=0.0,
            )[0]

            mask_hw = (masks[0, 0] > 0).numpy()
            frame_out = green_overlay(frame_bgr, mask_hw, 0.5)

            writer.write(frame_out)
            pbar.update(1)

            fi += 1

    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())