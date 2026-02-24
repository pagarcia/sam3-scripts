# scripts/sam3_video_demo.py
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
            repo_root.parent / "sam3",
            repo_root.parent / "sam3_repo",
            repo_root / "third_party" / "sam3",
        ]:
            if (cand / "sam3").exists():
                sys.path.insert(0, str(cand))
                return
        raise


def pick_video_file() -> Optional[str]:
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
        title="Select a video",
        filetypes=[("Videos", "*.mp4;*.mkv;*.avi;*.mov;*.m4v"), ("All files", "*.*")],
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


def _load_ckpt_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("Unexpected checkpoint format")


def _build_tracker_from_sam3_ckpt(checkpoint_path: str):
    """
    Build Sam3TrackerPredictor WITH a vision backbone and load weights:
      - tracker.* -> tracker predictor
      - detector.backbone.vision_backbone.* -> tracker.backbone.vision_backbone.*
    """
    from sam3.model_builder import build_tracker

    # Build tracker with its own backbone (so it can run standalone on video frames)
    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
    tracker = tracker.cuda().eval()

    sd_src = _load_ckpt_dict(checkpoint_path)
    sd = {}

    # Load tracker weights
    for k, v in sd_src.items():
        if k.startswith("tracker."):
            sd[k[len("tracker."):]] = v

    # Load vision backbone weights from detector into tracker backbone
    det_vis_prefix = "detector.backbone.vision_backbone."
    for k, v in sd_src.items():
        if k.startswith(det_vis_prefix):
            sd["backbone.vision_backbone." + k[len(det_vis_prefix):]] = v

    missing, unexpected = tracker.load_state_dict(sd, strict=False)

    # Make autocast sane on older GPUs (Pascal doesn't support bf16 well)
    try:
        tracker.bf16_context.__exit__(None, None, None)  # type: ignore[attr-defined]
    except Exception:
        pass
    major, _ = torch.cuda.get_device_capability()
    cast_dtype = torch.bfloat16 if major >= 8 else torch.float16
    tracker.bf16_context = torch.autocast(device_type="cuda", dtype=cast_dtype)  # type: ignore[attr-defined]
    tracker.bf16_context.__enter__()  # type: ignore[attr-defined]

    print(f"[INFO] Tracker weights loaded. missing={len(missing)} unexpected={len(unexpected)} autocast={cast_dtype}")
    return tracker


def _mask_for_obj(video_res_masks, obj_ids: List[int], obj_id: int) -> Optional[np.ndarray]:
    if video_res_masks is None:
        return None
    m = video_res_masks
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)

    # Shapes seen: [N,1,H,W] or [N,H,W]
    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0]
    if m.ndim != 3:
        return None

    if obj_id in obj_ids:
        idx = obj_ids.index(obj_id)
        return (m[idx] > 0)
    return (np.any(m > 0, axis=0))


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 video demo (SAM2-style tracker: click frame 0 -> propagate)")
    ap.add_argument("--video", default=None, help="Optional MP4 path. If omitted, a file dialog opens.")
    ap.add_argument("--checkpoint", default=os.getenv("SAM3_CHECKPOINT", ""), help="Optional local sam3.pt path.")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=1200)
    ap.add_argument("--max-frames", type=int, default=0, help="0=all")
    ap.add_argument("--out", default="", help="Optional output mp4 (overlay). Default: no saving.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this demo.")

    _ensure_sam3_on_path()
    from sam3.model_builder import download_ckpt_from_hf

    video_path = args.video or pick_video_file()
    if not video_path:
        raise SystemExit("No video selected.")

    # Read original first frame for UI (no resizing)
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    ok, first = cap0.read()
    cap0.release()
    if not ok or first is None:
        raise SystemExit("Cannot read first frame.")
    H0, W0 = first.shape[:2]

    # Checkpoint
    ckpt = args.checkpoint.strip()
    if not ckpt:
        ckpt = download_ckpt_from_hf()  # uses HF cache; no prompt if already logged in
    tracker = _build_tracker_from_sam3_ckpt(ckpt)

    # Tracker inference state (loads and resizes frames internally to image_size=1008)
    # Offloading frames to CPU saves VRAM; keep default True for 12GB cards.
    state = tracker.init_state(video_path=video_path, offload_video_to_cpu=True, async_loading_frames=False)

    # Interactive UI on frame 0 (mask updates on each click)
    base, scale = compute_display_base(first, max_side=args.max_side)
    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    last_mask_disp: Optional[np.ndarray] = None

    def redraw():
        vis = base.copy()
        if last_mask_disp is not None:
            vis = green_overlay(vis, last_mask_disp, alpha=0.5)
        for (px, py), lab in zip(points, labels):
            x = int(px * scale)
            y = int(py * scale)
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)
            cv2.circle(vis, (x, y), 6, col, -1)
        cv2.imshow("SAM3 Video (L=pos R=neg u=undo r=reset Enter=track ESC=quit)", vis)

    def clear_frame0():
        try:
            tracker.clear_all_points_in_frame(state, frame_idx=0, obj_id=int(args.obj_id), need_output=False)
        except Exception:
            pass

    def replay_all():
        nonlocal last_mask_disp
        clear_frame0()
        last_mask_disp = None
        out_mask = None
        for (px, py), lab in zip(points, labels):
            pt_rel = np.array([[px / float(W0), py / float(H0)]], dtype=np.float32)
            lab_arr = np.array([lab], dtype=np.int32)
            _, obj_ids, _, video_res_masks = tracker.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=int(args.obj_id),
                points=pt_rel,
                labels=lab_arr,
                clear_old_points=False,
                rel_coordinates=True,
            )
            out_mask = _mask_for_obj(video_res_masks, [int(x) for x in obj_ids], int(args.obj_id))

        if out_mask is not None:
            if scale != 1.0:
                last_mask_disp = cv2.resize(out_mask.astype(np.uint8), (base.shape[1], base.shape[0]),
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                last_mask_disp = out_mask.astype(bool)

    def add_click(px: int, py: int, lab: int):
        nonlocal last_mask_disp
        points.append((px, py))
        labels.append(lab)

        pt_rel = np.array([[px / float(W0), py / float(H0)]], dtype=np.float32)
        lab_arr = np.array([lab], dtype=np.int32)

        _, obj_ids, _, video_res_masks = tracker.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=int(args.obj_id),
            points=pt_rel,
            labels=lab_arr,
            clear_old_points=False,
            rel_coordinates=True,
        )
        out_mask = _mask_for_obj(video_res_masks, [int(x) for x in obj_ids], int(args.obj_id))
        if out_mask is None:
            last_mask_disp = None
            return

        if scale != 1.0:
            last_mask_disp = cv2.resize(out_mask.astype(np.uint8), (base.shape[1], base.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            last_mask_disp = out_mask.astype(bool)

    def undo_last():
        if not points:
            return
        points.pop()
        labels.pop()
        replay_all()

    def reset_all():
        nonlocal last_mask_disp
        points.clear()
        labels.clear()
        last_mask_disp = None
        clear_frame0()

    def mouse_cb(event, x, y, flags, param):
        px = int(x / scale)
        py = int(y / scale)
        px = max(0, min(W0 - 1, px))
        py = max(0, min(H0 - 1, py))

        if event == cv2.EVENT_LBUTTONDOWN:
            add_click(px, py, 1)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            add_click(px, py, 0)
            redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            reset_all()
            redraw()

    cv2.namedWindow("SAM3 Video (L=pos R=neg u=undo r=reset Enter=track ESC=quit)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 Video (L=pos R=neg u=undo r=reset Enter=track ESC=quit)", mouse_cb)

    print("[INFO] Controls: L=pos, R=neg, u=undo, r=reset, Enter=track, ESC/q=quit")
    redraw()

    # Wait for Enter / quit
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            return 0
        if k == ord("r"):
            reset_all()
            redraw()
        if k == ord("u"):
            undo_last()
            redraw()
        if k in (13, 10):  # Enter
            if not points:
                print("[WARN] No points. Add at least one positive click.")
                continue
            cv2.destroyAllWindows()
            break

    # Propagate forward
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = None
    if args.out.strip():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out.strip(), fourcc, fps, (W0, H0))
        if not writer.isOpened():
            raise SystemExit(f"Cannot open VideoWriter: {args.out.strip()}")

    cv2.namedWindow("SAM3 Tracking (ESC/q=stop)", cv2.WINDOW_AUTOSIZE)

    max_frames = None if int(args.max_frames) <= 0 else int(args.max_frames)

    for frame_idx, obj_ids, _low_res, video_res_masks, _obj_scores in tracker.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=max_frames,
        reverse=False,
        tqdm_disable=True,
        propagate_preflight=True,
    ):
        # Read original frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        mask = _mask_for_obj(video_res_masks, [int(x) for x in obj_ids], int(args.obj_id))
        if mask is not None:
            frame = green_overlay(frame, mask, alpha=0.5)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("SAM3 Tracking (ESC/q=stop)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            print("[INFO] Stopped early.")
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[OK] Wrote {args.out.strip()}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())