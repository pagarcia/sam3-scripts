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
        title="Select a video",
        filetypes=[
            ("Videos", "*.mp4;*.mkv;*.avi;*.mov;*.m4v"),
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


def _interactive_points(first_bgr: np.ndarray, max_side: int = 1200) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[int]]]:
    """
    Collect pos/neg clicks on the first frame.
      L-click = positive
      R-click = negative
      M-click or 'r' = reset
      'u' = undo
      Enter = accept
      ESC/q = cancel
    Returns (points_px, labels) or (None, None) on cancel.
    """
    base, scale = compute_display_base(first_bgr, max_side=max_side)
    H, W = first_bgr.shape[:2]

    points: List[Tuple[int, int]] = []
    labels: List[int] = []

    def redraw():
        vis = base.copy()
        for (px, py), lab in zip(points, labels):
            x = int(px * scale)
            y = int(py * scale)
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)
            cv2.circle(vis, (x, y), 6, col, -1)
        cv2.imshow("SAM3 Video – points (L=pos, R=neg, M/r=reset, u=undo, Enter=OK, ESC=cancel)", vis)

    def reset_all():
        points.clear()
        labels.clear()
        redraw()

    def undo_last():
        if points:
            points.pop()
            labels.pop()
        redraw()

    def cb(event, x, y, flags, param):
        # display -> original pixels
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
            redraw()
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            points.append((px, py))
            labels.append(0)
            redraw()
            return

    cv2.namedWindow("SAM3 Video – points (L=pos, R=neg, M/r=reset, u=undo, Enter=OK, ESC=cancel)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 Video – points (L=pos, R=neg, M/r=reset, u=undo, Enter=OK, ESC=cancel)", cb)

    print("[INFO] Select points on the first frame:")
    print("  L-click = positive, R-click = negative")
    print("  M-click / r = reset, u = undo")
    print("  Enter = accept, ESC/q = cancel")

    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):  # ESC/q
            cv2.destroyAllWindows()
            return None, None
        if k == ord("r"):
            reset_all()
        if k == ord("u"):
            undo_last()
        if k in (13, 10):  # Enter
            cv2.destroyAllWindows()
            break

    if not points:
        return None, None
    return points, labels


def _extract_mask(outputs: dict, target_obj_id: Optional[int] = None) -> Optional[np.ndarray]:
    """
    outputs usually contains:
      out_obj_ids: (K,)
      out_binary_masks: (K,H,W)
    Return a single HxW bool mask (union or specific obj_id).
    """
    if outputs is None:
        return None

    obj_ids = outputs.get("out_obj_ids", None)
    masks = outputs.get("out_binary_masks", None)
    if masks is None:
        return None

    m = np.asarray(masks)
    if m.ndim == 2:
        return m.astype(bool)

    if m.ndim != 3:
        return None

    if target_obj_id is None or obj_ids is None:
        return np.any(m.astype(bool), axis=0)

    ids = np.asarray(obj_ids).reshape(-1)
    hit = np.where(ids == int(target_obj_id))[0]
    if hit.size == 0:
        return np.any(m.astype(bool), axis=0)
    return m[int(hit[0])].astype(bool)


def _maybe_disable_hole_fill_for_old_gpu() -> None:
    """
    On sm_61 (Pascal) Triton CC postprocess can spam PTXAS errors.
    Disable by patching fill_holes_in_mask_scores to a no-op.
    """
    if not torch.cuda.is_available():
        return
    maj, _min = torch.cuda.get_device_capability()
    if maj >= 7:
        return

    try:
        import sam3.model.sam3_tracker_utils as stu

        def _no_fill(mask, max_area, fill_holes=True, remove_sprinkles=True):
            return mask

        stu.fill_holes_in_mask_scores = _no_fill
    except Exception:
        pass

    # sam3_video_inference imports the function by name; patch there too if loaded.
    try:
        import sam3.model.sam3_video_inference as svi

        def _no_fill(mask, max_area, fill_holes=True, remove_sprinkles=True):
            return mask

        svi.fill_holes_in_mask_scores = _no_fill
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 interactive video demo (points on frame 0 -> tracking)")
    ap.add_argument("--video", default=None, help="Optional video path. If omitted, a file dialog opens.")
    ap.add_argument("--checkpoint", default=os.getenv("SAM3_CHECKPOINT", ""),
                    help="Optional local sam3.pt path (avoids HF download).")
    ap.add_argument("--obj-id", type=int, default=1, help="Tracked object id.")
    ap.add_argument("--direction", choices=["forward", "backward", "both"], default="forward")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all, else cap propagation length.")
    ap.add_argument("--max-side", type=int, default=1200, help="Max display side for UI windows.")
    ap.add_argument("--out", default="", help="Optional output mp4 path (overlay). Default: no saving.")
    ap.add_argument("--no-show", action="store_true", help="Don’t show the overlay window during tracking.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for SAM3 video predictor (upstream calls .cuda() internally)."
        )

    _ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_video_predictor

    # Pick video
    video_path = args.video or pick_video_file()
    if not video_path:
        raise SystemExit("No video selected.")

    # Read first frame for UI
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    ok, first = cap0.read()
    cap0.release()
    if not ok or first is None:
        raise SystemExit("Cannot read first frame.")

    H, W = first.shape[:2]

    # Build predictor (loads model on GPU)
    ckpt = args.checkpoint.strip() or None
    predictor = build_sam3_video_predictor(checkpoint_path=ckpt) if ckpt else build_sam3_video_predictor()

    # Optional: disable the Triton CC hole-fill postprocess on Pascal (sm_61)
    _maybe_disable_hole_fill_for_old_gpu()

    # Start session
    sid = predictor.handle_request({"type": "start_session", "resource_path": video_path}).get("session_id")
    if not sid:
        raise SystemExit("start_session failed (no session_id returned).")

    try:
        # ---- prompt loop (allow redo) ----
        while True:
            points_px, labels = _interactive_points(first, max_side=args.max_side)
            if points_px is None or labels is None:
                return 0  # user cancelled

            # Convert pixel points -> relative [0..1] coords expected by video inference.
            points_rel = [[float(x) / float(W), float(y) / float(H)] for (x, y) in points_px]

            # Add prompt on frame 0 and get immediate output for preview
            resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": sid,
                "frame_index": 0,
                "points": points_rel,
                "point_labels": labels,
                "obj_id": int(args.obj_id),
            })
            out0 = resp.get("outputs", None)

            mask0 = _extract_mask(out0, target_obj_id=int(args.obj_id))
            base, scale = compute_display_base(first, max_side=args.max_side)
            vis = base.copy()
            if mask0 is not None:
                if scale != 1.0:
                    mask_disp = cv2.resize(mask0.astype(np.uint8),
                                           (base.shape[1], base.shape[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask_disp = mask0
                vis = green_overlay(vis, mask_disp, alpha=0.5)

            # draw points on preview
            for (px, py), lab in zip(points_px, labels):
                x = int(px * scale)
                y = int(py * scale)
                col = (0, 0, 255) if lab == 1 else (255, 0, 0)
                cv2.circle(vis, (x, y), 6, col, -1)

            cv2.imshow("Preview (Enter=track, r=redo points, ESC=quit)", vis)
            print("[INFO] Preview: Enter=track, r=redo, ESC/q=quit")
            while True:
                k = cv2.waitKey(20) & 0xFF
                if k in (27, ord("q")):
                    cv2.destroyAllWindows()
                    return 0
                if k == ord("r"):
                    cv2.destroyAllWindows()
                    predictor.handle_request({"type": "reset_session", "session_id": sid})
                    break  # back to point picking
                if k in (13, 10):
                    cv2.destroyAllWindows()
                    break  # accept and track
            if k in (13, 10):
                break  # accepted

        # ---- propagate + overlay ----
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        out_path = args.out.strip()
        writer = None
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

        stream_req = {
            "type": "propagate_in_video",
            "session_id": sid,
            "propagation_direction": args.direction,
            "start_frame_index": 0,
            "max_frame_num_to_track": (None if args.max_frames <= 0 else int(args.max_frames)),
        }

        if not args.no_show:
            cv2.namedWindow("SAM3 Tracking (ESC/q=stop)", cv2.WINDOW_AUTOSIZE)

        for item in predictor.handle_stream_request(stream_req):
            fi = int(item.get("frame_index", -1))
            outputs = item.get("outputs", None)
            if fi < 0 or outputs is None:
                continue

            # Seek to that frame and read it
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            mask = _extract_mask(outputs, target_obj_id=int(args.obj_id))
            if mask is not None:
                frame = green_overlay(frame, mask.astype(bool), alpha=0.5)

            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("SAM3 Tracking (ESC/q=stop)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    print("[INFO] Stopping early.")
                    break

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[OK] Wrote: {out_path}")

        if not args.no_show:
            cv2.destroyAllWindows()

        return 0

    finally:
        # Always close session
        try:
            predictor.handle_request({"type": "close_session", "session_id": sid})
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())