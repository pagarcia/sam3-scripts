# scripts/sam3_image_demo.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _overlay_union_mask(bgr: np.ndarray, mask_hw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    mask = mask_hw.astype(bool)
    color = np.zeros_like(bgr)
    color[mask] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0.0)


def _interactive_points(first_bgr: np.ndarray) -> Tuple[List[List[float]], List[int]]:
    """
    Collect positive/negative clicks in pixel coords.
    L-click=pos, R-click=neg, M-click=reset, ENTER=done, ESC=cancel.
    """
    disp = first_bgr.copy()
    pts: List[Tuple[int, int]] = []
    labs: List[int] = []

    def redraw():
        vis = disp.copy()
        for (x, y), lab in zip(pts, labs):
            col = (0, 0, 255) if lab == 1 else (255, 0, 0)
            cv2.circle(vis, (x, y), 6, col, -1)
        cv2.imshow("Points (ENTER=run, ESC=cancel)", vis)

    def cb(event, x, y, flags, param):
        nonlocal pts, labs
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); labs.append(1); redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            pts.append((x, y)); labs.append(0); redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            pts, labs = [], []; redraw()

    cv2.namedWindow("Points (ENTER=run, ESC=cancel)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Points (ENTER=run, ESC=cancel)", cb)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return [], []
        if k in (13, 10):
            cv2.destroyAllWindows()
            break

    return [[float(x), float(y)] for (x, y) in pts], labs


def _interactive_box(first_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) in pixels or None."""
    disp = first_bgr.copy()
    rect_s = rect_e = None
    drawing = False

    def redraw():
        vis = disp.copy()
        if rect_s and rect_e:
            cv2.rectangle(vis, rect_s, rect_e, (0, 255, 255), 2)
        cv2.imshow("Box (ENTER=run, ESC=cancel)", vis)

    def cb(event, x, y, flags, param):
        nonlocal rect_s, rect_e, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_s = rect_e = (x, y)
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_e = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            rect_e = (x, y)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            rect_s = rect_e = None
            redraw()

    cv2.namedWindow("Box (ENTER=run, ESC=cancel)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Box (ENTER=run, ESC=cancel)", cb)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if k in (13, 10):
            cv2.destroyAllWindows()
            break

    if not (rect_s and rect_e):
        return None

    x1, y1 = rect_s
    x2, y2 = rect_e
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (x1, y1, x2, y2)


def _boxes_xyxy_px_to_xywh_norm(box_xyxy: Tuple[int, int, int, int], w: int, h: int) -> List[List[float]]:
    x1, y1, x2, y2 = box_xyxy
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    # SAM3 video expects [xmin, ymin, width, height] normalized 0..1 :contentReference[oaicite:5]{index=5}
    return [[x1 / w, y1 / h, bw / w, bh / h]]


def _points_px_to_rel(points_px: List[List[float]], w: int, h: int) -> List[List[float]]:
    out = []
    for x, y in points_px:
        out.append([float(x) / float(w), float(y) / float(h)])
    return out


def _collect_outputs(predictor, session_id: str, direction: str, start_frame: int, max_frames: int) -> Dict[int, dict]:
    req = dict(
        type="propagate_in_video",
        session_id=session_id,
        propagation_direction=direction,
        start_frame_index=start_frame,
        max_frame_num_to_track=(None if max_frames <= 0 else int(max_frames)),
    )
    outputs: Dict[int, dict] = {}
    for resp in predictor.handle_stream_request(req):
        fi = resp.get("frame_index", None)
        out = resp.get("outputs", None)
        if fi is None or out is None:
            continue
        outputs[int(fi)] = out
    return outputs


def _write_overlay_video(video_path: str, outputs: Dict[int, dict], out_path: str,
                         only_obj_id: Optional[int] = None) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = outputs.get(idx, None)
        if out is not None:
            obj_ids = np.asarray(out.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
            masks = np.asarray(out.get("out_binary_masks", []))
            boxes = np.asarray(out.get("out_boxes_xywh", []))
            probs = np.asarray(out.get("out_probs", []))

            if masks.ndim == 3 and masks.shape[0] == len(obj_ids):
                if only_obj_id is None:
                    union = np.any(masks.astype(bool), axis=0)
                    frame = _overlay_union_mask(frame, union, alpha=0.5)
                else:
                    hit = np.where(obj_ids == int(only_obj_id))[0]
                    if hit.size:
                        union = masks[int(hit[0])].astype(bool)
                        frame = _overlay_union_mask(frame, union, alpha=0.5)

            # draw boxes as (xmin,ymin,w,h) normalized -> pixels :contentReference[oaicite:6]{index=6}
            if boxes.ndim == 2 and boxes.shape[1] == 4:
                for i in range(min(len(boxes), 50)):
                    x, y, bw, bh = boxes[i].tolist()
                    x1 = int(x * w); y1 = int(y * h)
                    x2 = int((x + bw) * w); y2 = int((y + bh) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    if i < len(obj_ids) and i < len(probs):
                        cv2.putText(frame, f"id={int(obj_ids[i])} p={float(probs[i]):.2f}",
                                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 video demo (text/box concept tracking OR point-based tracking)")
    ap.add_argument("--video", required=True, help="MP4 path (or a JPEG folder supported by SAM3).")
    ap.add_argument("--prompt", choices=["text", "box", "points"], default="text")
    ap.add_argument("--text", default=None, help="Text prompt for --prompt text (or combined with box).")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to add the initial prompt.")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all, else cap tracking length.")
    ap.add_argument("--direction", choices=["forward", "backward", "both"], default="forward")
    ap.add_argument("--checkpoint", default=os.getenv("SAM3_CHECKPOINT", ""),
                    help="Optional local sam3.pt path (avoids HF download).")
    ap.add_argument("--interactive-box", action="store_true", help="Draw a box on the prompt frame.")
    ap.add_argument("--box", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Box in pixel coords (xyxy) for --prompt box.")
    ap.add_argument("--obj-id", type=int, default=1,
                    help="Object id for --prompt points (and optional overlay filtering).")
    ap.add_argument("--overlay-obj-id", type=int, default=0,
                    help="0=union overlay, else overlay only this obj_id.")
    ap.add_argument("--out", default="sam3_out.mp4")
    args = ap.parse_args()

    _ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_video_predictor

    # Prepare first frame for interactive prompts (if needed)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Cannot read first frame.")

    H, W = first.shape[:2]

    # Build predictor (upstream API) :contentReference[oaicite:7]{index=7}
    ckpt = args.checkpoint.strip() or None
    predictor = build_sam3_video_predictor(checkpoint_path=ckpt, load_from_HF=(ckpt is None))

    # Start session
    resp = predictor.handle_request(dict(type="start_session", resource_path=args.video))
    session_id = resp["session_id"]

    # Add initial prompt
    if args.prompt == "text":
        if not args.text:
            raise SystemExit("--prompt text requires --text")
        _ = predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=int(args.frame),
            text=args.text,
        ))

    elif args.prompt == "box":
        box_xyxy = tuple(args.box) if args.box else None
        if args.interactive_box:
            sel = _interactive_box(first)
            if sel is not None:
                box_xyxy = sel
        if box_xyxy is None and not args.text:
            raise SystemExit("--prompt box needs --interactive-box or --box (or provide --text and use text mode).")

        boxes_norm = _boxes_xyxy_px_to_xywh_norm(box_xyxy, W, H) if box_xyxy is not None else None

        req = dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=int(args.frame),
            text=(args.text if args.text else None),
        )
        if boxes_norm is not None:
            req["bounding_boxes"] = boxes_norm
            req["bounding_box_labels"] = [1]
        _ = predictor.handle_request(req)

    else:  # points (tracker-style)
        pts_px, labs = _interactive_points(first)
        if not pts_px:
            raise SystemExit("No points selected.")
        pts_rel = _points_px_to_rel(pts_px, W, H)

        _ = predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=int(args.frame),
            points=pts_rel,                 # rel coords by default :contentReference[oaicite:8]{index=8}
            point_labels=labs,
            obj_id=int(args.obj_id),
        ))

    # Propagate + collect
    outputs = _collect_outputs(
        predictor,
        session_id=session_id,
        direction=args.direction,
        start_frame=int(args.frame),
        max_frames=int(args.max_frames),
    )

    # Close session (best practice)
    _ = predictor.handle_request(dict(type="close_session", session_id=session_id))

    only_obj = None if args.overlay_obj_id == 0 else int(args.overlay_obj_id)
    _write_overlay_video(args.video, outputs, args.out, only_obj_id=only_obj)
    print(f"[OK] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())