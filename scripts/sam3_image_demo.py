# scripts/sam3_image_demo.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

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
            repo_root / "third_party" / "sam3"  # optional submodule style
        ]:
            if (cand / "sam3").exists():
                sys.path.insert(0, str(cand))
                return
        raise


def _overlay_union_mask(bgr: np.ndarray, mask_hw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Green overlay of a boolean mask."""
    mask = mask_hw.astype(bool)
    color = np.zeros_like(bgr)
    color[mask] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0.0)


def _interactive_box_select(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) in pixels or None."""
    disp = bgr.copy()
    rect_s = rect_e = None
    drawing = False

    def redraw():
        vis = disp.copy()
        if rect_s and rect_e:
            cv2.rectangle(vis, rect_s, rect_e, (0, 255, 255), 2)
        cv2.imshow("Select box (ENTER=accept, ESC=cancel)", vis)

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

    cv2.namedWindow("Select box (ENTER=accept, ESC=cancel)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Select box (ENTER=accept, ESC=cancel)", cb)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        if k in (13, 10):  # ENTER
            cv2.destroyAllWindows()
            break

    if not (rect_s and rect_e):
        return None

    x1, y1 = rect_s
    x2, y2 = rect_e
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (x1, y1, x2, y2)


def main() -> int:
    ap = argparse.ArgumentParser("SAM3 image demo (text and/or box prompt)")
    ap.add_argument("--image", required=True, help="Path to an image file.")
    ap.add_argument("--text", default=None, help="Text prompt, e.g. 'person'.")
    ap.add_argument("--box", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Box prompt in pixel coords (xyxy).")
    ap.add_argument("--interactive-box", action="store_true", help="Draw a box in an OpenCV window.")
    ap.add_argument("--checkpoint", default=os.getenv("SAM3_CHECKPOINT", ""),
                    help="Optional local sam3.pt path (avoids HF download).")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for masks.")
    ap.add_argument("--out", default="sam3_out.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.text and not args.box and not args.interactive_box:
        ap.error("Provide --text and/or --box/--interactive-box.")

    _ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Load image
    img_pil = Image.open(args.image).convert("RGB")
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]

    # Optional interactive box
    box_xyxy = tuple(args.box) if args.box else None
    if args.interactive_box:
        sel = _interactive_box_select(bgr)
        if sel is not None:
            box_xyxy = sel

    ckpt = args.checkpoint.strip() or None

    # Build model + processor (SAM3 upstream API)
    model = build_sam3_image_model(
        checkpoint_path=ckpt,
        load_from_HF=(ckpt is None),
        device=args.device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.conf)

    state = processor.set_image(img_pil)

    # Text prompt
    if args.text:
        state = processor.set_text_prompt(prompt=args.text, state=state)

    # Box prompt (Sam3Processor expects normalized [cx,cy,w,h]) :contentReference[oaicite:2]{index=2}
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        cx = ((x1 + x2) * 0.5) / float(W)
        cy = ((y1 + y2) * 0.5) / float(H)
        bw = (x2 - x1) / float(W)
        bh = (y2 - y1) / float(H)

        state = processor.add_geometric_prompt(
            box=[cx, cy, bw, bh],
            label=True,
            state=state,
        )

    masks = state.get("masks", None)
    boxes = state.get("boxes", None)
    scores = state.get("scores", None)

    if masks is None or (hasattr(masks, "numel") and masks.numel() == 0):
        print("[WARN] No masks returned.")
        out_bgr = bgr
    else:
        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        # masks_np is typically [K,1,H,W] or [K,H,W]
        while masks_np.ndim > 3:
            masks_np = masks_np[:, 0]
        union = np.any(masks_np.astype(bool), axis=0)
        out_bgr = _overlay_union_mask(bgr, union, alpha=0.5)

        # Optional: draw boxes
        if boxes is not None:
            bx = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)
            sc = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else (np.asarray(scores) if scores is not None else None)
            for i in range(min(len(bx), 50)):
                x1, y1, x2, y2 = bx[i].astype(int).tolist()
                cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                if sc is not None:
                    cv2.putText(out_bgr, f"{sc[i]:.2f}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(args.out, out_bgr)
    print(f"[OK] Wrote {args.out}")

    if args.show:
        cv2.imshow("SAM3 image result", out_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())