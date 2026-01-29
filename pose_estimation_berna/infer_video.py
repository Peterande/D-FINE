#!/usr/bin/env python3
"""
Run D-FINE pose (per-query keypoints) on a video file or stream (RTSP/HTTP) and save an annotated mp4.

This is SSH-friendly: it does not open any UI window; it just writes output to disk.
"""

import argparse
import os
import sys
import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# Ensure imports work when run as a script (also under multiprocessing contexts)
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../pose_estimation_berna
repo_root_dir = os.path.dirname(current_dir)  # .../D-FINE-NEWBRINGER
src_dir = os.path.join(repo_root_dir, "src")
for p in [repo_root_dir, current_dir, src_dir]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from src.core import YAMLConfig  # noqa: E402


# COCO-17 skeleton in **0-indexed** keypoint order used by COCO annotations:
# [nose, leye, reye, lear, rear, lsho, rsho, lelb, relb, lwri, rwri, lhip, rhip, lkne, rkne, lank, rank]
COCO_SKELETON = [
    (15, 13),  # left ankle - left knee
    (13, 11),  # left knee - left hip
    (16, 14),  # right ankle - right knee
    (14, 12),  # right knee - right hip
    (11, 12),  # left hip - right hip
    (5, 11),   # left shoulder - left hip
    (6, 12),   # right shoulder - right hip
    (5, 6),    # left shoulder - right shoulder
    (5, 7),    # left shoulder - left elbow
    (7, 9),    # left elbow - left wrist
    (6, 8),    # right shoulder - right elbow
    (8, 10),   # right elbow - right wrist
    (0, 1),    # nose - left eye
    (0, 2),    # nose - right eye
    (1, 3),    # left eye - left ear
    (2, 4),    # right eye - right ear
    (3, 5),    # left ear - left shoulder
    (4, 6),    # right ear - right shoulder
    (1, 2),    # left eye - right eye
]

# Swap left/right keypoint indices for COCO-17 (useful for debugging if the model outputs are permuted).
COCO_KEYPOINT_FLIP_INDEX = [
    0,  # nose
    2, 1,  # left_eye <-> right_eye
    4, 3,  # left_ear <-> right_ear
    6, 5,  # left_shoulder <-> right_shoulder
    8, 7,  # left_elbow <-> right_elbow
    10, 9,  # left_wrist <-> right_wrist
    12, 11,  # left_hip <-> right_hip
    14, 13,  # left_knee <-> right_knee
    16, 15,  # left_ankle <-> right_ankle
]


def load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt


def draw_pose(
    img_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    keypoints: np.ndarray,
    color=(0, 255, 0),
    kpt_thr: float = 0.2,
    show_kpt_idx: bool = False,
):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1i = int(np.clip(round(x1), 0, max(0, w - 1)))
    y1i = int(np.clip(round(y1), 0, max(0, h - 1)))
    x2i = int(np.clip(round(x2), 0, max(0, w - 1)))
    y2i = int(np.clip(round(y2), 0, max(0, h - 1)))
    cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), color, 2)

    # keypoints: [17,3] (x,y,score)
    for i in range(keypoints.shape[0]):
        x, y, s = keypoints[i]
        if s < kpt_thr:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
        if show_kpt_idx:
            cv2.putText(
                img_bgr,
                str(int(i)),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    for a, b in COCO_SKELETON:
        xa, ya, sa = keypoints[a]
        xb, yb, sb = keypoints[b]
        if sa < kpt_thr or sb < kpt_thr:
            continue
        if xa < 0 or ya < 0 or xa >= w or ya >= h:
            continue
        if xb < 0 or yb < 0 or xb >= w or yb >= h:
            continue
        cv2.line(img_bgr, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


def open_capture(source: str) -> cv2.VideoCapture:
    # Source can be a path or stream URL (rtsp/http)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def _try_make_writer_cv2(out_path: str, fps: float, size_wh: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    w, h = size_wh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        return None
    return writer


class FFmpegWriter:
    def __init__(self, out_path: str, fps: float, size_wh: Tuple[int, int]):
        w, h = size_wh
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(w)}x{int(h)}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame_bgr: np.ndarray):
        if self.proc.stdin is None:
            return
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def main():
    # Local imports (after sys.path adjustments above)
    from pose_estimation_berna.core.tracking import IoUTracker, KalmanTracker  # noqa: WPS433
    from pose_estimation_berna.core.posture import PostureEstimator  # noqa: WPS433

    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        "-c",
        default="pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_pose.yml",
        help="D-FINE pose config (must enable num_keypoints=17)",
    )
    p.add_argument(
        "--checkpoint",
        "-r",
        default="outputs/standard/standard/best.pth",
        help="Checkpoint path (state_dict or dict with 'model'/'ema')",
    )
    p.add_argument("--input", "-i", required=True, help="Input video path or stream URL (rtsp/http)")
    p.add_argument("--out", "-o", default="pose_result.mp4", help="Output video path")
    p.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--score-thr", type=float, default=0.4)
    p.add_argument("--kpt-thr", type=float, default=0.5, help="Keypoint score threshold for drawing")
    p.add_argument("--show-kpt-idx", action="store_true", help="Draw keypoint indices (0-16) next to points")
    p.add_argument(
        "--swap-lr-kpts",
        action="store_true",
        help="Swap left/right keypoint indices before drawing (debug for permuted outputs)",
    )
    p.add_argument("--max-persons", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (debug)")
    p.add_argument("--fps", type=float, default=None, help="Override output fps (defaults to input fps)")
    p.add_argument("--use-ffmpeg", action="store_true", help="Use ffmpeg for writing mp4 (recommended)")
    p.add_argument("--verbose", action="store_true", help="Print per-frame debug (best score, drawn count)")
    p.add_argument("--log-every", type=int, default=30, help="Verbose log period in frames (default: 30)")
    p.add_argument(
        "--overlay-stats",
        action="store_true",
        help="Overlay simple debug stats on the output video (tracks/drawn/best_person_score).",
    )
    p.add_argument("--track", action="store_true", help="Enable simple IoU tracking (stable IDs over frames)")
    p.add_argument(
        "--track-method",
        choices=["iou", "kalman"],
        default="kalman",
        help="Tracking backend. 'kalman' predicts through short occlusions; 'iou' is simplest.",
    )
    p.add_argument("--track-iou", type=float, default=0.3, help="IoU threshold for matching tracks")
    p.add_argument(
        "--track-oks-thr",
        type=float,
        default=1.0,
        help="OKS-like threshold for matching tracks when keypoints are available (allows match even if IoU is low).",
    )
    p.add_argument("--track-max-age", type=int, default=30, help="Keep track alive for N missed frames")
    p.add_argument(
        "--track-score-thr",
        type=float,
        default=0.15,
        help="Low score threshold for tracker association (ByteTrack-lite). Drawing still uses --score-thr.",
    )
    p.add_argument(
        "--draw-lost",
        action="store_true",
        help="If set, also draw tracks that were not updated this frame (predicted through occlusion).",
    )
    p.add_argument(
        "--draw-lost-min-score",
        type=float,
        default=0.20,
        help="Minimum (possibly decayed) track score to draw when --draw-lost is enabled.",
    )
    p.add_argument(
        "--draw-lost-any-score",
        action="store_true",
        help="If set, draw lost tracks regardless of their (decayed) score when --draw-lost is enabled.",
    )
    p.add_argument(
        "--track-score-decay",
        type=float,
        default=0.98,
        help="When occluded (no update), decay track score by this factor each frame (kalman mode).",
    )
    p.add_argument(
        "--track-score-decay-grace",
        type=int,
        default=0,
        help="Number of lost frames to keep track score constant before applying --track-score-decay (kalman mode).",
    )
    p.add_argument(
        "--track-iou-weight",
        type=float,
        default=1.0,
        help="Association weight for IoU (kalman mode).",
    )
    p.add_argument(
        "--track-oks-weight",
        type=float,
        default=0.0,
        help="Association weight for OKS-like keypoint similarity (kalman mode).",
    )
    p.add_argument(
        "--track-kpt-thr",
        type=float,
        default=0.2,
        help="Keypoint score threshold used inside OKS-like association (kalman mode).",
    )
    p.add_argument(
        "--track-kpt-age-decay",
        type=float,
        default=0.97,
        help="Per-frame decay applied to OKS-like association when a track is lost (kalman mode).",
    )
    p.add_argument("--smooth-alpha", type=float, default=0.8, help="EMA smoothing alpha for tracking (higher=more smoothing)")
    p.add_argument("--no-smooth-boxes", action="store_true", help="Disable box smoothing in tracker")
    p.add_argument("--no-smooth-kpts", action="store_true", help="Disable keypoint smoothing in tracker")
    p.add_argument(
        "--track-min-area-ratio",
        type=float,
        default=0.6,
        help="Prevent box from shrinking too much under occlusion (min area ratio vs previous)",
    )
    p.add_argument(
        "--lock-track",
        action="store_true",
        help="Lock onto a single track_id once acquired (best for single-person videos / occlusion behind objects)",
    )
    p.add_argument(
        "--lock-track-id",
        type=int,
        default=None,
        help="If set, always draw only this track_id (requires --track).",
    )
    p.add_argument("--posture", action="store_true", help="Estimate posture per track_id (stand/crouch/lie)")
    p.add_argument("--posture-hold", type=int, default=15, help="Hold last posture probs for N frames when occluded")
    p.add_argument("--posture-decay", type=float, default=0.9, help="Decay posture probs towards uniform after hold")
    args = p.parse_args()
    if args.out is None or str(args.out).strip() == "":
        raise ValueError("--out must be a non-empty path (e.g. pose_result.mp4)")
    out_parent = os.path.dirname(os.path.abspath(str(args.out)))
    if out_parent and not os.path.exists(out_parent):
        os.makedirs(out_parent, exist_ok=True)

    cfg = YAMLConfig(args.config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    state = load_state_dict(args.checkpoint)
    cfg.model.load_state_dict(state, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = cfg.model.deploy().to(device).eval()
    post = cfg.postprocessor.to(device).eval()
    # Match training/eval convention: map contiguous labels -> MSCOCO category ids (person == 1)
    post.remap_mscoco_category = True

    tfm = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    tracker = None
    posture_est = None
    if args.track:
        if args.track_method == "kalman":
            tracker = KalmanTracker(
                iou_threshold=float(args.track_iou),
                oks_threshold=float(args.track_oks_thr),
                max_age=int(args.track_max_age),
                smooth_alpha=float(args.smooth_alpha),
                smooth_boxes=not bool(args.no_smooth_boxes),
                smooth_keypoints=not bool(args.no_smooth_kpts),
                min_area_ratio=float(args.track_min_area_ratio),
                score_decay=float(args.track_score_decay),
                score_decay_grace=int(args.track_score_decay_grace),
                iou_weight=float(args.track_iou_weight),
                oks_weight=float(args.track_oks_weight),
                kpt_thr=float(args.track_kpt_thr),
                kpt_age_decay=float(args.track_kpt_age_decay),
            )
        else:
            tracker = IoUTracker(
                iou_threshold=float(args.track_iou),
                max_age=int(args.track_max_age),
                smooth_alpha=float(args.smooth_alpha),
                smooth_boxes=not bool(args.no_smooth_boxes),
                smooth_keypoints=not bool(args.no_smooth_kpts),
                min_area_ratio=float(args.track_min_area_ratio),
            )
    if args.posture:
        posture_est = PostureEstimator(
            kpt_thr=float(args.kpt_thr),
            hold_frames=int(args.posture_hold),
            decay=float(args.posture_decay),
        )
    
    locked_track_id = int(args.lock_track_id) if args.lock_track_id is not None else None

    cap = open_capture(args.input)
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 1e-3:
        in_fps = 30.0
    out_fps = float(args.fps) if args.fps is not None else float(in_fps)

    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError("Could not read first frame from input.")

    base_h, base_w = frame_bgr.shape[:2]
    size_wh = (base_w, base_h)

    writer_cv2 = None
    writer_ff = None
    if args.use_ffmpeg:
        writer_ff = FFmpegWriter(args.out, out_fps, size_wh)
    else:
        writer_cv2 = _try_make_writer_cv2(args.out, out_fps, size_wh)
        if writer_cv2 is None:
            # fallback to ffmpeg
            writer_ff = FFmpegWriter(args.out, out_fps, size_wh)

    def _write(frame: np.ndarray):
        if writer_cv2 is not None:
            writer_cv2.write(frame)
        elif writer_ff is not None:
            writer_ff.write(frame)

    frame_idx = 0
    while True:
        if frame_idx > 0:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

        # Some sources (streams) may change resolution mid-stream; keep output stable and keep
        # orig_size consistent with the frame we actually run visualization on.
        h, w = frame_bgr.shape[:2]
        if (h != base_h) or (w != base_w):
            frame_bgr = cv2.resize(frame_bgr, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            h, w = base_h, base_w

        # preprocess (keep original resolution for visualization, but run 640x640 into model)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb)
        orig_size = torch.tensor([[w, h]], device=device)
        x = tfm(im_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            results = post(outputs, orig_size)

        det = results[0]
        labels = det["labels"].detach().cpu().numpy()
        boxes = det["boxes"].detach().cpu().numpy()
        scores = det["scores"].detach().cpu().numpy()
        keypoints = det.get("keypoints", None)
        if keypoints is not None:
            keypoints = keypoints.detach().cpu().numpy()

        # Two-threshold strategy:
        # - drawing uses args.score_thr (high)
        # - tracker association can use args.track_score_thr (low) to keep tracks alive through occlusion / low confidence.
        keep = scores >= args.score_thr
        # Depending on whether category remapping is enabled, person can be 1 (MSCOCO category_id)
        # or 0 (contiguous label). Keep both to avoid silent filtering issues.
        keep = keep & np.isin(labels, [0, 1])
        idxs = np.where(keep)[0]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        idxs = idxs[: args.max_persons]

        # Optional: track detections to stable IDs + smooth boxes/keypoints.
        track_outputs = None
        if tracker is not None:
            keep_trk = (scores >= float(args.track_score_thr)) & np.isin(labels, [0, 1])
            idxs_trk = np.where(keep_trk)[0]
            # Let the tracker see more candidates (don't cap too early), but still prioritize by score.
            idxs_trk = idxs_trk[np.argsort(scores[idxs_trk])[::-1]]
            cap_trk = max(int(args.max_persons) * 5, int(args.max_persons))
            idxs_trk = idxs_trk[:cap_trk]

            det_boxes = boxes[idxs_trk] if idxs_trk.size > 0 else np.zeros((0, 4), dtype=np.float32)
            det_scores = scores[idxs_trk] if idxs_trk.size > 0 else np.zeros((0,), dtype=np.float32)
            det_kpts = None
            if keypoints is not None and idxs_trk.size > 0:
                det_kpts = keypoints[idxs_trk]
            track_outputs = tracker.update(det_boxes, det_scores, det_kpts)

        if args.verbose and (frame_idx % max(1, int(args.log_every)) == 0):
            best_idx = int(np.argmax(scores)) if scores.size > 0 else -1
            best_score = float(scores[best_idx]) if best_idx >= 0 else float("nan")
            best_label = int(labels[best_idx]) if best_idx >= 0 else -1
            person_mask = np.isin(labels, [0, 1])
            best_person_score = float(np.max(scores[person_mask])) if np.any(person_mask) else float("nan")
            print(
                f"[infer_video] frame={frame_idx} best=(label={best_label}, score={best_score:.3f}) "
                f"best_person_score={best_person_score:.3f} score_thr={float(args.score_thr):.3f} "
                f"drawing={int(len(idxs))}"
            )

        out_frame = frame_bgr.copy()
        # Optional overlay to make it obvious whether the tracker has active tracks even when nothing is drawn.
        if bool(args.overlay_stats):
            person_mask = np.isin(labels, [0, 1])
            best_person_score = float(np.max(scores[person_mask])) if np.any(person_mask) else float("nan")
            n_tracks = int(len(track_outputs)) if track_outputs is not None else 0
            cv2.putText(
                out_frame,
                f"tracks={n_tracks} best_person_score={best_person_score:.3f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if track_outputs is None:
            for j in idxs:
                if keypoints is not None:
                    kpt = keypoints[j]
                    if args.swap_lr_kpts:
                        kpt = kpt[COCO_KEYPOINT_FLIP_INDEX, :]
                    draw_pose(
                        out_frame,
                        boxes[j],
                        kpt,
                        kpt_thr=float(args.kpt_thr),
                        show_kpt_idx=bool(args.show_kpt_idx),
                    )
                else:
                    x1, y1, x2, y2 = boxes[j].astype(int).tolist()
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    out_frame,
                    f"person {scores[j]:.2f}",
                    (int(boxes[j][0]), max(0, int(boxes[j][1]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        else:
            # Draw only a small, stable subset of tracks (prevents "box explosions" when matching is uncertain).
            # Prefer tracks updated this frame, then higher scores; cap by --max-persons.
            track_outputs = sorted(track_outputs, key=lambda t: (t.time_since_update, -t.score))
            if args.lock_track:
                if locked_track_id is None and len(track_outputs) > 0:
                    locked_track_id = int(track_outputs[0].track_id)
                if locked_track_id is not None:
                    track_outputs = [t for t in track_outputs if int(t.track_id) == int(locked_track_id)]
            # Filter what we draw:
            # - always allow the locked track (if any)
            # - otherwise, draw only confident tracks updated this frame
            # - optionally draw "lost" tracks (predicted) if --draw-lost is enabled
            draw_tracks = []
            for t in track_outputs:
                is_locked = locked_track_id is not None and int(t.track_id) == int(locked_track_id)
                if is_locked:
                    draw_tracks.append(t)
                    continue
                if int(t.time_since_update) == 0:
                    if float(t.score) >= float(args.score_thr):
                        draw_tracks.append(t)
                else:
                    if bool(args.draw_lost) and (
                        bool(args.draw_lost_any_score) or float(t.score) >= float(args.draw_lost_min_score)
                    ):
                        draw_tracks.append(t)

            draw_tracks = draw_tracks[: args.max_persons]
            for t in draw_tracks:
                box_draw = t.box_smooth if t.box_smooth is not None else t.box_xyxy
                kpt_draw = t.kpt_smooth if t.kpt_smooth is not None else t.keypoints
                if kpt_draw is not None and args.swap_lr_kpts:
                    kpt_draw = kpt_draw[COCO_KEYPOINT_FLIP_INDEX, :]
                if kpt_draw is not None:
                    draw_pose(
                        out_frame,
                        box_draw,
                        kpt_draw,
                        kpt_thr=float(args.kpt_thr),
                        show_kpt_idx=bool(args.show_kpt_idx),
                    )
                else:
                    x1, y1, x2, y2 = box_draw.astype(int).tolist()
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                lost_tag = f" LOST+{int(t.time_since_update)}" if int(t.time_since_update) > 0 else ""
                label = f"id {t.track_id}{lost_tag} person {t.score:.2f}"
                if posture_est is not None:
                    pr = posture_est.estimate(t.track_id, box_draw, kpt_draw)
                    label = f"id {t.track_id}{lost_tag} {pr.state} {pr.conf:.2f} | person {t.score:.2f}"
                cv2.putText(
                    out_frame,
                    label,
                    (int(box_draw[0]), max(0, int(box_draw[1]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        _write(out_frame)

        frame_idx += 1
        if args.max_frames is not None and frame_idx >= int(args.max_frames):
            break

    cap.release()
    if writer_cv2 is not None:
        writer_cv2.release()
    if writer_ff is not None:
        writer_ff.close()

    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()


