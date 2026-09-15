"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import collections
import contextlib
import ctypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
from collections import OrderedDict
from typing import List, Tuple

import cv2  # Added for video processing
import numpy as np
import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (1, 2),
]


class TRTInference(object):
    def __init__(
        self,
        engine_path,
        device="cuda:0",
        backend="torch",
        max_batch_size=1,
        verbose=False,
        plugin_libs=None,
    ):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.plugin_libs = list(plugin_libs or [])

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.trt_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        for p in self.plugin_libs:
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = list(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if len(shape) > 0 and shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            # For outputs, query concrete runtime shape after input shape is known.
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                rt_shape = list(context.get_tensor_shape(name))
                if all(int(d) > 0 for d in rt_shape):
                    shape = rt_shape
            data = torch.empty(tuple(int(d) for d in shape), dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype, device=device)
            bindings[name] = Binding(name, dtype, tuple(int(d) for d in shape), data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        batch_size = None
        for n in self.input_names:
            if blob[n].dtype != self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            in_shape = tuple(int(x) for x in blob[n].shape)
            if self.bindings[n].shape != in_shape:
                self.context.set_input_shape(n, in_shape)
                self.bindings[n] = self.bindings[n]._replace(shape=in_shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)
            if batch_size is None:
                batch_size = int(blob[n].shape[0])

        # Ensure output buffers match runtime output shapes for current input shape.
        for n in self.output_names:
            rt_shape = tuple(int(x) for x in self.context.get_tensor_shape(n))
            b = self.bindings[n]
            if rt_shape != b.shape:
                data = torch.empty(rt_shape, dtype=b.data.dtype, device=b.data.device)
                self.bindings[n] = b._replace(shape=rt_shape, data=data, ptr=data.data_ptr())

        # Bind pointers by tensor name to avoid any binding-order ambiguity.
        for n in self.input_names:
            self.context.set_tensor_address(n, int(blob[n].data_ptr()))
        for n in self.output_names:
            self.context.set_tensor_address(n, int(self.bindings[n].data.data_ptr()))

        if hasattr(self.context, "execute_async_v3") and torch.cuda.is_available():
            if self.trt_stream is not None:
                with torch.cuda.stream(self.trt_stream):
                    ok = self.context.execute_async_v3(stream_handle=self.trt_stream.cuda_stream)
                # Ensure default stream consumers see completed TRT writes.
                torch.cuda.current_stream().wait_stream(self.trt_stream)
            else:
                stream = torch.cuda.current_stream().cuda_stream
                ok = self.context.execute_async_v3(stream_handle=stream)
        else:
            # Fallback path for environments without v3 API / CUDA stream.
            bindings_addr = OrderedDict((n, int(self.bindings[n].data.data_ptr())) for n in self.bindings.keys())
            bindings_addr.update({n: int(blob[n].data_ptr()) for n in self.input_names})
            ok = self.context.execute_v2(list(bindings_addr.values()))
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned False")
        if batch_size is None:
            batch_size = 1
        outputs = {}
        for n in self.output_names:
            t = self.bindings[n].data
            # Only slice on dim-0 when output is actually batch-major.
            if t.ndim > 0 and int(t.shape[0]) == int(batch_size):
                outputs[n] = t[:batch_size]
            else:
                outputs[n] = t

        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

    return images


def overlay_seg(frame_bgr: np.ndarray, seg_map: np.ndarray, alpha: float) -> np.ndarray:
    palette = np.array(
        [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
        dtype=np.uint8,
    )
    seg_rgb = palette[np.clip(seg_map, 0, len(palette) - 1)]
    seg_bgr = seg_rgb[..., ::-1]
    h, w = frame_bgr.shape[:2]
    if seg_bgr.shape[:2] != (h, w):
        seg_bgr = cv2.resize(seg_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(frame_bgr, 1.0 - float(alpha), seg_bgr, float(alpha), 0.0)


def draw_pose(img: np.ndarray, box: np.ndarray, kpts: np.ndarray, kpt_thr: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    cv2.rectangle(img, (int(max(0, x1)), int(max(0, y1))), (int(min(w - 1, x2)), int(min(h - 1, y2))), (0, 255, 0), 2)

    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    diag = float(np.hypot(bw, bh))
    vx1, vy1 = x1 - 0.15 * bw, y1 - 0.20 * bh
    vx2, vy2 = x2 + 0.15 * bw, y2 + 0.20 * bh

    valid = np.zeros((kpts.shape[0],), dtype=bool)
    for i in range(kpts.shape[0]):
        x, y, s = kpts[i]
        if s < kpt_thr:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        if not (vx1 <= x <= vx2 and vy1 <= y <= vy2):
            continue
        valid[i] = True
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    max_limb = 0.65 * diag
    for a, b in COCO_SKELETON:
        if not (valid[a] and valid[b]):
            continue
        xa, ya, _ = kpts[a]
        xb, yb, _ = kpts[b]
        if float(np.hypot(xa - xb, ya - yb)) > max_limb:
            continue
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1[None, :])
    iy1 = np.maximum(ay1, by1[None, :])
    ix2 = np.minimum(ax2, bx2[None, :])
    iy2 = np.minimum(ay2, by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / union, 0.0).astype(np.float32)


def pose_boxes_from_keypoints(kpts: np.ndarray) -> np.ndarray:
    if kpts.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x = kpts[..., 0]
    y = kpts[..., 1]
    return np.stack([x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=-1).astype(np.float32)


def pose_boxes_from_keypoints_conf(
    kpts: np.ndarray, conf_thr: float, min_kpts: int
) -> tuple[np.ndarray, np.ndarray]:
    if kpts.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool)

    valid = kpts[..., 2] >= float(conf_thr)
    keep = valid.sum(axis=1) >= int(min_kpts)
    if not np.any(keep):
        return np.zeros((0, 4), dtype=np.float32), keep

    boxes = np.zeros((kpts.shape[0], 4), dtype=np.float32)
    for i in range(kpts.shape[0]):
        if not keep[i]:
            continue
        pts = kpts[i][valid[i]]
        boxes[i] = np.array(
            [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()],
            dtype=np.float32,
        )
    return boxes[keep], keep


def center_distance_ratio(det_box: np.ndarray, pose_box: np.ndarray) -> float:
    dcx = 0.5 * (det_box[0] + det_box[2])
    dcy = 0.5 * (det_box[1] + det_box[3])
    pcx = 0.5 * (pose_box[0] + pose_box[2])
    pcy = 0.5 * (pose_box[1] + pose_box[3])
    dist = float(np.hypot(dcx - pcx, dcy - pcy))
    ddiag = float(np.hypot(det_box[2] - det_box[0], det_box[3] - det_box[1]))
    if ddiag <= 1e-6:
        return 1e9
    return dist / ddiag


def _box_centers_xy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=-1).astype(np.float32)


def greedy_match(
    det_boxes: np.ndarray,
    pose_boxes: np.ndarray,
    pose_scores: np.ndarray,
    min_iou: float,
    max_center_dist_ratio: float,
) -> List[Tuple[int, int]]:
    if det_boxes.size == 0 or pose_boxes.size == 0:
        return []
    iou = iou_matrix(det_boxes, pose_boxes)
    det_centers = _box_centers_xy(det_boxes)
    pose_centers = _box_centers_xy(pose_boxes)
    det_diag = np.maximum(
        1.0,
        np.hypot(np.maximum(1.0, det_boxes[:, 2] - det_boxes[:, 0]), np.maximum(1.0, det_boxes[:, 3] - det_boxes[:, 1])),
    )
    center_dist = np.linalg.norm(det_centers[:, None, :] - pose_centers[None, :, :], axis=-1)
    center_dist_ratio = center_dist / det_diag[:, None]
    center_bonus = np.clip(1.0 - center_dist_ratio / max(1e-6, float(max_center_dist_ratio)), -1.0, 1.0)
    score = iou + 0.20 * pose_scores[None, :] + 0.20 * center_bonus
    pairs = []
    used_d, used_p = set(), set()
    for di, pi in np.dstack(np.unravel_index(np.argsort(score.ravel())[::-1], score.shape))[0]:
        di = int(di)
        pi = int(pi)
        if di in used_d or pi in used_p:
            continue
        if float(iou[di, pi]) < float(min_iou):
            continue
        if float(center_dist_ratio[di, pi]) > float(max_center_dist_ratio):
            continue
        pairs.append((di, pi))
        used_d.add(di)
        used_p.add(pi)
    return pairs


def fallback_assign_by_center(
    det_boxes: np.ndarray,
    pose_boxes: np.ndarray,
    pose_scores: np.ndarray,
    max_center_dist_ratio: float,
) -> List[Tuple[int, int]]:
    if det_boxes.size == 0 or pose_boxes.size == 0:
        return []
    det_centers = _box_centers_xy(det_boxes)
    pose_centers = _box_centers_xy(pose_boxes)
    det_diag = np.maximum(
        1.0,
        np.hypot(np.maximum(1.0, det_boxes[:, 2] - det_boxes[:, 0]), np.maximum(1.0, det_boxes[:, 3] - det_boxes[:, 1])),
    )
    center_dist = np.linalg.norm(det_centers[:, None, :] - pose_centers[None, :, :], axis=-1)
    center_dist_ratio = center_dist / det_diag[:, None]
    score = 1.0 - center_dist_ratio + 0.1 * pose_scores[None, :]
    pairs = []
    used_d, used_p = set(), set()
    for di, pi in np.dstack(np.unravel_index(np.argsort(score.ravel())[::-1], score.shape))[0]:
        di = int(di)
        pi = int(pi)
        if di in used_d or pi in used_p:
            continue
        if float(center_dist_ratio[di, pi]) > float(max_center_dist_ratio):
            continue
        pairs.append((di, pi))
        used_d.add(di)
        used_p.add(pi)
    return pairs


def _candidate_pose_quality(
    det_boxes: np.ndarray,
    pose_kpts: np.ndarray,
    kpt_thr: float,
    min_visible_kpts: int,
) -> float:
    if det_boxes.size == 0 or pose_kpts.size == 0:
        return 0.0
    pose_boxes, valid = pose_boxes_from_keypoints_conf(
        pose_kpts,
        conf_thr=float(kpt_thr),
        min_kpts=int(min_visible_kpts),
    )
    if pose_boxes.size == 0:
        return 0.0
    iou = iou_matrix(det_boxes, pose_boxes)
    if iou.size == 0:
        return 0.0
    return float(iou.max(axis=1).mean())


def _find_output_key(outputs, candidates):
    for k in candidates:
        if k in outputs:
            return k
    for k in outputs.keys():
        if any(k.endswith(c) for c in candidates):
            return k
    return None


def _get_detection_outputs(outputs, orig_size, det_post=None):
    # Case A: engine already returns postprocessed detection tensors
    if all(k in outputs for k in ["labels", "boxes", "scores"]):
        return outputs["labels"], outputs["boxes"], outputs["scores"]

    # Case B: singlepass raw detection head outputs -> run DFINE postprocessor
    logit_k = _find_output_key(outputs, ["det_pred_logits"])
    box_k = _find_output_key(outputs, ["det_pred_boxes"])
    if logit_k is not None and box_k is not None:
        if det_post is None:
            raise KeyError(
                f"Engine outputs raw detection tensors ({logit_k}, {box_k}) but no --det-config was provided."
            )
        det_dict = {"pred_logits": outputs[logit_k], "pred_boxes": outputs[box_k]}
        det_res = det_post(det_dict, orig_size)[0]
        return det_res["labels"], det_res["boxes"], det_res["scores"]

    raise KeyError(f"Could not find detection outputs. Available engine outputs: {list(outputs.keys())}")


def _as_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _box_center_dist_ratio(a: np.ndarray, b: np.ndarray) -> float:
    acx = 0.5 * (float(a[0]) + float(a[2]))
    acy = 0.5 * (float(a[1]) + float(a[3]))
    bcx = 0.5 * (float(b[0]) + float(b[2]))
    bcy = 0.5 * (float(b[1]) + float(b[3]))
    d = float(np.hypot(acx - bcx, acy - bcy))
    adiag = float(np.hypot(max(1.0, float(a[2] - a[0])), max(1.0, float(a[3] - a[1]))))
    if adiag <= 1e-6:
        return 1e9
    return d / adiag


def _ema_box(prev_box: np.ndarray, cur_box: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 0.99))
    return (a * prev_box + (1.0 - a) * cur_box).astype(np.float32)


def _ema_kpts(prev_kpts: np.ndarray, cur_kpts: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 0.99))
    out = cur_kpts.copy().astype(np.float32)
    # Smooth x/y only, keep current confidence.
    out[..., :2] = a * prev_kpts[..., :2] + (1.0 - a) * cur_kpts[..., :2]
    out[..., 2] = cur_kpts[..., 2]
    return out


def _bbox_from_kpts_xy(kpts: np.ndarray) -> np.ndarray:
    x = kpts[:, 0]
    y = kpts[:, 1]
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)


def _pose_update_is_reasonable(prev_kpts: np.ndarray, cur_kpts: np.ndarray, max_jump_ratio: float, max_scale_change: float) -> bool:
    prev_b = _bbox_from_kpts_xy(prev_kpts)
    cur_b = _bbox_from_kpts_xy(cur_kpts)
    jump = _box_center_dist_ratio(prev_b, cur_b)
    if jump > float(max_jump_ratio):
        return False
    prev_area = max(1.0, float((prev_b[2] - prev_b[0]) * (prev_b[3] - prev_b[1])))
    cur_area = max(1.0, float((cur_b[2] - cur_b[0]) * (cur_b[3] - cur_b[1])))
    ratio = cur_area / prev_area
    s = float(max_scale_change)
    if ratio > s or ratio < (1.0 / max(1e-6, s)):
        return False
    return True


def _open_video_sink(
    out_path: str,
    fps: float,
    width: int,
    height: int,
    writer_mode: str = "auto",
    ffmpeg_preset: str = "p4",
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    want_ffmpeg = writer_mode in ("auto", "ffmpeg")
    if want_ffmpeg and shutil.which("ffmpeg"):
        is_rtsp_out = str(out_path).lower().startswith("rtsp://")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            str(ffmpeg_preset),
            "-pix_fmt",
            "yuv420p",
        ]
        if is_rtsp_out:
            cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", out_path])
        else:
            cmd.append(out_path)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.stdin is not None:
                return {"mode": "ffmpeg", "proc": proc}
        except Exception:
            if writer_mode == "ffmpeg":
                raise

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {out_path}")
    return {"mode": "cv2", "writer": writer}


def _close_video_sink(sink):
    if sink["mode"] == "ffmpeg":
        proc = sink["proc"]
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        finally:
            proc.wait()
    else:
        sink["writer"].release()


def _sink_write_frame(sink, frame_bgr: np.ndarray):
    if sink["mode"] == "ffmpeg":
        proc = sink["proc"]
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
    else:
        sink["writer"].write(frame_bgr)


def _preprocess_frame_bgr_to_tensor(frame_bgr: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = frame_bgr.shape[:2]
    # OpenCV uses BGR; convert once to RGB and resize to model input.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_640 = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb_640).to(device=device, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().div_(255.0)
    orig_size_wh = torch.tensor([w, h], device=device, dtype=torch.float32).unsqueeze(0)
    orig_size_hw = torch.tensor([h, w], device=device, dtype=torch.float32).unsqueeze(0)
    return x, orig_size_wh, orig_size_hw


def process_image(m, file_path, device, det_post=None):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil)[None]

    blob = {
        "images": im_data.to(device),
        "orig_target_sizes": orig_size.to(device),
    }

    output = m(blob)
    labels, boxes, scores = _get_detection_outputs(output, orig_size, det_post=det_post)
    result_images = draw([im_pil], labels, boxes, scores)
    out_path = "trt_result.jpg"
    result_images[0].save(out_path)
    print(f"Image processing complete. Result saved as '{out_path}'.")


def process_video(
    m,
    file_path,
    device,
    det_post=None,
    pose_post=None,
    score_thr=0.4,
    pose_score_thr=0.4,
    kpt_thr=0.4,
    pose_box_kpt_thr=0.45,
    pose_box_min_kpts=6,
    match_min_iou=0.25,
    max_center_dist_ratio=0.30,
    fallback_min_match_rate=1.0,
    pose_orig_size_order="auto",
    max_persons=10,
    seg_alpha=0.3,
    track_max_missed=8,
    track_min_iou=0.20,
    track_max_center_dist_ratio=0.60,
    track_box_ema=0.60,
    track_kpt_ema=0.70,
    pose_max_jump_ratio=0.35,
    pose_max_scale_change=1.8,
    pose_topk=-1,
    writer_mode="auto",
    ffmpeg_preset="p4",
    io_queue_size=32,
    show=False,
    no_write=False,
    window_name="TRT Live",
    out_path="trt_result.mp4",
):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        if str(file_path).lower().startswith("rtsp://"):
            parsed = urlparse(str(file_path))
            stream_path = parsed.path.lstrip("/") or "<empty>"
            raise RuntimeError(
                "Could not open RTSP input: "
                f"{file_path}\n"
                f"- Requested stream path: '{stream_path}'\n"
                "- If your RTSP server logs show 'DESCRIBE failed: 404 Not Found', the path is not currently being published.\n"
                "- Make sure publisher and reader use the exact same path.\n"
                "  Example publish/read pair: rtsp://127.0.0.1:8554/pose_out"
            )
        raise RuntimeError(f"Could not open input video: {file_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        raise RuntimeError(f"Invalid input video size: {orig_w}x{orig_h} from {file_path}")

    sink = None
    out_is_url = "://" in str(out_path)
    if not bool(no_write):
        if not out_is_url:
            out_path = os.path.abspath(out_path)
        sink = _open_video_sink(
            out_path=out_path,
            fps=float(fps),
            width=int(orig_w),
            height=int(orig_h),
            writer_mode=str(writer_mode),
            ffmpeg_preset=str(ffmpeg_preset),
        )

    read_q: queue.Queue = queue.Queue(maxsize=max(2, int(io_queue_size)))
    write_q: queue.Queue = queue.Queue(maxsize=max(2, int(io_queue_size)))

    stop_evt = threading.Event()

    def _reader():
        try:
            while not stop_evt.is_set():
                ok, frm = cap.read()
                if not ok:
                    break
                read_q.put(frm)
        finally:
            read_q.put(None)
            cap.release()

    writer_error = {"exc": None}

    def _writer():
        while True:
            item = write_q.get()
            if item is None:
                break
            try:
                if sink is not None:
                    _sink_write_frame(sink, item)
            except Exception as e:
                writer_error["exc"] = e
                break

    rd_t = threading.Thread(target=_reader, daemon=True)
    wr_t = threading.Thread(target=_writer, daemon=True)
    rd_t.start()
    if sink is not None:
        wr_t.start()

    frame_count = 0
    tracks = {}
    next_track_id = 0
    t_start = time.perf_counter()
    print("Processing video frames...")
    writer_failed = None
    while True:
        if writer_error["exc"] is not None:
            writer_failed = writer_error["exc"]
            break
        frame_t0 = time.perf_counter()
        frame = read_q.get()
        if frame is None:
            break

        im_data, orig_size_wh, orig_size_hw = _preprocess_frame_bgr_to_tensor(frame, device=device)

        blob = {
            "images": im_data,
            "orig_target_sizes": orig_size_wh,
        }

        output = m(blob)

        labels, boxes, scores = _get_detection_outputs(output, orig_size_wh, det_post=det_post)
        labels = _as_numpy(labels)
        boxes = _as_numpy(boxes).astype(np.float32)
        scores = _as_numpy(scores).astype(np.float32)

        person_keep = np.where((scores >= float(score_thr)) & (labels == 1))[0]
        if person_keep.size > 0:
            keep = person_keep
        else:
            keep = np.where((scores >= float(score_thr)) & ((labels == 1) | (labels == 0)))[0]
        if keep.size > 0:
            keep = keep[np.argsort(scores[keep])[::-1][: int(max_persons)]]
        det_boxes_f = boxes[keep] if keep.size > 0 else np.zeros((0, 4), dtype=np.float32)
        det_scores_f = scores[keep] if keep.size > 0 else np.zeros((0,), dtype=np.float32)

        # Keep original BGR frame for drawing (avoid RGB<->PIL<->BGR roundtrip).
        frame = frame.copy()

        # Optional segmentation overlay
        seg_k = _find_output_key(output, ["seg_logits"])
        if seg_k is not None:
            seg_logits = output[seg_k]
            if isinstance(seg_logits, torch.Tensor):
                seg_map = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
                frame = overlay_seg(frame, seg_map, alpha=float(seg_alpha))

        # Optional pose drawing from raw pose outputs
        pose_logit_k = _find_output_key(output, ["pose_pred_logits"])
        pose_kpt_k = _find_output_key(output, ["pose_pred_keypoints"])
        if pose_logit_k is not None and pose_kpt_k is not None and pose_post is not None:
            if hasattr(pose_post, "num_top_queries"):
                try:
                    q_dim = int(_as_numpy(output[pose_logit_k]).shape[1])
                    if int(pose_topk) > 0:
                        pose_post.num_top_queries = int(min(int(pose_topk), q_dim))
                    else:
                        pose_post.num_top_queries = int(q_dim)
                except Exception:
                    pass
            pose_dict = {
                "pred_logits": output[pose_logit_k],
                "pred_keypoints": output[pose_kpt_k],
            }
            pose_res_wh = pose_post(pose_dict, orig_size_wh)[0]
            pose_scores_wh = _as_numpy(pose_res_wh["scores"]).astype(np.float32)
            pose_kpts_wh = _as_numpy(pose_res_wh["keypoints"]).astype(np.float32)
            pkeep_wh = np.where(pose_scores_wh >= float(pose_score_thr))[0]
            pose_scores_f = pose_scores_wh[pkeep_wh] if pkeep_wh.size > 0 else np.zeros((0,), dtype=np.float32)
            pose_kpts_f = pose_kpts_wh[pkeep_wh] if pkeep_wh.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)

            if pose_orig_size_order in ["hw", "auto"]:
                pose_res_hw = pose_post(pose_dict, orig_size_hw)[0]
                pose_scores_hw = _as_numpy(pose_res_hw["scores"]).astype(np.float32)
                pose_kpts_hw = _as_numpy(pose_res_hw["keypoints"]).astype(np.float32)
                pkeep_hw = np.where(pose_scores_hw >= float(pose_score_thr))[0]
                pose_scores_hw_f = pose_scores_hw[pkeep_hw] if pkeep_hw.size > 0 else np.zeros((0,), dtype=np.float32)
                pose_kpts_hw_f = pose_kpts_hw[pkeep_hw] if pkeep_hw.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)
                if pose_orig_size_order == "hw":
                    pose_scores_f = pose_scores_hw_f
                    pose_kpts_f = pose_kpts_hw_f
                else:
                    q_wh = _candidate_pose_quality(
                        det_boxes_f,
                        pose_kpts_f,
                        kpt_thr=float(pose_box_kpt_thr),
                        min_visible_kpts=int(pose_box_min_kpts),
                    )
                    q_hw = _candidate_pose_quality(
                        det_boxes_f,
                        pose_kpts_hw_f,
                        kpt_thr=float(pose_box_kpt_thr),
                        min_visible_kpts=int(pose_box_min_kpts),
                    )
                    if q_hw > q_wh:
                        pose_scores_f = pose_scores_hw_f
                        pose_kpts_f = pose_kpts_hw_f
            pose_boxes_all, pose_valid_mask = pose_boxes_from_keypoints_conf(
                pose_kpts_f, conf_thr=float(pose_box_kpt_thr), min_kpts=int(pose_box_min_kpts)
            )
            pose_kpts_f = pose_kpts_f[pose_valid_mask] if pose_valid_mask.size > 0 else np.zeros((0, 17, 3), dtype=np.float32)
            pose_scores_f = pose_scores_f[pose_valid_mask] if pose_valid_mask.size > 0 else np.zeros((0,), dtype=np.float32)
            pose_boxes_f = pose_boxes_all

            if det_boxes_f.shape[0] > 0 and pose_boxes_f.shape[0] > 0:
                pairs = greedy_match(
                    det_boxes_f,
                    pose_boxes_f,
                    pose_scores_f,
                    min_iou=float(match_min_iou),
                    max_center_dist_ratio=float(max_center_dist_ratio),
                )
                denom = max(1, min(det_boxes_f.shape[0], pose_boxes_f.shape[0]))
                match_rate = float(len(pairs)) / float(denom)
                if match_rate < float(fallback_min_match_rate):
                    fallback_pairs = fallback_assign_by_center(
                        det_boxes_f,
                        pose_boxes_f,
                        pose_scores_f,
                        max_center_dist_ratio=float(max_center_dist_ratio),
                    )
                    fallback_rate = float(len(fallback_pairs)) / float(denom)
                    if fallback_rate > match_rate:
                        pairs = fallback_pairs
                # Build current observations from matched det<->pose pairs.
                obs = []
                for di, pi in pairs:
                    obs.append(
                        {
                            "box": det_boxes_f[di].astype(np.float32),
                            "score": float(det_scores_f[di]),
                            "kpts": pose_kpts_f[pi].astype(np.float32),
                        }
                    )

                # Match observations to existing tracks (stable IDs).
                used_obs = set()
                used_tracks = set()
                track_items = list(tracks.items())
                if track_items and obs:
                    # Greedy by IoU / center-distance.
                    scores = []
                    for ti, (tid, t) in enumerate(track_items):
                        tbox = t["box"]
                        for oi, o in enumerate(obs):
                            iou = float(iou_matrix(tbox[None, :], o["box"][None, :])[0, 0])
                            cdr = _box_center_dist_ratio(tbox, o["box"])
                            s = iou - 0.15 * cdr
                            scores.append((s, ti, oi, iou, cdr))
                    scores.sort(key=lambda x: x[0], reverse=True)
                    for s, ti, oi, iou, cdr in scores:
                        tid, t = track_items[ti]
                        if tid in used_tracks or oi in used_obs:
                            continue
                        if iou < float(track_min_iou):
                            continue
                        if cdr > float(track_max_center_dist_ratio):
                            continue
                        o = obs[oi]
                        if not _pose_update_is_reasonable(
                            t["kpts"],
                            o["kpts"],
                            max_jump_ratio=float(pose_max_jump_ratio),
                            max_scale_change=float(pose_max_scale_change),
                        ):
                            continue
                        t["box"] = _ema_box(t["box"], o["box"], alpha=float(track_box_ema))
                        t["kpts"] = _ema_kpts(t["kpts"], o["kpts"], alpha=float(track_kpt_ema))
                        t["score"] = o["score"]
                        t["last_seen"] = frame_count
                        used_tracks.add(tid)
                        used_obs.add(oi)

                # Spawn tracks for unmatched observations.
                for oi, o in enumerate(obs):
                    if oi in used_obs:
                        continue
                    tid = int(next_track_id)
                    next_track_id += 1
                    tracks[tid] = {
                        "box": o["box"],
                        "kpts": o["kpts"],
                        "score": o["score"],
                        "last_seen": frame_count,
                    }

                # Draw active tracks.
                to_delete = []
                for tid, t in tracks.items():
                    if frame_count - int(t["last_seen"]) > int(track_max_missed):
                        to_delete.append(tid)
                        continue
                    box = t["box"]
                    cv2.putText(
                        frame,
                        f"id {tid} pose {float(t['score']):.2f}",
                        (int(box[0]), max(0, int(box[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    # Always render pose; when current update is unreliable we keep
                    # previous keypoints, so skeleton stays visible without jumping.
                    draw_pose(frame, box, t["kpts"], kpt_thr=float(kpt_thr))
                for tid in to_delete:
                    tracks.pop(tid, None)
            else:
                # Fallback: draw detections only
                for i, b in enumerate(det_boxes_f):
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"id {i} person {float(det_scores_f[i]):.2f}",
                        (int(b[0]), max(0, int(b[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
        else:
            # Detection-only fallback
            for i, b in enumerate(det_boxes_f):
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"id {i} person {float(det_scores_f[i]):.2f}",
                    (int(b[0]), max(0, int(b[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # Write the frame
        if show:
            cv2.imshow(str(window_name), frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        if sink is not None:
            write_q.put(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = max(1e-9, time.perf_counter() - t_start)
            fps_avg = frame_count / elapsed
            ms_frame = (elapsed * 1000.0) / frame_count
            frame_ms = (time.perf_counter() - frame_t0) * 1000.0
            print(
                f"Processed {frame_count} frames... "
                f"avg_fps={fps_avg:.2f} avg_ms={ms_frame:.2f} last_frame_ms={frame_ms:.2f}"
            )

    stop_evt.set()
    if sink is not None:
        write_q.put(None)
    rd_t.join()
    if sink is not None:
        wr_t.join()
        _close_video_sink(sink)
    if writer_failed is not None:
        raise RuntimeError(f"Video writer failed: {writer_failed}")
    if show:
        cv2.destroyAllWindows()
    elapsed = max(1e-9, time.perf_counter() - t_start)
    fps_avg = (frame_count / elapsed) if frame_count > 0 else 0.0
    ms_frame = (elapsed * 1000.0 / frame_count) if frame_count > 0 else 0.0
    out_size = (
        os.path.getsize(out_path)
        if (sink is not None and (not out_is_url) and os.path.isfile(out_path))
        else -1
    )
    print(
        f"Video processing complete. frames_written={frame_count} "
        f"file='{out_path}' size_bytes={out_size} "
        f"elapsed_s={elapsed:.3f} fps={fps_avg:.2f} ms_per_frame={ms_frame:.2f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-trt", "--trt", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--out", type=str, default="trt_result.mp4")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument(
        "--plugin-lib",
        action="append",
        default=[],
        help="Path to custom TensorRT plugin .so/.dll. May be repeated.",
    )
    parser.add_argument(
        "--det-config",
        type=str,
        default=None,
        help="Det config YAML for postprocessing raw det_pred_logits/det_pred_boxes outputs.",
    )
    parser.add_argument("--score-thr", type=float, default=0.4)
    parser.add_argument("--pose-score-thr", type=float, default=0.4)
    parser.add_argument("--kpt-thr", type=float, default=0.4)
    parser.add_argument("--pose-box-kpt-thr", type=float, default=0.45)
    parser.add_argument("--pose-box-min-kpts", type=int, default=6)
    parser.add_argument("--match-min-iou", type=float, default=0.25)
    parser.add_argument("--max-center-dist-ratio", type=float, default=0.30)
    parser.add_argument("--fallback-min-match-rate", type=float, default=1.0)
    parser.add_argument("--pose-orig-size-order", choices=["wh", "hw", "auto"], default="auto")
    parser.add_argument("--max-persons", type=int, default=10)
    parser.add_argument("--seg-alpha", type=float, default=0.3)
    parser.add_argument("--track-max-missed", type=int, default=8)
    parser.add_argument("--track-min-iou", type=float, default=0.20)
    parser.add_argument("--track-max-center-dist-ratio", type=float, default=0.60)
    parser.add_argument("--track-box-ema", type=float, default=0.60)
    parser.add_argument("--track-kpt-ema", type=float, default=0.70)
    parser.add_argument("--pose-max-jump-ratio", type=float, default=0.35)
    parser.add_argument("--pose-max-scale-change", type=float, default=1.8)
    parser.add_argument(
        "--pose-topk",
        type=int,
        default=-1,
        help="Top-k pose queries in postprocessor. -1 means auto (use model Q dim).",
    )
    parser.add_argument(
        "--writer-mode",
        choices=["auto", "ffmpeg", "cv2"],
        default="auto",
        help="Video writer backend. auto tries ffmpeg NVENC first, then cv2.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="p4",
        help="NVENC preset when --writer-mode uses ffmpeg (e.g. p1..p7).",
    )
    parser.add_argument(
        "--io-queue-size",
        type=int,
        default=32,
        help="Queue size for async decode/encode pipeline.",
    )
    parser.add_argument("--show", action="store_true", help="Show live window (requires X/GUI).")
    parser.add_argument("--no-write", action="store_true", help="Disable video file/RTSP output writing.")
    parser.add_argument("--window-name", type=str, default="TRT Live")

    args = parser.parse_args()

    m = TRTInference(args.trt, device=args.device, plugin_libs=args.plugin_lib)
    print(f"Engine outputs: {m.output_names}")

    det_post = None
    if args.det_config:
        from src.core import YAMLConfig
        from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor

        det_cfg = YAMLConfig(args.det_config)
        det_post = det_cfg.postprocessor.to(args.device).eval()
        if hasattr(det_post, "remap_mscoco_category"):
            det_post.remap_mscoco_category = True
        pose_post = DETRPosePostProcessor(
            num_classes=2,
            num_keypoints=17,
            num_top_queries=300,
            remap_mscoco_category=True,
        ).to(args.device).eval()
    else:
        pose_post = None

    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(m, file_path, args.device, det_post=det_post)
    else:
        # Process as video
        process_video(
            m,
            file_path,
            args.device,
            det_post=det_post,
            pose_post=pose_post,
            score_thr=float(args.score_thr),
            pose_score_thr=float(args.pose_score_thr),
            kpt_thr=float(args.kpt_thr),
            pose_box_kpt_thr=float(args.pose_box_kpt_thr),
            pose_box_min_kpts=int(args.pose_box_min_kpts),
            match_min_iou=float(args.match_min_iou),
            max_center_dist_ratio=float(args.max_center_dist_ratio),
            fallback_min_match_rate=float(args.fallback_min_match_rate),
            pose_orig_size_order=str(args.pose_orig_size_order),
            max_persons=int(args.max_persons),
            seg_alpha=float(args.seg_alpha),
            track_max_missed=int(args.track_max_missed),
            track_min_iou=float(args.track_min_iou),
            track_max_center_dist_ratio=float(args.track_max_center_dist_ratio),
            track_box_ema=float(args.track_box_ema),
            track_kpt_ema=float(args.track_kpt_ema),
            pose_max_jump_ratio=float(args.pose_max_jump_ratio),
            pose_max_scale_change=float(args.pose_max_scale_change),
            pose_topk=int(args.pose_topk),
            writer_mode=str(args.writer_mode),
            ffmpeg_preset=str(args.ffmpeg_preset),
            io_queue_size=int(args.io_queue_size),
            show=bool(args.show),
            no_write=bool(args.no_write),
            window_name=str(args.window_name),
            out_path=args.out,
        )
