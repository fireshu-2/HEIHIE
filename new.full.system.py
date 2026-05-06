import cv2
import csv
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from ultralytics import YOLO


# =========================
# YOLO + Simple IOU Zone Intrusion System
# 光照自适应 + 小行人优化 + 多人禁区过滤修正版
# =========================

# -------------------------
# 路径配置
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_VIDEOS_DIR = BASE_DIR / "test_videos"
OUTPUTS_DIR = BASE_DIR / "outputs"

LOG_CSV = BASE_DIR / "events_log_hybrid.csv"
MODEL_PATH = str(BASE_DIR / "yolov8n.pt")

VIDEO_SOURCE = str(TEST_VIDEOS_DIR / "epfl_passageway_light_change.mp4")
# VIDEO_SOURCE = 0

OUTPUT_VIDEO_PATH = Path("D:/project/code/outputs/result_final.mp4")
SAVE_OUTPUT_VIDEO = True


# -------------------------
# 分辨率处理
# 小视频也放大到 960 宽，提高小行人检出率
# -------------------------
PROCESS_WIDTH = 960


# -------------------------
# YOLO 检测参数
# -------------------------
CONF_THRES = 0.38
DARK_CONF_THRES = 0.30
BRIGHT_CONF_THRES = 0.32

IMG_SIZE = 640

# 正常光照下每 2 帧检测一次；
# 光照异常 / 光线突变时临时每帧检测。
NORMAL_DETECT_INTERVAL = 2
LIGHT_DETECT_INTERVAL = 1

# 光线突变后，连续多少帧使用每帧检测
LIGHT_BOOST_FRAMES = 45


# -------------------------
# 光照判断与增强参数
# -------------------------
ENABLE_LIGHT_ENHANCE_FOR_YOLO = True

DARK_BRIGHTNESS_THRES = 95
BRIGHT_BRIGHTNESS_THRES = 175

# 前后帧平均亮度变化超过该值，认为发生光线突变
LIGHT_CHANGE_DELTA_THRES = 18


# -------------------------
# 显示参数
# -------------------------
SHOW_DEBUG_INFO = False
SHOW_DETECTION_DETAIL = True
SHOW_TRAIL = False

FONT_SCALE_SMALL = 0.45
FONT_SCALE_NORMAL = 0.50
FONT_SCALE_ALARM = 0.70

TEXT_THICKNESS = 1
BOX_THICKNESS = 2
LINE_THICKNESS = 1

TRAIL_LEN = 20
DWELL_SECONDS = 3.0
ALARM_COOLDOWN = 2.0


# -------------------------
# 人形框过滤参数
# 针对 EPFL 小行人放宽，但保留极细长误检过滤
# -------------------------
MIN_BOX_AREA = 800
MIN_PERSON_WIDTH = 16
MIN_PERSON_HEIGHT = 38
MIN_PERSON_ASPECT = 1.05
MAX_PERSON_ASPECT = 5.20

# 多人靠近时，0.45 容易压掉相邻人框；这里放宽到 0.60
NMS_IOU_THRES = 0.60


# -------------------------
# 简单 ID 跟踪参数
# -------------------------
IOU_MATCH_THRES = 0.20
MAX_LOST = 3
MIN_HITS_FOR_EVENT = 2
MIN_HITS_TO_DRAW = 2


# -------------------------
# 人类运动特征过滤
# 当前先关闭，避免误杀正常行人
# -------------------------
ENABLE_HUMAN_MOTION_FILTER = False
H_MOTION_RATIO = 2.5
NON_HUMAN_MOTION_CONFIRM_FRAMES = 4


# -------------------------
# 报警确认
# -------------------------
ALARM_CONFIRM_FRAMES = 3


# -------------------------
# 默认警戒区域
# -------------------------
LINE_P1 = (220, 260)
LINE_P2 = (580, 260)

ZONE_POLYGON = np.array([
    [180, 280],
    [620, 280],
    [620, 460],
    [180, 460],
], dtype=np.int32)

ENABLE_MOUSE_REGION_SELECTION = True


# =========================
# 数据结构
# =========================
@dataclass
class EventRecord:
    timestamp: str
    track_id: int
    event_type: str
    detail: str


# =========================
# 日志
# =========================
class EventLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "track_id", "event_type", "detail"])

    def write(self, record: EventRecord):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp,
                record.track_id,
                record.event_type,
                record.detail
            ])


class AlarmManager:
    def __init__(self, cooldown: float = 2.0):
        self.cooldown = cooldown
        self.last_fire: Dict[Tuple[int, str], float] = {}

    def can_fire(self, track_id: int, event_type: str, now: float) -> bool:
        key = (track_id, event_type)
        last_t = self.last_fire.get(key, 0.0)

        if now - last_t >= self.cooldown:
            self.last_fire[key] = now
            return True

        return False


# =========================
# 入侵分析
# =========================
class IntrusionAnalyzer:
    def __init__(self, line_p1, line_p2, zone_polygon, dwell_seconds=3.0):
        self.line_p1 = np.array(line_p1, dtype=np.float32)
        self.line_p2 = np.array(line_p2, dtype=np.float32)
        self.zone_polygon = zone_polygon
        self.dwell_seconds = dwell_seconds

        self.prev_centers: Dict[int, Tuple[int, int]] = {}
        self.trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))

        self.in_zone_since: Dict[int, float] = {}
        self.zone_state: Dict[int, bool] = defaultdict(bool)
        self.zone_in_frames: Dict[int, int] = defaultdict(int)
        self.zone_out_frames: Dict[int, int] = defaultdict(int)

        self.non_human_motion_frames: Dict[int, int] = defaultdict(int)
        self.alarm_confirm_frames: Dict[Tuple[int, str], int] = defaultdict(int)
        self.pending_line_direction: Dict[int, str] = {}

        self.line_cross_count = 0
        self.zone_intrusion_count = 0
        self.dwell_alarm_count = 0

        self.line_alarm_count = 0
        self.zone_alarm_count = 0

    @staticmethod
    def bbox_anchor_xyxy(xyxy: np.ndarray) -> Tuple[int, int]:
        x1, y1, x2, y2 = xyxy.astype(int)
        return int((x1 + x2) / 2), int(y2)

    @staticmethod
    def side_of_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

    def is_human_like_motion(self, track_id: int, current_center: Tuple[int, int]) -> bool:
        if not ENABLE_HUMAN_MOTION_FILTER:
            return True

        if track_id not in self.prev_centers:
            return True

        prev_center = self.prev_centers[track_id]
        dx = abs(current_center[0] - prev_center[0])
        dy = abs(current_center[1] - prev_center[1])

        if dx + dy < 3:
            self.non_human_motion_frames[track_id] = 0
            return True

        if dx > max(1, dy) * H_MOTION_RATIO:
            self.non_human_motion_frames[track_id] += 1
        else:
            self.non_human_motion_frames[track_id] = 0

        return self.non_human_motion_frames[track_id] < NON_HUMAN_MOTION_CONFIRM_FRAMES

    def confirm_alarm_condition(self, track_id: int, event_type: str, condition: bool) -> bool:
        key = (track_id, event_type)

        if condition:
            self.alarm_confirm_frames[key] += 1
        else:
            self.alarm_confirm_frames[key] = 0

        return self.alarm_confirm_frames[key] >= ALARM_CONFIRM_FRAMES

    def confirm_line_cross(self, track_id: int, crossed: bool, direction: str, still_valid: bool) -> bool:
        key = (track_id, "line_cross")

        if crossed and direction is not None and still_valid:
            self.pending_line_direction[track_id] = direction
            self.alarm_confirm_frames[key] = 1
            return False

        if track_id in self.pending_line_direction:
            if still_valid:
                self.alarm_confirm_frames[key] += 1
            else:
                self.alarm_confirm_frames[key] = 0
                self.pending_line_direction.pop(track_id, None)
                return False

            if self.alarm_confirm_frames[key] >= ALARM_CONFIRM_FRAMES:
                self.alarm_confirm_frames[key] = 0
                return True

        return False

    def update_track(self, track_id: int, center: Tuple[int, int], now: float, box_in_zone_now: bool):
        self.trails[track_id].append(center)
        current_p = np.array(center, dtype=np.float32)

        is_human_motion = self.is_human_like_motion(track_id, center)

        crossed = False
        direction = None

        if track_id in self.prev_centers:
            prev_p = np.array(self.prev_centers[track_id], dtype=np.float32)
            s1 = self.side_of_line(prev_p, self.line_p1, self.line_p2)
            s2 = self.side_of_line(current_p, self.line_p1, self.line_p2)

            if s1 == 0:
                s1 = 1e-6
            if s2 == 0:
                s2 = 1e-6

            if s1 * s2 < 0:
                crossed = True
                direction = "enter" if s1 < 0 and s2 > 0 else "leave"
                self.line_cross_count += 1

        entered_zone = False

        if box_in_zone_now:
            self.zone_in_frames[track_id] += 1
            self.zone_out_frames[track_id] = 0
        else:
            self.zone_out_frames[track_id] += 1
            self.zone_in_frames[track_id] = 0

        if box_in_zone_now and not self.zone_state[track_id] and self.zone_in_frames[track_id] >= 1:
            self.zone_state[track_id] = True
            self.in_zone_since[track_id] = now
            entered_zone = True
            self.zone_intrusion_count += 1

        elif not box_in_zone_now and self.zone_state[track_id] and self.zone_out_frames[track_id] >= MAX_LOST:
            self.zone_state[track_id] = False
            self.in_zone_since.pop(track_id, None)

        in_zone = self.zone_state[track_id]

        dwell_alarm = False
        dwell_time = 0.0

        if in_zone:
            dwell_time = now - self.in_zone_since.get(track_id, now)
            if dwell_time >= self.dwell_seconds:
                dwell_alarm = True

        self.prev_centers[track_id] = center

        return {
            "crossed": crossed,
            "direction": direction,
            "in_zone": in_zone,
            "entered_zone": entered_zone,
            "dwell_alarm": dwell_alarm,
            "dwell_time": dwell_time,
            "is_human_motion": is_human_motion,
        }

    def clear_track(self, track_id: int):
        self.prev_centers.pop(track_id, None)
        self.trails.pop(track_id, None)
        self.in_zone_since.pop(track_id, None)
        self.zone_state.pop(track_id, None)
        self.zone_in_frames.pop(track_id, None)
        self.zone_out_frames.pop(track_id, None)
        self.non_human_motion_frames.pop(track_id, None)
        self.pending_line_direction.pop(track_id, None)

        for event_type in ["line_cross", "zone_intrusion", "dwell_alarm"]:
            self.alarm_confirm_frames.pop((track_id, event_type), None)


# =========================
# 工具函数
# =========================
def ensure_project_dirs():
    TEST_VIDEOS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)


def resize_for_process(frame):
    """
    统一处理到 PROCESS_WIDTH。
    小视频也放大，避免远处小行人太小。
    """
    if PROCESS_WIDTH is None:
        return frame

    h0, w0 = frame.shape[:2]

    if w0 == PROCESS_WIDTH:
        return frame

    scale = PROCESS_WIDTH / float(w0)
    new_h = int(h0 * scale)

    return cv2.resize(frame, (PROCESS_WIDTH, new_h))


def draw_zone(frame, polygon, color=(255, 200, 0), alpha=0.12):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, LINE_THICKNESS)


def draw_line(frame, p1, p2, color=(0, 255, 255)):
    cv2.line(frame, p1, p2, color, LINE_THICKNESS)
    cv2.putText(
        frame,
        "Line",
        (p1[0], max(16, p1[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE_SMALL,
        color,
        TEXT_THICKNESS
    )


def draw_status_panel(
    frame,
    fps,
    analyzer: IntrusionAnalyzer,
    current_alarm_text: str,
    current_mode: str,
    frame_idx: int,
    detect_count: int,
    track_count: int,
    current_interval: int,
    current_conf: float,
    brightness: float,
):
    infer_ratio = detect_count / max(frame_idx, 1)

    if not SHOW_DEBUG_INFO:
        lines = [
            f"FPS: {fps:.1f}",
            f"Mode: {current_mode}",
            f"Infer: {infer_ratio:.1%}",
            f"Alarm: {current_alarm_text if current_alarm_text else 'None'}",
        ]
    else:
        lines = [
            f"FPS: {fps:.1f}",
            f"Mode: {current_mode}",
            f"Frame: {frame_idx}",
            f"Detect: {detect_count}",
            f"Track: {track_count}",
            f"Infer: {infer_ratio:.2%}",
            f"Interval: {current_interval}",
            f"Conf: {current_conf:.2f}",
            f"Brightness: {brightness:.1f}",
            f"LineAlarm: {analyzer.line_alarm_count}",
            f"ZoneAlarm: {analyzer.zone_alarm_count}",
            f"DwellAlarm: {analyzer.dwell_alarm_count}",
            f"Alarm: {current_alarm_text if current_alarm_text else 'None'}",
        ]

    x, y = 12, 22
    line_gap = 18

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE_SMALL,
            (0, 255, 0),
            TEXT_THICKNESS
        )


def create_video_writer_for_frame(cap, output_path, frame):
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 1:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    return writer


def select_regions_by_mouse(frame):
    temp_line_points = []
    temp_zone_points = []

    canvas = frame.copy()
    window_name = "Select Regions"

    def redraw():
        nonlocal canvas
        canvas = frame.copy()

        instructions = [
            "Step 1: left click 2 points for warning line",
            "Step 2: left click 4 points for intrusion zone",
            "s = save | r = reset | q = quit"
        ]

        for i, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (16, 24 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE_NORMAL,
                (0, 255, 0),
                TEXT_THICKNESS
            )

        for i, pt in enumerate(temp_line_points):
            cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                f"L{i + 1}",
                (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE_SMALL,
                (0, 255, 255),
                TEXT_THICKNESS
            )

        if len(temp_line_points) == 2:
            cv2.line(canvas, temp_line_points[0], temp_line_points[1], (0, 255, 255), LINE_THICKNESS)

        for i, pt in enumerate(temp_zone_points):
            cv2.circle(canvas, pt, 4, (255, 200, 0), -1)
            cv2.putText(
                canvas,
                f"Z{i + 1}",
                (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE_SMALL,
                (255, 200, 0),
                TEXT_THICKNESS
            )

        if len(temp_zone_points) >= 2:
            for i in range(1, len(temp_zone_points)):
                cv2.line(canvas, temp_zone_points[i - 1], temp_zone_points[i], (255, 200, 0), LINE_THICKNESS)

        if len(temp_zone_points) == 4:
            poly = np.array(temp_zone_points, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], (255, 200, 0))
            cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
            cv2.polylines(canvas, [poly], True, (255, 200, 0), LINE_THICKNESS)

        cv2.imshow(window_name, canvas)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(temp_line_points) < 2:
                temp_line_points.append((x, y))
            elif len(temp_zone_points) < 4:
                temp_zone_points.append((x, y))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            temp_line_points.clear()
            temp_zone_points.clear()
            redraw()

        elif key == ord("s"):
            if len(temp_line_points) == 2 and len(temp_zone_points) == 4:
                cv2.destroyWindow(window_name)
                return (
                    temp_line_points[0],
                    temp_line_points[1],
                    np.array(temp_zone_points, dtype=np.int32)
                )
            else:
                print("[WARN] Please select 2 line points and 4 zone points first.")

        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            return None, None, None


def compute_iou(box_a, box_b) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def clip_box_to_frame(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    return x1, y1, x2, y2


def valid_person_box(box):
    x1, y1, x2, y2 = box

    w = max(1, int(x2 - x1))
    h = max(1, int(y2 - y1))
    area = w * h
    aspect = h / float(w)

    if area < MIN_BOX_AREA:
        return False

    if w < MIN_PERSON_WIDTH:
        return False

    if h < MIN_PERSON_HEIGHT:
        return False

    if not (MIN_PERSON_ASPECT <= aspect <= MAX_PERSON_ASPECT):
        return False

    # 只过滤特别小、特别细长的竖条，避免把小行人过滤掉
    if area < 1600 and w < 22 and aspect > 3.8:
        return False

    if w < 14 and aspect > 4.5:
        return False

    return True


def bbox_in_zone(box, polygon):
    """
    判断 person 框是否与警戒区相关。

    修正版：
    1. 不只看脚底点；
    2. 同时看脚底、小腿、下半身中心；
    3. 加入边界容错；
    4. 下半身与警戒区有明显重叠时，也认为在禁区内。
    """
    x1, y1, x2, y2 = box

    w = max(1, int(x2 - x1))
    h = max(1, int(y2 - y1))

    zone_margin = 12

    points = [
        # 脚底附近
        (int((x1 + x2) / 2), int(y2)),
        (int(x1 + 0.30 * w), int(y2)),
        (int(x1 + 0.70 * w), int(y2)),

        # 小腿 / 膝盖附近
        (int((x1 + x2) / 2), int(y1 + 0.85 * h)),
        (int(x1 + 0.35 * w), int(y1 + 0.85 * h)),
        (int(x1 + 0.65 * w), int(y1 + 0.85 * h)),

        # 下半身中心
        (int((x1 + x2) / 2), int(y1 + 0.70 * h)),
        (int(x1 + 0.35 * w), int(y1 + 0.70 * h)),
        (int(x1 + 0.65 * w), int(y1 + 0.70 * h)),
    ]

    inside_count = 0

    for p in points:
        dist = cv2.pointPolygonTest(polygon, p, True)

        if dist >= -zone_margin:
            inside_count += 1

    if inside_count >= 2:
        return True

    # 兜底：检查 bbox 下半部分和警戒区的重叠比例
    lower_y1 = int(y1 + 0.55 * h)
    lower_y2 = int(y2)

    lower_box = np.array([
        [x1, lower_y1],
        [x2, lower_y1],
        [x2, lower_y2],
        [x1, lower_y2],
    ], dtype=np.int32)

    bx1 = max(0, min(x1, x2))
    by1 = max(0, min(lower_y1, lower_y2))
    bx2 = max(x1, x2)
    by2 = max(lower_y1, lower_y2)

    if bx2 <= bx1 or by2 <= by1:
        return False

    mask_w = bx2 - bx1 + 1
    mask_h = by2 - by1 + 1

    zone_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    box_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

    shifted_polygon = polygon.copy()
    shifted_polygon[:, 0] -= bx1
    shifted_polygon[:, 1] -= by1

    shifted_lower_box = lower_box.copy()
    shifted_lower_box[:, 0] -= bx1
    shifted_lower_box[:, 1] -= by1

    cv2.fillPoly(zone_mask, [shifted_polygon], 255)
    cv2.fillPoly(box_mask, [shifted_lower_box], 255)

    inter = cv2.bitwise_and(zone_mask, box_mask)
    inter_area = float(np.count_nonzero(inter))
    lower_area = float(np.count_nonzero(box_mask))

    if lower_area <= 1:
        return False

    overlap_ratio = inter_area / lower_area

    return overlap_ratio >= 0.08


def get_frame_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def get_adaptive_conf(frame):
    brightness = get_frame_brightness(frame)

    if brightness < DARK_BRIGHTNESS_THRES:
        return DARK_CONF_THRES

    if brightness > BRIGHT_BRIGHTNESS_THRES:
        return BRIGHT_CONF_THRES

    return CONF_THRES


def is_light_abnormal(frame):
    brightness = get_frame_brightness(frame)
    return brightness < DARK_BRIGHTNESS_THRES or brightness > BRIGHT_BRIGHTNESS_THRES


def enhance_frame_for_yolo(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


# =========================
# YOLO 检测
# =========================
def detect_person_full_frame(model, frame, conf_thres, img_size, use_enhance=False):
    raw_boxes = []
    raw_scores = []

    detect_frames = [frame]

    if ENABLE_LIGHT_ENHANCE_FOR_YOLO and use_enhance:
        detect_frames.append(enhance_frame_for_yolo(frame))

    for detect_frame in detect_frames:
        results = model.predict(
            source=detect_frame,
            conf=conf_thres,
            imgsz=img_size,
            classes=[0],
            verbose=False
        )

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.xyxy is not None:
                xyxy_list = boxes.xyxy.cpu().numpy()
                conf_list = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy_list))

                for xyxy, conf in zip(xyxy_list, conf_list):
                    x1, y1, x2, y2 = xyxy[:4]

                    bbox = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )

                    bbox = clip_box_to_frame(bbox, frame.shape)

                    if not valid_person_box(bbox):
                        continue

                    raw_boxes.append(bbox)
                    raw_scores.append(float(conf))

    if not raw_boxes:
        return []

    nms_boxes_xywh = []

    for box in raw_boxes:
        x1, y1, x2, y2 = box
        nms_boxes_xywh.append([
            int(x1),
            int(y1),
            int(max(1, x2 - x1)),
            int(max(1, y2 - y1))
        ])

    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes_xywh,
        scores=raw_scores,
        score_threshold=conf_thres,
        nms_threshold=NMS_IOU_THRES
    )

    detections = []

    if len(indices) > 0:
        indices = np.array(indices).reshape(-1)

        for idx in indices:
            detections.append((raw_boxes[int(idx)], raw_scores[int(idx)]))

    return detections


# =========================
# 简单 IOU ID 管理器
# =========================
class SimpleZoneTrackManager:
    def __init__(self, max_lost=3, iou_thres=0.20):
        self.max_lost = max_lost
        self.iou_thres = iou_thres
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        removed = []

        matched_tracks = set()
        matched_dets = set()

        track_ids = list(self.tracks.keys())

        for det_idx, (det_box, det_conf) in enumerate(detections):
            best_iou = 0.0
            best_tid = None

            for tid in track_ids:
                if tid in matched_tracks:
                    continue

                if tid not in self.tracks:
                    continue

                trk = self.tracks[tid]
                iou = compute_iou(det_box, trk["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_thres:
                self.tracks[best_tid]["box"] = det_box
                self.tracks[best_tid]["conf"] = det_conf
                self.tracks[best_tid]["lost"] = 0
                self.tracks[best_tid]["hits"] += 1

                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)

        for det_idx, (det_box, det_conf) in enumerate(detections):
            if det_idx not in matched_dets:
                tid = self.next_id
                self.next_id += 1

                self.tracks[tid] = {
                    "box": det_box,
                    "conf": det_conf,
                    "lost": 0,
                    "hits": 1,
                }

        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1

                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
                    removed.append(tid)

        return removed

    def get_active(self):
        result = []

        for tid, trk in self.tracks.items():
            result.append((
                tid,
                trk["box"],
                trk["conf"],
                trk["hits"],
                trk["lost"]
            ))

        return result

    def empty(self):
        return len(self.tracks) == 0


# =========================
# 主函数
# =========================
def main():
    ensure_project_dirs()

    print(f"[INFO] Base dir: {BASE_DIR}")
    print(f"[INFO] Video source: {VIDEO_SOURCE}")
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] Output video: {OUTPUT_VIDEO_PATH}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if isinstance(VIDEO_SOURCE, str) and not Path(VIDEO_SOURCE).exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_SOURCE}")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video source: {VIDEO_SOURCE}\n"
            f"Please check: {TEST_VIDEOS_DIR}"
        )

    global LINE_P1, LINE_P2, ZONE_POLYGON

    if ENABLE_MOUSE_REGION_SELECTION:
        ok, first_frame = cap.read()

        if not ok:
            raise RuntimeError("Failed to read first frame for region selection.")

        first_frame = resize_for_process(first_frame)

        selected_line_p1, selected_line_p2, selected_zone_polygon = select_regions_by_mouse(first_frame)

        if selected_line_p1 is None:
            print("[INFO] Region selection cancelled.")
            cap.release()
            return

        LINE_P1 = selected_line_p1
        LINE_P2 = selected_line_p2
        ZONE_POLYGON = selected_zone_polygon

        cap.release()
        cap = cv2.VideoCapture(VIDEO_SOURCE)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to reopen video source: {VIDEO_SOURCE}")

    writer = None

    logger = EventLogger(LOG_CSV)
    alarm = AlarmManager(ALARM_COOLDOWN)
    analyzer = IntrusionAnalyzer(LINE_P1, LINE_P2, ZONE_POLYGON, DWELL_SECONDS)
    tracker = SimpleZoneTrackManager(
        max_lost=MAX_LOST,
        iou_thres=IOU_MATCH_THRES
    )

    prev_time = time.time()
    prev_brightness = None
    light_boost_until_frame = 0

    frame_idx = 0
    detect_count = 0
    track_count = 0
    current_mode = "IDLE"

    print("[INFO] Press q to quit.")

    while True:
        ok, raw_frame = cap.read()

        if not ok:
            print("[INFO] Video ended or frame read failed.")
            break

        raw_frame = resize_for_process(raw_frame)
        frame_idx += 1

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        alarm_text = ""

        brightness = get_frame_brightness(raw_frame)

        if prev_brightness is not None:
            brightness_delta = abs(brightness - prev_brightness)
            if brightness_delta >= LIGHT_CHANGE_DELTA_THRES:
                light_boost_until_frame = frame_idx + LIGHT_BOOST_FRAMES

        prev_brightness = brightness

        current_conf = get_adaptive_conf(raw_frame)

        light_abnormal = is_light_abnormal(raw_frame)
        light_boost_active = frame_idx <= light_boost_until_frame

        if light_abnormal or light_boost_active:
            current_detect_interval = LIGHT_DETECT_INTERVAL
            use_enhance = True
        else:
            current_detect_interval = NORMAL_DETECT_INTERVAL
            use_enhance = False

        frame = raw_frame.copy()

        draw_zone(frame, ZONE_POLYGON)
        draw_line(frame, LINE_P1, LINE_P2)

        need_detect = (frame_idx % current_detect_interval == 0)

        removed_ids = []

        if need_detect:
            current_mode = "DETECT"
            detect_count += 1

            detections_all = detect_person_full_frame(
                model,
                raw_frame,
                current_conf,
                IMG_SIZE,
                use_enhance=use_enhance
            )

            detections_zone = [
                (box, conf)
                for box, conf in detections_all
                if conf >= current_conf and bbox_in_zone(box, ZONE_POLYGON)
            ]

            removed_ids = tracker.update(detections_zone)

        else:
            current_mode = "TRACK" if not tracker.empty() else "IDLE"
            track_count += 1
            removed_ids = []

        for rid in removed_ids:
            analyzer.clear_track(rid)

        for tid, box, conf, hits, lost in tracker.get_active():
            # 稳定目标允许 lost=1 短暂显示，减少光照突变时闪烁；
            # 不长期保留，避免虚空框。
            if lost > 0:
                if not (lost <= 1 and hits >= 3):
                    continue

            if hits < MIN_HITS_TO_DRAW:
                continue

            x1, y1, x2, y2 = box
            xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
            center = analyzer.bbox_anchor_xyxy(xyxy)

            box_in_zone_now = bbox_in_zone(box, ZONE_POLYGON)

            if not box_in_zone_now:
                continue

            state = analyzer.update_track(tid, center, now, box_in_zone_now)

            human_motion_ok = state["is_human_motion"]
            stable_track = hits >= MIN_HITS_FOR_EVENT
            allow_alarm = stable_track and human_motion_ok

            color = (0, 165, 255)

            if state["dwell_alarm"]:
                color = (0, 0, 255)

            if not human_motion_ok:
                color = (128, 128, 128)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            if SHOW_DETECTION_DETAIL:
                label = f"person {tid} {conf:.2f}"
            else:
                label = f"person {tid}"

            cv2.putText(
                frame,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE_SMALL,
                color,
                TEXT_THICKNESS
            )

            cv2.circle(frame, center, 3, color, -1)

            if SHOW_TRAIL:
                pts = np.array(analyzer.trails[tid], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(180, 180, 180), thickness=1)

            line_confirmed = analyzer.confirm_line_cross(
                tid,
                state["crossed"] and allow_alarm,
                state["direction"],
                allow_alarm
            )

            zone_confirmed = analyzer.confirm_alarm_condition(
                tid,
                "zone_intrusion",
                box_in_zone_now and allow_alarm
            )

            dwell_confirmed = analyzer.confirm_alarm_condition(
                tid,
                "dwell_alarm",
                state["dwell_alarm"] and allow_alarm
            )

            if line_confirmed and alarm.can_fire(tid, "line_cross", now):
                direction = analyzer.pending_line_direction.get(tid, state["direction"])
                alarm_text = f"LineCross ID:{tid} {direction}"
                analyzer.line_alarm_count += 1

                logger.write(EventRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    track_id=tid,
                    event_type="line_cross",
                    detail=f"direction={direction}"
                ))

                analyzer.pending_line_direction.pop(tid, None)

            if zone_confirmed and alarm.can_fire(tid, "zone_intrusion", now):
                alarm_text = f"ZoneIntrusion ID:{tid}"
                analyzer.zone_alarm_count += 1

                logger.write(EventRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    track_id=tid,
                    event_type="zone_intrusion",
                    detail="entered restricted zone"
                ))

            if dwell_confirmed and alarm.can_fire(tid, "dwell_alarm", now):
                analyzer.dwell_alarm_count += 1
                alarm_text = f"DwellAlarm ID:{tid} {state['dwell_time']:.1f}s"

                logger.write(EventRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    track_id=tid,
                    event_type="dwell_alarm",
                    detail=f"dwell_time={state['dwell_time']:.2f}"
                ))

            if SHOW_DEBUG_INFO and state["in_zone"]:
                cv2.putText(
                    frame,
                    f"IN {state['dwell_time']:.1f}s",
                    (x1, min(frame.shape[0] - 8, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE_SMALL,
                    color,
                    TEXT_THICKNESS
                )

        draw_status_panel(
            frame,
            fps,
            analyzer,
            alarm_text,
            current_mode,
            frame_idx,
            detect_count,
            track_count,
            current_detect_interval,
            current_conf,
            brightness,
        )

        if alarm_text:
            cv2.putText(
                frame,
                alarm_text,
                (16, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE_ALARM,
                (0, 0, 255),
                2
            )

        if SAVE_OUTPUT_VIDEO and writer is None:
            writer = create_video_writer_for_frame(cap, OUTPUT_VIDEO_PATH, frame)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Final Intrusion System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    print(f"[INFO] Log saved to: {LOG_CSV}")

    if SAVE_OUTPUT_VIDEO:
        print(f"[INFO] Output video saved to: {OUTPUT_VIDEO_PATH}")

    print(f"[INFO] Total frames: {frame_idx}")
    print(f"[INFO] Detect frames: {detect_count}")
    print(f"[INFO] Track frames: {track_count}")
    print(f"[INFO] Inference ratio: {detect_count / max(frame_idx, 1):.2%}")


if __name__ == "__main__":
    main()