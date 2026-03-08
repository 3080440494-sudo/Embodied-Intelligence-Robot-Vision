import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# ==================================================
# 🔧 配置 (保持和你觉得好用的版本一致)
# ==================================================
MODEL_PATH = "box_grasp_project/yolov8s_train_run/weights/best.pt"
CONF_THRESHOLD = 0.25  # 保持低阈值，确保人拿着也能识别
Z_MIN_MM = 200
Z_MAX_MM = 1200


# ==================================================
# 🧠 核心工具函数
# ==================================================
def get_latest_packet(queue):
    """只取最新一帧，防止积压卡顿"""
    packets = queue.tryGetAll()
    if len(packets) > 0:
        return packets[-1]
    return None


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]);
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if inter_area > 0: return inter_area / min(box1_area, box2_area)
    return 0


def smart_group_boxes(raw_boxes):
    """合并碎框"""
    if len(raw_boxes) == 0: return []
    clean_boxes = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        clean_boxes.append([x1, y1, x2, y2, conf])
    groups = []
    while clean_boxes:
        current = clean_boxes.pop(0)
        found_group = False
        for group in groups:
            if any(compute_iou(current, g_box) > 0.1 for g_box in group):
                group.append(current);
                found_group = True;
                break
        if not found_group: groups.append([current])
    final_targets = []
    for group in groups:
        min_x1 = min(b[0] for b in group);
        min_y1 = min(b[1] for b in group)
        max_x2 = max(b[2] for b in group);
        max_y2 = max(b[3] for b in group)
        max_conf = max(b[4] for b in group)
        final_targets.append((min_x1, min_y1, max_x2, max_y2, max_conf))
    return final_targets


# ==================================================
# 📷 管道设置 (完全复制极速版的 RGB 配置)
# ==================================================
pipeline = dai.Pipeline()

# --- 1. RGB (配置完全同极速版) ---
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setIspScale(2, 3)
# 不设置 FPS，让它全速跑

# --- 2. 深度 (作为辅助挂载) ---
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# 深度对齐 RGB
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setSubpixel(True)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)  # 开启扩展视差以适应近距离

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# --- 3. 输出 ---
x_out_rgb = pipeline.createXLinkOut()
x_out_rgb.setStreamName("rgb")
cam_rgb.preview.link(x_out_rgb.input)

x_out_depth = pipeline.createXLinkOut()
x_out_depth.setStreamName("depth")
stereo.depth.link(x_out_depth.input)

# ==================================================
print(f"🚀 加载模型... (阈值: {CONF_THRESHOLD})")
model = YOLO(MODEL_PATH)

# 平滑队列
history_z = deque(maxlen=8)

with dai.Device(pipeline) as device:
    try:
        device.setIrLaserDotProjectorIntensity(1.0)
        print("🔦 红外散斑: ON")
    except:
        pass

    # 关键：非阻塞队列
    q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)

    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 640)
    fx = intrinsics[0][0];
    fy = intrinsics[1][1]
    cx = intrinsics[0][2];
    cy = intrinsics[1][2]

    print("✅ 终极融合版启动 (保留极速识别率 + 异步测距)")

    start_time = time.time()
    frame_count = 0
    last_depth_frame = None  # 深度图缓存

    while True:
        # 1. 获取数据 (极速清空模式)
        in_rgb = get_latest_packet(q_rgb)
        in_depth = get_latest_packet(q_depth)

        # 优先保证 RGB 存在
        if in_rgb is None:
            time.sleep(0.001)
            continue

        frame = in_rgb.getCvFrame()

        # 如果有新的深度图就更新，没有就用上一帧的 (异步策略)
        if in_depth is not None:
            last_depth_frame = in_depth.getFrame()

        # 2. YOLO 推理 (和极速版完全一致)
        results = model(frame, verbose=False, conf=CONF_THRESHOLD, iou=0.2, agnostic_nms=True)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        if elapsed > 2.0: frame_count = 0; start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制中心十字
        h, w = frame.shape[:2]
        screen_cx, screen_cy = w // 2, h // 2
        cv2.drawMarker(frame, (screen_cx, screen_cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # 3. 处理框 (智能聚类)
        raw_boxes = []
        if len(results) > 0:
            raw_boxes = results[0].boxes
        targets = smart_group_boxes(raw_boxes)

        # 4. 寻找 C 位目标
        best_idx = -1
        min_dist = 10000
        for i, t in enumerate(targets):
            cx_box = (t[0] + t[2]) // 2
            cy_box = (t[1] + t[3]) // 2
            dist = np.sqrt((cx_box - screen_cx) ** 2 + (cy_box - screen_cy) ** 2)
            if dist < min_dist: min_dist = dist; best_idx = i

        # 5. 遍历绘制
        for i, target in enumerate(targets):
            x1, y1, x2, y2, conf = target

            # 颜色：最佳=绿，其他=黄
            is_best = (i == best_idx)
            color = (0, 255, 0) if is_best else (0, 255, 255)

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # --- 深度计算部分 (仅在有深度图时进行) ---
            info_text = f"Box {conf:.2f}"

            if last_depth_frame is not None:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 避开边缘黑区
                if 50 < center_x < 590:
                    # 采样
                    roi_w = max(4, int((x2 - x1) * 0.2))
                    roi_h = max(4, int((y2 - y1) * 0.2))
                    d_x1 = max(0, center_x - roi_w // 2)
                    d_x2 = min(640, center_x + roi_w // 2)
                    d_y1 = max(0, center_y - roi_h // 2)
                    d_y2 = min(640, center_y + roi_h // 2)

                    depth_roi = last_depth_frame[d_y1:d_y2, d_x1:d_x2]
                    valid = depth_roi[depth_roi > 0]

                    if len(valid) > 10:
                        z_raw = int(np.median(valid))

                        if Z_MIN_MM < z_raw < Z_MAX_MM:
                            # 坐标转换
                            x_mm = int((center_x - cx) * z_raw / fx)
                            y_mm = int((center_y - cy) * z_raw / fy)

                            # 平滑
                            if is_best:
                                history_z.append(z_raw)
                                z_stable = int(sum(history_z) / len(history_z))
                            else:
                                z_stable = z_raw

                            info_text = f"X:{x_mm} Y:{y_mm} Z:{z_stable}"

                            if is_best:
                                print(f"🎯 坐标: {info_text}")

            cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Final System", frame)
        if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()