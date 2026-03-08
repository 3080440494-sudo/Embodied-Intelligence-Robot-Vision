import depthai as dai
import cv2
import numpy as np
import os
import threading
import queue
import time
from datetime import datetime

# --- 1. 配置路径 ---
base_dir = os.path.join(os.getcwd(), "box_dataset")
rgb_save_path = os.path.join(base_dir, "images", "train", "1")
depth_save_path = os.path.join(base_dir, "images", "train", "2")
labels_save_path = os.path.join(base_dir, "labels", "train", "3")

# 自动创建目录
for p in [rgb_save_path, depth_save_path, labels_save_path]:
    os.makedirs(p, exist_ok=True)

# --- 2. 全局变量与队列 ---
save_queue = queue.Queue(maxsize=100)
is_running = True

# 统计与历史记录
stats = {
    "positive": 0,
    "negative": 0,
    "total": 0
}

# 历史记录列表，用于实现 'a' 键撤销功能
history_lock = threading.Lock()
save_history = []


def save_worker():
    """后台保存线程"""
    global save_history
    print("💾 后台保存线程已启动...")

    while is_running or not save_queue.empty():
        try:
            task = save_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        rgb_frame, depth_frame, sample_type, frame_idx = task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"rgb_{timestamp}_{frame_idx:06d}"

        saved_files = []  # 记录本次生成的文件路径

        try:
            # 保存 RGB
            rgb_path = os.path.join(rgb_save_path, f"{base_filename}.jpg")
            cv2.imwrite(rgb_path, rgb_frame)
            saved_files.append(rgb_path)

            # 保存 Depth
            depth_path = os.path.join(depth_save_path, f"depth_{timestamp}_{frame_idx:06d}.png")
            cv2.imwrite(depth_path, depth_frame)
            saved_files.append(depth_path)

            # 保存 Label (仅负样本)
            if sample_type == "negative":
                label_path = os.path.join(labels_save_path, f"{base_filename}.txt")
                open(label_path, 'w').close()  # 创建空文件
                saved_files.append(label_path)

            # 更新统计和历史记录
            with history_lock:
                stats[sample_type] += 1
                stats["total"] += 1
                save_history.append({
                    'files': saved_files,
                    'type': sample_type,
                    'id': frame_idx
                })

            print(f"✅ 保存成功 #{frame_idx} [{sample_type}]")

        except Exception as e:
            print(f"❌ 保存出错: {e}")
        finally:
            save_queue.task_done()


# 启动线程
worker = threading.Thread(target=save_worker, daemon=True)
worker.start()


def delete_last_entry():
    """删除上一组文件的函数"""
    with history_lock:
        if not save_history:
            return False, "没有可删除的记录"

        last_entry = save_history.pop()
        deleted_files = []

        # 删除物理文件
        for file_path in last_entry['files']:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
                except Exception as e:
                    print(f"删除文件失败: {e}")

        # 回滚统计数据
        stats[last_entry['type']] = max(0, stats[last_entry['type']] - 1)
        stats['total'] = max(0, stats['total'] - 1)

        return True, f"已撤销 #{last_entry['id']} ({last_entry['type']})"


# --- 3. Pipeline 设置 ---
pipeline = dai.Pipeline()

# 配置 RGB 相机
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 400)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 配置 左 黑白相机
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setCamera("left")

# 配置 右 黑白相机
mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setCamera("right")

# 配置 深度引擎
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# 配置 输出
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")

# --- 这里是刚才漏掉的关键连接代码 ---
cam_rgb.preview.link(xout_rgb.input)
mono_left.out.link(stereo.left)  # 👈 关键修复：连接左眼
mono_right.out.link(stereo.right)  # 👈 关键修复：连接右眼
stereo.depth.link(xout_depth.input)

# --- 4. 主循环 ---
print("📷 相机初始化完成")
print("控制: 's'=正样本, 'n'=负样本, 'a'=撤销上一张, 'q'=退出")

frame_idx_counter = 0
last_ui_message = "就绪"
last_ui_time = time.time()

# 缓存变量
latest_rgb_frame = None
latest_depth_frame = None

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # 获取数据
        in_rgb = q_rgb.tryGet()
        in_depth = q_depth.tryGet()

        # 更新缓存
        if in_rgb is not None:
            latest_rgb_frame = in_rgb.getCvFrame()

        if in_depth is not None:
            latest_depth_frame = in_depth.getFrame()

        # 显示画面
        if latest_rgb_frame is not None:
            display_img = cv2.resize(latest_rgb_frame, (640, 360))

            # 绘制 UI
            cv2.putText(display_img, f"Total: {stats['total']} (Pos:{stats['positive']} Neg:{stats['negative']})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示操作消息
            if time.time() - last_ui_time < 3.0:  # 消息显示3秒
                color = (0, 0, 255) if "撤销" in last_ui_message else (255, 255, 0)
                # 处理中文可能的乱码问题，用拼音代替或者确保系统支持，这里简单用英文
                msg_display = last_ui_message.replace("已撤销", "Revoked").replace("保存成功", "Saved")
                cv2.putText(display_img, msg_display, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 显示队列积压情况
            q_size = save_queue.qsize()
            if q_size > 0:
                cv2.putText(display_img, f"Saving... Q:{q_size}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 165, 255), 2)

            cv2.imshow("RGB", display_img)

        if latest_depth_frame is not None:
            depth_vis = cv2.normalize(latest_depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(np.uint8(depth_vis), cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_vis)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('a'):
            # 处理撤销
            success, msg = delete_last_entry()
            last_ui_message = msg
            last_ui_time = time.time()
            print(f"🔙 {msg}")

        elif key in [ord('s'), ord('n')]:
            # 检查缓存是否都不为空
            if latest_rgb_frame is not None and latest_depth_frame is not None:
                sample_type = "positive" if key == ord('s') else "negative"
                frame_idx_counter += 1

                # 创建任务入队 (使用 copy 防止内存覆盖)
                save_queue.put((latest_rgb_frame.copy(), latest_depth_frame.copy(), sample_type, frame_idx_counter))

                last_ui_message = f"Saving #{frame_idx_counter}..."
                last_ui_time = time.time()
            else:
                last_ui_message = "Wait for camera..."
                last_ui_time = time.time()
                print("⚠️ 相机数据尚未准备好 (RGB或Depth为空)")

# 清理
print("正在等待剩余保存任务...")
is_running = False
worker.join()
print("程序退出")
cv2.destroyAllWindows()