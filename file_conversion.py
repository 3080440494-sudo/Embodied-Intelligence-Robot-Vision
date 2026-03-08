from ultralytics import YOLO
import blobconverter

# 1. 设置路径 (请修改为你 best.pt 的实际路径)
model_path = 'box_grasp_project/yolov8s_train_run/weights/best.pt'

print("⏳ 正在加载模型...")
model = YOLO(model_path)

# 2. 导出为 ONNX
print("🔄 正在导出为 ONNX (输入尺寸 640x640)...")
# opset=12 是对 OAK-D 兼容性最好的版本
model.export(format='onnx', imgsz=640, opset=12)

onnx_path = model_path.replace('.pt', '.onnx')
print(f"✅ ONNX 导出成功: {onnx_path}")

# 3. 将 ONNX 转换为 Blob
print("🔄 正在将 ONNX 转换为 Blob (这需要联网)...")
blob_path = blobconverter.from_onnx(
    model=onnx_path,
    data_type="FP16",    # OAK-D 需要 FP16 精度
    shaves=6,            # 使用 6 个计算核心
    use_cache=False,
    output_dir="."       # 保存在当前目录
)

print(f"\n🎉 转换完成！请找到文件: {blob_path}")
print("👉 请把这个 .blob 文件的路径复制下来，填入下面的推理代码中。")