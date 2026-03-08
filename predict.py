from ultralytics import YOLO
import os


def evaluate_and_predict():
    """
    一个完整的函数，用于：
    1. 评估模型在整个验证集上的性能指标 (mAP, Precision, Recall)。
    2. 对验证集中的每一张图片进行预测，并保存可视化的结果图。
    """
    print("🚀 === 开始模型的全面评估与预测 === 🚀")

    # --- 路径配置 ---
    # 训练好的模型权重路径
    MODEL_PATH = 'runs/detect/train/weights/best.pt'
    # 数据集配置文件路径 (用于 model.val() 自动寻找验证集)
    DATA_CONFIG_PATH = 'box_data.yaml'
    # 验证集图片文件夹路径 (用于 model.predict() 指定预测源)
    VALIDATION_IMAGE_DIR = './box_dataset/images/val/RGB/'
    # ------------------

    # 检查所有必需的文件和路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在于 '{MODEL_PATH}'")
        return
    if not os.path.exists(DATA_CONFIG_PATH):
        print(f"❌ 错误: 数据配置文件不存在于 '{DATA_CONFIG_PATH}'")
        return
    if not os.path.exists(VALIDATION_IMAGE_DIR):
        print(f"❌ 错误: 验证集图片文件夹不存在于 '{VALIDATION_IMAGE_DIR}'")
        return

    print("✅ 所有路径检查通过！")

    # 加载您训练好的最佳模型
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ 成功加载模型: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # --- 步骤 1: 评估模型整体性能 (计算 mAP) ---
    print("\n--- 步骤 1: 正在评估模型在整个验证集上的整体性能... ---")
    try:
        # model.val() 会自动使用 YAML 文件中定义的验证集进行评估
        metrics = model.val(data=DATA_CONFIG_PATH, split='val')

        print("\n--- 📊 性能评估结果 📊 ---")
        print(f"   mAP50-95: {metrics.box.map:.4f}")  # 综合性能（最常用）
        print(f"   mAP50:    {metrics.box.map50:.4f}")  # 宽松标准下的性能
        print(f"   Precision:  {metrics.box.p[0]:.4f}")  # 查准率
        print(f"   Recall:     {metrics.box.r[0]:.4f}")  # 查全率
        print("------------------------------")

    except Exception as e:
        print(f"❌ 在评估过程中发生错误: {e}")

    # --- 步骤 2: 对验证集中的每一张图片进行预测并保存结果 ---
    print("\n--- 步骤 2: 正在对验证集中的每一张图片进行预测... ---")
    try:
        # model.predict() 会对指定文件夹下的所有图片进行检测
        results = model.predict(
            source=VALIDATION_IMAGE_DIR,
            conf=0.25,  # 置信度阈值，可以根据需要调整
            save=True,  # 必须为 True 才会保存结果图片
            project='runs/detect',  # 指定结果保存的根目录
            name='full_validation_predictions',  # 为这次预测创建一个专门的文件夹
            exist_ok=True  # 如果文件夹已存在，则覆盖
        )

        # 预测过程是惰性的，需要迭代才会真正执行
        for _ in results:
            pass

        print(f"\n🎉 预测完成！所有带检测框的图片已保存至文件夹: runs/detect/full_validation_predictions")

    except Exception as e:
        print(f"❌ 在预测过程中发生错误: {e}")

    print("\n✅ === 所有任务执行完毕 === ✅")


if __name__ == "__main__":
    evaluate_and_predict()