from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt


class ArchiveBoxTrainer:
    def __init__(self, data_yaml, model_type='yolov8s.pt'):
        self.data_yaml = data_yaml
        self.model_type = model_type
        self.project = 'box_grasp_project'
        self.name = 'yolov8s_train_run'
        self.cwd = os.getcwd()

    def check_dataset(self):
        """
        深度检查数据集：文件存在性、配对情况、空标注
        """
        print(f"\n🔍 [1/4] 正在根据 {self.data_yaml} 深度检查数据集...")

        try:
            with open(self.data_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ 读取配置文件失败: {e}")
            return False

        base_path = Path(config.get('path', ''))
        # 如果path是绝对路径直接用，如果是相对路径则不拼接
        if not base_path.is_absolute():
            # 这里假设yaml里的path是相对于yaml文件或者当前执行目录的
            # 为了保险，建议在yaml里写绝对路径，或者在这里处理
            pass

        status = True
        total_images = 0

        for split in ['train', 'val']:
            if split not in config:
                continue

            img_dir = base_path / config[split]
            # 兼容绝对路径写法
            if os.path.isabs(config[split]):
                img_dir = Path(config[split])

            if not img_dir.exists():
                print(f"❌ {split} 图像目录不存在: {img_dir}")
                status = False
                continue

            # 查找图片
            imgs = list(img_dir.rglob('*.jpg')) + list(img_dir.rglob('*.png')) + list(img_dir.rglob('*.jpeg'))
            total_images += len(imgs)

            missing_labels = 0
            empty_labels = 0

            print(f"   📂 {split}集: 发现 {len(imgs)} 张图片")

            for img in imgs:
                # 寻找对应的 label
                # 逻辑: .../images/train/1.jpg -> .../labels/train/1.txt
                try:
                    p = list(img.parts)
                    idx = len(p) - 1 - p[::-1].index('images')
                    p[idx] = 'labels'
                    label_path = Path(*p).with_suffix('.txt')
                except ValueError:
                    # 尝试同级目录
                    label_path = img.with_suffix('.txt')

                if not label_path.exists():
                    # 允许一定容错吗？对于抓取任务，建议不允许
                    print(f"      ❌ 缺失标注文件: {img.name}")
                    missing_labels += 1
                else:
                    if label_path.stat().st_size == 0:
                        empty_labels += 1

            if missing_labels > 0:
                print(f"      ⛔ {split}集 有 {missing_labels} 张图片没有对应的txt文件！")
                status = False
            if empty_labels > 0:
                print(f"      ⚠️ {split}集 有 {empty_labels} 张图片是空标注(将被视为背景/负样本)。")

        print(f"✅ 数据集检查结束。共 {total_images} 张图片。状态: {'通过' if status else '失败'}")
        return status

    def train(self):
        """
        执行训练
        """
        print(f"\n🚀 [2/4] 开始加载模型 {self.model_type} 并训练...")
        model = YOLO(self.model_type)

        # 针对档案盒抓取的超参数优化
        # imgsz=640: OAK-D 标准输入
        # epochs=150: 500张图属于小数据集，需要多轮次
        # patience=30: 早停机制
        # box=7.5: 增加边框损失的权重(默认7.5)，为了抓取，边框精度至关重要
        try:
            self.results = model.train(
                data=self.data_yaml,
                epochs=150,
                imgsz=640,
                batch=16,
                workers=4,
                device='0',  # 自动检测GPU
                project=self.project,
                name=self.name,
                exist_ok=True,
                patience=30,
                save=True,
                verbose=True,
                seed=42,  # 固定随机种子以便复现
                close_mosaic=10  # 最后10轮关闭马赛克增强，提升边框定位精度
            )
            print("✅ 训练过程完成。")
            return True
        except Exception as e:
            print(f"❌ 训练中断: {e}")
            return False

    def analyze_results(self):
        """
        自动分析训练日志，给出诊断结论
        """
        print(f"\n📊 [3/4] 正在分析训练指标...")

        result_csv = os.path.join(self.project, self.name, 'results.csv')
        if not os.path.exists(result_csv):
            print("❌ 找不到结果文件 results.csv，无法分析。")
            return

        # 读取CSV，去除列名的空格
        df = pd.read_csv(result_csv)
        df.columns = [x.strip() for x in df.columns]

        # 获取关键指标
        last_epoch = df.iloc[-1]
        best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
        best_epoch_data = df.iloc[best_epoch_idx]

        train_box_loss = last_epoch['train/box_loss']
        val_box_loss = last_epoch['val/box_loss']
        map50 = best_epoch_data['metrics/mAP50(B)']
        map50_95 = best_epoch_data['metrics/mAP50-95(B)']

        print("\n" + "=" * 40)
        print("      🤖 AI 训练诊断报告")
        print("=" * 40)
        print(f"最佳轮次: Epoch {best_epoch_data['epoch']}")
        print(f"最终 mAP@50    : {map50:.4f} (目标 > 0.90)")
        print(f"最终 mAP@50-95 : {map50_95:.4f} (目标 > 0.70)")
        print(f"最终 训练Box Loss: {train_box_loss:.4f}")
        print(f"最终 验证Box Loss: {val_box_loss:.4f}")
        print("-" * 40)

        conclusions = []

        # 1. 判断 mAP (精度是否达标)
        if map50 < 0.8:
            conclusions.append(
                "🔴 **欠拟合/数据困难**: 模型没能很好地识别物体。mAP@50 低于 0.8。可能原因：数据量太少、标注错误、或者物体特征极不明显。")
        elif map50 > 0.95:
            conclusions.append("🟢 **精度优秀**: 模型识别准确率极高。")
        else:
            conclusions.append("🟡 **精度尚可**: 能用，但可能还有提升空间。")

        # 2. 判断过拟合 (验证集Loss远大于训练集Loss，或者验证集Loss开始反弹)
        # 获取最小验证损失及其所在轮次
        min_val_loss = df['val/box_loss'].min()
        min_val_idx = df['val/box_loss'].idxmin()
        current_epoch = df['epoch'].max()

        # 如果最好的loss出现在很久以前，且现在loss变大了
        if (val_box_loss > min_val_loss * 1.1) and (current_epoch - min_val_idx > 10):
            conclusions.append(
                f"🔴 **检测到过拟合 (Overfitting)**: 验证集Loss在第 {min_val_idx} 轮最低，之后开始上升，说明模型在'死记硬背'训练集。")

        # 简单比较 gap
        elif val_box_loss > train_box_loss * 1.3:  # 经验阈值
            conclusions.append("🟠 **轻微过拟合风险**: 验证集Loss明显高于训练集Loss，泛化能力可能受限。")
        else:
            conclusions.append("🟢 **泛化能力良好**: 训练集与验证集Loss差距在正常范围内。")

        # 输出结论
        print("诊断结论:")
        for c in conclusions:
            print(c)
        print("=" * 40)

        return df


def main():
    # 初始化训练器
    trainer = ArchiveBoxTrainer(data_yaml='box_data.yaml', model_type='yolov8s.pt')

    # 1. 检查数据
    if not trainer.check_dataset():
        return

    # 2. 训练
    success = trainer.train()
    if not success:
        return

    # 3. 自动分析
    trainer.analyze_results()

    print(f"\n💡 下一步提示:")
    print(f"1. 最佳模型保存在: {os.path.join(trainer.project, trainer.name, 'weights', 'best.pt')}")
    print(f"2. 请使用 convert_model.py 或 Luxonis 在线工具将 .pt 转换为 .blob 以部署到 OAK-D")


if __name__ == '__main__':
    main()