import os


def create_empty_labels_by_suffix():
    images_dir = r"D:\aoak\bottle_dataset\images\train\RGB"
    labels_dir = r"D:\aoak\bottle_dataset\labels\train"

    os.makedirs(labels_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    if not image_files:
        print("没有找到图像文件！")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 提取文件后缀编号并排序
    files_with_suffix = []
    for file in image_files:
        if file.startswith('rgb_') and file.endswith('.jpg'):
            # 提取最后6位数字编号
            parts = file.split('_')
            if len(parts) >= 5:
                suffix = parts[-1].replace('.jpg', '')  # 获取000000这样的编号
                try:
                    number = int(suffix)
                    files_with_suffix.append((number, file))
                except ValueError:
                    continue

    # 按编号排序
    files_with_suffix.sort()

    if not files_with_suffix:
        print("没有找到符合格式的文件！")
        return

    print(f"找到 {len(files_with_suffix)} 个带编号的文件")
    print(f"编号范围: {files_with_suffix[0][0]:06d} 到 {files_with_suffix[-1][0]:06d}")

    # 选择编号90到109的图片
    target_numbers = range(90, 110)  # 90到109

    created_count = 0
    print(f"\n正在为编号 {target_numbers[0]:06d} 到 {target_numbers[-1]:06d} 创建空标注文件：")

    for number, image_file in files_with_suffix:
        if number in target_numbers:
            image_name = os.path.splitext(image_file)[0]
            label_file = f"{image_name}.txt"
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                with open(label_path, 'w') as f:
                    pass
                created_count += 1
                print(f"✅ 创建: {label_file} (编号: {number:06d})")
            else:
                print(f"📁 已存在: {label_file} (编号: {number:06d})")

    # 检查是否所有目标文件都找到了
    found_numbers = [num for num, _ in files_with_suffix if num in target_numbers]
    missing_numbers = set(target_numbers) - set(found_numbers)

    if missing_numbers:
        print(f"\n⚠️  警告：未找到以下编号的文件: {sorted(missing_numbers)}")

    print(f"\n完成！")
    print(f"创建了 {created_count} 个空标注文件")


create_empty_labels_by_suffix()