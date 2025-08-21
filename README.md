import os

# 设置CUDA内存分配配置（在导入torch之前）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import gc
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path


# GPU内存清理函数
def cleanup_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    # 重置峰值内存统计
    torch.cuda.reset_peak_memory_stats()


def print_gpu_memory():
    """打印GPU内存使用情况"""
    print("\nGPU内存汇总:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        print(f"GPU {i} - 已分配: {allocated:.2f} GB | 已预留: {reserved:.2f} GB | 可用: {total - allocated:.2f} GB")


# 数据集路径
DATASET_PATH = Path('yolo_dataset')

# 确保数据集配置文件存在
data_yaml_path = DATASET_PATH / 'data.yaml'
if not data_yaml_path.exists():
    # 创建data.yaml文件
    data_config = {
        'path': str(DATASET_PATH),  # 数据集根目录
        'train': 'images/train',  # 训练图像相对路径
        'val': 'images/val',  # 验证图像相对路径
        'test': 'images/val',  # 测试图像相对路径
        'names': {}  # 类别名称字典
    }

    # 读取类别列表
    with open(DATASET_PATH / 'classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # 更新类别名称
    for i, cls in enumerate(classes):
        data_config['names'][i] = cls

    # 写入YAML文件
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"已创建数据配置文件: {data_yaml_path}")


# 主训练函数
def train_yolo():
    # 在开始前清理内存
    cleanup_gpu_memory()
    print_gpu_memory()

    # 设置训练参数
    # 使用较大的batch size和较小的imgsz来减少内存占用
    batch_size = 16  # 双GPU可以使用更大的批量
    image_size = 1280  # 适合1920*1080比例
    epochs = 300

    # 加载预训练模型
    model = YOLO("yolo11n.pt")

    print("开始训练...")

    # 开始训练
    results = model.train(
        data=str(data_yaml_path),  # 数据集配置
        epochs=epochs,  # 训练轮数
        imgsz=image_size,  # 图像尺寸
        batch=batch_size,  # 批量大小
        device='0',  # 使用的GPU设备
        workers=8,  # 数据加载工作进程
        pretrained=True,  # 使用预训练权重
        optimizer='SGD',  # 优化器
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率因子
        momentum=0.937,  # SGD动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 热身轮数
        warmup_momentum=0.8,  # 热身动量
        warmup_bias_lr=0.1,  # 热身偏置学习率
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 类别损失权重
        dfl=1.5,  # 分布焦点损失权重
        cos_lr=True,  # 余弦学习率调度
        close_mosaic=10,  # 最后10个轮次关闭马赛克增强
        amp=True,  # 自动混合精度训练
        fraction=1.0,  # 使用的数据集部分
        profile=False,  # 是否进行性能分析
        freeze=None,  # 冻结层 (可选: [0, 1, 2, ...])
        seed=0,  # 随机种子
        verbose=True,  # 详细输出
        resume=False,  # 从中断处恢复训练
        project='train',  # 项目名称
        name='detector',  # 运行名称
        exist_ok=False,  # 是否覆盖现有运行
        val=True,  # 训练期间进行验证
    )

    # 训练后清理内存
    cleanup_gpu_memory()
    print_gpu_memory()

    # 返回结果
    return results, model


# 主函数
if __name__ == '__main__':
    try:
        # 打印CUDA是否可用及GPU信息
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # 开始训练
        results, model = train_yolo()

        # 打印最佳结果
        print("\n训练完成!")
        print(f"最佳mAP50-95: {results.fitness}")

        # 在验证集上评估模型
        print("\n在验证集上评估模型...")
        val_results = model.val(data=str(data_yaml_path))
        print(f"验证集mAP50-95: {val_results.box.map}")
        print(f"验证集mAP50: {val_results.box.map50}")

        print("\n训练成功完成!")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        # 发生错误时也清理内存
        cleanup_gpu_memory()
