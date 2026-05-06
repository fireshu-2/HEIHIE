# Rock-Paper-Scissors Classifier

基于 MobileNetV1 的石头/剪刀/布（RPS）手势分类项目，支持负样本（无手势）识别，并提供完整的训练、导出、推理及数据集预处理流水线。

## 项目结构

```
.
├── train_rps_classifier.py         # 模型训练脚本
├── export_onnx.py                  # 导出 ONNX 模型
├── infer_rps_classifier.py         # 单张图片推理
├── prepare_video_dataset.py        # 视频转图片数据集（按类别裁剪）
├── generate_negative_dataset.py    # 从视频中采样负样本（非手势区域）
├── save_bin.py                     # 生成校准/评估用的二进制张量
├── outputs/rps_mobilenetv1/        # 训练输出目录（模型检查点、元数据）
└── calibrate_datasets_bin/         # 校准与评估数据集输出目录
```

## 环境依赖

```bash
pip install torch torchvision timm opencv-python-headless numpy Pillow onnx
```

## 数据集准备

### 1. 视频数据集目录结构

```
datasets/
├── R/                    # 石头手势视频
│   ├── video1.mp4
│   └── ...
├── P/                    # 布手势视频
│   └── ...
└── S/                    # 剪刀手势视频
    └── ...
```

### 2. 视频转图片帧

从视频中按固定间隔抽取帧，并裁剪出手势区域：

```bash
python prepare_video_dataset.py \
    --dataset_dir /path/to/video/datasets \
    --output_dir /path/to/processed_dataset \
    --crop 210 270 750 810 \
    --frame_step 3
```

参数说明：
- `--crop`: 裁剪区域 `(x1, y1, x2, y2)`
- `--frame_step`: 每隔 N 帧保存一帧，默认为 3

执行后会在 `output_dir` 下生成 `R/`, `P/`, `S/` 三个类别文件夹。

### 3. 生成负样本（可选但推荐）

从视频的非手势区域采样，构建负样本类别 `N`：

```bash
python generate_negative_dataset.py \
    --dataset_dir /path/to/video/datasets \
    --output_dir /path/to/processed_dataset/N \
    --target_crop 210 270 750 810 \
    --samples_per_video 100
```

负样本会自动避开手势目标区域，从画面其他位置随机裁剪。

### 4. 最终数据集结构

```
processed_dataset/
├── N/                          # 负样本（无手势）
│   └── *.png
├── P/                          # 布
│   └── *.png
├── R/                          # 石头
│   └── *.png
└── S/                          # 剪刀
    └── *.png
```

## 模型训练

### 启动训练

```bash
python train_rps_classifier.py \
    --dataset_dir /path/to/processed_dataset \
    --output_dir outputs/rps_mobilenetv1 \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-3 \
    --pretrained
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_dir` | `/mnt/hdd16t0/dataset/rps_dataset/processed_dataset` | 数据集根目录 |
| `--output_dir` | `outputs/rps_mobilenetv1` | 检查点输出目录 |
| `--epochs` | 30 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-3 | 初始学习率 |
| `--weight_decay` | 1e-4 | AdamW 权重衰减 |
| `--val_ratio` | 0.2 | 验证集比例 |
| `--head_hidden_dim` | 256 | 分类头隐藏层维度 |
| `--dropout` | 0.2 | Dropout 概率 |
| `--threshold` | 0.5 | Sigmoid 判定阈值 |
| `--pretrained` | True | 是否加载 timm 预训练权重 |

### 训练输出

- `best.pt` — 验证集表现最优的检查点
- `last.pt` — 最后一轮的检查点
- `metadata.json` — 训练配置与数据信息

### 模型架构

- **Backbone**: `mobilenetv1_100`（来自 timm，去掉全局池化和分类层）
- **Input**: `1 x 3 x 320 x 320`
- **Head**:
  - Conv2d (channels -> hidden_dim, 1x1)
  - BatchNorm2d + ReLU
  - Dropout2d
  - Conv2d (hidden_dim -> 3, 全图卷积)
  - Flatten
- **Output**: 3 维 sigmoid 置信度 `[P, R, S]`

输出采用多标签 sigmoid 而非 softmax，支持：
- 正样本：三者中置信度最高的类别为预测手势
- 负样本：三者置信度均低于阈值时判定为无手势

### 训练细节

**损失函数**

- `BCEWithLogitsLoss` — 对 3 路输出独立计算二元交叉熵

**优化器与调度**

- 优化器：`AdamW`，初始学习率 `1e-3`，权重衰减 `1e-4`
- 学习率调度：`CosineAnnealingLR`，`T_max` 等于训练总轮数

**训练数据增强**

1. `Resize((352, 352))` — 先放大
2. `RandomResizedCrop(320, scale=(0.75, 1.0), ratio=(0.9, 1.1))`
3. `RandomHorizontalFlip(p=0.5)`
4. `RandomRotation(degrees=18)`
5. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)`
6. `RandomPerspective(distortion_scale=0.15, p=0.2)`
7. `ToTensor()`
8. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
9. `RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.0))`

**验证数据预处理**

- `Resize((320, 320))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**验证指标**

| 指标 | 说明 |
|------|------|
| `exact_match` | 所有类别预测与标签完全一致的比例 |
| `positive_top1` | 正样本（P/R/S）中 Top-1 分类准确率 |
| `negative_recall` | 负样本（N）中无任何类别超过阈值的比例 |

**最佳模型选择**

验证阶段以 `(positive_top1 + negative_recall) / 2` 作为综合得分，得分最高时保存为 `best.pt`。

## 导出 ONNX

将训练好的 PyTorch 模型导出为 ONNX 格式：

```bash
python export_onnx.py \
    --checkpoint outputs/rps_mobilenetv1/best.pt \
    --output_path outputs/rps_mobilenetv1/best.onnx \
    --opset 18
```

导出后的 ONNX 模型输出为 sigmoid 后的置信度，节点名为 `confidence`。

## 推理测试

对单张图片进行推理：

```bash
python infer_rps_classifier.py \
    --image_path path/to/image.jpg \
    --checkpoint outputs/rps_mobilenetv1/best.pt
```

输出示例：

```json
{
  "P": 0.9234,
  "R": 0.0123,
  "S": 0.0456
}
```

## 生成校准/评估张量

用于后续 NPU 量化校准和精度评估：

```bash
python save_bin.py \
    --dataset_dir /path/to/processed_dataset \
    --output_dir calibrate_datasets_bin \
    --cal_num 50 \
    --eval_num 20 \
    --output_format bin
```

输出：
- `calibrate_datasets/` — 校准集张量
- `evaluate_datasets/` — 评估集张量
- `*_manifest.txt` — 文件清单

每张图片预处理流程：
- 读取为 RGB
- 缩放到 `320 x 320`
- 归一化到 `[0, 1]`
- 减去 mean `[0.485, 0.456, 0.406]`，除以 std `[0.229, 0.224, 0.225]`
- 布局为 `NCHW`
- 保存为 `float32` 格式
