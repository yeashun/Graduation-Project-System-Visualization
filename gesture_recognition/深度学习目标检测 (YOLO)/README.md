# 方案四：深度学习目标检测（YOLOv8）

这个目录提供一套基于 `YOLOv8` 的手势目标检测方案，用于在复杂背景中直接定位并识别手势。

## 目录结构

```text
深度学习目标检测 (YOLO)
├─ dataset/
│  ├─ images/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ labels/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ runs/
├─ 一、标注数据检查脚本.py
├─ 二、数据集配置.yaml
├─ 三、训练脚本.py
├─ 四、实时检测脚本.py
└─ README.md
```

## 标注要求

每张图片都需要：

1. 一个边界框，框住手势区域。
2. 一个类别标签，表示手势类别。

YOLO 标注文件格式为每行 5 个值：

```text
class_id x_center y_center width height
```

其中坐标是相对图片宽高归一化后的结果，范围应在 `0~1` 之间。

## 类别定义

当前默认使用 10 类数字手势：

- `0`
- `1`
- `2`
- `3`
- `4`
- `5`
- `6`
- `7`
- `8`
- `9`

如果你的类别不是数字手势，只需要修改 [二、数据集配置.yaml](D:/桌面/研0/毕业设计/可视化展示界面/gesture_recognition/深度学习目标检测%20(YOLO)/二、数据集配置.yaml) 中的 `names` 即可。

## 推荐流程

1. 使用 `labelImg`、`Label Studio` 或 `CVAT` 标注数据。
2. 按 YOLO 目录结构放入 `dataset/images/...` 和 `dataset/labels/...`。
3. 运行 `一、标注数据检查脚本.py` 检查标注是否有缺失或越界。
4. 运行 `三、训练脚本.py` 开始训练。
5. 使用 `四、实时检测脚本.py` 打开摄像头做实时识别。

## 环境依赖

建议安装：

```bash
pip install ultralytics opencv-python
```

如果使用 GPU，还需要安装与你环境匹配的 `PyTorch`。

## 运行示例

训练：

```bash
python 三、训练脚本.py
```

如果当前主要用 CPU，推荐直接使用脚本默认参数，它已经调整为更省时的配置：

- `epochs=30`
- `imgsz=320`
- `batch=8`
- `workers=0`
- `patience=10`

如果你想进一步加快速度，可以这样运行：

```bash
python 三、训练脚本.py --cache
```

实时检测：

```bash
python 四、实时检测脚本.py --weights runs/train/gesture_yolo/weights/best.pt
```

如果你已经有分类目录 `new_gesture_data`，可以先自动转换成检测数据集：

```bash
python 五、分类数据转YOLO检测数据.py
```

这个脚本会优先调用 MediaPipe 自动找手部区域；如果当前环境没有 MediaPipe，或者某张图检测失败，它会退回为“整张图就是一个目标框”。对于已经裁好手部 ROI 的分类数据，这种转换方式可以直接用。

如果你想做真正适合目标检测的数据，不建议只用裁剪后的分类图。建议直接运行：

```bash
python 六、原图采集并自动标注YOLO.py --split train --label 0
```

这个脚本会保存原始摄像头画面，并同时生成对应的 YOLO 检测框标签，更适合复杂背景中的检测任务。

采集完成后，可以用下面的脚本把 `dataset_raw` 自动切分成训练集、验证集和测试集：

```bash
python 七、原图数据集切分脚本.py --clear-target
```

## 方案特点

- 优点：定位和识别一步完成，适合复杂背景和多目标场景。
- 缺点：标注成本更高，对训练算力和数据质量要求更高。
