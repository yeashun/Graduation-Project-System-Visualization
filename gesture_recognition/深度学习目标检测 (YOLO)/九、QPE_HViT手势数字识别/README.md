# QPE_HViT 手势数字识别

这个目录提供一套用于手势数字分类的训练脚手架，默认读取当前目录下的：

`new_gesture_data`

## 目录说明

- `qpe_hvit_model.py`
  你自己的 `QPE_HViT` 模型定义文件。
- `一、数据划分脚本.py`
  将 `new_gesture_data` 划分为 train/val/test。
- `二、训练脚本.py`
  训练 QPE_HViT 分类模型。
- `三、评估脚本.py`
  在测试集上评估准确率。
- `四、实时识别脚本.py`
  摄像头实时分类识别。

## 使用顺序

1. 先把你自己的 `QPE_HViT` 模型类放进 `qpe_hvit_model.py`
2. 将数据集放到当前目录下的 `new_gesture_data/`
3. 运行 `一、数据划分脚本.py`
4. 运行 `二、训练脚本.py`
5. 运行 `三、评估脚本.py`
6. 运行 `四、实时识别脚本.py`

## 模型接口要求

`qpe_hvit_model.py` 中需要暴露：

```python
class QPE_HViT(nn.Module):
    def __init__(self, num_classes: int = 10):
        ...
```

输入默认是 `3 x 224 x 224` 图像，输出是 `num_classes` 维 logits。
