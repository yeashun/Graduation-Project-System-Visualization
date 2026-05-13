"""
静态展示数据配置。

后续维护方式：
1. 修改实验数字：更新 PERFORMANCE_DATA 和 ABLATION_DATA。
2. 替换理论框架图或可视化图片：把图片放到 assets/presentation 对应目录，
   或在界面里手动选择本地图片。
"""
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets" / "presentation"


def asset_path(*parts):
    return str((ASSET_DIR.joinpath(*parts)).resolve())


APP_META = {
    "window_title": "QPE-HViT 本科毕业设计可视化演示系统",
    "title": "基于量子位置编码的多尺度视觉Transformer可视化系统",
    "subtitle": (
        "系统围绕理论方法、性能对比实验、消融实验与可视化实验四个维度展开，"
        "适用于本科毕业设计答辩、组会汇报与论文展示场景。"
    ),
    "research_points": [
        "统一组织论文核心内容，减少答辩时在 PPT 与实验图之间来回切换。",
        "用静态数据和静态图片驱动界面展示，避免现场推理带来的不稳定因素。",
        "通过清晰的层次化页面结构，帮助老师快速理解模型思想、实验结果与可解释性证据。",
    ],
}


THEORY_DATA = {
    "overview": [
        "QPE-HViT 以 HViT 作为主干网络，在层次化视觉 Transformer 结构中融入量子先验增强机制。",
        "QPE 模块利用量子相位先验为 token 表征提供更强的位置编码能力，增强空间关系建模效果。",
        "QCSA 模块通过量子启发的通道-空间协同注意机制，提升关键区域响应与多尺度特征融合效率。",
        "最终模型在多个公开数据集上实现精度与特征表达能力的同步提升，并具备较好的可解释性表现。",
    ],
    "key_points": [
        ("主干结构", "以 HViT 的层次化表示为基础，兼顾全局依赖建模与局部细节表达。"),
        ("QPE 模块", "引入量子相位先验，强化位置编码的判别性与稳定性。"),
        ("QCSA 模块", "联合建模空间与通道信息，提升显著区域响应强度。"),
        ("协同机制", "QPE 强化先验表征，QCSA 优化选择性聚合，两者共同提升最终性能。"),
    ],
    "framework_title": "QPE-HViT 方法框架图",
    "framework_caption": "建议放置论文中的总体框架图、网络结构图或方法流程图。",
    "framework_image": asset_path("theory", "framework.png"),
}


PERFORMANCE_DATA = {
    "datasets": [
        "CIFAR-10",
        "CIFAR-100",
        "ImageNet-1k",
        "Oxford-IIIT Pet",
        "Flowers102",
        "Stanford Cars",
        "DUT-Anti-UAV",
    ],
    "models": [
        {
            "name": "ViT-B/16",
            "family": "ViT",
            "params": "86.61M",
            "flops": "11.28G",
            "accuracy": {
                "CIFAR-10": 67.42,
                "CIFAR-100": 41.41,
                "ImageNet-1k": 77.27,
                "Oxford-IIIT Pet": 90.10,
                "Flowers102": 93.42,
                "Stanford Cars": 88.31,
                "DUT-Anti-UAV": 76.84,
            },
        },
        {
            "name": "Swin-B",
            "family": "Hierarchical ViT",
            "params": "88.04M",
            "flops": "10.22G",
            "accuracy": {
                "CIFAR-10": 84.26,
                "CIFAR-100": 70.05,
                "ImageNet-1k": 83.56,
                "Oxford-IIIT Pet": 94.15,
                "Flowers102": 97.08,
                "Stanford Cars": 91.82,
                "DUT-Anti-UAV": 82.94,
            },
        },
        {
            "name": "MViTv2",
            "family": "Multi-scale Transformer",
            "params": "52.30M",
            "flops": "10.10G",
            "accuracy": {
                "CIFAR-10": 92.48,
                "CIFAR-100": 73.12,
                "ImageNet-1k": 84.10,
                "Oxford-IIIT Pet": 94.88,
                "Flowers102": 97.32,
                "Stanford Cars": 92.04,
                "DUT-Anti-UAV": 84.10,
            },
        },
        {
            "name": "CrossViT-B",
            "family": "Hybrid",
            "params": "105.01M",
            "flops": "15.51G",
            "accuracy": {
                "CIFAR-10": 75.00,
                "CIFAR-100": 49.94,
                "ImageNet-1k": 82.21,
                "Oxford-IIIT Pet": 92.22,
                "Flowers102": 95.83,
                "Stanford Cars": 90.47,
                "DUT-Anti-UAV": 80.92,
            },
        },
        {
            "name": "ConvNeXt-B",
            "family": "CNN",
            "params": "89.32M",
            "flops": "15.41G",
            "accuracy": {
                "CIFAR-10": 93.06,
                "CIFAR-100": 72.00,
                "ImageNet-1k": 83.81,
                "Oxford-IIIT Pet": 95.11,
                "Flowers102": 97.58,
                "Stanford Cars": 92.36,
                "DUT-Anti-UAV": 83.05,
            },
        },
        {
            "name": "QPE-HViT",
            "family": "Proposed",
            "params": "88.16M",
            "flops": "87.92G*",
            "highlight": True,
            "accuracy": {
                "CIFAR-10": 95.24,
                "CIFAR-100": 75.49,
                "ImageNet-1k": 84.45,
                "Oxford-IIIT Pet": 96.44,
                "Flowers102": 98.17,
                "Stanford Cars": 93.58,
                "DUT-Anti-UAV": 86.73,
            },
        },
    ],
    "note": (
        "* 当前数值为静态展示模板。答辩前可直接替换为论文最终实验结果；"
        "若个别数据集暂未确定，只需修改对应字段，不需要改界面代码。"
    ),
}


ABLATION_DATA = {
    "dataset": "CIFAR-100",
    "variants": [
        {
            "name": "基础模型 HViT（不含 QPE / QCSA）",
            "qpe": False,
            "qcsa": False,
            "acc": 68.32,
            "params": "87.50M",
            "flops": "14.86G",
        },
        {
            "name": "HViT + QPE",
            "qpe": True,
            "qcsa": False,
            "acc": 72.18,
            "params": "87.80M",
            "flops": "48.53G",
        },
        {
            "name": "HViT + QCSA",
            "qpe": False,
            "qcsa": True,
            "acc": 71.92,
            "params": "88.00M",
            "flops": "50.17G",
        },
        {
            "name": "完整模型 QPE-HViT",
            "qpe": True,
            "qcsa": True,
            "acc": 75.49,
            "params": "88.16M",
            "flops": "87.92G*",
            "highlight": True,
        },
    ],
}


VISUALIZATION_DATA = [
    {
        "name": "Grad-CAM",
        "description": "比较基础模型与 QPE-HViT 在关键判别区域上的关注范围、响应强度与定位精度。",
        "baseline_title": "基础模型 / Grad-CAM",
        "baseline_caption": "建议放置基础 HViT 的 Grad-CAM 热力图。",
        "baseline_image": asset_path("visualizations", "gradcam_baseline.png"),
        "ours_title": "QPE-HViT / Grad-CAM",
        "ours_caption": "建议放置 QPE-HViT 的 Grad-CAM 热力图。",
        "ours_image": asset_path("visualizations", "gradcam_ours.png"),
    },
    {
        "name": "t-SNE 特征空间",
        "description": "观察降维后的特征类间分离度与类内聚合度，直观展示特征空间优化效果。",
        "baseline_title": "基础模型 / t-SNE",
        "baseline_caption": "建议放置基础模型导出的 t-SNE 散点图。",
        "baseline_image": asset_path("visualizations", "tsne_baseline.png"),
        "ours_title": "QPE-HViT / t-SNE",
        "ours_caption": "建议放置 QPE-HViT 导出的 t-SNE 散点图。",
        "ours_image": asset_path("visualizations", "tsne_ours.png"),
    },
    {
        "name": "多尺度特征响应",
        "description": "比较不同层级特征图或响应热力图，说明量子先验对多尺度表征的增强作用。",
        "baseline_title": "基础模型 / 多尺度特征响应",
        "baseline_caption": "建议放置基础模型的多尺度响应图。",
        "baseline_image": asset_path("visualizations", "multiscale_baseline.png"),
        "ours_title": "QPE-HViT / 多尺度特征响应",
        "ours_caption": "建议放置 QPE-HViT 的多尺度响应图。",
        "ours_image": asset_path("visualizations", "multiscale_ours.png"),
    },
]
