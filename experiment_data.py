"""
静态实验数据与图片路径配置。

后续维护建议：
1. 指标数据：直接修改本文件中的数字即可。
2. 图片素材：将导出的图片放到 assets/visualizations 与 assets/presentation 对应目录即可自动显示。
"""
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VISUALIZATION_ASSET_DIR = BASE_DIR / "assets" / "visualizations"
PRESENTATION_ASSET_DIR = BASE_DIR / "assets" / "presentation"


def visualization_asset_path(*parts):
    return str((VISUALIZATION_ASSET_DIR.joinpath(*parts)).resolve())


def presentation_asset_path(*parts):
    return str((PRESENTATION_ASSET_DIR.joinpath(*parts)).resolve())


APP_META = {
    "window_title": "QPE-HViT 本科毕业设计可视化演示系统",
    "title": "基于量子位置编码的多尺度视觉Transformer可视化系统",
    "subtitle": (
        "系统围绕理论方法、性能对比实验、消融实验与可视化实验四个模块展开，"
        "适用于本科毕业设计答辩、组会汇报与论文展示场景。"
    ),
    "research_points": [
        "理论方法页负责讲清 QPE-HViT 的设计动机、核心模块和整体推理流程，避免答辩时只展示结论、不解释方法来源。",
        "性能对比页面向整体性能评估，在统一训练配置下比较 QPE-HViT 与主流视觉模型的精度、参数量与计算量。",
        "消融实验页面向模块贡献验证，通过基础模型、仅 QPE、仅 QCSA 与完整模型四个变体分析结构增益来源。",
        "可视化实验页面向可解释性展示，采用 Grad-CAM、t-SNE 和多尺度特征响应解释性能提升原因。",
    ],
}


THEORY_DATA = {
    "overview": [
        "QPE-HViT 以层次化视觉 Transformer 为主干，在多尺度视觉编码框架中引入量子先验增强机制。",
        "QPE 模块将量子相位信息转化为位置先验，增强 token 之间的相对位置感知与远距离依赖建模能力。",
        "QCSA 模块联合建模通道与空间的重要性分布，强化关键区域响应并抑制冗余特征。",
        "模型最终通过多尺度融合输出分类结果，同时为 Grad-CAM、t-SNE 等可解释性分析提供更稳定的特征表征。",
    ],
    "innovation_cards": [
        {
            "title": "分层 HViT 主干",
            "description": "通过层次化 token 编码兼顾局部纹理与全局语义，为后续模块增强提供稳定的多尺度表征基础。",
            "accent": False,
        },
        {
            "title": "QPE 量子位置先验",
            "description": "将量子相位先验注入位置编码链路，提升模型对空间关系和目标布局的辨别能力。",
            "accent": True,
        },
        {
            "title": "QCSA 协同注意",
            "description": "联合聚合空间与通道注意信息，突出目标区域、边界与细粒度判别特征。",
            "accent": True,
        },
        {
            "title": "多尺度融合输出",
            "description": "融合浅层细节与深层语义特征，为分类预测与可视化解释提供一致且可读的输出。",
            "accent": False,
        },
    ],
    "pipeline": [
        {
            "title": "输入图像",
            "description": "完成预处理、Patch 划分与初始序列构建。",
            "tone": "soft",
        },
        {
            "title": "Patch / Token 嵌入",
            "description": "保留局部纹理并建立基础位置感知。",
            "tone": "primary",
        },
        {
            "title": "分层 HViT 编码",
            "description": "在多尺度层级中提取局部到全局的视觉表征。",
            "tone": "primary",
        },
        {
            "title": "QPE + QCSA 增强",
            "description": "在编码阶段注入量子先验并进行通道-空间协同建模。",
            "tone": "accent",
        },
        {
            "title": "融合预测与解释",
            "description": "输出分类结果，并支持 Grad-CAM、t-SNE 等解释分析。",
            "tone": "soft",
        },
    ],
    "framework_title": "论文方法框图 / Framework",
    "framework_caption": (
        "默认读取 assets/presentation/theory/framework.png。"
        "如果你的论文里已经有正式方法框图，可在界面中临时替换。"
    ),
    "framework_image": presentation_asset_path("theory", "framework.png"),
    "talk_track": [
        "讲解建议：先顺着主流程讲一遍，再单独强调 QPE 与 QCSA 两个创新模块的作用。",
        "如果右侧还未放正式论文框图，左侧内置示意图也可以直接用于答辩讲解。",
    ],
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
            "name": "ResNet34",
            "family": "CNN",
            "params": "21.28M",
            "flops": "3.68G",
            "accuracy": {
                "CIFAR-10": 60.74,
                "CIFAR-100": 33.75,
                "ImageNet-1k": 70.04,
            },
        },
        {
            "name": "ResNet101",
            "family": "CNN",
            "params": "44.62M",
            "flops": "8.52G",
            "accuracy": {
                "CIFAR-10": 68.22,
                "CIFAR-100": 40.10,
                "ImageNet-1k": 77.37,
            },
        },
        {
            "name": "EfficientNet-B7",
            "family": "CNN",
            "params": "66.34M",
            "flops": "19.77G",
            "accuracy": {
                "CIFAR-10": 65.91,
                "CIFAR-100": 36.58,
                "ImageNet-1k": 84.38,
            },
        },
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
            "name": "DeiT-B",
            "family": "ViT",
            "params": "86.66M",
            "flops": "17.96G",
            "accuracy": {
                "CIFAR-10": 92.00,
                "CIFAR-100": 70.20,
                "ImageNet-1k": 81.85,
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
            "name": "CTA-Net",
            "family": "CNN-Attention",
            "params": "20.32M",
            "flops": "2.83G",
            "accuracy": {
                "CIFAR-10": 86.76,
                "CIFAR-100": 59.43,
                "ImageNet-1k": 76.86,
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
            "name": "MambaVision-B",
            "family": "State Space",
            "params": "97.70M",
            "flops": "15.01G",
            "accuracy": {
                "CIFAR-10": 93.82,
                "CIFAR-100": 72.45,
                "ImageNet-1k": 84.28,
            },
        },
        {
            "name": "QPE-HViT",
            "family": "Proposed",
            "params": "88.16M",
            "flops": "87.92G*",
            "highlight": True,
            "accuracy": {
                "CIFAR-10": 85.24,
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
        "* QPE-HViT 的 FLOPs 基于当前 classical simulation setting 统计，"
        "用于统一实验条件下比较量子增强机制的算法有效性，"
        "不宜直接等同为原生量子硬件上的实际部署开销。"
    ),
    "insights": [
        {
            "title": "分类精度分析",
            "text": (
                "在任务较简单、分辨率较低的 CIFAR-10 上，QPE-HViT（85.24%）未能超过 "
                "MambaVision-B 等最新架构。一个可能的原因是，CIFAR-10 的图像分辨率较低"
                "（32×32）且语义结构相对简单，使得 QPE 所带来的高维表征优势未能充分体现。"
                "尽管如此，QPE-HViT 仍明显优于表 2 中的 ResNet 系列与 ViT-B/16，"
                "并在 CIFAR-100 与 ImageNet-1K 上取得了更具竞争力的结果。"
                "这说明该方法更适合在类别更丰富、空间结构更复杂的任务上进一步评估。"
            ),
        },
        {
            "title": "计算开销与效率分析",
            "text": (
                "在参数量方面，QPE-HViT（88.16M）保持在 Base 量级，"
                "与 ConvNeXt-B 和 MambaVision-B 接近，说明量子模块并未显著增加参数规模。"
                "另一方面，表 2 中 QPE-HViT 的 FLOPs（87.92G）明显高于多数经典模型。"
                "这里需要指出，该数值反映的是经典硬件对量子态演化进行模拟时产生的额外开销，"
                "主要用于在统一实验条件下验证量子增强机制的算法有效性。"
                "因此，本文报告的 FLOPs 更适合作为当前 classical simulation setting 下的参考，"
                "而不宜直接等同为原生量子硬件上的实际部署开销。"
                "关于 QPU 场景下的复杂度收益，仍有待后续结合具体硬件条件进一步评估。"
            ),
        },
    ],
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
    "hyperparameter_study": {
        "title": "量子架构超参数消融实验结果（CIFAR-100）",
        "note": "表 4 固定量子线路层数为 2，比较不同量子比特配置下的性能与模拟代价。",
        "columns": ["Qubits", "Layers", "Top-1 Acc(%)", "FLOPs(G)", "Note"],
        "variants": [
            {
                "qubits": 2,
                "layers": 2,
                "acc": 71.36,
                "flops": 42.85,
                "note": "Lightweight",
            },
            {
                "qubits": 4,
                "layers": 2,
                "acc": 75.49,
                "flops": 87.92,
                "note": "Default",
                "highlight": True,
            },
            {
                "qubits": 8,
                "layers": 2,
                "acc": 75.83,
                "flops": 172.64,
                "note": "Heavy",
            },
        ],
        "insight": (
            "表 4 探索了超参数对模型性能的影响（固定线路层数为 2）。"
            "随着量子比特从 2 增加至 4，模型性能提升 4.13%；"
            "当进一步增加至 8 比特时，准确率仅进一步提高 0.34%，"
            "但 classic simulation 下的 FLOPs 增至 172.64G。"
            "综合性能与模拟代价，本文选择 4 Qubits 作为默认配置。"
        ),
    },
}


VISUALIZATION_DATA = [
    {
        "name": "空间判别注意力（Grad-CAM）",
        "figure_label": "图 5 空间判别注意力（Grad-CAM）",
        "description": "观察不同模型在关键目标区域上的响应分布，比较主体定位能力与背景抑制效果。",
        "insight": (
            "从图 5 可以看到，ResNet34 与 ViT 的激活区域相对分散，Swin-B 也存在一定冗余响应。"
            "相比之下，QPE-HViT 的热力图在当前示例中表现出更集中的目标响应："
            "在 Oxford-IIIT Pet 中更关注面部区域，在 DUT-Anti-UAV 场景下也较多聚焦于无人机主体。"
            "这一现象提示量子引导机制可能有助于减弱部分背景干扰，"
            "但相关结论仍需结合更多定量指标进一步验证。"
        ),
        "result_caption": "当前示例重点展示 Oxford-IIIT Pet 与 DUT-Anti-UAV 场景中的目标关注区域。",
        "baseline_title": "基础模型 / Grad-CAM",
        "baseline_caption": "建议放置基础 HViT 的 Grad-CAM 热力图。",
        "baseline_image": visualization_asset_path("gradcam", "baseline.png"),
        "ours_title": "QPE-HViT / Grad-CAM",
        "ours_caption": "建议放置 QPE-HViT 的 Grad-CAM 热力图。",
        "ours_image": visualization_asset_path("gradcam", "ours.png"),
    },
    {
        "name": "特征流形空间分布（t-SNE）",
        "figure_label": "图 6 CIFAR-10 t-SNE 特征分布图",
        "description": "观察高维特征映射后的类内聚集和类间边界，辅助判断特征表达质量。",
        "insight": (
            "如图 6 所示，基线 HViT 的高层特征存在一定的类间重叠。"
            "引入量子编码后，QPE-HViT 在当前可视化结果中呈现出更紧凑的类内聚集趋势"
            "和相对更清晰的类间边界。"
            "这说明在本文实验设定下，高维映射可能有助于提升特征表达能力，"
            "但该结论仍主要基于当前可视化结果。"
        ),
        "result_caption": "图中对比基线 HViT 与 QPE-HViT 在 CIFAR-10 上的类内聚集与类间分离情况。",
        "baseline_title": "基础模型 / t-SNE",
        "baseline_caption": "建议放置基础模型导出的 t-SNE 散点图。",
        "baseline_image": visualization_asset_path("tsne", "baseline.png"),
        "ours_title": "QPE-HViT / t-SNE",
        "ours_caption": "建议放置 QPE-HViT 导出的 t-SNE 散点图。",
        "ours_image": visualization_asset_path("tsne", "ours.png"),
    },
    {
        "name": "多尺度特征层级演化",
        "figure_label": "图 7 多尺度特征响应可视化",
        "description": "提取不同阶段的特征响应，观察模型如何从局部纹理逐步汇聚到全局判别语义。",
        "insight": (
            "提取网络内不同阶段的特征响应（图 7）揭示了 QPE-HViT 的层级建模逻辑。"
            "Stage 1 呈现高频离散响应，主要捕捉边缘与底层纹理；"
            "Stage 2 实现了向核心结构的初步聚拢与语义过渡；"
            "Stage 3 则显著抑制了背景干扰，输出高纯度的全局判别性语义。"
            "这一演化过程表明，QCSA 机制有效地引导了从局部纹理到全局语义的有序信息流转。"
        ),
        "result_caption": "当前结果依次展示 Stage 1、Stage 2 与 Stage 3 的多尺度响应演化。",
        "baseline_title": "基础模型 / 多尺度特征响应",
        "baseline_caption": "建议放置基础模型的多尺度响应图。",
        "baseline_image": visualization_asset_path("multiscale", "baseline.png"),
        "ours_title": "QPE-HViT / 多尺度特征响应",
        "ours_caption": "建议放置 QPE-HViT 的多尺度响应图。",
        "ours_image": visualization_asset_path("multiscale", "ours.png"),
    },
]
