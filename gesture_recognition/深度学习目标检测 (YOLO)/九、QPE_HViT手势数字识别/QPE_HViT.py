# # """
# # ========================================
# # @FileName:    QPE-MViT
# # @Author:      ye_shun
# # @Email:       2942613675@qq.com
# # @Created:     2026/3/1 13:44
# # @Description: 完整QPE-MViT
# # ========================================
# # """
#
# import torch
# import torch.nn as nn
# import pennylane as qml
# from functools import partial
# from collections import OrderedDict
# # ============================================================
# # DropPath 正则化
# # ============================================================
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()
#     return x.div(keep_prob) * random_tensor
#
#
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# # ============================================================
# # 1-单尺度 Patch Embedding
# # ============================================================
#
# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size #输入图片尺寸224*224
#         self.patch_size = patch_size #每个补丁尺寸16*16
#         self.grid_size = img_size // patch_size #单边补丁数224/16=14
#         self.num_patches = self.grid_size * self.grid_size #总补丁数14*14
#
#         self.proj = nn.Conv2d(in_c, embed_dim,kernel_size=patch_size,stride=patch_size) # 输入 [B, 3, 224, 224] → 输出 [B, 768, 14, 14]
#
#     def forward(self, x):
#         x = self.proj(x).flatten(2).transpose(1, 2) # B,3,224,224->B,768,14,14 -> B，768，196 ->B ,196,768
#         return x
#
#
# # ============================================================
# # 2-多尺度 Patch Embedding（MViT）
# # ============================================================
#
# class MultiScalePatchEmbed(nn.Module):
#     """
#     8×8 / 16×16 / 32×32 三种尺寸的补丁
#     """
#
#     def __init__(self, img_size=224, in_c=3, embed_dim=768):
#         super().__init__()
#
#         # 8×8补丁：224/8=28 → 输出特征图尺寸 28×28
#         self.patch_small = nn.Conv2d(in_c, embed_dim, 8, 8)
#         # 16×16补丁：224/16=14 → 输出特征图尺寸 14×14
#         self.patch_medium = nn.Conv2d(in_c, embed_dim, 16, 16)
#         # 32×32补丁：224/32=7 → 输出特征图尺寸 7×7
#         self.patch_large = nn.Conv2d(in_c, embed_dim, 32, 32)
#
#         # 计算所有尺度的总补丁数量
#         self.num_patches = (
#             (img_size // 8) ** 2 +
#             (img_size // 16) ** 2 +
#             (img_size // 32) ** 2
#         )
#
#     def forward(self, x):
#         # 输入[B,3,224,224] → 卷积后[B,768,28,28] → flatten(2)后[B,768,784] → transpose后[B,784,768]
#         s = self.patch_small(x).flatten(2).transpose(1, 2)
#         # 输入[B,3,224,224] → 卷积后[B,768,14,14] → flatten(2)后[B,768,196] → transpose后[B,196,768]
#         m = self.patch_medium(x).flatten(2).transpose(1, 2)
#         # 输入[B,3,224,224] → 卷积后[B,768,7,7] → flatten(2)后[B,768,49] → transpose后[B,49,768]
#         l = self.patch_large(x).flatten(2).transpose(1, 2)
#
#         #在补丁维度（dim = 1）拼接三个尺度的嵌入结果
#         # [B,784,768] + [B,196,768] + [B,49,768] → [B,1029,768]
#         return torch.cat([s, m, l], dim=1)
#
#
# # ============================================================
# # 3-Quantum Positional Encoding (QPE)
# # ============================================================
#
# # class QuantumPositionalEncoding(nn.Module):
# #     """
# #     插件式量子位置增强
# #     x = x + gamma * Q(x)
# #     其中Q(x)是量子神经网络对输入特征的变换，gamma是可学习的权重系数
# #     """
# #
# #     def __init__(self, embed_dim, n_qubits=4, n_layers=2):
# #         """
# #         初始化量子位置增强模块
# #         Args:
# #             embed_dim (int): 输入嵌入向量的维度（如768），也是输出维度
# #             n_qubits (int): 量子比特数量（默认4），决定量子电路的规模
# #             n_layers (int): 量子纠缠层的数量（默认2），影响量子电路的表达能力
# #         """
# #         super().__init__()
# #         # 1.下投影层：将高维嵌入向量（embed_dim）降维到量子比特数维度（n_qubits）
# #         # 目的：适配量子电路的输入维度
# #         self.down_proj = nn.Linear(embed_dim, n_qubits)
# #
# #         dev = qml.device("default.qubit", wires=n_qubits)
# #
# #         @qml.qnode(dev, interface="torch")
# #         def circuit(inputs, weights):
# #             """
# #             量子电路核心逻辑：将经典输入编码为量子态，通过纠缠层变换后测量
# #             Args:
# #                 inputs (torch.Tensor): 经典输入张量，形状为 [batch_size, n_qubits]
# #                 weights (torch.Tensor): 量子电路的可学习参数，形状由weight_shapes定义
# #             Returns:
# #                 list: 每个量子比特的Pauli-Z算符期望值，长度为n_qubits
# #             """
# #             # 角度嵌入：将经典输入值（角度）编码到量子比特的旋转中
# #             # 输入值范围通常为[0, π]，对应量子态的旋转角度
# #             qml.AngleEmbedding(inputs, wires=range(n_qubits))
# #             # 强纠缠层：引入量子比特间的纠缠，增强表达能力
# #             # weights参数：(n_layers, n_qubits, 3) → 每层每个量子比特有3个旋转参数
# #             qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
# #             # 测量：返回每个量子比特的Pauli-Z算符期望值（经典输出，范围[-1,1]）
# #             return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
# #
# #         # weights形状：(n_layers, n_qubits, 3) → 层数×量子比特数×每个比特的旋转参数数
# #         weight_shapes = {"weights": (n_layers, n_qubits, 3)}
# #         # 将量子电路封装为PyTorch兼容的层，可直接集成到NN中
# #         self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
# #
# #         # 2.上投影层：将量子电路输出的n_qubits维向量升维回原嵌入维度
# #         # 目的：将量子变换后的低维特征映射回高维嵌入空间，与原输入相加
# #         self.up_proj = nn.Linear(n_qubits, embed_dim)
# #
# #         #可学习的权重系数gamma：控制量子增强的强度，初始值为0（无增强）
# #         # nn.Parameter：将张量转为可训练参数，参与反向传播
# #         self.gamma = nn.Parameter(torch.zeros(1))
# #
# #     def forward(self, x):
# #         """
# #         前向传播：实现量子位置增强的核心逻辑 x = x + gamma * Q(x)
# #         Args:
# #             x (torch.Tensor): 输入嵌入张量，形状为 [batch_size, num_patches, embed_dim]
# #         Returns:
# #             torch.Tensor: 增强后的嵌入张量，形状与输入一致 [batch_size, num_patches, embed_dim]
# #         """
# #         identity = x # shape: [B, N, D] (B=批次, N=补丁数, D=嵌入维度)
# #
# #         q_in = torch.tanh(self.down_proj(x)) * 3.1415926 #tanh激活：将值限制在[-1,1]，乘以π后角度范围为[-π, π]，适配量子角度编码
# #         q_out = self.q_layer(q_in)
# #         q_out = self.up_proj(q_out)
# #
# #         return identity + self.gamma * q_out
#
# class QuantumPositionalEncoding(nn.Module):
#     def __init__(self, embed_dim, n_qubits=4, n_layers=2):
#         super().__init__()
#         self.n_qubits = n_qubits
#         self.down_proj = nn.Linear(embed_dim, n_qubits)
#
#         # 保持 default.qubit (默认为 CPU 模拟器)
#         dev = qml.device("default.qubit", wires=n_qubits)
#
#         @qml.qnode(dev, interface="torch")
#         def circuit(inputs, weights):
#             qml.AngleEmbedding(inputs, wires=range(n_qubits))
#             qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
#             return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
#
#         weight_shapes = {"weights": (n_layers, n_qubits, 3)}
#         self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
#
#         self.up_proj = nn.Linear(n_qubits, embed_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         identity = x
#         orig_device = x.device
#         orig_dtype = x.dtype
#
#         # 1. 准备量子输入
#         q_in = torch.tanh(self.down_proj(x)) * 3.1415926
#
#         # 2. 【核心修复】将计算强行搬移到 CPU
#         # 因为 default.qubit 模拟器在 Windows GPU 环境下极难通过 Torch 自动同步设备
#         q_in_cpu = q_in.cpu().float()
#
#         # 确保量子层的权重也在 CPU 上
#         if self.q_layer.weights.device != torch.device('cpu'):
#             self.q_layer = self.q_layer.cpu()
#
#         # 3. 在 CPU 上执行量子电路
#         q_out_cpu = self.q_layer(q_in_cpu)
#
#         # 4. 将结果搬回原设备并恢复精度
#         q_out = q_out_cpu.to(device=orig_device, dtype=orig_dtype)
#
#         # 5. 上投影并加权求和
#         q_out = self.up_proj(q_out)
#
#         return identity + self.gamma * q_out
#
#
# # ============================================================
# # Attention / MLP / Block
# # ============================================================
#
# class Attention(nn.Module):
#     """
#     标准多头自注意力模块（Multi-Head Self-Attention）
#     实现Transformer核心的注意力机制，将输入特征通过Q/K/V变换后计算注意力权重，
#     最终加权求和得到注意力输出。
#     """
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads # 注意力头数量
#         head_dim = dim // num_heads # 每个注意力头的维度（如768//8=96）
#         # 缩放因子：1/√d_k，用于防止注意力分数过大导致softmax梯度消失
#         self.scale = head_dim ** -0.5
#
#         # QKV投影层：一次性生成查询(Query)、键(Key)、值(Value)，输出维度为dim*3
#         # 等价于3个独立的Linear层（q_proj, k_proj, v_proj），但合并后计算效率更高
#         self.qkv = nn.Linear(dim, dim * 3)
#         # 输出投影层：将多头注意力的拼接结果映射回原维度
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): 输入特征张量，形状为 [B, N, C]
#                               B=批次大小，N=序列长度（如补丁数量），C=特征维度
#
#         Returns:
#             torch.Tensor: 注意力输出张量，形状与输入一致 [B, N, C]
#         """
#         # 获取输入维度：B=批次，N=序列长度，C=特征维度
#         B, N, C = x.shape
#
#         # 步骤1：生成Q/K/V并拆分多头
#         # 1.1 QKV投影：[B, N, C] → [B, N, 3*C]
#         # 1.2 维度重塑：[B, N, 3*C] → [B, N, 3, num_heads, head_dim]
#         #     其中 head_dim = C // num_heads
#         # 1.3 维度置换：[3, B, num_heads, N, head_dim]
#         #     置换后第0维对应Q/K/V，第1维是批次，第2维是注意力头，第3维是序列长度
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#
#         # 步骤2：拆分Q/K/V
#         # q: [B, num_heads, N, head_dim]  查询向量
#         # k: [B, num_heads, N, head_dim]  键向量
#         # v: [B, num_heads, N, head_dim]  值向量
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         # 步骤3：计算注意力分数（Scaled Dot-Product Attention）
#         # 3.1 矩阵乘法：q @ k^T → [B, num_heads, N, N]，每个位置(i,j)表示第i个token对第j个token的注意力分数
#         # 3.2 缩放：乘以1/√d_k，防止分数过大导致softmax饱和
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#
#         # 3.3 Softmax归一化：在最后一维（序列维度）归一化，得到注意力权重（和为1）
#         attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
#
#         # 步骤4：加权求和（注意力权重 × 值向量）
#         # [B, num_heads, N, N] @ [B, num_heads, N, head_dim] → [B, num_heads, N, head_dim]
#         x = (attn @ v)
#
#         # 步骤5：拼接多头结果
#         # 5.1 维度置换：[B, num_heads, N, head_dim] → [B, N, num_heads, head_dim]
#         # 5.2 维度重塑：[B, N, num_heads*head_dim] = [B, N, C]
#         x = x.transpose(1, 2).reshape(B, N, C)
#
#         # 步骤6：输出投影（线性变换）
#         x = self.proj(x)  # [B, N, C]
#
#         return x
#
#
# class Mlp(nn.Module):
#     """
#     Transformer中的多层感知机模块（MLP/Feed-Forward Network）
#     核心结构：Linear → GELU → Linear
#     作用：对注意力模块输出的特征进行非线性变换和维度映射，增强模型的表达能力
#     """
#     def __init__(self, in_features, hidden_features=None):
#         super().__init__()
#         hidden_features = hidden_features or in_features * 4 #隐藏层维度，默认值为in_features×4（Transformer经典设计）
#         # 第一层全连接：升维 → 扩展特征空间，增加表达能力
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         # 激活函数：GELU，相比ReLU更平滑，是Transformer的标配
#         self.act = nn.GELU()
#         # 第二层全连接：降维 → 将扩展后的特征映射回原输入维度
#         self.fc2 = nn.Linear(hidden_features, in_features)
#
#     def forward(self, x):
#         x = self.act(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# class Block(nn.Module):
#     """
#     Transformer基础块（Transformer Block）
#     核心结构：Pre-LN架构的残差块 → LayerNorm → Attention → 残差连接 → LayerNorm → MLP → 残差连接
#     作用：将注意力机制和多层感知机结合，通过残差连接和层归一化保证训练稳定性，是Transformer的核心单元
#     """
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         # 第一个层归一化：应用在注意力模块之前
#         self.norm1 = nn.LayerNorm(dim)
#         # 多头自注意力模块：捕捉序列中token间的依赖关系
#         self.attn = Attention(dim, num_heads)
#         # 第二个层归一化：应用在MLP模块之前
#         self.norm2 = nn.LayerNorm(dim)
#         # MLP模块：对注意力输出进行非线性特征变换
#         self.mlp = Mlp(dim)
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x
#
#
# # ============================================================
# # 4-QPE-MViT 主模型
# # ============================================================
#
# class QPE_MViT(nn.Module):
#     """
#     QPE_MViT：融合量子位置增强（QPE）和多尺度补丁嵌入的视觉Transformer模型
#     核心改进点：
#     1. 支持单尺度/多尺度补丁嵌入，适配不同粒度的图像特征提取
#     2. 在传统位置编码基础上增加量子位置增强，提升位置信息表达能力
#     3. 沿用ViT的经典架构（CLS Token + Transformer Blocks + 分类头）
#     """
#     def __init__(self,
#                  img_size=224,          # 输入图像尺寸（默认224×224）
#                  patch_size=16,         # 单尺度补丁尺寸（仅multi_scale=False时生效）
#                  in_c=3,                # 输入图像通道数（默认3，RGB图像）
#                  num_classes=10,        # 分类任务的类别数（默认10类）
#                  embed_dim=768,         # 补丁嵌入维度（Transformer特征维度）
#                  depth=12,              # Transformer Block的堆叠数量
#                  num_heads=12,          # 多头注意力的头数（需满足 embed_dim % num_heads == 0）
#                  multi_scale=True):     # 是否使用多尺度补丁嵌入（True=8/16/32，False=单尺度）
#         super().__init__()
#
#         # 选择单尺度 or 多尺度
#         if multi_scale:
#             self.patch_embed = MultiScalePatchEmbed(img_size, in_c, embed_dim)
#         else:
#             self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
#
#         # 获取补丁总数（多尺度为各尺度补丁数之和，单尺度为(img_size/patch_size)²）
#         num_patches = self.patch_embed.num_patches
#
#         #CLS_Token：ViT的分类令牌，最终通过该令牌输出分类结果
#         # nn.Parameter：转为可训练参数，初始值为全0
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         #传统位置编码：可训练的位置嵌入，维度为[1, num_patches+1, embed_dim]（+1是CLS Token）
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#
#         # QPE 插件：在传统位置编码基础上增加量子增强的位置信息
#         self.qpe = QuantumPositionalEncoding(embed_dim)
#
#         #Transformer编码器：堆叠depth个Block，实现特征的深度建模
#         self.blocks = nn.Sequential(*[
#             Block(embed_dim, num_heads)
#             for _ in range(depth)
#         ])
#
#         #最终层归一化：稳定训练，提升模型泛化能力
#         self.norm = nn.LayerNorm(embed_dim)
#         #分类头：将CLS_Token的特征映射到类别数维度，完成分类
#         self.head = nn.Linear(embed_dim, num_classes)
#
#     def forward_features(self, x):
#
#         x = self.patch_embed(x)
#
#         cls_token = self.cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#
#         # 原ViT位置编码
#         x = x + self.pos_embed
#
#         # 量子位置增强
#         x = self.qpe(x)
#
#         x = self.blocks(x)
#         x = self.norm(x)
#
#         return x[:, 0]
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         return self.head(x)
#
#
# # ------------------------------ 测试代码（验证模型可用性） ------------------------------
# if __name__ == "__main__":
#     # 初始化模型：多尺度补丁嵌入，10分类，768维嵌入，12层Transformer，12个头
#     model = QPE_MViT(
#         img_size=224,
#         num_classes=10,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         multi_scale=True
#     )
#
#     # 模拟输入：批次大小=2，3通道，224×224图像
#     dummy_img = torch.randn(2, 3, 224, 224)
#
#     # 前向传播
#     output = model(dummy_img)
#
#     # 打印关键信息
#     print(f"输入图像形状: {dummy_img.shape}")  # [2, 3, 224, 224]
#     print(f"模型输出形状: {output.shape}")  # [2, 10]（批次大小×类别数）
#     print(f"总补丁数量: {model.patch_embed.num_patches}")  # 多尺度下为784+196+49=1029



# ========================================
# @FileName:    QPE-MViT
# @Author:      ye_shun
# @Email:       2942613675@qq.com
# @Created:     2026/3/1 13:44
# @Description: 量子位置编码增强的多尺度视觉Transformer（QPE-MViT）
#               核心改进：
#               1. 基于补丁坐标的量子位置编码（而非特征直接编码）
#               2. 量子引导的跨尺度注意力机制
#               3. 支持量子/经典模块分阶段训练
#               4. 兼容NISQ时代量子硬件的噪声缓解策略
# ========================================

import torch
import torch.nn as nn
import pennylane as qml
from functools import partial
import warnings
import sys

warnings.filterwarnings("ignore")


# ============================================================
# 1. DropPath 正则化
# ============================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================
# 2. 多尺度补丁嵌入 (MultiScalePatchEmbed)
# ============================================================
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=224, in_c=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # 三种尺度卷积
        self.patch_small = nn.Conv2d(in_c, embed_dim // 4, kernel_size=8, stride=8)
        self.patch_medium = nn.Conv2d(in_c, embed_dim // 2, kernel_size=16, stride=16)
        self.patch_large = nn.Conv2d(in_c, embed_dim, kernel_size=32, stride=32)

        self.num_patches = (img_size // 8) ** 2 + (img_size // 16) ** 2 + (img_size // 32) ** 2

    def forward(self, x):
        s = self.patch_small(x).flatten(2).transpose(1, 2)
        m = self.patch_medium(x).flatten(2).transpose(1, 2)
        l = self.patch_large(x).flatten(2).transpose(1, 2)

        # 统一维度拼接
        s = nn.functional.pad(s, (0, self.embed_dim - s.shape[-1]))
        m = nn.functional.pad(m, (0, self.embed_dim - m.shape[-1]))
        return torch.cat([s, m, l], dim=1)


# ============================================================
# 3. 量子位置编码 (QuantumPositionalEncoding)
# ============================================================
class QuantumPositionalEncoding(nn.Module):
    def __init__(self, img_size=224, embed_dim=768, n_qubits=4, n_layers=2, noise_mitigation=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.noise_mitigation = noise_mitigation
        self.gamma = nn.Parameter(torch.zeros(1))

        # 预生成坐标
        self.register_buffer("patch_coords", self._generate_patch_coords(img_size))

        # 量子电路
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # 优化：随机初始化量子权重
        nn.init.uniform_(self.q_layer.weights, a=0, b=2 * torch.pi)

        self.up_proj = nn.Linear(n_qubits, embed_dim)
        self.measure_repeats = 5 if noise_mitigation else 1  # 降低测量次数平衡速度

    def _generate_patch_coords(self, img_size):
        coords = []
        for s in [8, 16, 32]:
            grid = img_size // s
            for i in range(grid):
                for j in range(grid):
                    coords.append([(i / grid) * torch.pi, (j / grid) * torch.pi, 0.0, 0.0])
        return torch.tensor(coords, dtype=torch.float32)

    def forward(self, x):
        B, N, D = x.shape
        cls_token, patch_tokens = x[:, 0:1, :], x[:, 1:, :]

        # 准备输入并确保设备对齐
        q_in = self.patch_coords.unsqueeze(0).expand(B, -1, -1).to(x.device)

        # 处理量子层在 CPU/GPU 之间的同步 (PennyLane默认在CPU运行)
        q_in_cpu = q_in.cpu().float()
        self.q_layer = self.q_layer.cpu()

        q_out_list = []
        for _ in range(self.measure_repeats):
            q_out_cpu = self.q_layer(q_in_cpu)
            q_out_list.append(q_out_cpu.to(x.device).to(x.dtype))

        q_out = torch.stack(q_out_list).mean(dim=0)
        q_out = self.up_proj(q_out)

        # 使用类型安全的 gamma 注入
        patch_tokens = patch_tokens + self.gamma.to(x.dtype) * q_out
        return torch.cat([cls_token, patch_tokens], dim=1)


# ============================================================
# 4. 量子引导跨尺度注意力 (QuantumGuidedCrossScaleAttention)
# ============================================
class QuantumGuidedCrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # 优化：每个头拥有独立的纠缠学习权重
        self.entanglement_weight = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if N == 1030:
            # 建立跨尺度掩码
            mask = torch.ones(N, N, device=x.device, dtype=x.dtype)
            # 强化尺度间关联 (8x8 <-> 16x16, 16x16 <-> 32x32)
            mask[1:785, 785:981] *= self.entanglement_weight.mean()
            mask[785:981, 1:785] *= self.entanglement_weight.mean()
            mask[785:981, 981:1030] *= self.entanglement_weight.mean()
            mask[981:1030, 785:981] *= self.entanglement_weight.mean()

            # 注入多头纠缠修正
            attn = attn * self.entanglement_weight

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ============================================================
# 5. Transformer Block & 主模型 QPE_MViT
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = QuantumGuidedCrossScaleAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QPE_MViT(nn.Module):
    def __init__(self, img_size=224, num_classes=10, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = MultiScalePatchEmbed(img_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.qpe = QuantumPositionalEncoding(img_size, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dpr[i]) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # 量子增强
        x = self.qpe(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# ============================================================
# 6. 执行与验证
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QPE_MViT().to(device)

    # 模拟输入
    dummy = torch.randn(1, 3, 224, 224).to(device)
    out = model(dummy)

    print(f"模型加载成功！设备: {device}")
    print(f"输出维度: {out.shape}")  # Expected: [1, 10]
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")