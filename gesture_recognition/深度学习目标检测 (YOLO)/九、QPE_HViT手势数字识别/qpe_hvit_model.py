"""
QPE_HViT/QPE_MViT 适配层。

当前目录下的 `QPE_HViT.py` 实际暴露的主模型类名为 `QPE_MViT`，
这里统一包装成训练脚手架使用的 `QPE_HViT` 接口。
"""
from __future__ import annotations

import types
import torch
import torch.nn as nn

try:
    from QPE_HViT import QPE_MViT
except ModuleNotFoundError as exc:
    if exc.name == "pennylane":
        raise ModuleNotFoundError(
            "当前环境缺少 pennylane。请先安装后再训练，例如：pip install pennylane"
        ) from exc
    raise


class QPE_HViT(QPE_MViT):
    def __init__(self, num_classes: int = 10):
        super().__init__(img_size=224, num_classes=num_classes, embed_dim=768, depth=12, num_heads=12)
        self._qpe_cache_cpu: torch.Tensor | None = None

    def _build_qpe_cache_cpu(self) -> torch.Tensor:
        qpe = self.qpe
        q_in_cpu = qpe.patch_coords.unsqueeze(0).float().cpu()
        qpe.q_layer = qpe.q_layer.cpu()

        repeats = max(1, int(getattr(qpe, "measure_repeats", 1)))
        outputs = []
        with torch.no_grad():
            for _ in range(repeats):
                outputs.append(qpe.q_layer(q_in_cpu))
            q_out_cpu = torch.stack(outputs).mean(dim=0)
            q_out_cpu = qpe.up_proj(q_out_cpu).cpu()
        return q_out_cpu

    def enable_realtime_optimization(self) -> None:
        qpe = self.qpe
        if hasattr(qpe, "measure_repeats"):
            qpe.measure_repeats = 1

        def cached_forward(module, x):
            bsz = x.shape[0]
            cls_token, patch_tokens = x[:, 0:1, :], x[:, 1:, :]

            if self._qpe_cache_cpu is None:
                self._qpe_cache_cpu = self._build_qpe_cache_cpu()

            q_out = self._qpe_cache_cpu.to(device=x.device, dtype=x.dtype).expand(bsz, -1, -1)
            patch_tokens = patch_tokens + module.gamma.to(device=x.device, dtype=x.dtype) * q_out
            return torch.cat([cls_token, patch_tokens], dim=1)

        qpe.forward = types.MethodType(cached_forward, qpe)
