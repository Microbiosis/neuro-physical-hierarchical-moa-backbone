#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端任务基准：六阶段 14 算子栈 vs 仅 Stage III 三柱 vs 纯底座

这是此前完全缺失的一环——五维独立性审计只证明「14 个算子互不干扰」，
不证明「合起来有用」。本脚本用真实任务（SST-2 情感分类，真实语义标签）
给出 held-out 准确率。

三个变体（同底座、同数据、同种子、同头、同优化器）：
  A  full    : UnifiedInterdisciplinaryMoALayer（Stage 0 → VI，14 算子）
  B  cortex  : 仅 Stage III（MoA 注意力 + 3 个异构新皮层柱 + 聚合 + 迭代精炼）
  C  base    : 纯底座（冻结编码器 → mean-pool → 分类头）

底座冻结（fp16），只训练算子栈 + 分类头 —— 这样比较的是「算子栈本身带来什么」。

用法：
    python3 bench_end2end.py --epochs 2 --train-size 20000 --batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hybrid_moa_vl_operator_suite"))

from hybrid_moa_vl_operator_suite.unified_hybrid_moa_architecture import (  # noqa: E402
    UnifiedHybridArchitectureConfig, UnifiedInterdisciplinaryMoALayer)
from hybrid_moa_vl_operator_suite.moa_operators import (  # noqa: E402
    CrossProposalAggregatorOp, CrossModalVisionLanguageFusionSubOp, MixtureOfAttentionOp,
    MoAOperatorConfig)
from hybrid_moa_vl_operator_suite.neuro_nonlinear_operators import (  # noqa: E402
    NeuroNonlinearConfig, NonlinearDendriticNeuroOp)
from hybrid_moa_vl_operator_suite.local_self_modifying_operators import (  # noqa: E402
    LocalSelfModConfig, LocalNonlinearSelfModifyingOp)
from hybrid_moa_vl_operator_suite.moa_operators import RMSNorm  # noqa: E402


# ------------------------------------------------------------------ 模型 ------
class CortexOnlyStack(nn.Module):
    """只保留 Stage III：MoA 注意力 + 3 个异构新皮层柱 + 聚合 + 迭代精炼。"""

    def __init__(self, d_model: int = 768, n_heads: int = 12, num_anchors: int = 16,
                 num_groups: int = 16, local_rank: int = 4, refinement: int = 2,
                 top_k: int = 2):
        super().__init__()
        self.refinement = max(int(refinement), 1)
        self.moa_cfg = MoAOperatorConfig(d_model=d_model, n_heads=n_heads,
                                         num_proposers=3, top_k_proposers=top_k,
                                         num_cross_modal_anchors=num_anchors)
        self.attn_norm = RMSNorm(d_model, eps=1e-6)
        self.mo_attention = MixtureOfAttentionOp(self.moa_cfg)
        self.col_norm = RMSNorm(d_model, eps=1e-6)
        self.col0 = CrossModalVisionLanguageFusionSubOp(d_model=d_model, num_anchors=num_anchors)
        self.col1 = NonlinearDendriticNeuroOp(NeuroNonlinearConfig(d_model=d_model, n_heads=n_heads))
        self.col2 = LocalNonlinearSelfModifyingOp(LocalSelfModConfig(
            d_model=d_model, num_local_groups=num_groups, local_rank=local_rank))
        self.branch_norm = RMSNorm(d_model, eps=1e-6)
        self.branch_scale = nn.Parameter(torch.ones(1, 3, 1, d_model))
        self.aggregator = CrossProposalAggregatorOp(self.moa_cfg)

    def forward(self, x):
        attn_out, _ = self.mo_attention(self.attn_norm(x))
        h_attn = x + attn_out
        h = h_attn
        aux_total = torch.zeros((), device=x.device, dtype=x.dtype)
        metrics = {}
        for _ in range(self.refinement):
            s3 = self.col_norm(h)
            c0 = self.col0(s3)
            c1, d1 = self.col1(s3)
            c2, d2 = self.col2(s3)
            props = torch.stack([c0, c1, c2], dim=1)
            props = self.branch_norm(props) * self.branch_scale
            syn, step_aux, metrics = self.aggregator(s3, props)
            h = h_attn + syn
            aux_total = aux_total + step_aux
        return h, aux_total / self.refinement, metrics


class Head(nn.Module):
    def __init__(self, d_model: int, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, backbone, variant: str, d_model: int, n_cls: int, cfg_kwargs: dict):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.variant = variant
        if variant == "full":
            self.stack = UnifiedInterdisciplinaryMoALayer(
                UnifiedHybridArchitectureConfig(d_model=d_model, **cfg_kwargs))
        elif variant == "cortex":
            self.stack = CortexOnlyStack(d_model=d_model, **cfg_kwargs)
        else:
            self.stack = None
        self.head = Head(d_model, n_cls)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.stack is not None:
            h, aux, metrics = self.stack(out)
        else:
            h, aux, metrics = out, torch.zeros((), device=out.device), {}
        pooled = h.mean(dim=1)          # 三变体统一 mean-pool，排除池化差异
        return self.head(pooled), aux, metrics


# ------------------------------------------------------------------ 数据 ------
def load_sst2(train_size: int, max_len: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = None
    for spec in [("nyu-mll/glue", "sst2"), ("SetFit/sst2", None), ("glue", "sst2")]:
        try:
            ds = load_dataset(spec[0], spec[1]) if spec[1] else load_dataset(spec[0])
            break
        except Exception as e:  # noqa: BLE001
            print(f"  dataset {spec} 失败: {str(e)[:80]}", flush=True)
    if ds is None:
        raise RuntimeError("SST-2 加载失败")

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def enc(batch):
        return tok(batch["sentence"], truncation=True, max_length=max_len, padding="max_length")

    train = ds["train"].shuffle(seed=42).select(range(min(train_size, len(ds["train"]))))
    valid = ds["validation"] if "validation" in ds else ds["test"]
    # 只删文本/索引列，保留 label（此前把 label 一起删掉导致 set_format 报错）
    keep = {"label"}
    tr_drop = [c for c in train.column_names if c not in keep]
    va_drop = [c for c in valid.column_names if c not in keep]
    train = train.map(enc, batched=True, remove_columns=tr_drop)
    valid = valid.map(enc, batched=True, remove_columns=va_drop)
    train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    valid.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return train, valid


# ------------------------------------------------------------------ 训练 ------
def run(variant: str, train, valid, cfg_kwargs: dict, args, seed: int = 0):
    from transformers import AutoModel
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = AutoModel.from_pretrained("distilbert-base-uncased")
    model = Model(backbone, variant, 768, 2, cfg_kwargs).cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    tr_loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True,
                                            drop_last=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(valid, batch_size=64, shuffle=False, num_workers=0)

    t0 = time.time()
    log = []
    for ep in range(args.epochs):
        model.train()
        run_loss, seen, correct = 0.0, 0, 0
        for i, b in enumerate(tr_loader):
            ids = b["input_ids"].cuda()
            mask = b["attention_mask"].cuda()
            y = b["label"].cuda()
            logits, aux, _ = model(ids, mask)
            loss = loss_fn(logits, y) + 0.01 * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            run_loss += float(loss) * y.size(0)
            seen += y.size(0)
            correct += int((logits.argmax(-1) == y).sum())
            if i % 200 == 0:
                print(f"    [{variant}] ep{ep} step{i}/{len(tr_loader)} "
                      f"loss={float(loss):.4f} acc={correct / seen:.4f}", flush=True)
        tr_acc = correct / max(seen, 1)
        va_acc = evaluate(model, va_loader)
        log.append({"epoch": ep, "train_loss": run_loss / max(seen, 1),
                    "train_acc": tr_acc, "val_acc": va_acc})
        print(f"  [{variant}] epoch {ep}: loss={run_loss / max(seen, 1):.4f} "
              f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f}", flush=True)

    return {"variant": variant, "trainable_params": n_train, "log": log,
            "final_val_acc": log[-1]["val_acc"], "minutes": (time.time() - t0) / 60}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = seen = 0
    for b in loader:
        ids = b["input_ids"].cuda()
        mask = b["attention_mask"].cuda()
        y = b["label"].cuda()
        logits, _, _ = model(ids, mask)
        correct += int((logits.argmax(-1) == y).sum())
        seen += y.size(0)
    return correct / max(seen, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--variants", default="full,cortex,base")
    ap.add_argument("--out", default="/kaggle/working/bench_end2end.json")
    args = ap.parse_args()

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    train, valid = load_sst2(args.train_size, args.max_len)
    print(f"  train={len(train)} valid={len(valid)} max_len={args.max_len}", flush=True)

    cfg_kwargs = dict(n_heads=12, num_proposers=3, top_k_proposers=2, num_refinement_steps=2,
                      num_vacuum_modes=16, num_cross_modal_anchors=16, num_isa_anchors=16,
                      num_extra_dims=7, num_local_groups=16, local_rank=4, num_tulpas=4,
                      num_dream_anchors=16)
    cortex_kwargs = dict(n_heads=12, num_anchors=16, num_groups=16, local_rank=4,
                         refinement=2, top_k=2)

    results = []
    for v in args.variants.split(","):
        v = v.strip()
        print(f"\n=== 变体 {v} ===", flush=True)
        kw = cfg_kwargs if v == "full" else (cortex_kwargs if v == "cortex" else {})
        try:
            results.append(run(v, train, valid, kw, args))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"variant": v, "error": f"{type(e).__name__}: {e}"})

    print("\n=== 汇总 ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['variant']}: FAILED {r['error']}", flush=True)
        else:
            print(f"  {r['variant']:8s} held-out acc = {r['final_val_acc']:.4f} | "
                  f"可训练参数 {r['trainable_params'] / 1e6:.2f}M | {r['minutes']:.1f} min", flush=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print("saved", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
