#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渲染器：暗色数据报告风（基准 moe_mechanism_full.mp4）

近黑底 / 左上标题 / 每章一行场景说明 / 主体全是真实数据图表 /
等宽琥珀色标公式 / 底部结论横幅 / 最底下章节进度条。无装饰图形。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out")
FRAMES = os.path.join(OUTDIR, "frames")
DATA = os.path.join(HERE, "data.npz")
os.makedirs(FRAMES, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---- 注册中文字体（matplotlib 默认不索引这些路径）----
_FONT_CANDS = [
    "/workspace/fonts/NotoSansSC-Regular.ttf",
    "/workspace/fonts/NotoSansSC-Bold.ttf",
    "/kaggle/working/fonts/NotoSansSC-Regular.otf",
    "/kaggle/working/fonts/NotoSansSC-Regular.ttf",
    "/root/.fonts/NotoSansSC-VF.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for _p in _FONT_CANDS:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
        except Exception:  # noqa: BLE001
            pass
_have_cn = any("Noto Sans SC" == f.name or "Noto Sans CJK SC" == f.name
               for f in font_manager.fontManager.ttflist)
if not _have_cn:  # 让 fc-match 兜底
    try:
        import subprocess as _sp

        out = _sp.run(["fc-match", "-f", "%{file}", "sans-serif:lang=zh"],
                      capture_output=True, text=True, timeout=20).stdout.strip()
        if out and os.path.exists(out):
            font_manager.fontManager.addfont(out)
            _have_cn = True
    except Exception:  # noqa: BLE001
        pass
if not _have_cn:
    raise RuntimeError(
        "找不到中文字体：请把 NotoSansSC-Regular.ttf 放到 /workspace/fonts/ 或 "
        "/kaggle/working/fonts/，或先 `apt-get install fonts-noto-cjk`")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------ 风格 ------
BG = "#0a0c10"
PANEL = "#0e1218"
EDGE = "#1b222c"
FG = "#f4f6fa"
DIM = "#8a93a6"
FAINT = "#4b5462"
AMBER = "#e2a33c"
MONO = "DejaVu Sans Mono"
CN = "Noto Sans SC"
W_IN, H_IN, DPI, FPS = 19.2, 12.0, 100, 30

CH = [  # (编号字符, 章名, 场景说明, 强调色, 秒)
    ("①", "真空基底", "64 个真空锚点替代 L² 全交互：O(L·R·D)，零距离衰减", "#6366f1", 7.0),
    ("②", "物质起源", "正反物质对消之后，剩余净重子不对称穿过", "#3b82f6", 6.0),
    ("③", "时空几何", "DBI 阻尼随张力增强：越拉越压", "#06b6d4", 6.0),
    ("④", "皮质下", "Pyragas 闭环镇定混沌，G_shut 闸门常态全开", "#10b981", 7.0),
    ("⑤", "三柱 MoA", "top-k 之外，权重与梯度被精确归零", "#f59e0b", 9.0),
    ("⑥", "代理与睡眠", "弱印迹被类淋巴清除，强印迹保留", "#a855f7", 8.0),
    ("⑦", "两种机制", "扰动算子 j，算子 i 输出逐位不变", "#ef4444", 9.0),
    ("⑧", "真实代价", "T4 实测：每阶段前向 + 反向时延", "#94a3b8", 7.0),
]
LABELS = [c[0] + " " + c[1] for c in CH]


def cmap_for(hexcolor):
    cm = LinearSegmentedColormap.from_list("c", [BG, hexcolor])
    cm.set_bad(BG)
    return cm


def new_fig():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def header(fig, i, color, params):
    no, name, desc, _c, _s = CH[i]
    fig.text(0.026, 0.963, "六阶十一域混合架构", fontsize=23, color=FG,
             weight="bold", family=CN)
    fig.text(0.026, 0.936, f"第 {i + 1} 章 / 8 · {name}", fontsize=12, color=DIM, family=CN)
    fig.text(0.026, 0.893, f"{no}  {desc}", fontsize=16.5, color=color,
             weight="bold", family=CN)
    fig.text(0.974, 0.958, params, fontsize=10.5, color=FAINT, family=CN, ha="right")
    fig.text(0.974, 0.936, "真实前向 / 反向 / 计时", fontsize=10, color=FAINT,
             family=CN, ha="right")


def banner(fig, text, color):
    ax = fig.add_axes([0.026, 0.034, 0.948, 0.058])
    ax.set_facecolor("none")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.004,rounding_size=0.03",
                                fc="#11151c", ec=color, lw=1.3, transform=ax.transAxes,
                                clip_on=False))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16.5, color=FG,
            weight="bold", family=CN)


def progress(fig, active, color):
    ax = fig.add_axes([0.026, 0.010, 0.948, 0.016])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(CH)
    for i in range(n):
        x0 = i / n
        w = 1 / n - 0.005
        c = color if i == active else "#151a21"
        ax.add_patch(Rectangle((x0, 0.30), w, 0.42, fc=c, ec="none"))
        ax.text(x0 + w / 2, -0.25, LABELS[i], ha="center", va="top", fontsize=9,
                color=(color if i == active else "#333b47"), family=CN)


def panel(fig, rect, caption=None, cap="top"):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(EDGE)
        s.set_linewidth(1.0)
    ax.tick_params(colors=FAINT, labelsize=8.5, length=2)
    if caption:
        if cap == "top":
            ax.text(0.0, 1.025, caption, transform=ax.transAxes, fontsize=11,
                    color=DIM, family=CN)
        else:
            ax.text(0.0, -0.06, caption, transform=ax.transAxes, fontsize=11,
                    color=DIM, family=CN, va="top")
    return ax


def heat(ax, M, color, vmin=None, vmax=None, reveal=None, values=False, fmt="{:.2f}",
         thresh=0.55):
    M = np.asarray(M, dtype=float)
    if reveal is not None:
        k = int(round(reveal * M.shape[1]))
        M = M.copy()
        M[:, k:] = np.nan
    cm = cmap_for(color)
    ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    if values and M.shape[0] <= 13 and M.shape[1] <= 13:
        hi = vmax if vmax is not None else np.nanmax(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.6,
                        color=(BG if v > hi * thresh else FG), family=MONO)
    return ax


def bars(ax, vals, color, labels=None, frac=1.0, fmt="{:.2f}", horizontal=False):
    vals = np.asarray(vals, dtype=float) * frac
    idx = np.arange(len(vals))
    if horizontal:
        ax.barh(idx, vals, color=color, height=0.6)
        ax.set_yticks(idx)
        if labels:
            ax.set_yticklabels(labels, fontsize=8, color=DIM, family=CN)
        ax.invert_yaxis()
    else:
        ax.bar(idx, vals, color=color, width=0.62)
        ax.set_xticks(idx)
        if labels:
            ax.set_xticklabels(labels, fontsize=8, color=DIM, family=CN)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.tick_params(colors=FAINT, labelsize=8, length=2)
    return ax


def curve(ax, xs, ys, color, frac=1.0, lw=2.4, label=None, marker=False):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = max(2, int(round(frac * len(xs))))
    ax.plot(xs[:n], ys[:n], color=color, lw=lw, label=label,
            marker="o" if marker and n < 30 else None, ms=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.tick_params(colors=FAINT, labelsize=8.5, length=2)
    if label:
        ax.legend(facecolor=PANEL, edgecolor=EDGE, labelcolor=DIM, fontsize=9)
    return ax


def bignum(ax, value, label, color, frac=1.0, fmt="{:.4f}", size=40, sub=None):
    ax.axis("off")
    txt = fmt.format(value * frac) if frac < 1.0 else fmt.format(value)
    ax.text(0.5, 0.62, txt, ha="center", va="center", fontsize=size, color=color,
            weight="bold", family=MONO)
    ax.text(0.5, 0.24, label, ha="center", va="center", fontsize=12, color=DIM, family=CN)
    if sub:
        ax.text(0.5, 0.08, sub, ha="center", va="center", fontsize=10, color=FAINT,
                family=CN)


def mono(ax, s, y=0.5, x=0.5, size=11.5, color=AMBER, ha="center", fam=None):
    ax.axis("off")
    if fam is None:
        # 纯 ASCII/公式用等宽；含中文的混排用中文字体（DejaVu Mono 无 CJK 字形）
        fam = MONO if all(ord(ch) < 0x2E80 for ch in s) else CN
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, family=fam)


# ================================================================== 场景 ======
def sc_open(fig, t, d):
    a = min(1.0, t / 0.4)
    fig.text(0.5, 0.555, "六阶十一域混合架构", ha="center", fontsize=44, color=FG,
             weight="bold", family=CN, alpha=a)
    fig.text(0.5, 0.465, "6 阶段 × 12 算子 × 2 机制", ha="center", fontsize=16,
             color=DIM, family=CN, alpha=a)
    note = "算子来自 hybrid_moa_vl_operator_suite 本体"
    if "gpu_name" in d:
        note += f" · {str(d['gpu_name'][0])} 实测"
    fig.text(0.5, 0.395, note, ha="center", fontsize=11.5, color="#7d6836",
             family=CN, alpha=a)


def sc_end(fig, t, d):
    a = min(1.0, t / 0.4)
    fig.text(0.5, 0.555, "机制", ha="center", fontsize=44, color=FG, weight="bold",
             family=CN, alpha=a)
    fig.text(0.5, 0.465, "6 阶段 × 12 算子 × 2 机制", ha="center", fontsize=16,
             color=DIM, family=CN, alpha=a)
    fig.text(0.5, 0.395, "全部数值来自真实前向、反向与时延测量", ha="center",
             fontsize=11.5, color="#7d6836", family=CN, alpha=a)
    banner(fig, "Stage 0 真空 → Stage V 睡眠：一条前向，走完六个尺度", "#ef4444")


def sc1(fig, t, d):
    c = CH[0][3]
    header(fig, 0, c, f"B={d['np_x_in'].shape[0]} · L={d['np_x_in'].shape[1]} · D={d['np_x_in'].shape[2]}")
    ax = panel(fig, [0.026, 0.44, 0.60, 0.36],
               "attn_abs [B=0] · 每个 token 对全部真空锚点的读出（逐列点亮）")
    heat(ax, d["vac_attn_abs"][0], c, reveal=max(0.02, t), vmin=0)
    ax2 = panel(fig, [0.66, 0.44, 0.314, 0.36], "Bogoliubov 挤压：q,p → q_sq,p_sq")
    q, p = d["zpe_q"][0].ravel(), d["zpe_p"][0].ravel()
    r = float(d["zpe_r_squeeze"][0])
    qs = q * np.cosh(r) + p * np.sinh(r)
    ps = p * np.cosh(r) + q * np.sinh(r)
    n = max(20, int(t * len(q)))
    ax2.scatter(q[:n], p[:n], s=3, color=FAINT, alpha=0.5, label="挤压前")
    ax2.scatter(qs[:n], ps[:n], s=3, color=c, alpha=0.8, label="挤压后")
    ax2.legend(facecolor=PANEL, edgecolor=EDGE, labelcolor=DIM, fontsize=8.5, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax3 = panel(fig, [0.026, 0.16, 0.60, 0.24],
                "E_vac 逐通道最小值：q² + p² + ω² + ε，永不低于零点能下限")
    e = d["zpe_q"] ** 2 + d["zpe_p"] ** 2 + d["zpe_omega"][None, None, :] ** 2 + 0.05
    mn, mx = e.min(axis=(0, 1)), e.max(axis=(0, 1))
    xs = np.arange(len(mn))
    ax3.fill_between(xs, mn, mx, color=c, alpha=0.25)
    ax3.plot(xs, mn, color=c, lw=2.0)
    ax3.axhline(0.05, color=AMBER, lw=1.4, ls="--")
    ax3.text(len(mn) * 0.99, 0.05, " ε = 0.05 下限", color=AMBER, fontsize=9,
             family=MONO, ha="right", va="bottom")
    ax3.set_xlabel("channel", color=FAINT, fontsize=9)
    ax4 = panel(fig, [0.66, 0.16, 0.314, 0.24], "锚点数 R_vac")
    bignum(ax4, float(d["vac_attn_abs"].shape[2]), "个共享潜锚点", c, fmt="{:.0f}", size=44)
    banner(fig, f"{d['vac_attn_abs'].shape[2]} 个锚点把交互从 O(L²D) 压到 O(L·R·D)，零距离衰减", c)
    progress(fig, 0, c)


def sc2(fig, t, d):
    c = CH[1][3]
    header(fig, 1, c, f"B={d['np_x_in'].shape[0]} · L={d['np_x_in'].shape[1]}")
    ax = panel(fig, [0.026, 0.44, 0.475, 0.36], "baryon_out [B=0] · 重子生成阶段输出")
    heat(ax, d["np_baryon_out"][0], c, reveal=max(0.02, t))
    ax2 = panel(fig, [0.53, 0.44, 0.444, 0.36], "逐通道净不对称（对 L 取均值）")
    net = d["np_baryon_out"].mean(axis=(0, 1))
    bars(ax2, np.abs(net), c, frac=min(1.0, t * 1.4))
    ax2.set_xlabel("channel", color=FAINT, fontsize=9)
    ax3 = panel(fig, [0.026, 0.16, 0.30, 0.24], "Stage 0.A → 0.B 输出范数")
    mono(ax3, f"‖x_matter‖ = {np.linalg.norm(d['np_x_matter']):.1f}\n"
              f"‖baryon_out‖ = {np.linalg.norm(d['np_baryon_out']):.1f}", y=0.5, size=13)
    ax4 = panel(fig, [0.35, 0.16, 0.30, 0.24], "输入 → 输出范数比")
    bignum(ax4, float(np.linalg.norm(d["np_baryon_out"]) / np.linalg.norm(d["np_x_matter"])),
           "输出 / 输入", c, fmt="{:.3f}", size=40)
    ax5 = panel(fig, [0.674, 0.16, 0.30, 0.24], "净不对称均值")
    bignum(ax5, float(d.get("diag_stage0b_baryon_asymmetry", np.abs(net).mean())),
           "stage0b_baryon_asymmetry", c, fmt="{:.4f}", size=40)
    banner(fig, "绝大部分对消，剩余一股净重子不对称穿过闸门", c)
    progress(fig, 1, c)


def sc3(fig, t, d):
    c = CH[2][3]
    header(fig, 2, c, f"L={d['np_x_in'].shape[1]} · D={d['np_x_in'].shape[2]}")
    ax = panel(fig, [0.026, 0.44, 0.475, 0.36], "brane_out [B=0] · 超膜紧致化后")
    heat(ax, d["np_brane_out"][0], c, reveal=max(0.02, t))
    ax2 = panel(fig, [0.53, 0.44, 0.444, 0.36], "KK 紧致化模态塔（w_kk × ω_kk）")
    kk = np.asarray(d["kk_w"]) * np.asarray(d["kk_omega"])
    bars(ax2, kk, c, labels=[f"m{i}" for i in range(len(kk))], frac=min(1.0, t * 1.5))
    ax3 = panel(fig, [0.026, 0.16, 0.475, 0.24], "x_spacetime → brane_out：DBI 阻尼后的压缩")
    a_, b_ = np.linalg.norm(d["np_x_spacetime"]), np.linalg.norm(d["np_brane_out"])
    bars(ax3, [a_, b_], [c, FAINT], labels=["输入", "输出"], frac=min(1.0, t * 1.4))
    mono(ax3, f"‖in‖={a_:.1f}  →  ‖out‖={b_:.1f}   压缩 {b_ / a_:.3f}×", y=0.06, x=0.02,
         size=12, ha="left")
    ax4 = panel(fig, [0.53, 0.16, 0.444, 0.24], "Stage I.B 量子潮汐 WKB 隧穿率")
    bignum(ax4, float(d.get("diag_stage1b_quantum_wkb_tunneling", 0.0)), "quantum_wkb_tunneling",
           c, fmt="{:.4f}", size=40)
    banner(fig, "张力越大，DBI 阻尼越把膜振压住；潮汐只让一部分概率隧穿", c)
    progress(fig, 2, c)


def sc4(fig, t, d):
    c = CH[3][3]
    header(fig, 3, c, f"L={d['np_x_in'].shape[1]} · D={d['np_x_in'].shape[2]}")
    ax = panel(fig, [0.026, 0.44, 0.44, 0.36], "spinal_out [B=0] · 脊髓背角门控输出")
    heat(ax, d["np_spinal_out"][0], c, reveal=max(0.02, t))
    ax2 = panel(fig, [0.50, 0.44, 0.474, 0.36],
                "皮质脊髓混沌：输入幅度扫 48 倍，输出幅度几乎不变")
    xs, ys = d["bound_in_chaos"], d["bound_out_chaos"]
    n = max(2, int(round(t * len(xs))))
    ax2.plot(xs[:n], ys[:n], color=c, lw=2.6, marker="o", ms=4, label="输出范数")
    ax2.plot(xs[:n], xs[:n], color=FAINT, lw=1.4, ls="--", label="等比参考线")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(facecolor=PANEL, edgecolor=EDGE, labelcolor=DIM, fontsize=9, loc="upper left")
    ax2.set_xlabel("输入范数（log）", color=FAINT, fontsize=9)
    ax3 = panel(fig, [0.026, 0.16, 0.30, 0.24], "G_shut 闸门（常态）")
    bignum(ax3, float(d["np_g_shut_gate"].mean()), "mean(G_shut)", c, fmt="{:.4f}", size=40)
    ax4 = panel(fig, [0.35, 0.16, 0.30, 0.24], "G_shut 闸门（+50 冲击）")
    bignum(ax4, float(d["np_g_shut_shock"][:, 0, :].mean()), "mean(G_shut | shock)",
           "#ef4444", fmt="{:.2e}", size=34)
    ax5 = panel(fig, [0.674, 0.16, 0.30, 0.24], "创伤通路被挡在认知流之外")
    mono(ax5, "trauma_out 只从 (1-G_shut) 进输出\n认知阶段对创伤参数零灵敏", y=0.5, size=12)
    banner(fig, "混沌被 Pyragas 闭环压住；常态闸门全开，冲击下瞬间切换", c)
    progress(fig, 3, c)


def sc5(fig, t, d):
    c = CH[4][3]
    header(fig, 4, c, f"L={d['np_proposals_raw'].shape[2]} · D={d['np_proposals_raw'].shape[3]}")
    names = ["Col0 跨模态 VL 双线性", "Col1 锥体树突 NMDA-FHN", "Col2 突触快权重 DeltaOja"]
    for i in range(3):
        ax = panel(fig, [0.026 + i * 0.318, 0.42, 0.30, 0.38], names[i])
        heat(ax, d["np_proposals_raw"][0, i], c, reveal=max(0.02, t))
    ax = panel(fig, [0.026, 0.16, 0.30, 0.22], "router_logits [B=0] · 逐 token 三柱得分")
    heat(ax, d["np_router_logits"][0], c, values=True, fmt="{:.1f}")
    ax2 = panel(fig, [0.35, 0.16, 0.30, 0.22], "top-k 之后：未选中精确归零")
    aw = np.asarray(d["np_active_weights"])[0].copy()
    heat(ax2, aw, c, values=True, fmt="{:.2f}")
    ax3 = panel(fig, [0.674, 0.16, 0.30, 0.22], "专家负载 f_i 与均衡系数")
    fi = np.asarray(d["np_f_i"])
    bars(ax3, fi, c, labels=["C0", "C1", "C2"], frac=min(1.0, t * 1.4))
    bal = 1.0 / (len(fi) * float(np.sum(fi ** 2))) if float(np.sum(fi ** 2)) > 0 else 0.0
    mono(ax3, f"均衡系数 = {bal:.4f}", y=0.06, x=0.02, size=12, ha="left")
    banner(fig, "top-k 之外，权重与梯度都被精确归零——不是近似，是逐位为 0", c)
    progress(fig, 4, c)


def sc6(fig, t, d):
    c = CH[5][3]
    header(fig, 5, c, f"R_dream={d['dream_attn_bc'].shape[2]} · D={d['np_x_in'].shape[2]}")
    ax = panel(fig, [0.026, 0.44, 0.44, 0.36], "tulpa_out [B=0] · 多子人格审议输出")
    heat(ax, d["np_tulpa_out"][0], c, reveal=max(0.02, t))
    ax2 = panel(fig, [0.50, 0.44, 0.474, 0.36], "梦境锚点广播 attn_bc [B=0]")
    heat(ax2, d["dream_attn_bc"][0], c, reveal=max(0.02, t))
    ax3 = panel(fig, [0.026, 0.16, 0.30, 0.24], "SHY 突触稳态下调 inv_shy 均值")
    bignum(ax3, float(d["sleep_inv_shy"].mean()), "mean(inv_shy)", c, fmt="{:.4f}", size=38)
    ax4 = panel(fig, [0.35, 0.16, 0.30, 0.24], "类淋巴冲刷滤子 W_glymph [B=0]")
    heat(ax4, d["sleep_w_glymph"][0], c, vmin=0, vmax=1)
    ax5 = panel(fig, [0.674, 0.16, 0.30, 0.24], "冲刷前后印迹强度")
    xw, wg = d["sleep_x_wake"][0], d["sleep_w_glymph"][0]
    before = float(np.linalg.norm(xw))
    after = float(np.linalg.norm(xw * wg))
    bars(ax5, [before, after], [FAINT, c], labels=["冲刷前", "冲刷后"], frac=min(1.0, t * 1.4))
    mono(ax5, f"保留 {after / before:.3f}×", y=0.06, x=0.02, size=12, ha="left")
    banner(fig, "SHY 先整体压暗，类淋巴再清掉弱印迹——强印迹保留", c)
    progress(fig, 5, c)


def sc7(fig, t, d):
    c = CH[6][3]
    header(fig, 6, c, "12 个领域算子 × 12 组参数")
    ax = panel(fig, [0.026, 0.40, 0.52, 0.40],
               "参数交叉灵敏矩阵 ‖∂Y_i / ∂Θ_j‖_inf（对角=自灵敏，非对角=串扰）")
    J = np.asarray(d["jacobian"])
    heat(ax, J, c, values=True, fmt="{:.3f}", vmin=0, vmax=max(J.max(), 1e-9))
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels([str(i + 1) for i in range(12)], fontsize=8, color=DIM)
    ax.set_yticklabels([str(i + 1) for i in range(12)], fontsize=8, color=DIM)
    ax2 = panel(fig, [0.57, 0.40, 0.424, 0.185], "非对角交叉干扰")
    off = float(np.max(J - np.diag(np.diag(J))))
    bignum(ax2, off, "max off-diagonal", "#ef4444", fmt="{:.2e}", size=36)
    ax3 = panel(fig, [0.57, 0.19, 0.424, 0.185], "G_shut：常态 vs +50 冲击")
    g1 = float(d["np_g_shut_gate"].mean())
    g2 = float(d["np_g_shut_shock"][:, 0, :].mean())
    bars(ax3, [g1, g2], [c, "#ef4444"], labels=[f"常态 {g1:.4f}", f"冲击 {g2:.1e}"],
         frac=min(1.0, t * 1.4), horizontal=True)
    ax4 = panel(fig, [0.026, 0.16, 0.968, 0.20], "")
    mono(ax4, "扰动算子 j 的参数 → 算子 i 的输出逐位不变（非对角精确 0）\n"
              "常态走稳态认知相，冲击挂起认知相、切到 DPDR 缓冲相",
         y=0.5, size=14)
    banner(fig, f"非对角精确 {off:.2e}：这是「异构算子柱」可测的证据，不是修辞", c)
    progress(fig, 6, c)


def sc8(fig, t, d):
    c = CH[7][3]
    dev = str(d["gpu_name"][0]) if "gpu_name" in d else "CPU"
    header(fig, 7, c, f"{dev} · 前向 + 反向 · 5 次平均")
    ax = panel(fig, [0.026, 0.44, 0.62, 0.36], "每阶段前向 + 反向时延（ms）")
    ms = np.asarray(d["timing_ms"])
    bars(ax, ms, c, labels=[f"{i + 1}" for i in range(len(ms))], frac=min(1.0, t * 1.4))
    ax.set_xlabel("算子编号（1=真空 … 12=睡眠）", color=FAINT, fontsize=9)
    ax2 = panel(fig, [0.674, 0.44, 0.30, 0.36], "每阶段参数量（M）")
    par = np.asarray(d["timing_params"]) / 1e6
    bars(ax2, par, c, labels=[f"{i + 1}" for i in range(len(par))], frac=min(1.0, t * 1.4))
    ax3 = panel(fig, [0.026, 0.16, 0.30, 0.24], "12 阶段总时延")
    bignum(ax3, float(ms.sum()), "ms / step", c, fmt="{:.1f}", size=40)
    ax4 = panel(fig, [0.35, 0.16, 0.30, 0.24], "12 阶段总参数")
    bignum(ax4, float(np.asarray(d["timing_params"]).sum() / 1e6), "M params", c,
           fmt="{:.2f}", size=40)
    ax5 = panel(fig, [0.674, 0.16, 0.30, 0.24], "最慢 / 最快阶段")
    i_s, i_f = int(np.argmax(ms)), int(np.argmin(ms))
    mono(ax5, f"{i_s + 1} 号 {ms[i_s]:.1f} ms\n{i_f + 1} 号 {ms[i_f]:.1f} ms",
         y=0.5, size=14)
    banner(fig, f"{dev} 实测：12 阶段串行前向 + 反向共 {ms.sum():.1f} ms / step", c)
    progress(fig, 7, c)


SCENES = [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8]


def main():
    d = dict(np.load(DATA))
    total = sum(c[4] for c in CH) + 3.5 + 4.0
    print(f"[render] {len(CH)} 章 + 片头片尾，共 {total:.1f}s @ {FPS}fps")
    segs = []

    def emit(name, dur, fn):
        n = int(round(dur * FPS))
        sub = os.path.join(FRAMES, name)
        os.makedirs(sub, exist_ok=True)
        for i in range(n):
            fig = new_fig()
            fn(fig, i / max(1, n - 1), d)
            fig.savefig(os.path.join(sub, f"{i:04d}.png"), dpi=DPI, facecolor=BG)
            plt.close(fig)
            if i % 30 == 0:
                print(f"    {name}: {i}/{n}", flush=True)
        seg = os.path.join(OUTDIR, f"{name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                        "-i", os.path.join(sub, "%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", seg], check=True)
        segs.append(seg)
        print(f"    {name}: {n} 帧 -> {seg}", flush=True)

    emit("s0_open", 3.5, sc_open)
    for i, fn in enumerate(SCENES):
        emit(f"s{i + 1}", CH[i][4], fn)
    emit("s9_end", 4.0, sc_end)

    lst = os.path.join(OUTDIR, "list.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{os.path.basename(s)}'\n")
    final = os.path.join(OUTDIR, "v6_mechanism.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", final], check=True)
    sz = os.path.getsize(final) / 1e6
    print(f"[render] done -> {final}  ({sz:.2f} MB, {total:.1f}s)")


if __name__ == "__main__":
    main()
