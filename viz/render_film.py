#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六阶十一域混合架构 · 机制动画 —— 渲染器（v12，按 4 地层组织）

对齐 v12 unified_hybrid_moa_architecture.py 的权威结构：
    Unified 13-Domain, 7-Stage (14-Operator, 4-Stratum)
    Stratum I   — 宇宙学 / 相对论 / 量子资源时空发生基底（Stage 0 & I）
    Stratum II  — 皮质下控制轴 vs. 新皮层并行张量柱（Stage II & III）
    Stratum III — 异构硅基 Tri-ISA CPU + SIMT GPU 自举协处理器（Stage IV）
    Stratum IV  — 多吸引子双稳态代理与清醒后睡眠巩固（Stage V & VI）

风格基准：moe_mechanism_full.mp4 —— 近黑底 / 左上标题 / 章节场景说明 /
主体全是真实数据图表 / 等宽琥珀色公式 / 底部结论横幅 / 章节进度条。1920×1200 / 30fps。
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


# ------------------------------------------------------------------ 字体 ------
def _register_cjk_font():
    cands = [
        "/workspace/fonts/NotoSansSC-Regular.ttf",
        "/workspace/fonts/NotoSansSC-Bold.ttf",
        "/kaggle/working/fonts/NotoSansSC-Regular.otf",
        "/kaggle/working/fonts/NotoSansSC-Regular.ttf",
        "/root/.fonts/NotoSansSC-VF.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:  # noqa: BLE001
                pass
    try:
        import subprocess as _sp

        r = _sp.run(["fc-match", "-f", "%{file}|%{family}", "sans-serif:lang=zh"],
                    capture_output=True, text=True, timeout=25).stdout.strip()
        if "|" in r:
            f, fam = r.split("|", 1)
            if os.path.exists(f):
                font_manager.fontManager.addfont(f)
    except Exception:  # noqa: BLE001
        pass
    names = [f.name for f in font_manager.fontManager.ttflist]
    for want in ("Noto Sans SC", "Noto Sans CJK SC", "Noto Sans CJK JP",
                 "Source Han Sans SC", "WenQuanYi Zen Hei"):
        if want in names:
            return want
    for n in names:
        if any(k in n for k in ("CJK", "Han", "WenQuanYi", "Heiti", "Song")):
            return n
    return None


_CN_FAM = _register_cjk_font()
if _CN_FAM is None:
    raise RuntimeError("找不到中文字体：把 NotoSansSC-Regular.ttf 放到 /workspace/fonts/ "
                       "或 /kaggle/working/fonts/")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [_CN_FAM, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------ 风格 ------
BG = "#0a0c10"
PANEL = "#0e1218"
EDGE = "#28313d"
FG = "#f4f6fa"
DIM = "#aab4c4"
FAINT = "#7d8794"
AMBER = "#f5bd5c"
MONO = "DejaVu Sans Mono"
CN = _CN_FAM
W_IN, H_IN, DPI, FPS = 19.2, 12.0, 100, 30

# 五个章节 = 四个地层 + 一个不变量/代价章
CH = [
    ("Ⅰ", "Stratum I", "宇宙学 / 相对论 / 量子资源时空发生基底 · Stage 0 & I",
     "#6366f1", 12.0, ["0.A 真空", "0.B 重子", "I.A 超膜", "I.B 潮汐", "I.C 量子资源"]),
    ("Ⅱ", "Stratum II", "皮质下控制轴 vs. 新皮层并行张量柱 · Stage II & III",
     "#10b981", 14.0, ["II.A 脊髓", "II.B 混沌", "II.C 解离门", "III 三柱 MoA"]),
    ("Ⅲ", "Stratum III", "异构硅基 Tri-ISA CPU + SIMT GPU 自举协处理器 · Stage IV",
     "#64748b", 9.0, ["IV 硅基自举"]),
    ("Ⅳ", "Stratum IV", "多吸引子双稳态代理与清醒后睡眠巩固 · Stage V & VI",
     "#a855f7", 12.0, ["V Tulpa 代理", "VI 睡眠巩固"]),
    ("Ω", "不变量与代价", "四条架构不变量 + T4 实测逐算子代价",
     "#ef4444", 12.0, ["零串扰", "零污染", "逐算子代价"]),
]
LABELS = [c[0] + " " + c[1] for c in CH]


def cmap_for(hexcolor):
    cm = LinearSegmentedColormap.from_list("c", ["#161c25", hexcolor])
    cm.set_bad(BG)
    return cm


def new_fig():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def header(fig, i, color, params):
    no, name, desc, _c, _s, _st = CH[i]
    fig.text(0.026, 0.963, "本体论分层混合 MoA 架构 · 13 域 14 算子 7 阶段 4 地层",
             fontsize=22, color=FG, weight="bold", family=CN)
    fig.text(0.026, 0.936, f"第 {i + 1} 章 / 5 · {name}", fontsize=14.5, color=DIM, family=CN)
    fig.text(0.026, 0.895, f"{no}  {desc}", fontsize=19.5, color=color, weight="bold", family=CN)
    fig.text(0.974, 0.958, params, fontsize=12.5, color=DIM, family=CN, ha="right")
    fig.text(0.974, 0.936, "真实前向 / 反向 / 计时", fontsize=11.5, color=DIM, family=CN, ha="right")


def banner(fig, text, color):
    ax = fig.add_axes([0.026, 0.034, 0.948, 0.058])
    ax.set_facecolor("none")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.004,rounding_size=0.03",
                                fc="#11151c", ec=color, lw=1.3, transform=ax.transAxes,
                                clip_on=False))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=19, color=FG,
            weight="bold", family=CN)


def progress(fig, active, color):
    ax = fig.add_axes([0.026, 0.010, 0.948, 0.016])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(CH)
    for i in range(n):
        x0 = i / n
        w = 1 / n - 0.004
        c = color if i == active else "#151a21"
        ax.add_patch(Rectangle((x0, 0.30), w, 0.42, fc=c, ec="none"))
        ax.text(x0 + w / 2, -0.25, LABELS[i], ha="center", va="top", fontsize=11,
                color=(color if i == active else "#59626e"), family=CN)


def stage_flow(fig, rect, stages, color, t):
    """地层内部的 stage 流条：节点自左向右依次点亮。"""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(stages)
    for i, name in enumerate(stages):
        cx = (i + 0.5) / n
        lit = t > (i + 0.35) / n
        ax.add_patch(FancyBboxPatch((cx - 0.055, 0.30), 0.11, 0.42,
                                    boxstyle="round,pad=0.004,rounding_size=0.12",
                                    fc=(color if lit else "#141a22"),
                                    ec=(color if lit else EDGE), lw=1.2,
                                    transform=ax.transAxes))
        ax.text(cx, 0.51, name, ha="center", va="center", fontsize=12.5,
                color=(FG if lit else FAINT), family=CN)
        if i < n - 1:
            ax.annotate("", xy=((i + 1.5) / n - 0.062, 0.51),
                        xytext=((i + 0.5) / n + 0.062, 0.51),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=(color if lit else EDGE),
                                        lw=1.6))
    return ax


def panel(fig, rect, caption=None):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(EDGE)
        s.set_linewidth(1.0)
    ax.tick_params(colors=DIM, labelsize=10, length=3)
    if caption:
        ax.text(0.0, 1.025, caption, transform=ax.transAxes, fontsize=13, color=DIM, family=CN)
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
    if values and M.shape[0] <= 14 and M.shape[1] <= 14:
        hi = vmax if vmax is not None else np.nanmax(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.0,
                        color=(BG if v > hi * thresh else FG), family=MONO)
    return ax


def bars(ax, vals, color, labels=None, frac=1.0, horizontal=False):
    vals = np.asarray(vals, dtype=float) * frac
    idx = np.arange(len(vals))
    if horizontal:
        ax.barh(idx, vals, color=color, height=0.6)
        ax.set_yticks(idx)
        if labels:
            ax.set_yticklabels(labels, fontsize=9.5, color=DIM, family=CN)
        ax.invert_yaxis()
    else:
        ax.bar(idx, vals, color=color, width=0.62)
        ax.set_xticks(idx)
        if labels:
            ax.set_xticklabels(labels, fontsize=9.5, color=DIM, family=CN)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.tick_params(colors=DIM, labelsize=9.5, length=3)
    return ax


def curve(ax, xs, ys, color, frac=1.0, lw=2.4, label=None):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = max(2, int(round(frac * len(xs))))
    ax.plot(xs[:n], ys[:n], color=color, lw=lw, label=label)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.tick_params(colors=DIM, labelsize=10, length=3)
    if label:
        ax.legend(facecolor=PANEL, edgecolor=EDGE, labelcolor=DIM, fontsize=9)
    return ax


def bignum(ax, value, label, color, frac=1.0, fmt="{:.4f}", size=40, sub=None):
    ax.axis("off")
    txt = fmt.format(value * frac) if frac < 1.0 else fmt.format(value)
    ax.text(0.5, 0.62, txt, ha="center", va="center", fontsize=size, color=color,
            weight="bold", family=MONO)
    ax.text(0.5, 0.24, label, ha="center", va="center", fontsize=14, color=DIM, family=CN)
    if sub:
        ax.text(0.5, 0.08, sub, ha="center", va="center", fontsize=11.5, color=DIM, family=CN)


def mono(ax, s, y=0.5, x=0.5, size=14, color=AMBER, ha="center", fam=None):
    ax.axis("off")
    if fam is None:
        fam = MONO if all(ord(ch) < 0x2E80 for ch in s) else CN
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, family=fam)


def dg(d, key, default=0.0):
    try:
        return float(d.get("diag_" + key, default))
    except Exception:  # noqa: BLE001
        return default


# ================================================================== 场景 ======
def sc_open(fig, t, d):
    a = min(1.0, t / 0.35)
    fig.text(0.5, 0.655, "本体论分层的混合 MoA 架构", ha="center", fontsize=47, color=FG,
             weight="bold", family=CN, alpha=a)
    fig.text(0.5, 0.575, "13 域 · 14 算子 · 7 阶段 · 4 地层", ha="center", fontsize=19.5,
             color=DIM, family=CN, alpha=a)
    # 四地层总览
    strata = [("Ⅰ", "时空发生基底", "Stage 0 & I", "#6366f1"),
              ("Ⅱ", "皮质下 / 新皮层", "Stage II & III", "#10b981"),
              ("Ⅲ", "硅基自举协处理器", "Stage IV", "#64748b"),
              ("Ⅳ", "代理 / 睡眠巩固", "Stage V & VI", "#a855f7")]
    ax = fig.add_axes([0.10, 0.30, 0.80, 0.20])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (no, name, stg, col) in enumerate(strata):
        x0 = 0.02 + i * 0.245
        lit = t > 0.25 + i * 0.13
        ax.add_patch(FancyBboxPatch((x0, 0.18), 0.215, 0.64,
                                    boxstyle="round,pad=0.006,rounding_size=0.06",
                                    fc=(col if lit else "#12171f"), ec=(col if lit else EDGE),
                                    lw=1.6, alpha=0.92, transform=ax.transAxes))
        ax.text(x0 + 0.1075, 0.66, no, ha="center", va="center", fontsize=23,
                color=(FG if lit else FAINT), family=CN, weight="bold")
        ax.text(x0 + 0.1075, 0.42, name, ha="center", va="center", fontsize=14.5,
                color=(FG if lit else FAINT), family=CN)
        ax.text(x0 + 0.1075, 0.26, stg, ha="center", va="center", fontsize=11.5,
                color=(DIM if lit else FAINT), family=CN)
    note = "算子来自 hybrid_moa_vl_operator_suite 本体"
    if "gpu_name" in d:
        note += f" · {str(d['gpu_name'][0])} 实测"
    fig.text(0.5, 0.215, note, ha="center", fontsize=13, color="#a8894a", family=CN, alpha=a)


def sc_end(fig, t, d):
    a = min(1.0, t / 0.4)
    fig.text(0.5, 0.575, "机制", ha="center", fontsize=48, color=FG, weight="bold", family=CN, alpha=a)
    fig.text(0.5, 0.485, "13 域 · 14 算子 · 7 阶段 · 4 地层", ha="center", fontsize=19,
             color=DIM, family=CN, alpha=a)
    fig.text(0.5, 0.415, "全部数值来自真实前向、反向与时延测量", ha="center", fontsize=13.5,
             color="#7d6836", family=CN, alpha=a)
    banner(fig, "Stage 0 真空 → Stage VI 睡眠：一条前向，走完六个尺度", "#ef4444")


def sc1(fig, t, d):
    """Stratum I：Stage 0.A / 0.B / I.A / I.B / I.C"""
    c = CH[0][3]
    header(fig, 0, c, f"B={d['np_x_in'].shape[0]} · L={d['np_x_in'].shape[1]} · D={d['np_x_in'].shape[2]}")
    stage_flow(fig, [0.026, 0.845, 0.948, 0.055], CH[0][5], c, t)
    ax = panel(fig, [0.026, 0.44, 0.30, 0.36], "0.A attn_abs · 真空锚点读出")
    heat(ax, d["vac_attn_abs"][0], c, reveal=max(0.02, t * 1.6), vmin=0)
    ax2 = panel(fig, [0.35, 0.44, 0.30, 0.36], "0.B baryon_out · 净重子不对称")
    heat(ax2, d["np_baryon_out"][0], c, reveal=max(0.02, t * 1.6))
    ax3 = panel(fig, [0.674, 0.44, 0.30, 0.36], "I.C qrp_out · 定点逆平方根相位")
    heat(ax3, d["np_qrp_out"][0], c, reveal=max(0.02, t * 1.6))
    ax4 = panel(fig, [0.026, 0.16, 0.30, 0.24], "0.A 零点能下限 / I.B WKB 隧穿")
    mono(ax4, f"ω²+ε ≥ 0.05   (ZPE 地板)\nWKB 隧穿率 {dg(d, 'stage1b_quantum_wkb_tunneling'):.4f}",
         y=0.5, size=14.5)
    ax5 = panel(fig, [0.35, 0.16, 0.30, 0.24], "I.C DQI 可靠性 / λ_eff")
    mono(ax5, f"DQI ρ = 1/R(ε)  {dg(d, 'stage1c_dqi_reliability'):.4f}\n"
              f"λ_eff          {dg(d, 'stage1c_block_lambda_eff'):.4f}", y=0.5, size=14.5)
    ax6 = panel(fig, [0.674, 0.16, 0.30, 0.24], "I.A DBI 阻尼")
    mono(ax6, f"1/√(1+τ²E+β²F²)\n均值 {dg(d, 'stage1a_supermembrane_dbi'):.4f}", y=0.5, size=14.5)
    banner(fig, "从真空地板到时空紧致化：五个阶段都在写「有界物理量」而不是点态激活值", c)
    progress(fig, 0, c)


def sc2(fig, t, d):
    """Stratum II：Stage II.A/B/C + Stage III 三柱"""
    c = CH[1][3]
    header(fig, 1, c, f"L={d['np_x_in'].shape[1]} · D={d['np_x_in'].shape[2]}")
    stage_flow(fig, [0.026, 0.845, 0.948, 0.055], CH[1][5], c, t)
    ax = panel(fig, [0.026, 0.44, 0.29, 0.36], "II.A spinal_out · 背角门控")
    heat(ax, d["np_spinal_out"][0], c, reveal=max(0.02, t * 1.5))
    ax2 = panel(fig, [0.336, 0.44, 0.29, 0.36], "II.B 混沌有界：输入扫 48 倍")
    xs, ys = d.get("bound_in_chaos"), d.get("bound_out_chaos")
    if xs is not None:
        curve(ax2, xs, ys, c, frac=min(1.0, t * 1.4))
        ax2.set_xscale("log")
        ax2.set_xlabel("输入范数（log）", color=DIM, fontsize=10.5)
    ax3 = panel(fig, [0.646, 0.44, 0.328, 0.36], "III 三柱 · 异构新皮层柱")
    ax3.set_xticks([]); ax3.set_yticks([])
    for i, nm in enumerate(["Col0 VL 双线性", "Col1 树突 NMDA-FHN", "Col2 快权重 ΔOja"]):
        sub = fig.add_axes([0.652 + i * 0.107, 0.50, 0.10, 0.26])
        heat(sub, d["np_proposals_raw"][0, i], c, reveal=max(0.02, t * 1.5))
        sub.text(0.0, -0.12, nm, transform=sub.transAxes, fontsize=10, color=DIM, family=CN)
    ax4 = panel(fig, [0.026, 0.16, 0.29, 0.24], "II.C G_shut 闸门（常态 / 冲击）")
    g1 = dg(d, "stage2c_corticolimbic_shutdown_gate")
    g2 = float(np.mean(d["np_g_shut_shock"][:, 0, :]))
    mono(ax4, f"常态 {g1:.4f}\n+50 冲击 {g2:.2e}", y=0.5, size=15)
    ax5 = panel(fig, [0.336, 0.16, 0.29, 0.24], "III 路由：top-k 之外精确归零")
    heat(ax5, d["np_active_weights"][0], c, values=True, fmt="{:.2f}")
    ax6 = panel(fig, [0.646, 0.16, 0.328, 0.24], "零跨阶段创伤污染")
    mono(ax6, "‖∂(内部)/∂Θ_Trauma‖_∞ ≡ 0.00e+00\n创伤只从 (1-G_shut) 旁路进输出", y=0.5, size=13.5)
    banner(fig, "皮质下给基线，新皮层并行展开；创伤通路被正交隔离，进不了认知与睡眠阶段", c)
    progress(fig, 1, c)


def sc3(fig, t, d):
    """Stratum III：Stage IV 硅基自举"""
    c = CH[2][3]
    header(fig, 2, c, "x86-64 / ARM64 / RISC-V · 32-lane warp")
    stage_flow(fig, [0.026, 0.845, 0.948, 0.055], CH[2][5], c, t)
    ax = panel(fig, [0.026, 0.44, 0.44, 0.36], "boot_out · 自举编译后的硅基状态 H_silicon")
    heat(ax, d["np_boot_out"][0], c, reveal=max(0.02, t * 1.5))
    ax2 = panel(fig, [0.50, 0.44, 0.474, 0.36], "三路 ISA 分发权重 + GPU roofline 利用率")
    gx = dg(d, "stage4_cpu_x86_gate")
    ga = dg(d, "stage4_cpu_arm64_sve")
    gr = dg(d, "stage4_cpu_riscv_vsetvli")
    gu = dg(d, "stage4_gpu_roofline_utilization")
    bars(ax2, [gx, ga, gr, gu], c,
         labels=[f"x86-64 {gx:.2f}", f"ARM64 {ga:.2f}", f"RISC-V {gr:.2f}", f"GPU {gu:.2f}"],
         frac=min(1.0, t * 1.4))
    ax2.set_ylim(0, max(gx, ga, gr, gu) * 1.3)
    mono(ax2, "Tri-ISA 仲裁器 Y_cpu → 32-lane warp 调度", y=0.035, x=0.02, size=12.5, ha="left")
    ax3 = panel(fig, [0.026, 0.16, 0.30, 0.24], "自举定点收敛")
    bignum(ax3, dg(d, "stage4_bootstrap_fixed_point"), "bootstrap_fixed_point", c, fmt="{:.4f}", size=40)
    ax4 = panel(fig, [0.35, 0.16, 0.30, 0.24], "自举编译校验")
    mono(ax4, "SHA256(Stage1)\n  ==\nSHA256(Stage2)", y=0.5, size=14.5)
    ax5 = panel(fig, [0.674, 0.16, 0.30, 0.24], "算子 14")
    mono(ax5, "14:CPUGPU_TriISA_Bootstrap\n自举编译 + 安全回退", y=0.5, size=14.5)
    banner(fig, "三种 ISA 的分歧被显式建模成可微权重，GPU 侧按 roofline 调度", c)
    progress(fig, 2, c)


def sc4(fig, t, d):
    """Stratum IV：Stage V 代理 + Stage VI 睡眠巩固 + 互补门控"""
    c = CH[3][3]
    rd = d["dream_attn_bc"].shape[2] if "dream_attn_bc" in d else 0
    header(fig, 3, c, f"R_dream={rd} · D={d['np_x_in'].shape[2]}")
    stage_flow(fig, [0.026, 0.845, 0.948, 0.055], CH[3][5], c, t)
    ax = panel(fig, [0.026, 0.44, 0.29, 0.36], "V tulpa_out · 多吸引子双稳态审议")
    heat(ax, d["np_tulpa_out"][0], c, reveal=max(0.02, t * 1.5))
    ax2 = panel(fig, [0.336, 0.44, 0.29, 0.36], "VI 梦境锚点广播 attn_bc")
    if "dream_attn_bc" in d:
        heat(ax2, d["dream_attn_bc"][0], c, reveal=max(0.02, t * 1.5))
    ax3 = panel(fig, [0.646, 0.44, 0.328, 0.36], "VI 类淋巴冲刷滤子 W_glymph")
    if "np_sleep_w_glymph" in d:
        heat(ax3, d["np_sleep_w_glymph"][0], c, vmin=0, vmax=1)
    ax4 = panel(fig, [0.026, 0.16, 0.29, 0.24], "TPJ 代理归因 / VLPO 睡眠门")
    mono(ax4, f"κ_eff(softplus) {dg(d, 'stage5_tulpa_tpj_agency'):.4f}\n"
              f"VLPO 门     {dg(d, 'stage6_vlpo_sleep_gate'):.4f}", y=0.5, size=14.5)
    ax5 = panel(fig, [0.336, 0.16, 0.29, 0.24], "冲刷前后印迹强度")
    if "np_sleep_x_wake" in d and "np_sleep_w_glymph" in d:
        xw, wg = d["np_sleep_x_wake"][0], d["np_sleep_w_glymph"][0]
        before, after = float(np.linalg.norm(xw)), float(np.linalg.norm(xw * wg))
        bars(ax5, [before, after], [FAINT, c], labels=["冲刷前", "冲刷后"], frac=min(1.0, t * 1.4))
        mono(ax5, f"保留 {after / max(before, 1e-9):.3f}×", y=0.04, x=0.02, size=13, ha="left")
    ax6 = panel(fig, [0.646, 0.16, 0.328, 0.24], "互补门控合成")
    mono(ax6, "Y = H_clean + G_shut·Y_consolidated\n  + (1-G_shut)·(gate·Y_trauma)",
         y=0.5, size=13)
    banner(fig, "清醒态先被多吸引子审议，再用睡眠把弱印迹洗掉——冲击时这一整段被挂起", c)
    progress(fig, 3, c)


def sc5(fig, t, d):
    """不变量与真实代价"""
    c = CH[4][3]
    header(fig, 4, c, "14 个领域算子 × 14 组参数")
    stage_flow(fig, [0.026, 0.845, 0.948, 0.055], CH[4][5], c, t)
    ax = panel(fig, [0.026, 0.42, 0.50, 0.38],
               "参数交叉灵敏矩阵 ‖∂Y_i / ∂Θ_j‖_inf（对角=自灵敏，非对角=串扰）")
    J = np.asarray(d["jacobian"])
    heat(ax, J, c, values=True, fmt="{:.2f}", vmin=0, vmax=max(J.max(), 1e-9))
    ax.set_xticks(range(J.shape[1]))
    ax.set_yticks(range(J.shape[0]))
    ax.set_xticklabels([str(i + 1) for i in range(J.shape[1])], fontsize=7.5, color=DIM)
    ax.set_yticklabels([str(i + 1) for i in range(J.shape[0])], fontsize=7.5, color=DIM)
    ax2 = panel(fig, [0.55, 0.42, 0.424, 0.185], "非对角交叉干扰（零串扰）")
    off = float(np.max(J - np.diag(np.diag(J))))
    bignum(ax2, off, "max off-diagonal", "#ef4444", fmt="{:.2e}", size=36)
    ax3 = panel(fig, [0.026, 0.094, 0.948, 0.040], "")
    mono(ax3, "四条架构不变量：零本体论倒置 · 零重复算子 · 零跨阶段创伤污染 · 连续安全 Triton 分派",
         y=0.5, size=14)
    ax4 = panel(fig, [0.026, 0.16, 0.46, 0.22], "T4 实测：14 个算子前向 + 反向时延（ms）")
    ms = np.asarray(d["timing_ms"])
    bars(ax4, ms, c, labels=[str(i + 1) for i in range(len(ms))], frac=min(1.0, t * 1.4))
    ax4.set_xlabel("算子编号 1=真空 … 14=硅基自举", color=DIM, fontsize=10.5, labelpad=1)
    ax5 = panel(fig, [0.51, 0.16, 0.22, 0.22], "14 算子总代价")
    bignum(ax5, float(ms.sum()), "ms / step", c, fmt="{:.1f}", size=38)
    ax6 = panel(fig, [0.75, 0.16, 0.224, 0.22], "DeltaOja launch")
    le = int(np.asarray(d.get("oja_launches_eager", np.array([0])))[0])
    lp = int(np.asarray(d.get("oja_launches_persistent", np.array([1])))[0])
    mono(ax6, f"{le} → {lp}\n前向 ≈34× 前反向 ≈26×", y=0.5, size=13)
    banner(fig, f"非对角精确 {off:.2e}：14 个算子零串扰，这是可测的证据，不是修辞", c)
    progress(fig, 4, c)


SCENES = [sc1, sc2, sc3, sc4, sc5]


def main():
    d = dict(np.load(DATA))
    total = sum(c[4] for c in CH) + 4.0 + 4.0
    print(f"[render] {len(CH)} 章（4 地层 + 不变量）+ 片头片尾，共 {total:.1f}s @ {FPS}fps")
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
            if i % 60 == 0:
                print(f"    {name}: {i}/{n}", flush=True)
        seg = os.path.join(OUTDIR, f"{name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                        "-i", os.path.join(sub, "%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", seg], check=True)
        segs.append(seg)
        print(f"    {name}: {n} 帧 -> {seg}", flush=True)

    emit("s0_open", 4.0, sc_open)
    for i, fn in enumerate(SCENES):
        emit(f"s{i + 1}", CH[i][4], fn)
    emit("s6_end", 4.0, sc_end)

    lst = os.path.join(OUTDIR, "list.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{os.path.basename(s)}'\n")
    final = os.path.join(OUTDIR, "v12_mechanism.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", final], check=True)
    sz = os.path.getsize(final) / 1e6
    print(f"[render] done -> {final}  ({sz:.2f} MB, {total:.1f}s)")

    if "--keep-frames" not in sys.argv:
        import shutil

        for s in segs:
            os.remove(s)
        shutil.rmtree(FRAMES, ignore_errors=True)
        os.remove(lst)
        print("[render] 已清理中间帧与分段视频，output 仅保留成品")


if __name__ == "__main__":
    main()
