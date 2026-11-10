#!/usr/bin/env python3
"""病毒核酸侵入算子 · 隐式耦合状态更新版（v2：机制保真重构）。

用法
    op = create_viral_invasion_operator(CoupledViralInvasionConfig(d_model=32))
    y, info = op(x, return_aux=True)   # x, y 形状 [B, L, D]
    q = info["state_trajectory"]       # [B, L, D, 7]
    print(op.backend, op.device)

算子契约
    签名 f: R^{B×L×D} → R^{B×L×D}。逐 (b, l) 求一个 7 坐标潜在量 q 的不动点，
    样本之间独立，沿 L 因果（第 t 个 token 的输出不依赖 t' > t），同输入同输出。
    T 把单纯形映回自身且 Lipschitz 常数 < 1，故不动点存在唯一且迭代收敛。
    PyTorch 路径端到端可微；NumPy 路径用复步微分给出 dy/dx 与 dy/dθ，精度到机器精度。
    下面列出的状态名与边名是这套算子的标注：标注对应一组结构断言，由 _MECHANISMS
    在构造期强制；算子自身成立的性质是数学的，标注是否对应真实对象要靠实验判定。

对象与范围
    宿主定为真核（人）细胞，病原定为 HIV-1 一类逆转录病毒。模块只实现该范围内的
    状态与转移。原核 CRISPR 适应性免疫、噬菌体、非逆转录病毒一律不在模型内，
    也不与本模块的 memory 共用变量——上一版把两者压成一个"记忆"是错的。

状态（7 个，均为潜在占比，每个 token 的占比和为 1）
    available   尚可动用的宿主容量：未占用的受体/融合能力、可动用的核苷酸与表达
                资源、空载的整合位点。SAMHD1 水解 dNTP 即收缩该容量
                （Laguette 2011 Nature, PMID 21613998）。
    rna         融合后进入胞质的病毒基因组 RNA（Wilen 2012, PMID 22908191）。
    cdna        逆转录产物，胞质 pre-integration DNA；RNase H 清除 RNA:DNA 杂合体
                中的 RNA 链属于该步（Hu & Hughes 2012, PMID 23028129）。
                该状态是 cGAS 的配体来源。
    integrated  共价整合进宿主基因组的原病毒 DNA；跨细胞周期存续，是模板而非耗材
                （Craigie & Bushman 2012, PMID 22762018；Burdick 2020,
                PMID 32094182）。
    expressed   由原病毒模板驱动的活跃表达产物（病毒 RNA 与蛋白）。其物质来自
                available 池，不从 integrated 扣减（Siliciano & Greene 2011,
                PMID 22229121）。
    variant     复制群体的序列分化；RT 错误率 3.4e-5/碱基/复制轮
                （Mansky & Temin 1995, PMID 7541846）。
    memory      先天免疫激活态（cGAS–STING → IFN/ISG），自限性回落，且持续激活会
                加速自身关闭（去敏化）。cGAS 识别
                逆转录产生的病毒 DNA（Gao 2013 Science, PMID 23929945；
                Lahaye 2013 Immunity, PMID 24269171），HIV-1 衣壳可部分规避
                （Rasaiyaah 2013 Nature, PMID 24196705），IFN 同时限制进入与表达
                （Doyle 2015 Nat Rev Microbiol, PMID 25915633）。

十一条边（源 → 目标；右列为该边在模型中的语义）
    entry                 available → rna        受体结合与膜融合
    reverse_transcription rna → cdna             逆转录（含 RNase H 清除 RNA 链）
    integration           cdna → integrated      整合
    expression            available → expressed  原病毒模板驱动的表达（门控 = integrated）
    mutagenesis           rna → variant          错误倾向的逆转录产生变异 cDNA
    dna_sensing           cdna → memory          整合前 DNA 触发 cGAS–STING
    variant_sensing       variant → memory       变异 DNA 的感知
    memory_turnover       memory → available     IFN/ISG 反应衰减、限制解除（慢尺度）
    clearance             integrated → available 感染细胞死亡、原病毒清除
    release               expressed → available  子代释放与表达产物降解
    variant_integration   variant → integrated   变异 cDNA 同样可整合

机制约束
    与文献对应的那些结构要求不是文档里的句子，而是模块内的 _MECHANISMS：13 条
    (名称, PMID, 规则) 三元组，覆盖边的连接方式、门控、必需缺失的边、唯一出口与
    调节符号。_topology() 构造时逐条校验，任何一条不成立就抛 MechanismViolation，
    违背机制的实例无法被构造出来。describe() 把边、门控、调节符号、依赖集与机制
    清单一起导出，供外部按同一份数据核对。

可分性
    依赖关系不在文档里手写，由 describe()["dependencies"] 在运行时从边表与反馈矩阵
    算出。结论：7 个坐标是一个强连通块——没有封闭真子集，也不存在能把方程化成块三角
    的坐标顺序，每个 token 的 7 个坐标只能联立求解。注意"依赖"与"影响"互为转置：
    扰动坐标 i 看哪些坐标变化，得到的是依赖表的逆关系。速率取的是该 token 各通道状态的
    算术均值（无加权系数），所以通道 d 还依赖各通道 d' 的同名坐标，只会更稠密。

收敛
    转移比例落在 [0, 0.40]，σ' ≤ 1/4。单条边的导数上界为
    0.40 * 0.25 * 0.60 * |Wf[:, e]|_1；每个目标坐标收到的 |Wf| 列和最大为 2.4，
    源质量 ≤ 1，故速率项 ≤ 0.144；模板门控用 tanh(1.0 q)，斜率上界 1.0，最高
    转移比例 0.40，门控项 ≤ 0.40 * 1.0 = 0.40。合计 L ≤ 0.544 < 1。
    T 把单纯形映回自身（每个坐标至多两条出边，每条 ≤ 0.84），由 Banach 不动点
    定理，每个 token 的不动点唯一。迭代 N 步的偏差 ≤ 2 * 0.564**N：FP64 用 56 步
    （≤ 3e-14），FP32 用 28 步（≤ 2e-7，与单精度 eps 同量级）。这两个步数是
    最坏情况上限；实际迭代带早停（tolerance 默认 FP64 1e-13、FP32 1e-6），
    收敛后立即退出，通常十几步以内结束。

长期效应
    记忆的回落边带 0.15 的速率跨度倍率，半衰期落在数十个 token；同时记忆对自身的回落
    有正反馈，表示持续激活会加速关闭（去敏化）。两者合起来给出三个可观察后果：
    持续给药只把病毒负担压低一部分而非清零；同一剂量下抑制的边际效益随暴露时间递减；
    停药后抑制以数步到数十步的尺度消退。Doyle 2015 记录的"明显但短暂"的临床病毒载量
    下降与这一时间尺度一致；去敏化的具体通路（SOCS/USP18 一类）是建模设定，未逐条引用。

未标定、不声称
    转移比例、反馈矩阵、门控增益都是设定值，没有用感染数据拟合。本模块证明的是：
    代码正确求解了这里写下的方程，且这些方程保留了已发表实验的因果方向与作用范围，
    不构成对真实感染进程的定量预测。

执行
    自动试跑可用加速器的完整前向与反向，失败则回退到 PyTorch CPU 或 NumPy CPU。
    PyTorch 路径端到端可求导；NumPy 路径给出 FP64 前向。两条路径共用同一份公式。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import warnings

import numpy as np

try:
    import torch
    from torch import nn
except (ImportError, OSError) as error:
    _TORCH_IMPORT_ERROR = error
    torch = None
    nn = None


STATE_NAMES = ("available", "rna", "cdna", "integrated", "expressed", "variant", "memory")
EDGE_NAMES = (
    "entry", "reverse_transcription", "integration", "expression", "mutagenesis",
    "dna_sensing", "variant_sensing", "memory_turnover", "clearance", "release",
    "variant_integration",
)
_STATE_COUNT = len(STATE_NAMES)
_EDGE_COUNT = len(EDGE_NAMES)
_EDGE_SOURCE = (0, 1, 2, 0, 1, 2, 5, 6, 3, 4, 5)
_EDGE_TARGET = (1, 2, 3, 4, 5, 6, 6, 0, 0, 0, 3)
_EDGE_SOURCE_INDEX = list(_EDGE_SOURCE)
_GATED_EDGE = 3          # expression
_GATE_STATE = 3          # integrated
_RATE_FLOOR = 0.0
_RATE_SPAN = 0.40
_FEEDBACK_STRENGTH = 0.60
_GATE_GAIN = 1.0
# 每条边的速率跨度倍率。memory_turnover 取 0.15：记忆的半衰期约数十个 token，
# 停药后抑制不会立刻消失，这是"长期效应"的时间尺度来源。
_EDGE_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 1.0, 1.0, 1.0])
_INITIAL_MASS = (0.70, 0.06, 0.05, 0.05, 0.05, 0.04, 0.05)
_PARAMETER_NAMES = ("input_gain_raw", "rate_bias_raw", "initial_logits",
                    "readout_raw", "readout_gate_raw")



__all__ = [
    "CoupledViralInvasionConfig",
    "MechanismViolation",
    "NumpyCoupledViralInvasionOperator",
    "TorchCoupledViralInvasionOperator",
    "create_viral_invasion_operator",
]

@dataclass(frozen=True)
class CoupledViralInvasionConfig:
    d_model: int = 32

    def __post_init__(self):
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int) or self.d_model < 1:
            raise ValueError("d_model must be a positive integer")


class MechanismViolation(Exception):
    """算子结构违背了它自己声明的机制。构造时抛出，不允许存在违背机制的实例。"""


_GATE_OF_EDGE = {_GATED_EDGE: _GATE_STATE}

# (名称, PMID, 规则)。规则只约束结构，不约束数值：
#   ("edge", 边名, 源, 目标, 门控或 None)   这条边必须存在且连接方式如此
#   ("absent", 源, 目标)                    这条边必须不存在
#   ("sole_output", 状态, 边名)             该状态只能经由这条边离开
#   ("regulator", 状态, 边名, ±1)           该状态对该条边对数比的贡献必须是这个符号
_MECHANISMS = (
    ("entry", "22908191", ("edge", "entry", "available", "rna", None)),
    ("reverse_transcription", "23028129",
     ("edge", "reverse_transcription", "rna", "cdna", None)),
    ("integration", "22762018", ("edge", "integration", "cdna", "integrated", None)),
    ("expression_is_templated", "22229121",
     ("edge", "expression", "available", "expressed", "integrated")),
    ("expression_does_not_consume_provirus", "22229121", ("absent", "expressed", "integrated")),
    ("provirus_persists", "9371822", ("sole_output", "integrated", "clearance")),
    ("dna_sensing_precedes_integration", "23929945",
     ("edge", "dna_sensing", "cdna", "memory", None)),
    ("variant_sensing", "24269171", ("edge", "variant_sensing", "variant", "memory", None)),
    ("mutagenesis_at_reverse_transcription", "7541846",
     ("edge", "mutagenesis", "rna", "variant", None)),
    ("variant_integrates", "22762018",
     ("edge", "variant_integration", "variant", "integrated", None)),
    ("ifn_restricts_entry", "25915633", ("regulator", "memory", "entry", -1)),
    ("ifn_restricts_expression", "25915633", ("regulator", "memory", "expression", -1)),
    ("ifn_desensitises", "25915633", ("regulator", "memory", "memory_turnover", +1)),
)


def _enforce_mechanisms(edges, feedback):
    """按 _MECHANISMS 校验边表与反馈符号；任何一条不成立就不允许构造。"""
    if len(edges) != _EDGE_COUNT:
        raise MechanismViolation(f"edge table has {len(edges)} rows, expected {_EDGE_COUNT}")
    for name, pmid, rule in _MECHANISMS:
        kind = rule[0]
        if kind == "edge":
            _, edge, source, target, gate = rule
            index = EDGE_NAMES.index(edge)
            expected = (STATE_NAMES.index(source), STATE_NAMES.index(target))
            if tuple(edges[index]) != expected:
                raise MechanismViolation(
                    f"{name} (PMID {pmid}): {edge} is {edges[index]}, expected {expected}")
            expected_gate = None if gate is None else STATE_NAMES.index(gate)
            if _GATE_OF_EDGE.get(index) != expected_gate:
                raise MechanismViolation(
                    f"{name} (PMID {pmid}): gate of {edge} is {_GATE_OF_EDGE.get(index)}, "
                    f"expected {expected_gate}")
        elif kind == "absent":
            _, source, target = rule
            if (STATE_NAMES.index(source), STATE_NAMES.index(target)) in set(map(tuple, edges)):
                raise MechanismViolation(f"{name} (PMID {pmid}): edge {source} -> {target} exists")
        elif kind == "sole_output":
            _, state, edge = rule
            index = STATE_NAMES.index(state)
            outlets = [EDGE_NAMES[edge_id] for edge_id in range(_EDGE_COUNT)
                       if edges[edge_id][0] == index]
            if outlets != [edge]:
                raise MechanismViolation(
                    f"{name} (PMID {pmid}): {state} leaves via {outlets}, expected [{edge}]")
        elif kind == "regulator":
            _, state, edge, sign = rule
            value = feedback[STATE_NAMES.index(state), EDGE_NAMES.index(edge)]
            if sign * value <= 0.0:
                raise MechanismViolation(
                    f"{name} (PMID {pmid}): regulator {state} on {edge} is {value}, "
                    f"expected sign {sign}")
        else:
            raise MechanismViolation(f"unknown rule kind {kind}")


def describe():
    """算子自身的机器可读定义：状态、边、门控、调节符号、依赖集、机制清单。"""
    feedback = _topology()
    edges = [{"name": EDGE_NAMES[index],
              "source": STATE_NAMES[_EDGE_SOURCE[index]],
              "target": STATE_NAMES[_EDGE_TARGET[index]],
              "gate": (None if index not in _GATE_OF_EDGE
                       else STATE_NAMES[_GATE_OF_EDGE[index]]),
              "regulators": {STATE_NAMES[state]: feedback[state, index].item()
                             for state in range(_STATE_COUNT) if feedback[state, index]}}
             for index in range(_EDGE_COUNT)]
    dependencies = {}
    for state in range(_STATE_COUNT):
        needed = set()
        for index in range(_EDGE_COUNT):
            if state not in (_EDGE_SOURCE[index], _EDGE_TARGET[index]):
                continue
            needed.update(int(i) for i in range(_STATE_COUNT) if feedback[i, index])
            if index in _GATE_OF_EDGE:
                needed.add(_GATE_OF_EDGE[index])
        dependencies[STATE_NAMES[state]] = [STATE_NAMES[i] for i in sorted(needed)]
    return {"states": list(STATE_NAMES), "edges": edges, "dependencies": dependencies,
            "mechanisms": [{"name": name, "pmid": pmid, "rule": rule}
                           for name, pmid, rule in _MECHANISMS]}


def _topology():
    """反馈矩阵 Wf[i, e]：第 i 个坐标对第 e 条边对数比的贡献。负为抑制，正为促进。"""
    feedback = np.zeros((_STATE_COUNT, _EDGE_COUNT))
    feedback[1, 0] = -1.0   # RNA 堆积抑制进入（超感染排斥）
    feedback[6, 0] = -0.9   # ISG 限制进入
    feedback[2, 1] = -1.0   # cDNA 堆积抑制逆转录
    feedback[3, 2] = -0.9   # 已占整合位点抑制整合
    feedback[6, 2] = -0.6   # ISG 效应子抑制整合
    feedback[4, 3] = -0.9   # 表达产物反馈
    feedback[6, 3] = -0.9   # IFN 抑制原病毒转录
    feedback[5, 4] = -1.0   # 变异 cDNA 堆积抑制错误倾向的逆转录
    feedback[6, 5] = -1.0   # 记忆饱和抑制 DNA 感知
    feedback[6, 6] = -1.0   # 记忆饱和抑制变异感知
    feedback[0, 7] = -0.8   # 容量充裕时记忆回落更快
    feedback[6, 7] = +0.8   # 去敏化：持续激活加速自身关闭（SOCS/USP18 类负反馈）
    feedback[6, 8] = +0.6   # ISG 促进感染细胞清除
    feedback[0, 9] = -0.8   # 容量充裕时释放回流更快
    feedback[3, 10] = -0.9  # 已占整合位点抑制变异 cDNA 的整合
    _enforce_mechanisms(tuple(zip(_EDGE_SOURCE, _EDGE_TARGET)), feedback)
    return feedback


def _flux_matrix():
    """[E, 7] 通量矩阵：每条边在源坐标扣 1、在目标坐标加 1，逐行和为 0。"""
    flux = np.zeros((_EDGE_COUNT, _STATE_COUNT))
    for edge, (source, target) in enumerate(zip(_EDGE_SOURCE, _EDGE_TARGET)):
        flux[edge, source] -= 1.0
        flux[edge, target] += 1.0
    return flux


def _gate_mask():
    mask = np.zeros(_EDGE_COUNT)
    mask[_GATED_EDGE] = 1.0
    return mask


def _contraction_bound(feedback):
    """由常量与拓扑给出的解析上界：速率项 + 门控项。"""
    sigmoid_slope = 0.25
    per_unit = _RATE_SPAN * sigmoid_slope * _FEEDBACK_STRENGTH
    target_load = np.zeros(_STATE_COUNT)
    for edge in range(_EDGE_COUNT):
        target_load[_EDGE_TARGET[edge]] += _EDGE_SCALE[edge] * np.abs(feedback[:, edge]).sum()
    rate_term = target_load.max().item() * per_unit
    gate_term = (_RATE_FLOOR + _RATE_SPAN) * _GATE_GAIN   # tanh 门控斜率上界 _GATE_GAIN
    return rate_term + gate_term, rate_term, gate_term


def _initial_parameters(config, seed):
    rng = np.random.default_rng(seed)
    width = config.d_model
    gains = rng.uniform(0.20, 0.65, (width, _EDGE_COUNT))
    gains[:, 7:10] *= 0.5
    initial = np.broadcast_to(np.log(_INITIAL_MASS), (width, _STATE_COUNT)).copy()
    readout = np.linspace(-0.65, 0.75, _STATE_COUNT) + rng.normal(0, 0.03, (width, _STATE_COUNT))
    return gains, np.zeros_like(gains), initial, readout, np.full(width, -0.2)


def _sigmoid(x, xp):
    return 0.5 + 0.5 * xp.tanh(0.5 * x)


def _input_drive(x, gain, bias, xp):
    return 2.0 * (xp.tanh(x)[..., None] * xp.tanh(gain) + bias)


def _transition_rates(candidate, drive, feedback, xp):
    # 转移速率是 token 级的：由该 token 的通道均值决定，通道之间不存在加权系数。
    pooled = candidate.mean(1)[:, None, :]
    logits = drive + _FEEDBACK_STRENGTH * (pooled @ feedback)
    rates = _RATE_FLOOR + _RATE_SPAN * _sigmoid(logits, xp)
    # 模板门控在 integrated = 0 处严格为 0：没有原病毒就没有表达。
    gate = xp.tanh(_GATE_GAIN * pooled[..., _GATE_STATE:_GATE_STATE + 1])
    return rates, gate


def _transfer(previous, rates, gate, gate_mask, edge_scale, flux, xp):
    effective = (rates + gate_mask * rates * (gate - 1.0)) * edge_scale
    flow = previous[..., _EDGE_SOURCE_INDEX] * effective
    return previous + flow @ flux


def _solve_step(previous, drive, feedback, gate_mask, edge_scale, flux, steps, xp,
                start=None, tolerance=None):
    """不动点迭代。tolerance 给定时按实测步长提前退出，steps 仍是最坏情况下的上限。"""
    state = previous if start is None else start
    for _ in range(steps):
        rates, gate = _transition_rates(state, drive, feedback, xp)
        updated = _transfer(previous, rates, gate, gate_mask, edge_scale, flux, xp)
        if tolerance is not None:
            change = xp.abs(updated - state).max().item()
            state = updated
            if change <= tolerance:
                break
        else:
            state = updated
    return state / state.sum(-1)[..., None]


def _sequence(x, parameters, feedback, gate_mask, edge_scale, flux, steps, tolerance,
              return_aux, xp):
    gain, bias, initial_logits, readout, output_gate = parameters
    init_mass = xp.exp(initial_logits - xp.amax(initial_logits, -1)[..., None])
    prior = 0.99 * (init_mass / init_mass.sum(-1)[..., None]) + 0.01 / _STATE_COUNT
    state = xp.broadcast_to(prior, (x.shape[0], x.shape[2], _STATE_COUNT))
    drives = _input_drive(x, gain, bias, xp)
    decoder = xp.tanh(readout)
    gate = _sigmoid(output_gate, xp)

    corrections = []
    trajectory, rates_history, residuals = [], [], []
    for t in range(x.shape[1]):
        previous = state
        drive = drives[:, t]
        state = _solve_step(previous, drive, feedback, gate_mask, edge_scale, flux, steps, xp,
                            tolerance=tolerance)
        corrections.append((state * decoder).sum(-1) * gate)
        if return_aux:
            rates, gate_value = _transition_rates(state, drive, feedback, xp)
            implied = _transfer(previous, rates, gate_value, gate_mask, edge_scale, flux, xp)
            trajectory.append(state)
            rates_history.append(rates * (1.0 + gate_mask * (gate_value - 1.0)))
            residuals.append(xp.abs(state - implied).sum(-1))

    y = x + xp.stack(corrections, 1)
    if not return_aux:
        return y
    return y, {
        "state_trajectory": xp.stack(trajectory, 1),
        "transition_rates": xp.stack(rates_history, 1),
        "solver_residual": xp.stack(residuals, 1),
        "final_state": state,
    }


def _check_shape(x, config):
    if x.ndim != 3 or not x.shape[0] or not x.shape[1] or x.shape[2] != config.d_model:
        raise ValueError(f"Expected nonempty [B, L, {config.d_model}], got {tuple(x.shape)}")


class NumpyCoupledViralInvasionOperator:
    """FP64 CPU forward with the same joint transition as the Torch path."""

    backend = "numpy"
    device = "cpu"
    compute_dtype = np.dtype("float64")
    solver_steps = 56
    solver_tolerance = 1e-13

    def __init__(self, config, seed=2026):
        self.config = config
        for name, value in zip(_PARAMETER_NAMES, _initial_parameters(config, seed)):
            setattr(self, name, value)
        self.feedback = _topology()
        self.flux = _flux_matrix()
        self.gate_mask = _gate_mask()
        self.edge_scale = _EDGE_SCALE.copy()

    def forward(self, x, *, return_aux=False):
        x = np.asarray(x)
        if x.dtype.kind not in "fiu":
            raise TypeError("Expected real-valued input")
        x = np.asarray(x, dtype=np.float64)
        _check_shape(x, self.config)
        params = tuple(getattr(self, name) for name in _PARAMETER_NAMES)
        return _sequence(x, params, self.feedback, self.gate_mask, self.edge_scale,
                         self.flux, self.solver_steps, self.solver_tolerance, return_aux, np)

    __call__ = forward

    def jacobian(self, x, *, wrt="input"):
        """复步微分给出的精确导数，不依赖自动微分框架。

        wrt="input"      返回 dy/dx，形状 [输出数, 输入数]
        wrt="parameters" 返回 {参数名: [输出数, 参数数]}
        """
        x = np.asarray(x, dtype=np.float64)
        _check_shape(x, self.config)
        params = [np.asarray(getattr(self, name), dtype=np.complex128) for name in _PARAMETER_NAMES]
        probe_x = x.astype(complex)

        def run(px, pp):
            return _sequence(px, pp, self.feedback, self.gate_mask, self.edge_scale,
                             self.flux, self.solver_steps, None, False, np)

        if wrt == "input":
            flat = x.reshape(-1)
            collected = []
            for index in range(flat.size):
                shifted = flat.astype(complex)
                shifted[index] += 1e-20j
                collected.append(run(shifted.reshape(x.shape), params).imag.reshape(-1) / 1e-20)
            return np.stack(collected, axis=1)

        if wrt == "parameters":
            result = {}
            for slot, name in enumerate(_PARAMETER_NAMES):
                collected = []
                for index in range(params[slot].size):
                    probe = [array.copy() for array in params]
                    probe[slot].reshape(-1)[index] += 1e-20j
                    collected.append(run(probe_x, probe).imag.reshape(-1) / 1e-20)
                result[name] = np.stack(collected, axis=1)
            return result

        raise ValueError('wrt must be "input" or "parameters"')


def _accelerator_candidates():
    devices, diagnostics = [], []
    for name, runtime in (
        ("cuda", torch.cuda),  # NVIDIA CUDA and AMD ROCm share this device interface.
        ("xpu", getattr(torch, "xpu", None)),
        ("mps", getattr(torch.backends, "mps", None)),
    ):
        try:
            if runtime is not None and runtime.is_available():
                devices.append(torch.device(name))
        except RuntimeError as error:
            diagnostics.append(f"{name}: {error}")
    current = getattr(getattr(torch, "accelerator", None), "current_accelerator", None)
    if current is not None:
        try:
            detected = current(check_available=True)
        except (RuntimeError, TypeError) as error:
            diagnostics.append(f"torch.accelerator: {error}")
        else:
            if detected is not None and all(d.type != detected.type for d in devices):
                devices.append(detected)
    devices.append(torch.device("cpu"))
    return devices, diagnostics


if torch is not None:
    class TorchCoupledViralInvasionOperator(nn.Module):
        """Device-native autograd through the complete joint state transition."""

        backend = "torch"

        def __init__(self, config, seed=2026, *, device="auto", dtype=None):
            super().__init__()
            self.config = config
            dtype = torch.float32 if dtype is None else dtype
            if dtype not in (torch.float32, torch.float64):
                raise ValueError("dtype must be torch.float32 or torch.float64")
            # None 关闭早停，退回固定步数（torch.compile / CUDA graph 下更好追踪）
            self.solver_tolerance = 1e-13 if dtype == torch.float64 else 1e-6
            automatic = device == "auto"
            if automatic:
                candidates, diagnostics = _accelerator_candidates()
            else:
                candidates, diagnostics = [torch.device(device)], []

            arrays = _initial_parameters(config, seed)
            flux = _flux_matrix()
            gate_mask = _gate_mask()
            for candidate in candidates:
                if candidate.type == "meta":
                    raise ValueError("The operator needs a device that executes tensor kernels")
                try:
                    with torch.inference_mode(False):
                        if candidate.type == "mps" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
                            raise RuntimeError("MPS is configured to fall back to CPU")
                        for name, value in zip(_PARAMETER_NAMES, arrays):
                            tensor = torch.tensor(value, device=candidate, dtype=dtype)
                            self.register_parameter(name, nn.Parameter(tensor))
                        self.register_buffer("feedback", torch.tensor(_topology(), device=candidate, dtype=dtype))
                        self.register_buffer("flux", torch.tensor(flux, device=candidate, dtype=dtype))
                        self.register_buffer("gate_mask", torch.tensor(gate_mask, device=candidate, dtype=dtype))
                        self.register_buffer(
                            "edge_scale", torch.tensor(_EDGE_SCALE, device=candidate, dtype=dtype)
                        )
                        if candidate.type != "cpu":
                            self._probe()
                except (RuntimeError, TypeError, AssertionError) as error:
                    if not automatic or candidate.type == "cpu":
                        raise
                    diagnostics.append(f"{candidate}: {error}")
                else:
                    break
            self.device_probe_failures = tuple(diagnostics)
            if diagnostics:
                warnings.warn("Skipped unavailable paths: " + " | ".join(diagnostics), RuntimeWarning, stacklevel=2)

        @property
        def device(self):
            return self.rate_bias_raw.device

        @property
        def compute_dtype(self):
            return torch.float64 if self.rate_bias_raw.dtype == torch.float64 else torch.float32

        @property
        def solver_steps(self):
            return 56 if self.compute_dtype == torch.float64 else 28


        def _probe(self):
            with torch.enable_grad(), warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                sample = self.rate_bias_raw.new_full((1, 2, self.config.d_model), 0.35, requires_grad=True)
                y, aux = self(sample, return_aux=True)
                grads = torch.autograd.grad(y.square().mean(), (sample, *self.parameters()))
                values = (y, *aux.values(), *grads)
                if any(v.device != self.device for v in values):
                    raise RuntimeError("The device probe returned a tensor on another device")
                if not bool(torch.stack([v.isfinite().all() for v in values]).all()):
                    raise RuntimeError("The device probe produced nonfinite values")
            for notice in messages:
                message = str(notice.message).lower().replace(" ", "")
                if "cpu" in message and ("fallback" in message or "fallingback" in message):
                    raise RuntimeError(str(notice.message))

        def forward(self, x, *, return_aux=False):
            if isinstance(x, torch.Tensor) and not x.is_floating_point():
                raise TypeError("Expected a floating-point input tensor")
            if isinstance(x, np.ndarray) and x.dtype.kind not in "fiu":
                raise TypeError("Expected real-valued input")
            x = torch.as_tensor(x, device=self.device, dtype=self.compute_dtype)
            _check_shape(x, self.config)

            context = nullcontext()
            amp_mode = getattr(getattr(torch, "amp", None), "autocast_mode", None)
            autocast_available = getattr(amp_mode, "is_autocast_available", None)
            if autocast_available is not None and autocast_available(self.device.type):
                context = torch.autocast(self.device.type, enabled=False)
            with context:
                params = tuple(getattr(self, name).to(dtype=x.dtype) for name in _PARAMETER_NAMES)
                return _sequence(
                    x, params, self.feedback.to(dtype=x.dtype), self.gate_mask.to(dtype=x.dtype),
                    self.edge_scale.to(dtype=x.dtype), self.flux.to(dtype=x.dtype),
                    self.solver_steps, self.solver_tolerance, return_aux, torch,
                )


def create_viral_invasion_operator(config=None, seed=2026, *, backend="auto", device="auto", dtype=None):
    """Create the standalone coupled operator with device selection at construction."""
    cfg = CoupledViralInvasionConfig() if config is None else config
    if not isinstance(cfg, CoupledViralInvasionConfig):
        raise TypeError("Expected CoupledViralInvasionConfig")
    if backend not in ("auto", "torch", "numpy"):
        raise ValueError("backend must be auto, torch or numpy")
    if torch is not None and backend != "numpy":
        return TorchCoupledViralInvasionOperator(cfg, seed, device=device, dtype=dtype)
    if backend == "torch":
        raise ImportError("Install a usable PyTorch build") from _TORCH_IMPORT_ERROR
    if str(device) not in ("auto", "cpu"):
        raise ValueError("An accelerator device requires PyTorch")
    if dtype is not None:
        raise ValueError("NumPy uses float64. Omit dtype")
    return NumpyCoupledViralInvasionOperator(cfg, seed)


