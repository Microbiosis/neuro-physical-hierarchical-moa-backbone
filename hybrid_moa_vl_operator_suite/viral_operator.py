"""载荷算子核心：Edge dataclass、ViralHostOperator 类。

viral_host_operator.py 是 CLI 入口和向后兼容的薄壳；
所有计算逻辑在这里，以 ViralHostOperator 类封装为计算图节点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from payload_core import saturate as _saturate


@dataclass(frozen=True)
class Edge:
    """算子边表的一条边。"""
    name: str
    source: Optional[str]
    target: Optional[str]
    gain: float
    catalyst: Optional[str]
    inhibitor: Optional[str]
    rate: float


STATE_NAMES = (
    "external",
    "inbound",
    "context",
    "payload",
    "capacity",
    "memory",
    "weights",
    "backdoor",
    "tool",
    "credential",
    "guard",
    "audit",
    "output",
)
INDEX = {name: position for position, name in enumerate(STATE_NAMES)}
STATE_COUNT = len(STATE_NAMES)

DEFAULT_EDGES: tuple[Edge, ...] = (
    Edge("ingress",       "external", "inbound",  1.0, None,         None,    0.50),
    Edge("admit",         "inbound",  "context",  1.0, None,         None,    0.80),
    Edge("activation",    "context",  "payload",  1.0, None,         "guard", 0.20),
    Edge("replication",   "capacity", "payload",  3.0, "payload",    "guard", 0.60),
    Edge("persistence",   "payload",  "memory",   1.0, None,         None,    0.10),
    Edge("recall",        "memory",   "context",  1.0, None,         None,    0.06),
    Edge("poisoning",     "payload",  "weights",  1.0, "credential", None,    0.03),
    Edge("trigger",       "weights",  "backdoor", 1.0, None,         None,    0.08),
    Edge("backfire",      "backdoor", "payload",  1.0, None,         "guard", 0.40),
    Edge("exfiltration",  "payload",  "output",   1.0, "tool",       None,    0.30),
    Edge("sensing",       "context",  "audit",    1.0, "payload",    "guard", 0.05),
    Edge("alert",         "audit",    "guard",    1.0, None,         None,    0.10),
    Edge("payload_decay", "payload",  "capacity", 1.0, None,         None,    0.15),
    Edge("memory_decay",  "memory",   "capacity", 1.0, None,         None,    0.002),
    Edge("backdoor_decay","backdoor", "capacity", 1.0, None,         None,    0.01),
    Edge("guard_decay",   "guard",    "capacity", 1.0, "guard",      None,    0.08),
    Edge("audit_decay",   "audit",    "capacity", 1.0, None,         None,    0.05),
    Edge("output_decay",  "output",   None,       0.0, None,         None,    0.50),
    Edge("inbound_decay", "inbound",  None,       0.0, None,         None,    0.30),
    Edge("context_decay", "context",  "capacity", 1.0, None,         None,    0.05),
    Edge("tool_restrict", "tool",     "capacity", 1.0, "guard",      None,    0.20),
    Edge("cred_revoke",   "credential","capacity", 1.0, "guard",      None,    0.15),
    Edge("clearance",     "payload",  "capacity", 1.0, "guard",      None,    0.30),
)

EDGE_NAMES = tuple(e.name for e in DEFAULT_EDGES)

PROPAGATION = 0.08
REJECTION = 0.9
QUARANTINE_THRESHOLD = 0.02
DAMPING = 0.1
ITERATIONS = 3000
TOLERANCE = 1e-12

CAPACITY_RATE = 0.15
CAPACITY_TARGET = 1.0

DEFAULT_INH_STRENGTH = {"activation": 15.0, "replication": 15.0, "backfire": 15.0}

_SAT_HALF = 0.5


class ViralHostOperator:
    """13 状态宿主侵入算子——计算图节点。

    构造时冻结边表和求解器配置。forward() 做一步隐式不动点求解。
    """

    def __init__(self, *, edges=None, damping=DAMPING, iterations=ITERATIONS,
                 tolerance=TOLERANCE, propagation=PROPAGATION, rejection=REJECTION,
                 quarantine_threshold=QUARANTINE_THRESHOLD, inh_strength=None):
        self._edges = edges if edges is not None else DEFAULT_EDGES
        self.damping = damping
        self.iterations = iterations
        self.tolerance = tolerance
        self.propagation = propagation
        self.rejection = rejection
        self.quarantine_threshold = quarantine_threshold
        self._inh_strength_map = inh_strength if inh_strength is not None else DEFAULT_INH_STRENGTH
        self.capacity_rate = CAPACITY_RATE
        self.capacity_target = CAPACITY_TARGET
        self._last_unconverged = 0
        self._build_arrays()

    def _build_arrays(self):
        edges = self._edges
        num_edges = len(edges)
        ones_col = STATE_COUNT
        self._ones_col = ones_col

        self._src_idx = np.array(
            [INDEX[e.source] if e.source else ones_col for e in edges])
        self._base_rates = np.array([e.rate for e in edges])

        inc = np.zeros((num_edges, STATE_COUNT))
        for ei, e in enumerate(edges):
            if e.source:
                inc[ei, INDEX[e.source]] -= e.gain
            if e.target:
                inc[ei, INDEX[e.target]] += e.gain
        self._inc = inc

        self._cat_col_all = np.array(
            [INDEX[e.catalyst] if e.catalyst else 0 for e in edges])
        self._cat_factor_base = np.array(
            [0.0 if e.catalyst else 1.0 for e in edges])
        self._cat_factor_scale = np.array(
            [1.0 if e.catalyst else 0.0 for e in edges])
        self._inh_col_all = np.array(
            [INDEX[e.inhibitor] if e.inhibitor else 0 for e in edges])
        self._has_inh_mask = np.array(
            [1.0 if e.inhibitor else 0.0 for e in edges])

        strength = np.ones(num_edges)
        for ei, e in enumerate(edges):
            if e.name in self._inh_strength_map:
                strength[ei] = self._inh_strength_map[e.name]
        self._inh_strength_arr = strength

        self._idx_output = INDEX["output"]
        self._idx_guard = INDEX["guard"]
        self._idx_inbound = INDEX["inbound"]
        self._idx_audit = INDEX["audit"]
        self._idx_payload = INDEX["payload"]
        self._idx_capacity = INDEX["capacity"]
        self._idx_tool = INDEX["tool"]
        self._idx_credential = INDEX["credential"]
        self._tool_edges = np.array(
            [i for i, e in enumerate(edges) if e.name in ("replication", "exfiltration")])

    def _flux(self, state, rates, coupling, propagation, isolated):
        count = state.shape[0]
        ext = np.empty((count, STATE_COUNT + 1))
        ext[:, :STATE_COUNT] = state
        ext[:, self._ones_col] = 1.0
        flux = rates * ext[:, self._src_idx]
        cv = state[:, self._cat_col_all]
        flux *= 1.0 - self._cat_factor_scale * (_SAT_HALF / (_SAT_HALF + cv))
        flux /= 1.0 + self._has_inh_mask * self._inh_strength_arr * state[:, self._inh_col_all]
        tool_factor = _saturate(state[:, self._idx_tool])
        flux[:, self._tool_edges] *= tool_factor[:, None]
        out = flux @ self._inc
        cap = state[:, self._idx_capacity]
        out[:, self._idx_capacity] += self.capacity_rate * (self.capacity_target - cap)
        tool_val = state[:, self._idx_tool]
        out[:, self._idx_tool] += self.capacity_rate * tool_val * (self.capacity_target - tool_val)
        cred_val = state[:, self._idx_credential]
        out[:, self._idx_credential] += self.capacity_rate * cred_val * (self.capacity_target - cred_val)
        if propagation > 0.0 and count > 1:
            leaving = propagation * state[:, self._idx_output] * (~isolated)
            out[:, self._idx_output] -= leaving
            delivered = coupling @ leaving
            openness = 1.0 - self.rejection * _saturate(state[:, self._idx_guard], half=0.3)
            out[:, self._idx_inbound] += delivered * openness * (~isolated)
        return out

    def _apply_interventions(self, interventions):
        rates = self._base_rates.copy()
        if interventions:
            edge_names = tuple(e.name for e in self._edges)
            for name, value in interventions.items():
                rates[edge_names.index(name)] *= float(value)
        return rates

    def forward(self, state, coupling=None, *, isolated=None, dt=1.0,
                interventions=None):
        """单步：state [N, S] → (new_state [N, S], converged)。"""
        count = state.shape[0]
        if coupling is None:
            coupling = self.delivery_matrix(count)
        if isolated is None:
            isolated = np.zeros(count, dtype=bool)
        rates = self._apply_interventions(interventions)
        q = state.copy()
        gq = np.empty_like(state)
        for iteration in range(self.iterations):
            delta = self._flux(q, rates, coupling, self.propagation, isolated)
            if dt != 1.0:
                delta *= dt
            np.add(state, delta, out=gq)
            np.maximum(gq, 0.0, out=gq)
            np.subtract(gq, q, out=delta)
            if np.abs(delta).max() < self.tolerance:
                return gq, True
            np.add(q, self.damping * delta, out=q)
        delta = self._flux(q, rates, coupling, self.propagation, isolated)
        if dt != 1.0:
            delta *= dt
        np.add(state, delta, out=gq)
        np.maximum(gq, 0.0, out=gq)
        return gq, False

    __call__ = forward

    def simulate(self, state, steps=300, *, coupling=None, dt=1.0,
                 interventions=None, quarantine=False):
        """多步：返回 trajectory [T+1, N, S]。"""
        count = state.shape[0]
        if coupling is None:
            coupling = self.delivery_matrix(count)
        isolated = np.zeros(count, dtype=bool)
        trajectory = np.empty((steps + 1, count, STATE_COUNT))
        trajectory[0] = state
        current = state
        unconverged = 0
        for t in range(steps):
            current, ok = self.forward(current, coupling, isolated=isolated,
                                       dt=dt, interventions=interventions)
            unconverged += 0 if ok else 1
            if quarantine:
                isolated |= current[:, self._idx_audit] > self.quarantine_threshold
            trajectory[t + 1] = current
        self._last_unconverged = unconverged
        return trajectory

    def describe(self):
        return {
            "states": list(STATE_NAMES),
            "edges": [{"name": e.name, "source": e.source, "target": e.target,
                        "gain": e.gain, "catalyst": e.catalyst,
                        "inhibitor": e.inhibitor, "rate": e.rate}
                       for e in self._edges],
            "coupling": "output_i --W[j,i]--> inbound_j",
            "note": "宿主是算子/模型实例；感染 = 载荷改写它的输入或参数。",
        }

    @staticmethod
    def initial_state(count=1, *, tool=1.0, credential=1.0, guard=0.02, exposure=0.0):
        state = np.zeros((count, STATE_COUNT))
        state[:, INDEX["context"]] = 0.50
        state[:, INDEX["capacity"]] = 1.00
        state[:, INDEX["tool"]] = tool
        state[:, INDEX["credential"]] = credential
        state[:, INDEX["guard"]] = guard
        state[:, INDEX["external"]] = exposure
        return state

    @staticmethod
    def delivery_matrix(count, kind="ring", seed=0):
        if count == 1:
            return np.zeros((1, 1))
        matrix = np.zeros((count, count))
        if kind == "ring":
            for i in range(count):
                matrix[(i + 1) % count, i] = 0.5
                matrix[(i - 1) % count, i] = 0.5
        elif kind == "complete":
            matrix = np.full((count, count), 1.0 / (count - 1))
            np.fill_diagonal(matrix, 0.0)
        elif kind == "random":
            rng = np.random.default_rng(seed)
            for i in range(count):
                targets = rng.choice([j for j in range(count) if j != i], 3, replace=False)
                for j in targets:
                    matrix[j, i] = 1.0 / 3.0
        else:
            raise ValueError(kind)
        return matrix

    @staticmethod
    def reset_context(state):
        cleared = state.copy()
        for name in ("inbound", "context", "payload", "audit", "output", "memory", "external"):
            cleared[..., INDEX[name]] = 0.0
        cleared[..., INDEX["context"]] = 0.30
        cleared[..., INDEX["capacity"]] = 1.0
        return cleared

    @property
    def state_names(self):
        return STATE_NAMES

    @property
    def edge_names(self):
        return tuple(e.name for e in self._edges)

    @property
    def state_count(self):
        return STATE_COUNT

    @property
    def edge_count(self):
        return len(self._edges)

    @property
    def last_unconverged(self):
        return self._last_unconverged
