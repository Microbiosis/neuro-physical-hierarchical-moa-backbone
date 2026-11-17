"""NBSC-Ω 3.6 tensor transition with causal spikes and a stretch reflex.

The emission coefficient is part of the in-flight state. Local release and
reception use the payload that actually arrives. Continuous paths use
autograd, and sampled events supply a score term through their log probability.
"""

from itertools import accumulate
from math import comb, exp, expm1, isclose, isfinite
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = ["NBSCOperator", "NBSCState", "NBSCOutput"]


class NBSCState(NamedTuple):
    """Batched state with neuron N, synapse E and joint K axes.

    The floating payload queue has shape (B, H). Each edge occupies ages
    zero through its propagation delay, newest first. Zero means no event.
    Values in (0, 1] carry the emission-time transmission gain.
    Synaptic current and postsynaptic trace share the target-neuron axis.
    Presynaptic traces are per edge since emission blocking is edge-specific.
    Voltage is normalized and refractory counters are integers. Activation
    and reflex activation are (B, K, 2), flexor first. Joint angle and
    velocity are in rad and rad/s.
    Fiber reference (B, K) is the extensor's reference strain, with its
    negative on the flexor. The three return-gate fields are (B, 1).
    """

    voltage: Tensor
    refractory: Tensor
    payload_history: Tensor
    utilization: Tensor
    resource: Tensor
    synaptic_current: Tensor
    pre_trace: Tensor
    post_trace: Tensor
    plasticity: Tensor
    activation: Tensor
    reflex_activation: Tensor
    angle: Tensor
    velocity: Tensor
    fiber_reference: Tensor
    return_difference: Tensor
    return_power: Tensor
    return_strong: Tensor


class NBSCOutput(NamedTuple):
    """Joint torque excludes damping and both constraint impulses.

    The two impulse fields have shape (B, K) and units N m s. The speed
    limiter acts on the drift velocity; a perfectly inelastic stop then
    arrests the remaining motion when the drift reaches the angle limit.
    """

    state: NBSCState
    spikes: Tensor
    spike_probability: Tensor
    release_count: Tensor
    release: Tensor
    edge_weight: Tensor
    muscle_force: Tensor
    joint_torque: Tensor
    log_probability: Tensor
    arrived_payload: Tensor
    speed_limit_impulse: Tensor
    joint_limit_impulse: Tensor


def _topology(segments, return_delay_s):
    groups = ("f", "e", "I", "p", "h", "m", "zmP", "zpP", "zmT", "zpT")
    neurons = ("c", "r") + tuple(f"{group}:{i}" for i in range(segments) for group in groups)
    edges = [("c", "r", 0.44, 0.018, "")]
    for i in range(segments):
        flex, extend, inh, prop, touch, motor = (f"{group}:{i}" for group in groups[:6])
        terminal = i == segments - 1
        edges.extend(((prop, "c", 1.10, 0.023, ""), (inh, "c", -10 if i == 0 else -0.12, 0.022, ""),
                      (touch, "r", 0.20, 0.019, "")))
        for dst, wc, wr in ((flex, 0.62, 0.43), (extend, 0.40, 0.56),
                            (inh, 10 if i == 0 else 0.30, 10 if terminal else 0.28)):
            edges.extend((("c", dst, wc, 0.024, ""), ("r", dst, wr, 0.015, "")))
        edges.extend(((prop, flex, 0.20, 0.008, "P"), (prop, extend, 0.20, 0.008, "P"),
                      (touch, extend, 0.02, 0.012, "T"), ("r", motor, 0.40, 0.017, ""),
                      (prop, motor, 0.10, 0.016, "")))
        for kind, sensory in (("P", prop), ("T", touch)):
            for sign, mult in (("m", 1), ("p", -1)):
                dst = f"z{sign}{kind}:{i}"
                edges.extend((("c", dst, 0.64 * mult, 0.022, ""), ("r", dst, 0.45 * mult, 0.016, ""),
                              (inh, dst, 0.48 * mult, 0.011, ""), (sensory, dst, 0.38 * mult, 0.012, "")))
    edges.append((f"I:{segments - 1}", "r", -10.5, return_delay_s, ""))
    for i in range(segments):
        for j in (i - 1, i + 1):
            if 0 <= j < segments:
                edges.extend(((f"f:{i}", f"I:{j}", 1.25 * .35, .021, ""),
                              (f"e:{i}", f"I:{j}", 1.05 * .35, .023, ""),
                              (f"I:{i}", f"f:{j}", -1.30 * .35, .017, ""),
                              (f"I:{i}", f"e:{j}", -1.10 * .35, .019, "")))
    edges.extend((f"f:{i}", "r", .38 * .35, .031, "") for i in range(segments))

    # Former instantaneous rate couplings become causal, plastic synapses.
    local = (("f", "f", .77), ("e", "e", .75), ("e", "f", -.67), ("f", "e", -.65),
             ("I", "f", -.50), ("I", "e", -.50), ("f", "I", .63), ("e", "I", .63),
             ("I", "I", -.43), ("m", "f", .22))
    for i in range(segments):
        edges.extend((f"{src}:{i}", f"{dst}:{i}", w, .001, "") for src, dst, w in local)
        for j in (i - 1, i + 1):
            if 0 <= j < segments:
                edges.extend((f"{name}:{i}", f"{name}:{j}", .16, .001, "") for name in ("f", "e"))
    return neurons, tuple(edges)


class NBSCOperator(nn.Module):
    """One explicit brain-spinal-muscle transition with trainable parameters.

    Drive has shape (B, N) and load (B, K), in N m. Optional uniform inputs
    (B, N) and (B, E, M) select sampled spikes and release counts. Omitting
    them uses a fixed voltage threshold and fixed release quantiles. The
    score counts only the branches supplied with uniform samples. Its log
    probability uses float32 accumulation with float16 or bfloat16 state.
    Spike voltage scales control the sampled law in normalized voltage units.
    The optional sensory scale applies to proprioceptive and touch cells.

    Optional send and receive controls have shape (B, E) and range [0, 1].
    The former affects newly emitted payloads. The latter scales local
    transmission after vesicle release, alongside the local synaptic weight.
    Defaults are research examples.
    """

    spike_voltage_scale = .07
    sensory_spike_voltage_scale = None
    _spindle_payload_gain = 18.
    _spindle_recruitment = 24.
    _spindle_ascending_gain = 28.
    _spindle_sensitivity = 10.
    _motor_unit_recruitment = .00625

    def __init__(self, segments=4, step_s=.001, release_sites=4, return_delay_s=.023,
                 return_weak=.50, return_thresholds=(.000005, .000012), return_memory_s=6.,
                 return_filter_s=1., *, spike_voltage_scale=.07, sensory_spike_voltage_scale=None):
        super().__init__()
        if type(segments) is not int or segments < 4:
            raise ValueError("segments must be an integer of at least four")
        if not isfinite(step_s) or not 0 < step_s <= .001:
            raise ValueError("step_s must be finite and in (0, 0.001]")
        if type(release_sites) is not int or not 1 <= release_sites <= 64:
            raise ValueError("release_sites must be an integer in [1, 64]")
        if not isfinite(return_delay_s) or return_delay_s <= 0:
            raise ValueError("return_delay_s must be finite and positive")
        low, high = return_thresholds
        if not 0 < return_weak < 1 or not 0 < low < high < 1:
            raise ValueError("return gate requires 0 < weak < 1 and 0 < low < high < 1")
        if not isfinite(return_memory_s) or return_memory_s <= 0:
            raise ValueError("return_memory_s must be finite and positive")
        if not isfinite(return_filter_s) or return_filter_s <= 0:
            raise ValueError("return_filter_s must be finite and positive")
        if not isfinite(spike_voltage_scale) or spike_voltage_scale <= 0:
            raise ValueError("spike_voltage_scale must be finite and positive")
        if sensory_spike_voltage_scale is not None and (
                not isfinite(sensory_spike_voltage_scale) or sensory_spike_voltage_scale <= 0):
            raise ValueError("sensory_spike_voltage_scale must be finite and positive")
        self.segments, self.step_s, self.release_sites = segments, step_s, release_sites
        self.return_weak, self.return_thresholds = return_weak, (low, high)
        self.return_memory_s = return_memory_s
        self.return_filter_s = return_filter_s
        self.spike_voltage_scale = spike_voltage_scale
        self.sensory_spike_voltage_scale = sensory_spike_voltage_scale
        self.neurons, self.edges = _topology(segments, return_delay_s)
        self.neuron_count, self.edge_count = len(self.neurons), len(self.edges)
        index = {name: i for i, name in enumerate(self.neurons)}
        source, target, weight, delay, modality = zip(*self.edges)
        ticks = [round(value / step_s) for value in delay]
        if any(n < 1 or not isclose(n * step_s, d, rel_tol=0, abs_tol=1e-12) for n, d in zip(ticks, delay)):
            raise ValueError("every delay must be a positive integer number of steps")
        source_index = [index[s] for s in source]
        target_index = [index[t] for t in target]
        target_neurons = sorted(set(target_index))
        target_slots = {neuron: slot for slot, neuron in enumerate(target_neurons)}
        offsets = list(accumulate((delay_steps + 1 for delay_steps in ticks), initial=0))
        self.history_length = offsets[-1]
        self.refractory_steps = round(.003 / step_s)
        self.register_buffer("source", torch.tensor(source_index), persistent=False)
        self.register_buffer("target", torch.tensor(target_index), persistent=False)
        self.register_buffer("target_neurons", torch.tensor(target_neurons), persistent=False)
        self.register_buffer("target_slot", torch.tensor(
            [target_slots[neuron] for neuron in target_index]
        ), persistent=False)
        self.register_buffer("delay", torch.tensor(ticks), persistent=False)
        self.register_buffer("send_index", torch.tensor(offsets[:-1]), persistent=False)
        self.register_buffer("arrival_index", torch.tensor(offsets[1:]) - 1, persistent=False)
        edge_index = {(s, t): i for i, (s, t) in enumerate(zip(source, target))}
        terminal = f"I:{segments - 1}"
        self.register_buffer("control_edges", torch.tensor(
            [edge_index["c", terminal], edge_index["r", terminal]]
        ), persistent=False)
        self.register_buffer("return_edge", torch.tensor([edge_index[terminal, "r"]]), persistent=False)
        self.register_buffer("stretch_edges", torch.tensor([
            [edge_index[f"p:{i}", f"{name}:{i}"] for name in ("f", "e")] for i in range(segments)
        ]), persistent=False)
        self.register_buffer("spindle_ascending", torch.tensor([
            edge_index[f"p:{i}", "c"] for i in range(segments)
        ]), persistent=False)
        self.register_buffer("release_quantiles",
                             (torch.arange(release_sites, dtype=torch.float64) + .5) / release_sites,
                             persistent=False)
        self.register_buffer("base_weight", torch.tensor(weight), persistent=False)
        self.register_buffer("binomial_coefficient", torch.tensor([comb(release_sites, k)
                             for k in range(release_sites + 1)], dtype=torch.long), persistent=False)
        gate_inh, gate_fac, alpha, beta = [], [], [], []
        for name, kind in zip(source, modality):
            segment = name.rpartition(":")[2]
            gate_inh.append(target_slots[index[f"zm{kind}:{segment}"]] if kind else 0)
            gate_fac.append(target_slots[index[f"zp{kind}:{segment}"]] if kind else 0)
            alpha.append(.55 if kind == "P" else .50 if kind == "T" else 0.)
            beta.append(.45 if kind == "P" else .35 if kind == "T" else 0.)
        self.register_buffer("gate_inh", torch.tensor(gate_inh), persistent=False)
        self.register_buffer("gate_fac", torch.tensor(gate_fac), persistent=False)
        self.register_buffer("gate_alpha", torch.tensor(alpha), persistent=False)
        self.register_buffer("gate_beta", torch.tensor(beta), persistent=False)
        sensory = [index[f"{group}:{i}"] for i in range(segments) for group in ("p", "h")]
        motor = [[index[f"f:{i}"], index[f"e:{i}"]] for i in range(segments)]
        self.register_buffer("sensory", torch.tensor(sensory), persistent=False)
        if sensory_spike_voltage_scale is not None:
            sensory_mask = torch.zeros(self.neuron_count, dtype=torch.bool)
            sensory_mask[self.sensory] = True
            self.register_buffer("_sensory_spike_mask", sensory_mask, persistent=False)
        self.register_buffer("motor", torch.tensor(motor), persistent=False)
        self.register_buffer("fiber_direction", torch.tensor([-1., 1.]), persistent=False)
        self.weight_logits = nn.Parameter(torch.zeros(self.edge_count))
        self.utilization_logits = nn.Parameter(torch.zeros(self.edge_count))
        self.stdp_logits = nn.Parameter(torch.zeros(self.edge_count, 2))
        self.bias_logits = nn.Parameter(torch.zeros(self.neuron_count))
        self.force_logits = nn.Parameter(torch.zeros(segments, 2))

    def initial_state(self, batch_size):
        """Allocate a fresh, silent prehistory on the module's device and dtype."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        n, e, k = self.neuron_count, self.edge_count, self.segments
        target_count = self.target_neurons.numel()
        zeros = self.weight_logits.new_zeros
        device = self.weight_logits.device
        return NBSCState(
            zeros((batch_size, n)), torch.zeros(batch_size, n, dtype=torch.long, device=device),
            zeros((batch_size, self.history_length)),
            zeros((batch_size, e)), zeros((batch_size, e)) + 1, zeros((batch_size, target_count)),
            zeros((batch_size, e)), zeros((batch_size, target_count)),
            zeros((batch_size, e)) + .5,
            zeros((batch_size, k, 2)), zeros((batch_size, k, 2)),
            zeros((batch_size, k)), zeros((batch_size, k)),
            zeros((batch_size, k)), zeros((batch_size, 1)), zeros((batch_size, 1)),
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
        )

    def forward(self, state, drive, load, spike_uniform=None, release_uniform=None,
                *, send_gate=None, receive_gate=None):
        h = self.step_s
        batch = state.voltage.shape[0]
        neuron_shape = (batch, self.neuron_count)
        history = getattr(state, "payload_history", None)
        if history is None:
            raise ValueError("state needs emission-time payloads; allocate it with initial_state")
        reflex_activation = getattr(state, "reflex_activation", None)
        if reflex_activation is None:
            raise ValueError("state needs stretch-reflex activation; allocate it with initial_state")
        target_shape = (batch, self.target_neurons.numel())
        edge_shape = (batch, self.edge_count)
        if (state.voltage.shape != neuron_shape or state.refractory.shape != neuron_shape
                or history.shape != (batch, self.history_length) or not history.is_floating_point()
                or state.utilization.shape != edge_shape
                or state.synaptic_current.shape != target_shape or state.post_trace.shape != target_shape
                or state.pre_trace.shape != edge_shape or state.fiber_reference.shape != load.shape
                or reflex_activation.shape != (batch, self.segments, 2)
                or state.return_difference.shape != (batch, 1) or state.return_power.shape != (batch, 1)
                or state.return_strong.shape != (batch, 1)
                or drive.shape != neuron_shape
                or (spike_uniform is not None and spike_uniform.shape != neuron_shape)
                or load.shape != (batch, self.segments)
                or (release_uniform is not None and release_uniform.shape != (*edge_shape, self.release_sites))
                or (send_gate is not None and send_gate.shape != edge_shape)
                or (receive_gate is not None and receive_gate.shape != edge_shape)):
            raise ValueError("state and inputs must match the emission-payload layout")
        # Fail fast on dtype drift. Without this the mismatch surfaces much later as
        # "index_copy_(): self and source expected to have the same dtype".
        dtype = state.voltage.dtype
        for label, tensor in (("drive", drive), ("load", load)):
            if tensor.dtype != dtype:
                raise TypeError(
                    f"{label} dtype {tensor.dtype} does not match state dtype {dtype}; "
                    f"cast the inputs (e.g. drive.to(state.voltage.dtype))"
                )
        for label, tensor in (("spike_uniform", spike_uniform),
                              ("release_uniform", release_uniform),
                              ("send_gate", send_gate), ("receive_gate", receive_gate)):
            if tensor is not None and tensor.dtype != dtype:
                raise TypeError(f"{label} dtype {tensor.dtype} does not match state dtype {dtype}")

        # Arrival uses the stored transmission gain. Synaptic efficacy is local.
        payload = history[:, self.arrival_index]
        arrivals = payload > 0
        u_decay = exp(-h / .30) * state.utilization
        utilization = u_decay + arrivals * (.05 + .55 * self.utilization_logits.sigmoid()) * (1 - u_decay)
        recovered = 1 - exp(-h / .45) * (1 - state.resource)
        eps = torch.finfo(utilization.dtype).eps
        release_prob = torch.where(arrivals, utilization.clamp(eps, 1 - eps), .5)
        release_samples = self.release_quantiles if release_uniform is None else release_uniform
        count = (release_samples < release_prob.unsqueeze(-1)).sum(dim=-1)
        count = torch.where(arrivals, count, 0)
        count_float = count.to(recovered.dtype)
        release = recovered * (count_float / self.release_sites)
        resource = recovered - release
        score_dtype = torch.float32 if recovered.dtype in (torch.float16, torch.bfloat16) else recovered.dtype
        log_probability = recovered.new_zeros(batch, dtype=score_dtype)
        if release_uniform is not None:
            score_count = count.to(score_dtype)
            score_prob = release_prob.to(score_dtype)
            binomial = self.binomial_coefficient[count].to(score_dtype).log()
            binomial = (binomial + score_count * score_prob.log()
                        + (self.release_sites - score_count) * torch.log1p(-score_prob))
            log_probability = torch.where(arrivals, binomial, 0.).sum(dim=-1)

        edge_weight = self.base_weight * (.5 + self.weight_logits.sigmoid()) * (.5 + state.plasticity)
        sensory_gate = (1 - self.gate_alpha * state.post_trace[:, self.gate_inh]) * (
            1 + self.gate_beta * state.post_trace[:, self.gate_fac]
        )
        if receive_gate is not None:
            sensory_gate = sensory_gate * receive_gate.clamp(0., 1.)
        transmission = edge_weight * payload * sensory_gate * release
        current = (exp(-h / .012) * state.synaptic_current).scatter_add(
            1, self.target_slot.expand(batch, -1), transmission)
        input_current = torch.zeros_like(state.voltage).index_copy(1, self.target_neurons, current)

        # The terminal segment compares the two descending signals after transit.
        descending = payload[:, self.control_edges]
        mismatch = descending[:, :1] - descending[:, 1:]
        difference = (exp(-h / self.return_filter_s) * state.return_difference
                      - expm1(-.003 / self.return_filter_s) * mismatch).clamp(-1., 1.)
        power = state.return_power + (-expm1(-h / self.return_memory_s)) * (
            difference.square() - state.return_power
        )
        low, high = self.return_thresholds
        strong = (power >= high) | ((power > low) & state.return_strong)
        gain = (self.return_weak + (1 - self.return_weak) * state.return_strong.to(power.dtype)) * (
            1 - .1 * state.return_power
        )
        emission = torch.ones_like(state.plasticity).index_copy(1, self.return_edge, gain)

        # The same fiber geometry supplies force and the next neural drive.
        arm, rest_length = .025, .10
        fiber_velocity = arm * state.velocity.unsqueeze(-1) * self.fiber_direction
        strain = (arm / rest_length) * state.angle.unsqueeze(-1) * self.fiber_direction
        maximum_force = 80 * (.5 + self.force_logits.sigmoid())
        force_length = torch.exp(-(strain / .25).square())
        force_velocity = 1 + .55 * torch.tanh(fiber_velocity / .25)
        reference = state.fiber_reference
        relative_strain = strain - reference.unsqueeze(-1) * self.fiber_direction
        short_strain = .025 * torch.tanh(relative_strain / .025)
        active_force = maximum_force * force_length * force_velocity * (1 + 23.4 * short_strain)
        fiber_reference = reference + (-expm1(-h / .030)) * (strain[..., 1] - reference)
        passive_force = 3 * maximum_force * strain.clamp_min(0).square()
        lengthening = torch.tanh(fiber_velocity.clamp_min(0.) / .005)
        prior_activation = state.activation + (1 - state.activation) * reflex_activation * lengthening
        prior_tension = prior_activation * active_force + passive_force
        # One sampled spindle-population volley drives two directional branches.
        stretch = 3 * relative_strain + 5 * fiber_velocity
        differential = stretch[..., 1] - stretch[..., 0]
        magnitude = torch.sqrt(differential.square() + .002 ** 2) - .002
        prop = .215 * torch.tanh(self._spindle_sensitivity * magnitude)
        ascending_payload = .30 + .70 * torch.tanh(self._spindle_ascending_gain * magnitude)
        stretch_payload = torch.tanh(self._spindle_payload_gain * stretch.clamp_min(0.))
        emission = emission.index_copy(1, self.stretch_edges.flatten(), stretch_payload.flatten(1))
        emission = emission.index_copy(1, self.spindle_ascending, ascending_payload)
        if send_gate is not None:
            emission = emission * send_gate.clamp(0., 1.)
        touch = .18 * torch.tanh(prior_tension.sum(dim=-1) / 80)
        sensory_input = torch.stack((prop, touch), dim=-1).reshape(batch, -1)
        input_current = (.16 * input_current).scatter_add(1, self.sensory.expand(batch, -1), sensory_input)

        # Noisy threshold, reset and integer absolute refractory interval.
        available = state.refractory == 0
        membrane_decay = exp(-h / .025)
        equilibrium = .28 + .30 * torch.tanh(self.bias_logits) + drive + input_current
        proposed = membrane_decay * state.voltage + (1 - membrane_decay) * equilibrium
        voltage_scale = self.spike_voltage_scale
        if self.sensory_spike_voltage_scale is not None:
            voltage_scale = torch.where(self._sensory_spike_mask,
                                        proposed.new_tensor(self.sensory_spike_voltage_scale),
                                        proposed.new_tensor(voltage_scale))
        spike_logit = (proposed - .5) / voltage_scale
        # Convert the reference 1 ms Bernoulli law to a physical survival rate.
        log_survival = (h / .001) * F.logsigmoid(-spike_logit)
        event_probability = -torch.expm1(log_survival)
        spike_probability = torch.where(available, event_probability, 0.)
        fires = proposed >= .5 if spike_uniform is None else spike_uniform < spike_probability
        spikes = available & fires
        voltage = torch.where(spikes | ~available, 0., proposed)
        refractory = torch.where(spikes, self.refractory_steps, (state.refractory - 1).clamp_min(0))
        if spike_uniform is not None:
            fire_log_probability = torch.where(spikes, event_probability, 1.).to(score_dtype).log()
            spike_log_probability = torch.where(spikes, fire_log_probability, log_survival.to(score_dtype))
            log_probability = log_probability + torch.where(available, spike_log_probability, 0.).sum(dim=-1)
        history = history.roll(1, dims=1).index_copy(1, self.send_index, spikes[:, self.source] * emission)

        # Arrival-relative STDP writes a bounded slow state for every synapse.
        pre = exp(-h / .020) * state.pre_trace
        post = exp(-h / .020) * state.post_trace
        rates = .01 + .10 * self.stdp_logits.sigmoid()
        ltp = rates[:, 0] * pre * spikes[:, self.target]
        ltd = rates[:, 1] * post[:, self.target_slot] * arrivals
        slow = .5 + exp(-h / 120.) * (state.plasticity - .5)
        plasticity = (1 - ltp - ltd) * slow + ltp
        pre_trace = torch.where(arrivals, 1., pre)
        post_trace = torch.where(spikes[:, self.target_neurons], 1., post)

        # A delayed spindle release recruits a separate fast part of the
        # available muscle capacity. Its force acts on the lengthening fiber.
        decay = exp(-h / .045) * state.activation
        motor_twitch = self._motor_unit_recruitment * spikes[:, self.motor].to(decay.dtype)
        activation = decay + motor_twitch * (1 - decay)
        reflex_decay = exp(-h / .025) * reflex_activation
        reflex_twitch = -torch.expm1(-self._spindle_recruitment * transmission[:, self.stretch_edges])
        reflex_activation = reflex_decay + reflex_twitch * (1 - reflex_decay)
        total_activation = activation + (1 - activation) * reflex_activation * lengthening
        force = total_activation * active_force + passive_force
        torque = arm * (force[..., 0] - force[..., 1]) - load - .6 * state.angle
        inertia, damping = .045, .08
        free_velocity = (state.velocity + h * torque / inertia) / (1 + h * damping / inertia)
        drift_velocity = free_velocity.clamp(-10., 10.)
        speed_impulse = (inertia + h * damping) * (drift_velocity - free_velocity)
        proposed_angle = state.angle + h * drift_velocity
        angle = proposed_angle.clamp(-1.2, 1.2)
        arrested_velocity = torch.where(proposed_angle.abs() >= 1.2, drift_velocity, 0.)
        stop_impulse = -inertia * arrested_velocity
        joint_velocity = drift_velocity - arrested_velocity

        updated = NBSCState(voltage, refractory, history, utilization, resource, current,
                            pre_trace, post_trace, plasticity, activation, reflex_activation, angle, joint_velocity,
                            fiber_reference, difference, power, strong)
        return NBSCOutput(updated, spikes, spike_probability, count, release, edge_weight,
                          force, torque, log_probability, payload, speed_impulse, stop_impulse)


# ==============================================================================
# Standalone verification suite
# ==============================================================================
def run_nbsc_omega_verification() -> None:
    """自检：构造、脉冲、静默性、边界与可微性。"""
    import numpy as np  # noqa: F401  (仅用于可选路径)

    try:
        import torch
    except ImportError:
        print("[SKIP] torch is required for the NBSC-Omega verification suite.")
        return

    op = NBSCOperator(segments=4)
    n_par = sum(p.numel() for p in op.parameters())
    print(f"NBSC-Omega operator: {op.neuron_count} neurons, {op.edge_count} edges, "
          f"{n_par} trainable parameters")

    state = op.initial_state(2)
    drive = torch.zeros(2, op.neuron_count)
    load = torch.zeros(2, op.segments)

    out = op(state, drive, load)
    print(f"[PASS] single step: spikes={tuple(out.spikes.shape)}, "
          f"torque={tuple(out.joint_torque.shape)}")

    # 无驱动必须完全静默
    st = state
    spikes = 0
    for _ in range(100):
        o = op(st, drive, load)
        spikes += int(o.spikes.sum())
        st = o.state
    assert spikes == 0, f"silent path produced {spikes} spikes"
    assert torch.all(st.angle == 0), "silent path moved the joint"
    print("[PASS] silent when undriven: 100 steps, zero spikes, zero motion")

    # 有驱动必须产生脉冲，且全程有界
    torch.manual_seed(0)
    drive = torch.zeros(2, op.neuron_count)
    drive[:, :2] = 1.5
    drive[:, 2:] = 0.8
    load = torch.full((2, op.segments), 0.5)
    st = state
    spikes = 0
    for _ in range(300):
        o = op(st, drive, load)
        spikes += int(o.spikes.sum())
        st = o.state
    assert spikes > 0, "driven path produced no spikes"
    for field in ("voltage", "utilization", "resource", "angle", "velocity"):
        assert torch.isfinite(getattr(st, field)).all(), f"{field} not finite"
    print(f"[PASS] driven: 300 steps, {spikes} spikes, all state fields finite")

    # 反向可达
    op.zero_grad(set_to_none=True)
    st = state
    for _ in range(30):
        o = op(st, drive, load)
        st = o.state
    o.joint_torque.abs().sum().backward()
    got = [n for n, p in op.named_parameters() if p.grad is not None]
    assert got, "no parameter received a gradient"
    print(f"[PASS] backward reaches {len(got)}/{len(list(op.named_parameters()))} parameter tensors")
    print("ALL NBSC-OMEGA VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_nbsc_omega_verification()
