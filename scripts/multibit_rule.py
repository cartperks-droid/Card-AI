"""The discrete strategic transformer at b bits per weight and activation (b >= 2), for scripts/learning_rule_bench.py.

    python3 scripts/learning_rule_bench.py --rule scripts/multibit_rule.py --freeze-language \\
        --rule-args --precision 4 --bits 1024 --strength 6 --weak-block 0.001 --init-spread 1

The same network as scripts/discrete_rule.py (tokens, field layers, attention, FFN, init, continuous tail), with every
1-bit operation replaced by its b-bit form on the fixed-point grid {0, 1/M, ..., 1}, M = 2^b - 1. Each form is the
1-bit operation exactly at b = 1, and an ordinary continuous operation as b grows (user, 2026-10-19):
  AND (input x link)      w x
  OR over inputs          saturating sum: min(1, sum)
  NOT                     1 - y
  XOR (residual stream)   addition modulo 1 on the grid: (X + C) mod 2^b (XOR is addition in Z/2)
  neuron output sampled   stochastic rounding of y to the grid (at b = 1: Bernoulli(y))
  attention score         sum q k;   NOR mixing across tokens: 1 - min(1, sum_j a_ij v_j)
Weights: b bits per weight, bit k (0 = least significant) on with p_k = sigma(2^k l_k): each added bit is twice as sharp
as the one below, so as b grows the weight's value lives in near-deterministic high bits and only the finest bit keeps
the 1-bit model's stochasticity (user: the weights become more deterministic as the network becomes continuous).
W = sum_k 2^k bit_k / M. The init is the 1-bit model's: every bit of a weight starts at its 1-bit probability, so the
network starts as the 1-bit network and learns intermediate values.

Sampling (low variance, per neuron per battle, as the 1-bit neuron sampling): a neuron's input sum is drawn directly
from its distribution, mean sum x E[W] and variance sum x^2 Var[W] (local reparameterization, Gaussian), instead of
drawing every weight bit. Learning: the draw is smooth in the bit probabilities, so backprop gives each bit its exact
gradient; the stochastic rounding of outputs is the only discrete step: straight-through for its exact linear part,
REINFORCE on what the linearisation misses (residual = battle loss minus its first-order rounding term, leave-one-out
batch baseline). The continuous tail by backprop + AdamW, as discrete_rule.
"""

import math
import sys

import torch
import torch.nn as nn

import discrete_rule as dr

CLAMP = dr.CLAMP


def bit_scale(K, device):
    return (2.0 ** torch.arange(K, device=device, dtype=torch.float32))  # beta_k = 2^k, also the bit's value


class MultiGate(dr.Gate):
    """A Gate whose weights (and bias) carry K bit logits each: w [K, (G,) nin, nout], b [K, (G,) nout]."""

    K = 4

    def initialise(self, a):
        super().initialise(a)  # the 1-bit derived init: one logit per weight
        K = self.K
        beta = bit_scale(K, self.w.device).view(K, *[1] * self.w.ndim)
        self.w = nn.Parameter((self.w.detach()[None] / beta).clamp(-CLAMP, CLAMP))  # sigma(beta l) = the 1-bit sigma(w)
        betab = bit_scale(K, self.b.device).view(K, *[1] * self.b.ndim)
        self.b = nn.Parameter((self.b.detach()[None] / betab).clamp(-CLAMP, CLAMP))

    def moments(self):
        """(E[W], Var[W]) of the weights and the bias, on the masked connections."""
        K = self.K
        M = 2 ** K - 1

        def stats(logits):
            v = bit_scale(K, logits.device).view(K, *[1] * (logits.ndim - 1))
            p = torch.sigmoid(v * logits)
            return (v * p).sum(0) / M, (v * v * p * (1 - p)).sum(0) / (M * M)
        W, V = stats(self.w)
        mask = self.mask if self.w.ndim == 3 else self.mask
        W, V = W * mask, V * mask
        if self.use_bias:
            Wb, Vb = stats(self.b)
        else:
            Wb = Vb = torch.zeros_like(self.b[0])
        return W, V, Wb, Vb

    def clamp_(self):
        with torch.no_grad():
            for t in (self.w, self.b):
                v = bit_scale(self.K, t.device).view(self.K, *[1] * (t.ndim - 1))
                t.copy_(torch.maximum(torch.minimum(t, CLAMP / v), -CLAMP / v))


class MultiFire:
    """Neurons at b bits: the input sum drawn per battle (mean + sd * noise), saturated, NOT / OR, stochastically
    rounded. Collects, per battle, the rounding decisions' log-probability and their first-order terms."""

    def __init__(self, K):
        self.K, self.M = K, 2 ** K - 1
        self.record = self.hard = False
        self.stats = {}
        self.rounding = []  # (straight-through tensor, deviation) per rounded tensor
        self.logp = None

    def round(self, y, name):
        M = self.M
        z = y.clamp(0, 1) * M
        lo = z.detach().floor()
        frac = (z - lo).clamp(0, 1)
        if self.hard:
            up = (frac.detach() >= 0.5).float()
        else:
            up = (torch.rand_like(frac) < frac.detach()).float()
        yr = (lo + up) / M
        out = y + (yr - y).detach()  # straight-through: the value is the rounded one, the gradient the linear part
        if self.record:
            f = frac.clamp(1e-6, 1 - 1e-6)
            lp = (up * f.log() + (1 - up) * (1 - f).log())
            lp = lp * ((z.detach() - lo) > 0)  # values already on the grid were not rounded: no choice, no score
            lp = lp.flatten(1).sum(1)
            self.logp = lp if self.logp is None else self.logp + lp
            if out.requires_grad:
                self.rounding.append((out, (yr - y).detach()))
            rate = self.stats.setdefault(name, [0.0, 0])
            rate[0] += float(yr.mean())
            rate[1] += 1
        return out

    def __call__(self, gate, x):
        if gate.w is None:
            gate.initialise(float(x.detach().mean()))
        if gate.any_and or set(gate.ops) - {"not", "linear"}:
            raise SystemExit("multibit: or-not and or-linear neurons only")
        W, V, Wb, Vb = gate.moments()
        if gate.w.ndim == 4:  # grouped [K, G, nin, nout] -> [G, nin, nout]
            mu = torch.einsum("btgi,gio->btgo", x, W) + Wb
            var = torch.einsum("btgi,gio->btgo", x * x, V) + Vb
        else:
            mu = x @ W + Wb
            var = (x * x) @ V + Vb
        s = mu if self.hard else mu + var.clamp_min(0).add(1e-12).sqrt() * torch.randn_like(mu)
        pre = s.clamp(0, 1)  # OR: the saturating sum
        y = pre if gate.ops == ["linear"] else 1 - pre  # NOT
        return self.round(y, gate.name)


def mod_add(x, c, M):
    """XOR at b bits: (X + C) mod 2^b on the grid, with gradient 1 through both (the wrap is a constant offset)."""
    total = x + c
    wrap = (total.detach() * M + 0.5).floor() >= M + 1  # X + C >= 2^b
    return total - wrap.float() * (M + 1) / M


class MultiLayer(dr.BitLayer):
    def forward(self, x, fire):
        B, T, _ = x.shape
        q, k, v = fire(self.q, x), fire(self.k, x), fire(self.v, x)
        qh, kh, vh = (t.view(B, T, self.H, self.S).transpose(1, 2) for t in (q, k, v))
        s = qh @ kh.transpose(-1, -2)
        s = (s - s.mean(-1, keepdim=True)) / self.tau
        att = torch.sigmoid(s - self.offset)
        u = fire.round(1 - (att @ vh).clamp(0, 1), "mixing").transpose(1, 2)  # NOR across tokens
        props = fire(self.prop, u)  # [B, T, H, D]
        change = 1 - props.sum(2).clamp(0, 1)  # NOR across heads
        x = mod_add(x, change, fire.M)
        g = fire(self.fg, x)
        flip = 1 - fire(self.ff, g)  # OR
        return mod_add(x, flip, fire.M)


class MultiStack(dr.BitStack):
    def __init__(self, model, cfg, layers, generator):
        super().__init__(model, cfg, layers, generator)
        self.fire = MultiFire(cfg.precision)

    def field_gates(self, kind, fields, device):
        gates = super().field_gates(kind, fields, device)
        for name, pair in gates.items():
            for i, gate in enumerate(pair):
                pair[i] = self.multi(gate)
        return gates

    def multi(self, gate):
        gate.__class__ = MultiGate
        gate.K = self.cfg.precision
        return gate

    def build(self, card_fields, other_fields):
        super().build(card_fields, other_fields)
        for layer in self.layers:
            layer.__class__ = MultiLayer
            for gate in (layer.q, layer.k, layer.v, layer.prop, layer.fg, layer.ff):
                self.multi(gate)
        if self.mids:
            raise SystemExit("multibit: --interleave is not supported")

    def embed(self, gates, fields):
        x = None
        for name, (first, second) in gates.items():
            bits = self.fire(second, self.fire(first, fields[name]))
            x = bits if x is None else mod_add(x, bits, self.fire.M)
        return x


class MultiRule(dr.DiscreteRule):
    def __init__(self, model, args):
        super().__init__(model, args)
        cfg = self.cfg
        if cfg.precision < 2:
            raise SystemExit("multibit: --precision >= 2 (b = 1 is scripts/discrete_rule.py)")
        if cfg.embed != "codes" or cfg.credit != "batch" or cfg.sampling != "neuron":
            raise SystemExit("multibit: code embeddings, batch credit, neuron sampling")
        old = self.stack
        self.stack = MultiStack(model, cfg, old.n_layers, old.generator).to(old.output.weight.device)
        model.strategy.blocks[0] = self.stack
        self.stack.output.requires_grad_(True)

    def ensure_built(self, forward, batch):
        if self.stack.built:
            return
        super().ensure_built(forward, batch)
        self.gate_params = [p for g in self.stack.gates() for p in (g.w, g.b)]

    def step(self, forward, batch):
        self.ensure_built(forward, batch)
        fire = self.stack.fire
        fire.record, fire.logp, fire.stats, fire.rounding = True, None, {}, []
        logits = forward(batch)
        fire.record = False
        per = -(batch["target"] * logits.float().log_softmax(-1)).sum(-1)
        value = per.mean()
        B = per.shape[0]
        # first-order rounding term per battle: sum over rounded values of dL_b/dy * (rounded - y)
        tensors = [t for t, _ in fire.rounding]
        grads = torch.autograd.grad(per.sum(), tensors, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            lin = torch.zeros(B, device=per.device)
            for g, (_, dev) in zip(grads, fire.rounding):
                if g is not None:
                    lin += (g * dev).flatten(1).sum(1)
            residual = per.detach() - lin
            adv = residual - (residual.sum() - residual) / (B - 1)  # leave-one-out baseline
        surrogate = (adv * fire.logp).mean() if fire.logp is not None else 0.0
        fire.rounding = []
        self.optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
        (value + surrogate).backward()
        torch.nn.utils.clip_grad_norm_(self.cont_params, 1.0)
        self.optimizer.step()
        self.gate_optimizer.step()
        for gate in self.stack.gates():
            gate.clamp_()
        self.steps += 1
        if self.steps % 100 == 1:
            rates = " ".join(f"{k}={v[0] / v[1]:.3f}" for k, v in fire.stats.items() if not k.startswith(("card-", "other-")))
            with torch.no_grad():  # how decided each bit position is: mean |2p - 1| over every weight
                K = self.cfg.precision
                dec = torch.zeros(K)
                n = 0
                for g in self.stack.gates():
                    v = bit_scale(K, g.w.device).view(K, *[1] * (g.w.ndim - 1))
                    p = torch.sigmoid(v * g.w)
                    dec += (2 * p - 1).abs().flatten(1).mean(1).cpu()
                    n += 1
                dec = " ".join(f"b{k}={float(dec[k]) / n:.3f}" for k in range(K))
            print(f"[multibit] step {self.steps} loss {float(value.detach()):.4f} decided: {dec} | {rates}", flush=True)
        return float(value.detach())


_parse = dr.rule_args


def rule_args(args):
    precision = 4
    rest = list(args.rule_args)
    if "--precision" in rest:
        i = rest.index("--precision")
        precision = int(rest[i + 1])
        del rest[i:i + 2]
    args.rule_args = rest
    cfg = _parse(args)
    cfg.precision = precision
    return cfg


dr.rule_args = rule_args  # the rule's own --precision on top of discrete_rule's arguments


def make_rule(model, args):
    rule = MultiRule(model, args)
    rule.args_lr = args.lr
    return rule
