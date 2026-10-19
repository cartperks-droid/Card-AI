"""A discrete (bitwise) equivalent of the strategic transformer, for scripts/learning_rule_bench.py.

    python3 scripts/learning_rule_bench.py --rule scripts/discrete_rule.py --freeze-language --rule-args --bits 1024

The battle's 15 tokens are built as bits, and all but the last 2 strategic layers are bitwise layers (the btx design,
bnn/btx.c). The last 2 continuous blocks, the outcome head and the stat prior are the model's own.

Gates: every learned connection is a transmission probability sigma(w); a neuron is an input-wide OR followed by NOT
(NOR) with an always-1 bias bit: P(fire) = (1 - sigma(b)) prod_j (1 - sigma(w_j) x_j), sampled.

Tokens (D bits each):
  - each input field through its own two layers (field bits -> D NOR -> D OR: no NOT before an XOR), the fields XORed (the model's summed
    embeddings). Card fields: identity, border, mutation, pack, classes, description sign bits, thermometer bins of
    the 37 stat inputs (the model's stat_inputs), slot, hidden. Other tokens: slot, support, tier, mode, hidden.
    Categorical fields are fixed random dense codes (one-hot inputs defeat the derived init: q > 1).
Bit layer (no mask, like the model):
  Q, K, V   NOR layers D -> D/4 bits, H heads of S = D/4/H bits.
  scores    s_ij = popcount(Q_i^h AND K_j^h), centred over j; a_ij = sigmoid((s - mean)/tau - offset).
  mixing    P(u_i[b] = 1) = prod_{j : V_j[b] = 1} (1 - a_ij)   (NOR across the attended tokens).
  proposal  per head a NOR layer S -> D; change = NOR of the heads' proposals; x <- x XOR change.
  FFN       g = NOR layer D -> 4D, flip = OR layer 4D -> D; x <- x XOR flip.
Then Linear(D, width) into the last 2 continuous blocks.

Learning: the continuous part (projection, the 2 blocks, the head) by backpropagation + AdamW, as the bench's backprop
rule (clip 1, weight decay 0.05). The gates by REINFORCE, one sample per battle (K = 1): advantage = the battle's loss
minus a baseline (the mean loss of the other battles in the batch, or an EMA), Adam on the logits, clamped to +-16.
Bits are samples: no gradient passes through them (no straight-through to the gates).
Init: the derived binary rule of bnn/btx.c (strong +S with probability min(q, 1), every other link silent at the clamp
-16, so each neuron starts as the Boolean function of its strong links; bias strong where the inputs cannot reach the
target firing rate p). Connectivity: each input node fully connected until 1024 outgoing connections.
Anti-regularisation (--antireg): logit *= 1 + lr * lambda after each step (biases: --antireg-bias).
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_rule_bench import loss

FANCAP = 1024
CLAMP = 16.0


def rule_args(args):
    parser = argparse.ArgumentParser(prog="discrete_rule")
    parser.add_argument("--bits", type=int, default=1024, help="D: bits per token")
    parser.add_argument("--heads", type=int, default=4, help="bit attention heads")
    parser.add_argument("--continuous", type=int, default=2, help="continuous blocks kept at the end")
    parser.add_argument("--glr", type=float, default=0.01, help="gate Adam learning rate (logits)")
    parser.add_argument("--p", type=float, default=0.5, help="target firing rate at init")
    parser.add_argument("--pflip", type=float, default=0.1, help="target change/flip rate at init")
    parser.add_argument("--strength", type=float, default=16.0, help="binary init: strong logit S")
    parser.add_argument("--bias", choices=["on", "off"], default="on", help="off: no bias bit in any gate")
    parser.add_argument("--weak-block", type=float, default=0.0,
                        help="binary init: what a neuron's non-strong links block together (a m w_weak); 0: silent at -16")
    parser.add_argument("--init-spread", type=float, default=0.0,
                        help="binary init: each logit drawn around its derived value with this standard deviation, so "
                             "links do not share one probability (0: every strong / weak link identical)")
    parser.add_argument("--route", type=float, default=0.0,
                        help="link structure: this fraction of each layer's neurons start as a pure NOT of one input")
    parser.add_argument("--sampling", choices=["neuron", "link"], default="neuron",
                        help="neuron: each neuron sampled from its exact probability, REINFORCE on that; link: every link "
                             "transmits by its own draw (hard sampled circuits) and REINFORCE scores each link's draw")
    parser.add_argument("--circuits", type=int, default=32, help="link sampling: circuits drawn per batch")
    parser.add_argument("--embed", choices=["codes", "learned"], default="codes",
                        help="categorical fields: fixed random codes, or learned tables of bit probabilities collapsed "
                             "(sampled) when the token is built")
    parser.add_argument("--credit", choices=["batch", "semiprop"], default="batch",
                        help="batch: one per-battle advantage for every gate; semiprop: Main's gates learn only from a "
                             "small discrete error network H (per-bit credit x a learned trust of the battle's input), and "
                             "H learns by long-term REINFORCE (eligibility trace over the future loss)")
    parser.add_argument("--h-depth", type=int, default=2, help="semiprop: H layers developing each layer's error")
    parser.add_argument("--h-trace", type=float, default=0.9, help="semiprop: eligibility trace decay (EMA of the future)")
    parser.add_argument("--h-lr", type=float, default=1.0,
                        help="semiprop: H's reward-modulated Hebbian rate (weight change = rate x reward x co-activity trace)")
    parser.add_argument("--sp-lr", type=float, default=1e-3, help="semiprop: H's credit projections / trust")
    parser.add_argument("--trust", choices=["regression", "hopfield", "boltzmann"], default="boltzmann",
                        help="xor: trust by a regression network on the input (collapsed to ~0: per-battle performance is "
                             "mostly noise), a modern Hopfield memory of past inputs and Main's performance on them (no hidden "
                             "units: it averaged to noise), or a Boltzmann-style network with hidden units, settled per "
                             "recall and learned by contrastive Hebbian learning")
    parser.add_argument("--trust-hidden", type=int, default=256, help="boltzmann trust: hidden units")
    parser.add_argument("--trust-rate", type=float, default=0.01, help="boltzmann trust: contrastive Hebbian rate")
    parser.add_argument("--hopfield-beta", type=float, default=8.0, help="hopfield trust: recall sharpness")
    parser.add_argument("--h-update", choices=["readout", "xor"], default="xor",
                        help="semiprop: readout = a continuous credit per Main bit (fitted to per-bit REINFORCE samples), "
                             "Main moved by trust x credit x dP; xor = discrete error bits (their chance to trigger lowered "
                             "by the pass's trust), Main driven toward its activation XOR H's error, H learning by modern "
                             "Hebbian learning with trust as the modulatory signal")
    parser.add_argument("--embed-logit", type=float, default=4.0, help="learned tables: initial |logit| of each bit")
    parser.add_argument("--interleave", type=int, default=0,
                        help="N > 0: a continuous block after every N bit layers (not after the last), its flips XORed in")
    parser.add_argument("--mid-bias", type=float, default=-4.0, help="interleaved blocks: initial flip logit")
    parser.add_argument("--att-offset", type=float, default=3.0)
    parser.add_argument("--bins", type=int, default=8, help="thermometer bins per stat input")
    parser.add_argument("--baseline", choices=["batch", "ema"], default="batch")
    parser.add_argument("--antireg", type=float, default=0.0, help="lambda: logit *= 1 + glr * lambda per step")
    parser.add_argument("--antireg-bias", choices=["same", "none", "lower"], default="none")
    parser.add_argument("--act", default="or-not",
                        help="neuron type(s), comma-separated to mix them (neuron k takes the k-th, cycling): " + ", ".join(ACTS))
    parser.add_argument("--save", help="checkpoint folder: best.pt whenever validation KL reaches a new low, and latest.pt "
                                       "(the final state, to extend the run) when the run ends")
    parser.add_argument("--eval-mode", choices=["sample", "hard"], default="sample",
                        help="evaluation: sampled bits, or the hardened model (each gate fires iff P > 1/2)")
    return parser.parse_args(args.rule_args)


# Neuron types: a global op over the inputs, then an activation. Global OR: g = OR of the transmitted active inputs and
# the bias; global AND: g = 1 unless a 0 input or the bias blocks (each with its transmission probability). Either way an
# inner event E = "nothing transmitted / nothing blocked" has P(E) = exp(acc) (acc over x for OR, over 1 - x for AND),
# and g = 1 - E (OR) or E (AND). Activations: not, linear, and pair ops with the next neuron's g (wrapping at the end).
ACTS = {  # name: (global AND?, activation, P(E) giving output rate t)
    "or-not": (False, "not", lambda t: t),
    "or-linear": (False, "linear", lambda t: 1 - t),
    "or-and": (False, "and", lambda t: 1 - t ** 0.5),
    "or-nand": (False, "nand", lambda t: 1 - (1 - t) ** 0.5),
    "and-not": (True, "not", lambda t: 1 - t),
    "and-linear": (True, "linear", lambda t: t),
    "and-or": (True, "or", lambda t: 1 - (1 - t) ** 0.5),
    "and-nor": (True, "nor", lambda t: 1 - t ** 0.5),
}
OPS = ("not", "linear", "and", "nand", "or", "nor")


def log1m_sig(w):
    return -F.softplus(w)  # ln(1 - sigma(w))


class Gate(nn.Module):
    """A layer of probabilistic gates nin -> nout (one per group when groups > 1: weights [G, nin, nout]), of the
    neuron types cfg.act (ACTS). `target` is the output firing rate the derived init aims at; each type's inner event
    gets the rate that produces it, and AND neurons see the rate of zeros, 1 - a. The weights are drawn on the gate's
    first input (initialise), from the measured input rate a, so every layer starts at its target whatever the layers
    before it fire at."""

    def __init__(self, name, nin, nout, target, cfg, generator, groups=1, act=None):
        super().__init__()
        self.name = name
        self.shape = (groups, nin, nout)
        self.target, self.strength, self.generator = target, cfg.strength, generator
        self.use_bias = cfg.bias == "on"
        self.weak_block, self.route, self.spread = cfg.weak_block, cfg.route, cfg.init_spread
        self.w = self.b = None
        fan = min(nout, FANCAP)
        mask = torch.zeros(nin, nout, dtype=torch.bool)
        if fan >= nout:
            mask[:] = True
        else:  # implicit balanced targets (A j + B c) mod nout, A and B coprime to nout (bnn.c)
            coprimes = [k for k in range(1, nout) if math.gcd(k, nout) == 1]
            A = coprimes[int(torch.randint(len(coprimes), (1,), generator=generator))]
            B = coprimes[int(torch.randint(len(coprimes), (1,), generator=generator))]
            j = torch.arange(nin)[:, None]
            c = torch.arange(fan)[None, :]
            mask[j.expand(-1, fan), (A * j + B * c) % nout] = True
        fanin = mask.sum(0).clamp_min(1).double()  # [nout]
        kinds = [ACTS[name] for name in (act or cfg.act).split(",")]
        per = [kinds[k % len(kinds)] for k in range(nout)]
        is_and = torch.tensor([kind[0] for kind in per])
        self.register_buffer("is_and", is_and, persistent=False)
        self.register_buffer("op", torch.tensor([OPS.index(kind[1]) for kind in per]), persistent=False)
        self.ops = sorted(set(kind[1] for kind in per))
        self.register_buffer("sign", torch.where(is_and, -1.0, 1.0), persistent=False)
        self.any_and = bool(is_and.any())
        self.inner = torch.tensor([kind[2](target) for kind in per], dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
        self.fanin = fanin
        self.register_buffer("mask", mask, persistent=False)

    def initialise(self, a):
        """The derived binary init (bnn/btx.c) for input rate a: each connection strong (+S) with probability min(q, 1),
        else silent (logit -16: the link exists and can learn, but starts transmitting ~1e-7), the bias strong where the
        inputs cannot reach the inner target alone."""
        groups, nin, nout = self.shape
        fanin, target, generator, S = self.fanin, self.inner, self.generator, self.strength
        a = max(a, 1e-6)
        a = torch.where(self.is_and.cpu(), max(1 - a, 1e-6), a).double()
        q = (1 - target ** (1 / fanin)) / a
        shape = self.shape
        strong = torch.rand(shape, generator=generator, dtype=torch.float64) < q.clamp(max=1)
        if self.weak_block > 0:  # weak links: together they block about weak_block (a m w_weak = weak_block)
            ww = (self.weak_block / (a * fanin)).clamp(max=0.5)
            weak = (ww.log() - (-ww).log1p()).float().expand(shape)
        else:
            weak = torch.full(shape, -CLAMP)
        w = torch.where(strong, torch.tensor(S, dtype=torch.float32), weak).float()
        if self.spread > 0:  # weak links on the sigmoid's exponential tail: centre lowered by spread^2 / 2 so their
            # expected transmission (what they block together) stays at its derived value
            w = torch.where(strong, w, w - self.spread ** 2 / 2) + self.spread * torch.randn(shape, generator=generator)
        pin = (1 - a * q.clamp(max=1)) ** fanin
        bias_strong = (pin > target * 1.0001) & (torch.rand((groups, nout), generator=generator, dtype=torch.float64)
                                                  < 1 - target / pin)
        # an off bias is one more weak link (an always-active input): the neuron's weak transmission, or silent; at -S it
        # blocked sigma(-S) of the time whatever the input (S=4: 1.8%, S=6: 0.25%, more than the weak links' 0.1%)
        off = weak[:, 0, :] - (self.spread ** 2 / 2 if self.weak_block > 0 else 0.0)
        b = torch.where(bias_strong, torch.tensor(S, dtype=torch.float32), off).float() if self.use_bias else torch.full((groups, nout), -CLAMP)
        if self.spread > 0 and self.use_bias:
            b = b + self.spread * torch.randn((groups, nout), generator=generator)
        w, b = w.clamp(-CLAMP, CLAMP), b.clamp(-CLAMP, CLAMP)
        if self.route > 0:  # pure NOT neurons: one strong link, nothing else, bias off (pass-through paths)
            every = max(1, round(1 / self.route))
            mask = self.mask.cpu()
            for k in range(0, nout, every):
                inputs = mask[:, k].nonzero().flatten()
                j = int(inputs[(k // every * 7919) % len(inputs)])
                w[:, :, k] = weak[:, :, k] if self.weak_block > 0 else -CLAMP
                w[:, j, k] = S
                b[:, k] = -S if self.use_bias else -CLAMP
        if groups == 1:
            w, b = w[0], b[0]
        self.w = nn.Parameter(w.to(self.mask.device))
        self.b = nn.Parameter(b.to(self.mask.device))

    def logs(self):
        """(ln(1 - sigma(w)) on the connections, ln(1 - sigma(b)))."""
        Lb = log1m_sig(self.b) if self.use_bias else torch.zeros_like(self.b)
        return log1m_sig(self.w) * self.mask, Lb


def accumulate(x, L, Lb, gate):
    """ln P(E) for 0/1 inputs x [B, T, nin] (or [B, T, G, nin] with grouped L [G, nin, nout]): sum of L over active
    inputs (OR neurons) or inactive ones (AND neurons: (1 - x) L = colsum(L) - x L)."""
    if gate.any_and:
        Lb = Lb + (L * gate.is_and).sum(-2)
        L = L * gate.sign
    if L.ndim == 3:
        return torch.einsum("btgi,gio->btgo", x, L) + Lb
    return x @ L + Lb


def activate(E, gate):
    """The neurons' outputs from their sampled inner events E [..., nout]."""
    g = torch.where(gate.is_and, E, 1 - E)
    if gate.ops == ["not"] and not gate.any_and:
        return E
    nxt = g.roll(-1, -1)  # the next neuron's g, wrapping at the end
    out = torch.empty_like(g)
    for name in gate.ops:
        value = {"not": lambda: 1 - g, "linear": lambda: g, "and": lambda: g * nxt, "nand": lambda: 1 - g * nxt,
                 "or": lambda: torch.maximum(g, nxt), "nor": lambda: 1 - torch.maximum(g, nxt)}[name]()
        out = torch.where(gate.op == OPS.index(name), value, out) if len(gate.ops) > 1 else value
    return out


class Score(torch.autograd.Function):
    """sum over a battle's gates of ln p(n), whose gradient is the REINFORCE score: d ln p / d acc = rho = (n - P)/(1 - P).
    Saves only the input bits and rho (half precision), as the btx tape, instead of the whole autograd graph."""

    @staticmethod
    def forward(ctx, L, Lb, x, rho, lp, sign, is_and):
        ctx.save_for_backward(x, rho, sign, is_and)
        ctx.grouped = L.ndim == 3
        return lp

    @staticmethod
    def backward(ctx, grad):  # grad [B]: the advantage / B
        x, rho, sign, is_and = ctx.saved_tensors  # d acc / d L[i, o] = sign_o x_i + [o is AND]
        g = rho.float() * grad.view(-1, *([1] * (rho.ndim - 1)))
        x = x.float()
        if ctx.grouped:
            gb = g.sum((0, 1))
            gL = torch.einsum("btgi,btgo->gio", x, g) * sign + (gb * is_and)[:, None, :]
        else:
            gb = g.flatten(0, -2).sum(0)
            gL = (x.flatten(0, -2).T @ g.flatten(0, -2)) * sign + gb * is_and
        return gL, gb, None, None, None, None, None


class LinkScore(torch.autograd.Function):
    """Per battle, the REINFORCE score of the link draws of its circuit: d ln p(T) / dw = T - sigma(w) for every link
    whose input was active in the battle (a link whose input never fired cannot have changed its loss); the bias always
    counts. Its value is 0; backward receives A_b / B."""

    @staticmethod
    def forward(ctx, w, b, T, tb, act, mask, use_bias):
        ctx.save_for_backward(w.detach(), b.detach(), T, tb, act, mask)
        ctx.use_bias = use_bias
        return w.new_zeros(act.shape[0])

    @staticmethod
    def backward(ctx, grad):
        w, b, T, tb, act, mask = ctx.saved_tensors
        G, B = T.shape[0], act.shape[0]
        U = (grad.view(G, B // G, *[1] * (act.ndim - 1)) * act.view(G, B // G, *act.shape[1:])).sum(1)  # [G, nin(, ..)]
        Tf = T.float()
        if w.ndim == 3:  # grouped [H, S, D]: U [G, H, S]
            gw = torch.einsum("ghs,ghsd->hsd", U, Tf) - U.sum(0)[..., None] * torch.sigmoid(w)
        else:
            gw = torch.einsum("gi,gio->io", U, Tf) - U.sum(0)[:, None] * torch.sigmoid(w)
        gw = gw * mask
        Ub = grad.view(G, B // G).sum(1)
        gb = (torch.einsum("g,g...->...", Ub, tb.float()) - Ub.sum() * torch.sigmoid(b)) if ctx.use_bias else torch.zeros_like(b)
        return gw, gb, None, None, None, None, None


class BitLayer(nn.Module):
    def __init__(self, D, H, cfg, generator):
        super().__init__()
        self.D, self.H, self.S = D, H, D // 4 // H
        self.tau = max(1.0, math.sqrt(self.S * cfg.p ** 2))
        self.offset = cfg.att_offset
        dq = D // 4
        self.q = Gate("Q", D, dq, cfg.p, cfg, generator)
        self.k = Gate("K", D, dq, cfg.p, cfg, generator)
        self.v = Gate("V", D, dq, cfg.p, cfg, generator)
        self.prop = Gate("proposal", self.S, D, 1 - cfg.pflip ** (1 / H), cfg, generator, groups=H)
        self.fg = Gate("ffn-NOR", D, 4 * D, cfg.p, cfg, generator)
        self.ff = Gate("ffn-flip", 4 * D, D, 1 - cfg.pflip, cfg, generator)

    def forward(self, x, fire):
        B, T, _ = x.shape
        q, k, v = fire(self.q, x), fire(self.k, x), fire(self.v, x)
        qh, kh, vh = (t.view(B, T, self.H, self.S).transpose(1, 2) for t in (q, k, v))  # [B, H, T, S]
        s = qh @ kh.transpose(-1, -2)  # popcount(Q_i AND K_j) [B, H, T, T]
        s = (s - s.mean(-1, keepdim=True)) / self.tau
        att = torch.sigmoid(s - self.offset)
        if fire.hard:
            att = (att > 0.5).float()
        logp = (-att).log1p().clamp_min(-32.0) @ vh  # ln P(u = 1) [B, H, T, S]
        u = fire.sample(logp).transpose(1, 2)  # [B, T, H, S]  (parameter-free: no score term)
        props = fire(self.prop, u)  # [B, T, H, D]
        change = 1 - props.amax(2)  # NOR across heads
        x = (x + change) % 2  # XOR
        g = fire(self.fg, x)
        flip = 1 - fire(self.ff, g)  # OR
        return (x + flip) % 2


class Firing:
    """Samples gates and collects their log-probabilities per battle (for the REINFORCE surrogate)."""

    def __init__(self, link=False, circuits=32):
        self.record = False
        self.hard = False
        self.logp = None
        self.stats = {}
        self.link, self.circuits = link, circuits
        self.layer, self.logp_layers = 0, {}
        self.tape = None
        self.hebb = False  # H's gates: collect centred co-activity (reward-modulated Hebbian learning)
        self.hebb_weight = None  # [B]: per-battle modulatory signal folded into the co-activity (xor: trust)

    def add(self, lp):
        """A gate's per-battle log-probability, into the total and into its layer's (semiprop credit)."""
        self.logp = lp if self.logp is None else self.logp + lp
        self.logp_layers[self.layer] = lp if self.layer not in self.logp_layers else self.logp_layers[self.layer] + lp

    def link_fire(self, gate, x):
        """True stochastic binary (or-not): each of `circuits` battle groups runs its own hard circuit, every link
        transmitting by its own draw T ~ Bernoulli(sigma(w)); a neuron fires iff none of its active links (nor its bias)
        transmitted. No neuron probability is computed: REINFORCE scores each link's draw (LinkScore)."""
        B, grouped = x.shape[0], gate.w.ndim == 3
        G = math.gcd(self.circuits, B)  # circuits must split the batch evenly (an odd last evaluation chunk: fewer)
        with torch.no_grad():
            sw = torch.sigmoid(gate.w)
            sb = torch.sigmoid(gate.b) if gate.use_bias else torch.zeros_like(gate.b)
            if self.hard:
                T = ((sw > 0.5) & gate.mask).expand(G, *sw.shape)
                tb = (sb > 0.5).expand(G, *sb.shape)
            else:
                T = (torch.rand(G, *sw.shape, device=sw.device) < sw) & gate.mask
                tb = torch.rand(G, *sb.shape, device=sb.device) < sb
            dtype = torch.float16 if x.is_cuda else torch.float32
            xg = x.reshape(G, -1, *x.shape[-2:]) if grouped else x.reshape(G, -1, x.shape[-1])
            if grouped:
                hits = torch.einsum("gnhs,ghsd->gnhd", xg.to(dtype), T.to(dtype))
                fired = ~((hits > 0) | tb[:, None])
            else:
                fired = ~((torch.bmm(xg.to(dtype), T.to(dtype)) > 0) | tb[:, None])
            out = fired.float().reshape(*x.shape[:-1], -1)
            if gate.ops == ["linear"]:  # an OR gate: fires iff some active link (or the bias) transmitted
                out = 1 - out
        if self.record:
            act = x.reshape(B, -1, *x.shape[2:]).amax(1) > 0  # input active for some token of the battle
            lp = LinkScore.apply(gate.w, gate.b, T, tb, act, gate.mask, gate.use_bias)
            self.add(lp)
            rate = self.stats.setdefault(gate.name, [0.0, 0])
            rate[0] += float(out.mean())
            rate[1] += 1
        return out

    def sample(self, acc):
        acc = acc.clamp(max=0.0)
        if self.hard:
            return (acc > -math.log(2)).float()
        return (torch.rand_like(acc) < acc.exp()).float()

    def __call__(self, gate, x):
        x = x.detach()
        if gate.w is None:
            gate.initialise(float(x.mean()))
        if self.link:
            return self.link_fire(gate, x)
        L, Lb = gate.logs()
        with torch.no_grad():
            acc = accumulate(x, L, Lb, gate)
            n = self.sample(acc)
        if self.tape is not None:  # semiprop: what Main's update from H needs (input, output bit, its probability)
            with torch.no_grad():
                P = acc.clamp(max=0.0).exp()
                out = activate(n, gate)
                self.tape.append((gate, self.layer, x.half(), out.bool(), (1 - P if gate.ops == ["linear"] else P).half()))
            return out
        if self.record and self.hebb:
            with torch.no_grad():
                post = n - n.flatten(0, -2 if gate.w.ndim == 2 else -3).mean(0)  # output bits, centred
                if self.hebb_weight is not None:
                    post = post * self.hebb_weight.view(-1, *[1] * (post.ndim - 1))
                count = n[..., 0].numel() if gate.w.ndim == 2 else n[..., 0, 0].numel()
                if gate.w.ndim == 3:
                    gate.hebb_w = torch.einsum("btgi,btgo->gio", x, post) / count
                    gate.hebb_b = post.sum((0, 1)) / count
                else:
                    gate.hebb_w = x.flatten(0, -2).T @ post.flatten(0, -2) / count
                    gate.hebb_b = post.flatten(0, -2).sum(0) / count
        elif self.record:
            with torch.no_grad():
                a = acc.clamp(max=-1e-6)
                P = a.exp()
                rho = ((n - P) / (1 - P).clamp_min(1e-4)).half()
                lp = (n * a + (1 - n) * torch.log(-torch.expm1(a))).flatten(1).sum(1)
                del acc, a, P
            lp = Score.apply(L, Lb, x.half(), rho, lp, gate.sign, gate.is_and.float())
            self.add(lp)
        out = activate(n, gate)
        if self.record:
            rate = self.stats.setdefault(gate.name, [0.0, 0])
            rate[0] += float(out.mean())
            rate[1] += 1
        return out


class BitStack(nn.Module):
    """Replaces the strategic blocks before the last `continuous`: bit tokens in, continuous tokens [B, 15, width] out.

    Each input field goes through its own two layers (field bits -> D NOR -> D OR), and a token's fields are XORed together,
    the bit counterpart of the model's summed embeddings. The second layers' target rate r makes the XOR of K fields
    fire at p: (1 - 2r)^K = 1 - 2p."""

    def __init__(self, model, cfg, layers, generator):
        super().__init__()
        self.cfg = cfg
        self.generator = generator
        self.D = cfg.bits
        self.width = model.strategy.config.width
        self.n_layers = layers
        self.fire = Firing(cfg.sampling == "link", cfg.circuits)
        self.built = False
        self.output = nn.Linear(self.D, self.width)
        self.config = model.strategy.config

    def field_gates(self, kind, fields, device):
        cfg, g, D = self.cfg, self.generator, self.D
        r = (1 - (1 - 2 * cfg.p) ** (1 / len(fields))) / 2
        gates = nn.ModuleDict()
        for name, bits in fields.items():
            first = (EmbedTable(f"{kind}-{name}-1", bits.shape[-1], D, cfg.embed_logit, name == "class", g)
                     if cfg.embed == "learned" and name in CATEGORICAL else Gate(f"{kind}-{name}-1", bits.shape[-1], D, cfg.p, cfg, g))
            # the second layer feeds the XOR: an OR, no NOT (as the FFN's flip)
            gates[name] = nn.ModuleList([first, Gate(f"{kind}-{name}-2", D, D, r, cfg, g, act="or-linear")]).to(device)
        return gates

    def build(self, card_fields, other_fields):
        """Gates are sized and initialised on the first batch (the input firing rates and the stat bins need data)."""
        device = next(iter(card_fields.values())).device
        self.card_gates = self.field_gates("card", card_fields, device)
        self.other_gates = self.field_gates("other", other_fields, device)
        layers = []
        for _ in range(self.n_layers):
            layers.append(BitLayer(self.D, self.cfg.heads, self.cfg, self.generator).to(device))
        self.layers = nn.ModuleList(layers)
        n = self.cfg.interleave
        self.mids = nn.ModuleDict({str(i): ContinuousFlip(self.D, self.config, self.cfg.mid_bias).to(device)
                                   for i in range(self.n_layers - 1) if n and (i + 1) % n == 0})
        self.built = True

    def gates(self):
        return [m for m in self.modules() if isinstance(m, Gate)]

    def embed(self, gates, fields):
        x = None
        for name, (first, second) in gates.items():
            inner = first(fields[name], self.fire) if isinstance(first, EmbedTable) else self.fire(first, fields[name])
            bits = self.fire(second, inner)
            x = bits if x is None else (x + bits) % 2  # XOR
        return x

    def forward(self, tokens):
        card_fields, other_fields = tokens
        if not self.built:
            raise RuntimeError("BitStack used before build")
        B = next(iter(card_fields.values())).shape[0]
        if not self.fire.record and B > EVAL_CHUNK:  # evaluation batches (4096) in chunks: the same result, less memory
            return torch.cat([self.forward(({k: v[i:i + EVAL_CHUNK] for k, v in card_fields.items()},
                                            {k: v[i:i + EVAL_CHUNK] for k, v in other_fields.items()}))
                              for i in range(0, B, EVAL_CHUNK)])
        self.fire.layer = 0
        card = self.embed(self.card_gates, card_fields)
        x = torch.empty(card.shape[0], 15, self.D, device=card.device)
        x[:, CARD_SLOTS] = card
        x[:, OTHER_SLOTS] = self.embed(self.other_gates, other_fields)
        self.states = [x]  # the main state after the embedding and after each bit layer (the critic's inputs)
        for i, layer in enumerate(self.layers):
            self.fire.layer = i + 1
            x = layer(x, self.fire)
            if str(i) in self.mids:
                x = self.mids[str(i)](x, self.fire)
            self.states.append(x)
        return self.output(x)


class EmbedTable(nn.Module):
    """A learned embedding of a categorical field as bit probabilities: row v holds value v's D logits, and the bits are
    sampled (collapsed) when the token is built, every pass. Several values at once (classes): the OR of their sampled
    rows. Learned by REINFORCE on each sampled bit (score n - sigma(z)); a row learns only where its value is used."""

    def __init__(self, name, count, D, logit, multi, generator):
        super().__init__()
        self.name, self.multi = name, multi
        self.w = nn.Parameter(torch.where(torch.rand(count, D, generator=generator) < 0.5, logit, -logit))

    def forward(self, select, fire):
        if self.multi:  # select [B, T, count] multi-hot: one draw per member value
            z = self.w.expand(*select.shape[:2], *self.w.shape)
            keep = select[..., None]
        else:
            z = self.w[select.argmax(-1)]  # select one-hot [B, T, count]
            keep = None
        with torch.no_grad():
            n = (z > 0).float() if fire.hard else (torch.rand_like(z) < torch.sigmoid(z)).float()
            if keep is not None:
                n = n * keep
        if fire.record:
            lp = n * F.logsigmoid(z) + (1 - n) * F.logsigmoid(-z)
            if keep is not None:
                lp = lp * keep
            lp = lp.flatten(1).sum(1)
            fire.add(lp)
        bits = n.amax(-2) if keep is not None else n
        if fire.record:
            rate = fire.stats.setdefault(self.name, [0.0, 0])
            rate[0] += float(bits.mean())
            rate[1] += 1
        return bits


class ErrorH(nn.Module):
    """SemiProp's H for the bit layers: a small discrete error network (D/8 bits per token, 1:1 with Main's tokens).

    It starts from a discrete error vector: an input projection of the exact error at the bit/continuous boundary,
    sampled to bits. It then runs down the depth beside Main: at each Main layer it reads Main's state (a NOR gate
    D -> D/8) XORed into its stream and spends --h-depth bit layers (Main's layer type) developing that layer's error.
    Its credit for every bit of every Main gate of the layer comes from a projection of its bits, 1:1 per bit. A trust
    network on the battle's input (Main's embedding bits) scales the credit and can go negative where H misleads.
    H's probabilistic parts (input bits, gates, sampled trust) learn by long-term REINFORCE; the projections are fitted
    to the per-bit REINFORCE samples of Main's bits."""

    def __init__(self, D, width, depth, n_states, tape, cfg, generator):
        super().__init__()
        Dh = D // 8
        self.Dh = Dh
        self.xor = cfg.h_update == "xor"
        self.input = nn.Linear(width, Dh)
        self.read = nn.ModuleList([Gate(f"H-read{l}", D, Dh, cfg.p, cfg, generator) for l in range(n_states)])
        self.layers = nn.ModuleList([nn.ModuleList([BitLayer(Dh, 2, cfg, generator) for _ in range(depth)])
                                     for _ in range(n_states)])
        self.projections = nn.ModuleList()
        for i, (gate, layer, x, out, P) in enumerate(tape):
            if self.xor:  # one error bit per Main bit: H's own gate (D/8 -> the Main gate's output size)
                self.projections.append(Gate(f"H-error{i}", Dh, out[0, 0].numel(), cfg.pflip, cfg, generator))
            else:
                projection = nn.Linear(Dh, out[0, 0].numel())
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                self.projections.append(projection)
        self.trust = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.trust[-1].weight)
        # readout: trust = 2 tanh(.), starting at 1, bounded in (-2, 2); xor: trust is signed, starting at 0
        nn.init.constant_(self.trust[-1].bias, 0.0 if self.xor else math.atanh(0.5))
        self.trust_sigma = 0.3
        self.fire = Firing()
        self.fire.hebb = True

    def forward(self, anchor, states):
        """H's bits at each Main layer (states: Main's state after its embedding and each bit layer) and the per-battle
        log-probability of H's samples (its input bits; its gates, collected by its own Firing)."""
        z = self.input(anchor)
        with torch.no_grad():
            h = (torch.rand_like(z) < torch.sigmoid(z)).float()
        with torch.no_grad():  # the input projection's centred co-activity (anchor error in, sampled bits out)
            post = h - h.flatten(0, 1).mean(0)
            if self.fire.hebb_weight is not None:
                post = post * self.fire.hebb_weight.view(-1, 1, 1)
            self.hebb_input = (post.flatten(0, 1).T @ anchor.flatten(0, 1) / post[..., 0].numel(),
                               post.flatten(0, 1).sum(0) / post[..., 0].numel())
        bits = [None] * len(states)
        for l in reversed(range(len(states))):  # down the depth, from the boundary to the embedding
            h = (h + self.fire(self.read[l], states[l])) % 2
            for layer in self.layers[l]:
                h = layer(h, self.fire)
            bits[l] = h
        return bits

    def credit(self, i, h, out):
        """H's credit for every bit of tape entry i's output [B, T(, heads), n], from H's bits at its layer."""
        tokens = out.shape[1]
        h = h[:, CARD_SLOTS] if tokens == 8 else h[:, OTHER_SLOTS] if tokens == 7 else h
        return self.projections[i](h).view(out.shape)

    def error_bits(self, i, h, out, modulation):
        """xor: H's error bit for every bit of tape entry i's output, each triggering with its own probability lowered
        by the pass's trust (P' = modulation x P); the co-activity (modulated by M) is kept for H's Hebbian update."""
        gate = self.projections[i]
        tokens = out.shape[1]
        h = h[:, CARD_SLOTS] if tokens == 8 else h[:, OTHER_SLOTS] if tokens == 7 else h
        if gate.w is None:
            gate.initialise(float(h.mean()))
        with torch.no_grad():
            L, Lb = gate.logs()
            P = accumulate(h, L, Lb, gate).clamp(max=0.0).exp() * modulation.view(-1, 1, 1)
            e = (torch.rand_like(P) < P).float()
            post = e - e.flatten(0, 1).mean(0)
            if self.fire.hebb_weight is not None:
                post = post * self.fire.hebb_weight.view(-1, 1, 1)
            count = e[..., 0].numel()
            gate.hebb_w = h.flatten(0, 1).T @ post.flatten(0, 1) / count
            gate.hebb_b = post.flatten(0, 1).sum(0) / count
        return e.view(out.shape)

    def trust_sample(self, x0):
        mu = 2 * torch.tanh(self.trust(x0.mean(1)).squeeze(-1))
        with torch.no_grad():
            t = mu + self.trust_sigma * torch.randn_like(mu)
        return t, -(t - mu) ** 2 / (2 * self.trust_sigma ** 2)


class HopfieldTrust(nn.Module):
    """Trust as a modern Hopfield memory: keys are past battles' input patterns (Main's embedding bits pooled, +-1,
    unit length), values Main's performance on them. Recall: trust(x) = softmax(beta keys . x) . values, Main's actual
    performance on similar inputs. Each new battle also moves the values of the memories it recalled toward its own
    performance (old inputs keep being judged by what happens on similar inputs later), then is stored (oldest
    overwritten)."""

    def __init__(self, D, capacity=8192, beta=8.0, rate=0.5):
        super().__init__()
        self.register_buffer("keys", torch.zeros(capacity, D))
        self.register_buffer("values", torch.zeros(capacity))
        self.capacity, self.beta, self.rate, self.filled, self.next = capacity, beta, rate, 0, 0

    @staticmethod
    def pattern(x0):
        return F.normalize(2 * x0.mean(1) - 1, dim=-1)

    def recall(self, x0):
        q = self.pattern(x0)
        if self.filled == 0:
            return torch.zeros(q.shape[0], device=q.device), q, None
        weights = torch.softmax(self.beta * q @ self.keys[:self.filled].T * math.sqrt(q.shape[-1]) / 8, -1)
        return weights @ self.values[:self.filled], q, weights

    def store(self, q, weights, trust, performance):
        if weights is not None:  # recalled memories move toward the new evidence (each by its share of the recall)
            self.values[:self.filled] += self.rate * (weights.T @ (performance - trust)) / weights.sum(0).clamp_min(1.0)
        for i in range(q.shape[0]):
            self.keys[self.next], self.values[self.next] = q[i], performance[i]
            self.next = (self.next + 1) % self.capacity
            self.filled = min(self.filled + 1, self.capacity)


class BoltzmannTrust(nn.Module):
    """Trust as a Boltzmann-style network with hidden units. Visible units: the battle's input bits (Main's embedding
    state, every token, +-1) and one performance unit; hidden units couple to both and are never clamped. Energy
    E = -v.W.h - p u.h - b.h - c p. Recall (free phase): clamp the input, settle hidden and performance units by mean
    field for `steps` updates; trust is the settled performance unit. Storage (clamped phase): also clamp the
    performance unit to Main's actual performance and settle the hidden units; every coupling moves by
    rate x (clamped co-activity - free co-activity), contrastive Hebbian learning (no gradient)."""

    def __init__(self, visible, hidden, rate, steps=5, generator=None):
        super().__init__()
        self.register_buffer("W", torch.randn(visible, hidden, generator=generator) / math.sqrt(visible))
        self.register_buffer("u", torch.zeros(hidden))
        self.register_buffer("b", torch.zeros(hidden))
        self.register_buffer("c", torch.zeros(()))
        self.rate, self.steps = rate, steps

    def settle(self, v, p=None):
        """Mean-field settling with the input clamped (and the performance unit too, if p is given)."""
        drive = v @ self.W + self.b
        free = p is None
        p = torch.zeros(v.shape[0], device=v.device) if free else p
        for _ in range(self.steps):
            h = torch.tanh(drive + p[:, None] * self.u)
            if free:
                p = torch.tanh(h @ self.u + self.c)
            elif _ == 0:
                break
        return h, p

    def recall(self, x0):
        v = 2 * x0.flatten(1) - 1
        h, p = self.settle(v)
        return p, v, h

    def store(self, v, h_free, p_free, performance):
        p_clamped = torch.tanh(performance / performance.std().clamp_min(1e-6))
        h_clamped, _ = self.settle(v, p_clamped)
        B = v.shape[0]
        self.W += self.rate * (v.T @ (h_clamped - h_free)) / B
        self.u += self.rate * (p_clamped @ h_clamped - p_free @ h_free) / B
        self.b += self.rate * (h_clamped - h_free).mean(0)
        self.c += self.rate * (p_clamped - p_free).mean()


class ContinuousFlip(nn.Module):
    """An interleaved continuous block: bits -> Linear(D, width) -> the model's TransformerBlock -> Linear(width, D) flip
    logits; the sampled flips are XORed into the tokens. The bits after it are samples, so it learns by REINFORCE: its
    flips' log-probability joins the gates' in the surrogate (autograd through the logits, same per-battle advantage)."""

    def __init__(self, D, config, bias):
        super().__init__()
        from card_engine.model.networks import TransformerBlock
        self.input = nn.Linear(D, config.width)
        self.block = TransformerBlock(config.width, config.attention_width, config.heads, config.feedforward_width, 0.0)
        self.flip = nn.Linear(config.width, D)
        nn.init.normal_(self.flip.weight, std=0.02)
        nn.init.constant_(self.flip.bias, bias)

    def forward(self, x, fire):
        z = self.flip(self.block(self.input(x))).float()
        if fire.hard:
            n = (z > 0).float()
        else:
            n = (torch.rand_like(z) < torch.sigmoid(z.detach())).float()
        if fire.record:
            lp = (n * F.logsigmoid(z) + (1 - n) * F.logsigmoid(-z)).flatten(1).sum(1)
            fire.add(lp)
            rate = fire.stats.setdefault("mid-flip", [0.0, 0])
            rate[0] += float(n.mean())
            rate[1] += 1
        return (x + n) % 2  # XOR


CARD_SLOTS = [3, 4, 5, 6, 9, 10, 11, 12]  # A cards 1..4, B cards 1..4
OTHER_SLOTS = [0, 1, 2, 7, 8, 13, 14]  # BOS, A red, A blue, B red, B blue, MODE, PREDICT
CODE_BITS = 64
CATEGORICAL = {"border", "mutation", "pack", "class", "identity", "slot", "hidden", "support", "tier", "mode"}
EVAL_CHUNK = 512


class Encoder:
    """Battle metadata -> input fields, as bits: card tokens {field: [B, 8, F]} and the other tokens {field: [B, 7, F]}.

    Categorical fields are fixed random dense codes (CODE_BITS bits, each on with probability 1/2) rather than one-hot:
    the derived init assumes independent inputs at rate a, and gives one-hot inputs every link strong (q > 1), which
    cannot tell the values apart. Description: the sign bits of the card vector. Stats: thermometer bins of the 37
    stat inputs (the model's stat_inputs), edges at the first batch's quantiles."""

    def __init__(self, strategy, bins, seed, learned=False):
        self.strategy, self.learned = strategy, learned
        self.bins = bins
        self.thresholds = None
        self.generator = torch.Generator().manual_seed(seed)
        self.tables = {}

    def code(self, name, ids, count):
        if self.learned:  # the learned tables take the value itself: one-hot
            return F.one_hot(ids.long().clamp(0, count - 1), count).float()
        if name not in self.tables:
            self.tables[name] = (torch.rand(count, CODE_BITS, generator=self.generator) < 0.5).float().to(ids.device)
        return self.tables[name][ids.long().clamp(0, count - 1)]

    def __call__(self, card_embeddings, border_ids, red_support_ids, blue_support_ids, *, pack_ids=None,
                 class_weights=None, mutation_ids=None, identity_keys=None, card_visible=None, support_visible=None,
                 mode_ids=None, support_tiers=None, card_stats=None):
        c = self.strategy.config
        B, device = card_embeddings.shape[0], card_embeddings.device
        cv = (torch.ones(B, 2, 4, dtype=torch.bool, device=device) if card_visible is None else card_visible.bool())
        sv = (torch.ones(B, 2, 2, dtype=torch.bool, device=device) if support_visible is None else support_visible.bool())
        seen = lambda ids: torch.where(cv, ids.long(), 0)  # hidden cards: value 0 in every field, and the hidden code
        cards = {"border": self.code("border", seen(border_ids), 17)}
        if mutation_ids is not None:
            cards["mutation"] = self.code("mutation", seen(mutation_ids), max(1, len(c.mutation_names)))
        if pack_ids is not None:
            cards["pack"] = self.code("pack", seen(pack_ids), 15)
        if class_weights is not None:  # several classes: the OR of their codes
            member = torch.where(cv[..., None], class_weights > 0, False).float()
            if self.learned:
                cards["class"] = member
            else:
                table = self.code("class", torch.arange(len(c.class_names), device=device), len(c.class_names))
                cards["class"] = ((member @ table) > 0).float()
        if identity_keys is not None:
            cards["identity"] = self.code("identity", seen(identity_keys), c.identity_capacity + 1)
        cards["description"] = torch.where(cv[..., None], card_embeddings > 0, False).float()
        if card_stats is not None:
            stats = self.strategy.stat_inputs(torch.where(cv[..., None], card_stats, 1.0), cv)  # [B, 2, 4, 37]
            if self.thresholds is None:
                flat = stats.reshape(-1, stats.shape[-1])
                qs = torch.linspace(0, 1, self.bins + 2, device=device)[1:-1]
                self.thresholds = torch.quantile(flat.float(), qs, dim=0).T.contiguous()  # [37, bins]
            cards["stats"] = torch.where(cv[..., None, None], stats[..., None] > self.thresholds, False).flatten(-2).float()
        cards["slot"] = self.code("card-slot", torch.arange(8, device=device).view(1, 2, 4).expand(B, -1, -1), 8)
        cards["hidden"] = self.code("card-hidden", (~cv).long(), 2)
        cards = {k: v.reshape(B, 8, -1) for k, v in cards.items()}

        # other tokens: BOS, A red, A blue, B red, B blue, MODE, PREDICT
        zero = torch.zeros(B, dtype=torch.long, device=device)
        tiers = support_tiers.long() if support_tiers is not None else torch.zeros(B, 2, 2, dtype=torch.long, device=device)
        red = torch.where(sv[..., 0], red_support_ids.long(), 0)  # 0 = none or hidden
        blue = torch.where(sv[..., 1], blue_support_ids.long(), 0)
        support = torch.stack([zero, red[:, 0], 29 + blue[:, 0], red[:, 1], 29 + blue[:, 1], zero, zero], 1)
        tr, tb = torch.where(sv[..., 0], tiers[..., 0], 0), torch.where(sv[..., 1], tiers[..., 1], 0)
        tier = torch.stack([zero, tr[:, 0], 6 + tb[:, 0], tr[:, 1], 6 + tb[:, 1], zero, zero], 1)
        if mode_ids is None:
            mode_ids = (~cv[:, 1]).any(-1).logical_or((~sv[:, 1]).any(-1)).long()
        mode = torch.stack([zero, zero, zero, zero, zero, 1 + mode_ids.long(), zero], 1)
        hidden = torch.stack([zero, ~sv[:, 0, 0], ~sv[:, 0, 1], ~sv[:, 1, 0], ~sv[:, 1, 1], zero, zero], 1).long()
        other = {"slot": self.code("other-slot", torch.arange(7, device=device).expand(B, -1), 7),
                 "support": self.code("support", support, 29 + 16), "tier": self.code("tier", tier, 12),
                 "mode": self.code("mode", mode, 3), "hidden": self.code("other-hidden", hidden, 2)}
        return cards, other


class DiscreteRule:
    def __init__(self, model, args):
        cfg = rule_args(args)
        self.cfg = cfg
        strategy = model.strategy
        blocks = list(strategy.blocks)
        if cfg.sampling == "link" and cfg.act != "or-not":
            raise SystemExit("--sampling link is implemented for or-not only")
        n_bit = len(blocks) - cfg.continuous
        if n_bit < 0:
            raise SystemExit("--continuous exceeds the model's layers")
        device = strategy.bos.device
        generator = torch.Generator().manual_seed(args.seed + 1)
        self.cfg_seed = args.seed
        model.requires_grad_(False)  # the token embeddings and the replaced blocks no longer take part
        self.stack = BitStack(model, cfg, n_bit, generator).to(device)
        strategy.blocks = nn.ModuleList([self.stack, *blocks[n_bit:]])
        strategy.build_sequence = Encoder(strategy, cfg.bins, args.seed + 2, cfg.embed == "learned")
        for module in [self.stack.output, *blocks[n_bit:], strategy.outcome_head]:
            module.requires_grad_(True)
        self.model = model
        self.stack.fire.hard = False
        self.optimizer = None
        self.gate_optimizer = None
        self.base = None
        self.steps = 0
        orig_train = model.train

        def train(mode=True):  # evaluation: sampled bits (default) or the hardened model
            self.stack.fire.hard = (not mode) and cfg.eval_mode == "hard"
            return orig_train(mode)
        model.train = train
        self.blocks = blocks[n_bit:]
        if cfg.credit == "semiprop":
            if cfg.act != "or-not" or cfg.sampling != "neuron" or cfg.embed != "codes" or cfg.interleave:
                raise SystemExit("--credit semiprop: or-not, neuron sampling, code embeddings, no interleave")
            self.stack.register_forward_hook(lambda m, args, out: setattr(self, "stack_out", out))

    def ensure_built(self, forward, batch):
        if self.stack.built:
            return
        strategy = self.model.strategy
        captured = {}
        # one forward with the encoder only, to size and initialise the gates from real input rates
        enc = strategy.build_sequence
        def probe(*a, **k):
            captured["bits"] = enc(*a, **k)
            raise _Probe
        strategy.build_sequence = probe
        try:
            with torch.no_grad():
                forward(batch)
        except _Probe:
            pass
        strategy.build_sequence = enc
        card_fields, other_fields = captured["bits"]
        self.stack.build(card_fields, other_fields)
        with torch.no_grad():  # the first real forward draws every gate's weights from its measured input rate
            forward(batch)
        bitwise = {id(q) for m in self.stack.modules() if isinstance(m, (Gate, EmbedTable)) for q in m.parameters(recurse=False)}
        cont = [p for p in self.model.parameters() if p.requires_grad and id(p) not in bitwise]
        self.cont_params = cont
        self.optimizer = torch.optim.AdamW(cont, lr=self.args_lr, weight_decay=0.05)
        self.gate_weights = [g.w for g in self.stack.gates()] + [t.w for t in self.stack.modules() if isinstance(t, EmbedTable)]
        self.gate_biases = [g.b for g in self.stack.gates() if g.use_bias]
        self.gate_optimizer = torch.optim.Adam(self.gate_weights + self.gate_biases, lr=self.cfg.glr)
        if self.cfg.credit == "semiprop":
            self.build_errorh(forward, batch)
        n_conn = sum(int(g.mask.sum()) * (g.w.shape[0] if g.w.ndim == 3 else 1) for g in self.stack.gates())
        print(f"[discrete] bit layers={self.stack.n_layers} D={self.stack.D} heads={self.cfg.heads} "
              "fields: " + " ".join(f"{k}={v.shape[-1]}@{float(v.mean()):.2f}" for k, v in {**card_fields, **other_fields}.items()) + " "
              f"gate connections={n_conn:,} continuous params={sum(p.numel() for p in cont):,}", flush=True)

    def build_errorh(self, forward, batch):
        """H is sized from one taped forward (its projections match Main's gates); its gates initialise on first use."""
        fire = self.stack.fire
        fire.tape = []
        with torch.no_grad():
            forward(batch)
            tape, states = fire.tape, [x.detach() for x in self.stack.states]
        fire.tape = None
        device = self.stack.output.weight.device
        self.errorh = ErrorH(self.stack.D, self.stack.width, self.cfg.h_depth, len(states), tape, self.cfg,
                             torch.Generator().manual_seed(self.cfg_seed + 3)).to(device)
        with torch.no_grad():
            self.errorh(torch.zeros(states[0].shape[0], 15, self.stack.width, device=device), states)
        H = self.errorh
        self.h_rl = list(H.trust.parameters())  # trust: REINFORCE on its sampled value (a continuous network)
        # plain SGD, so it moves with the trace's actual signal (Adam turned this noisy scalar reward into full steps)
        self.trust_optimizer = torch.optim.SGD(self.h_rl, lr=0.1)
        # H's gates and input projection: reward-modulated Hebbian traces (one per weight)
        self.hebb_params = [(m, m.w, m.b) for m in H.modules() if isinstance(m, Gate)] + [(None, H.input.weight, H.input.bias)]
        self.hebb_trace = None
        if H.xor:  # trust: fitted to Main's actual performance (supervised: Adam); initialise the error gates
            self.trust_optimizer = torch.optim.Adam(H.trust.parameters(), lr=self.cfg.sp_lr)
            if self.cfg.trust == "hopfield":
                self.hopfield = HopfieldTrust(self.stack.D, beta=self.cfg.hopfield_beta).to(device)
            elif self.cfg.trust == "boltzmann":
                self.boltzmann = BoltzmannTrust(15 * self.stack.D, self.cfg.trust_hidden, self.cfg.trust_rate,
                                                generator=torch.Generator().manual_seed(self.cfg_seed + 4)).to(device)
            bits = self.errorh(torch.zeros(states[0].shape[0], 15, self.stack.width, device=device), states)
            for i, (gate, layer, x, out, P) in enumerate(tape):
                H.error_bits(i, bits[layer], out, torch.ones(out.shape[0], device=device))
            self.hebb_params = [(m, m.w, m.b) for m in H.modules() if isinstance(m, Gate)] + [
                (None, H.input.weight, H.input.bias)]
        else:
            self.proj_optimizer = torch.optim.Adam(H.projections.parameters(), lr=self.cfg.sp_lr)
        self.trace, self.loss_level, self.h_stats = None, None, {}

    def xor_step(self, forward, batch):
        """Main learns only from H: each Main bit is driven toward its own activation XOR H's error bit (a likelihood
        step, each gate's update normalised); H learns by modern (three-factor) Hebbian learning, its modulatory signal
        M = trust, fitted to Main's actual performance on the battle's input; trust also lowers each error bit's
        chance to trigger."""
        fire, H, cfg = self.stack.fire, self.errorh, self.cfg
        fire.tape = []
        logits = forward(batch)
        tape, states = fire.tape, [x.detach() for x in self.stack.states]
        fire.tape = None
        per = -(batch["target"] * logits.float().log_softmax(-1)).sum(-1)
        value = per.mean()
        B = per.shape[0]
        anchor = torch.autograd.grad(per.sum(), self.stack_out, retain_graph=True)[0]  # exact error at the boundary
        anchor = anchor / anchor.pow(2).mean().sqrt().clamp_min(1e-12)
        self.optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
        self.trust_optimizer.zero_grad(set_to_none=True)
        value.backward()  # the continuous part (and the stack's output projection) by backprop

        performance = -(per - per.mean()).detach()  # loss minimisation: better than the batch's battles is positive
        if cfg.trust == "boltzmann":  # settle with the input clamped (recall), then store with the performance clamped
            with torch.no_grad():
                trust, v, h = self.boltzmann.recall(states[0])
                self.boltzmann.store(v, h, trust, performance)
        elif cfg.trust == "hopfield":  # Main's performance on similar past inputs (recalled before this batch is stored)
            with torch.no_grad():
                trust, q, weights = self.hopfield.recall(states[0])
                self.hopfield.store(q, weights, trust, performance)
        else:
            trust = H.trust(states[0].mean(1)).squeeze(-1)  # trust in this input: Main's predicted relative performance
            F.mse_loss(trust, performance).backward()
        M = trust.detach()
        modulation = torch.sigmoid(M / M.std().clamp_min(1e-6))  # lowers each error bit's chance on distrusted inputs
        H.fire.hebb_weight = M
        H.fire.record, H.fire.stats = True, {}
        bits = H(anchor.detach(), states)
        H.fire.record = False

        flips, count = 0.0, 0
        for i, (gate, layer, x, out, P) in enumerate(tape):
            e = H.error_bits(i, bits[layer], out, modulation)
            target = (out.float() + e) % 2  # Main's activation XOR H's error
            L, Lb = gate.logs()
            Pw = accumulate(x.float(), L, Lb, gate).clamp(max=0.0).exp()
            Pw = (1 - Pw if gate.ops == ["linear"] else Pw).clamp(1e-6, 1 - 1e-6)
            (F.binary_cross_entropy(Pw, target, reduction="sum") / B).backward()
            flips += float(e.sum())
            count += e.numel()
        with torch.no_grad():  # each Main gate's update normalised to unit RMS
            for q in self.gate_weights + self.gate_biases:
                if q.grad is not None:
                    q.grad /= q.grad.pow(2).mean().sqrt().clamp_min(1e-30)
        H.fire.hebb_weight = None

        # modern Hebbian: M-weighted co-activity into each link's eligibility trace, applied every step (anti-Hebbian
        # for NOR gates: a link stops its neuron, so a co-active pattern on a trusted input weakens the link)
        hebb = [(m.hebb_w, m.hebb_b) if m is not None else H.hebb_input for m, w, b in self.hebb_params]
        with torch.no_grad():
            self.hebb_trace = hebb if self.hebb_trace is None else [
                (cfg.h_trace * ew + hw, cfg.h_trace * eb + hb) for (ew, eb), (hw, hb) in zip(self.hebb_trace, hebb)]
            for (m, w, b), (ew, eb) in zip(self.hebb_params, self.hebb_trace):
                sign = -1.0 if m is not None else 1.0
                w.add_(sign * cfg.h_lr * ew * (m.mask if m is not None else 1))
                if m is None or m.use_bias:
                    b.add_(sign * cfg.h_lr * eb)

        torch.nn.utils.clip_grad_norm_(self.cont_params, 1.0)
        self.optimizer.step()
        self.gate_optimizer.step()
        if cfg.trust == "regression":
            self.trust_optimizer.step()
        with torch.no_grad():
            for w in self.gate_weights + self.gate_biases + [q for m in H.modules() if isinstance(m, Gate) for q in (m.w, m.b)]:
                w.clamp_(-CLAMP, CLAMP)
        self.steps += 1
        L_now = float(value.detach())
        if self.steps % 100 == 1:
            corr = float(torch.corrcoef(torch.stack([M, performance]))[0, 1]) if float(M.std()) > 0 else float("nan")
            print(f"[discrete] step {self.steps} loss {L_now:.4f} trust corr {corr:+.3f} M+-{float(M.std()):.4f} "
                  f"error-bit rate {flips / max(count, 1):.4f}", flush=True)
        return L_now

    def semiprop_step(self, forward, batch):
        fire, H, cfg = self.stack.fire, self.errorh, self.cfg
        fire.tape = []
        logits = forward(batch)
        tape, states = fire.tape, [x.detach() for x in self.stack.states]
        fire.tape = None
        per = -(batch["target"] * logits.float().log_softmax(-1)).sum(-1)
        value = per.mean()
        B = per.shape[0]
        anchor = torch.autograd.grad(per.sum(), self.stack_out, retain_graph=True)[0]  # exact error at the boundary
        anchor = anchor / anchor.pow(2).mean().sqrt().clamp_min(1e-12)
        self.optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
        self.proj_optimizer.zero_grad(set_to_none=True)
        value.backward()  # the continuous part (and the stack's output projection) by backprop

        H.fire.record, H.fire.logp, H.fire.logp_layers, H.fire.stats = True, None, {}, {}
        bits = H(anchor.detach(), states)
        H.fire.record = False
        trust, lp = H.trust_sample(states[0])

        with torch.no_grad():
            centred = per - (per.sum() - per) / (B - 1)  # the battle's loss against the others' (per-bit REINFORCE)
        fit, size = 0.0, 0.0
        for i, (gate, layer, x, out, P) in enumerate(tape):
            c = H.credit(i, bits[layer], out)
            with torch.no_grad():  # the per-bit REINFORCE sample of d loss / d P(bit) (H's regression target)
                P32, n = P.float(), out.float()
                score = ((n - P32) / (P32 * (1 - P32)).clamp_min(1e-6)).clamp(-20, 20)
                target = centred.view(-1, *[1] * (out.ndim - 1)) * score
            fit = fit + F.mse_loss(c, target)
            size = size + target.pow(2).mean()
            # Main learns only from H: d/dw of trust x credit x P(bit), P recomputed from the stored input
            L, Lb = gate.logs()
            Pw = accumulate(x.float(), L, Lb, gate).clamp(max=0.0).exp()
            Pw = 1 - Pw if gate.ops == ["linear"] else Pw
            weight = (trust.view(-1, *[1] * (out.ndim - 1)) * c.detach())
            ((weight * Pw).sum() / B).backward()
        fit.backward()  # the projections
        with torch.no_grad():  # each Main gate's update from H normalised to unit RMS (its raw size, credit x dP/dw with
            for q in self.gate_weights + self.gate_biases:  # dP/dw ~ 1e-7 on silent links, sat below Adam's epsilon)
                if q.grad is not None:
                    q.grad /= q.grad.pow(2).mean().sqrt().clamp_min(1e-30)

        # long-term REINFORCE: this step's loss judges the decisions in the trace (the past steps'), then this step's join
        L_now = float(value.detach())
        self.loss_level = L_now if self.loss_level is None else 0.9 * self.loss_level + 0.1 * L_now
        delta = L_now - self.loss_level
        grads = torch.autograd.grad(lp.sum() / B, self.h_rl, allow_unused=True)
        grads = [torch.zeros_like(q) if g is None else g for q, g in zip(self.h_rl, grads)]
        self.trust_optimizer.zero_grad(set_to_none=True)
        scale = math.copysign(math.sqrt(abs(delta)), delta)  # H's credit: sign(delta) sqrt|delta| (user)
        if self.trace is not None:
            for q, e in zip(self.h_rl, self.trace):
                q.grad = scale * e
            self.trust_optimizer.step()
        self.trace = grads if self.trace is None else [cfg.h_trace * e + g for e, g in zip(self.trace, grads)]
        # reward-modulated Hebbian: the reward (-scale: + when the loss fell below its level) times the co-activity
        # trace; NOR gates are anti-Hebbian (a link stops its neuron: a rewarded "input on, neuron fired" weakens it)
        hebb = [(m.hebb_w, m.hebb_b) if m is not None else H.hebb_input for m, w, b in self.hebb_params]
        with torch.no_grad():
            if self.hebb_trace is not None:
                for (m, w, b), (ew, eb) in zip(self.hebb_params, self.hebb_trace):
                    sign = -1.0 if m is not None else 1.0
                    w.add_(sign * cfg.h_lr * -scale * ew * (m.mask if m is not None else 1))
                    if m is None or m.use_bias:
                        b.add_(sign * cfg.h_lr * -scale * eb)
            self.hebb_trace = hebb if self.hebb_trace is None else [
                (cfg.h_trace * ew + hw, cfg.h_trace * eb + hb) for (ew, eb), (hw, hb) in zip(self.hebb_trace, hebb)]

        torch.nn.utils.clip_grad_norm_(self.cont_params, 1.0)
        self.optimizer.step()
        self.gate_optimizer.step()
        self.proj_optimizer.step()
        with torch.no_grad():
            for w in self.gate_weights + self.gate_biases + [q for m in H.modules() if isinstance(m, Gate) for q in (m.w, m.b)]:
                w.clamp_(-CLAMP, CLAMP)
        self.steps += 1
        if self.steps % 100 == 1:
            print(f"[discrete] step {self.steps} loss {L_now:.4f} delta {delta:+.4f} trust {float(trust.mean()):.3f}"
                  f"+-{float(trust.std()):.3f} H fit {float(fit) / max(float(size), 1e-12):.3f}", flush=True)
        return L_now

    def state(self, step):
        """Everything needed to continue: weights (the model's, every gate's), both optimizers, the step, the encoder's
        stat bins and codes."""
        if self.cfg.credit != "batch":
            raise SystemExit("--save: implemented for --credit batch")
        enc = self.model.strategy.build_sequence
        return {"step": step, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "gate_optimizer": self.gate_optimizer.state_dict(), "base": self.base, "rule_steps": self.steps,
                "thresholds": enc.thresholds, "tables": enc.tables, "rule_args": vars(self.cfg)}

    def on_eval(self, step, row):
        if not self.cfg.save:
            return
        from pathlib import Path
        folder = Path(self.cfg.save)
        folder.mkdir(parents=True, exist_ok=True)
        if row.get("kl") is not None and row["kl"] < getattr(self, "best_kl", float("inf")):  # saved on performance only
            self.best_kl = row["kl"]
            torch.save({**self.state(step), "metrics": row}, folder / "best.pt.tmp")
            (folder / "best.pt.tmp").replace(folder / "best.pt")
        track = getattr(self, "track", None)  # the bench's --track: also keep the best by it (e.g. upset accuracy)
        if track and track.lstrip("-") != "kl":
            sign, metric = (-1.0, track[1:]) if track.startswith("-") else (1.0, track)
            value = row.get(metric)
            if value is not None and sign * value > getattr(self, "best_track", -float("inf")):
                self.best_track = sign * value
                name = f"best_{metric}.pt"
                torch.save({**self.state(step), "metrics": row}, folder / (name + ".tmp"))
                (folder / (name + ".tmp")).replace(folder / name)

    def on_finish(self, step):
        if self.cfg.save:
            from pathlib import Path
            folder = Path(self.cfg.save)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save({**self.state(step), "best_kl": getattr(self, "best_kl", None)}, folder / "latest.pt.tmp")
            (folder / "latest.pt.tmp").replace(folder / "latest.pt")

    def load(self, path, forward, batch):
        """Restore a checkpoint (the gates are built first, so their shapes and masks exist); returns its step."""
        state = torch.load(path, map_location=self.stack.output.weight.device, weights_only=False)
        enc = self.model.strategy.build_sequence
        enc.thresholds, enc.tables = state["thresholds"], state["tables"]
        self.ensure_built(forward, batch)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.gate_optimizer.load_state_dict(state["gate_optimizer"])
        self.base, self.steps = state["base"], state["rule_steps"]
        best = state["metrics"]["kl"] if "metrics" in state else state.get("best_kl")
        if best is not None:  # the best so far carries over: a resumed run only replaces it with a better one
            self.best_kl = best
        return state["step"]

    def step(self, forward, batch):
        self.ensure_built(forward, batch)
        if self.cfg.credit == "semiprop":
            return self.xor_step(forward, batch) if self.cfg.h_update == "xor" else self.semiprop_step(forward, batch)
        fire = self.stack.fire
        fire.record, fire.logp, fire.stats = True, None, {}
        logits = forward(batch)
        fire.record = False
        per = -(batch["target"] * logits.float().log_softmax(-1)).sum(-1)  # the bench's loss, per battle
        value = per.mean()
        with torch.no_grad():
            B = per.shape[0]
            if self.cfg.baseline == "batch" and self.cfg.sampling == "link":
                # the baseline must not depend on the battle's own circuit: the mean loss of the other circuits' battles
                G = math.gcd(self.cfg.circuits, B)
                groups = per.view(G, B // G).sum(1)
                base = ((per.sum() - groups) / (B - B // G)).repeat_interleave(B // G)
            elif self.cfg.baseline == "batch":
                base = (per.sum() - per) / (B - 1)
            else:
                base = self.base if self.base is not None else per.mean()
                self.base = per.mean() if self.base is None else 0.99 * self.base + 0.01 * per.mean()
            adv = per - base
        surrogate = (adv * fire.logp).mean()  # REINFORCE: d/dw E[loss] ~ A d ln p / dw
        self.optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
        (value + surrogate).backward()
        torch.nn.utils.clip_grad_norm_(self.cont_params, 1.0)
        self.optimizer.step()
        self.gate_optimizer.step()
        with torch.no_grad():
            lam = self.cfg.antireg * self.cfg.glr
            bias_scale = {"same": 1.0, "lower": 0.1, "none": 0.0}[self.cfg.antireg_bias]
            for w in self.gate_weights:
                if lam:
                    w.mul_(1 + lam)
                w.clamp_(-CLAMP, CLAMP)
            for b in self.gate_biases:
                if lam and bias_scale:
                    b.mul_(1 + lam * bias_scale)
                b.clamp_(-CLAMP, CLAMP)
        self.steps += 1
        if self.steps % 100 == 1:
            rates = " ".join(f"{k}={v[0] / v[1]:.3f}" for k, v in fire.stats.items() if not k.startswith(("card-", "other-")))
            for kind in ("card", "other"):
                for layer in ("1", "2"):
                    vals = [v[0] / v[1] for k, v in fire.stats.items() if k.startswith(kind + "-") and k.endswith("-" + layer)]
                    rates += f" {kind}-embed{layer}={sum(vals) / max(1, len(vals)):.3f}"
            print(f"[discrete] step {self.steps} loss {float(value.detach()):.4f} firing: {rates}", flush=True)
        return float(value.detach())


class _Probe(Exception):
    pass


def make_rule(model, args):
    rule = DiscreteRule(model, args)
    rule.args_lr = args.lr
    return rule
