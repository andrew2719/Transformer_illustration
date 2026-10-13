"""Independent reference implementation (plain Python, no shared code with the JS engine).

Recomputes the toy encoder layer, the decoder mask grid and the next-token softmax from the
parameters and the raw sentence, then compares every intermediate with the page engine.

Run:  node tests/export-engine.mjs && python3 tests/reference.py
"""
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
data = json.loads((ROOT / "fixtures" / "engine-export.json").read_text())
params = data["params"]
D_MODEL, D_K = 4, 2

# tokenization: words with internal apostrophes, or single punctuation marks
tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", data["sentence"])
assert tokens == data["tokens"], (tokens, data["tokens"])


def pe(pos):
    out = []
    for i in range(D_MODEL // 2):
        angle = pos / (10000 ** (2 * i / D_MODEL))
        out += [math.sin(angle), math.cos(angle)]
    return out


def row_times(x, W):
    return [sum(x[r] * W[r][c] for r in range(len(x))) for c in range(len(W[0]))]


def layer_norm(x, eps=1e-6):
    mu = sum(x) / len(x)
    var = sum((v - mu) ** 2 for v in x) / len(x)
    return [(v - mu) / math.sqrt(var + eps) for v in x]


def softmax(row):
    e = [math.exp(v) for v in row]  # exp(-inf) == 0.0
    s = sum(e)
    return e, s, [v / s for v in e]


def encoder(position_on, scale):
    E = [params["embeddings"][t.lower()] for t in tokens]
    X = [[e[d] + (pe(i)[d] if position_on else 0.0) for d in range(D_MODEL)] for i, e in enumerate(E)]
    heads = []
    for h in params["heads"]:
        Q, K, V = ([row_times(x, h[w]) for x in X] for w in ("WQ", "WK", "WV"))
        raw = [[sum(q[c] * k[c] for c in range(D_K)) for k in K] for q in Q]
        scaled = [[s / scale for s in row] for row in raw]
        sm = [softmax(row) for row in scaled]
        A = [p for _, _, p in sm]
        weighted = [[[a * v for v in V[j]] for j, a in enumerate(row)] for row in A]
        Z = [[sum(w[j][c] for j in range(len(tokens))) for c in range(D_K)] for w in weighted]
        heads.append(dict(Q=Q, K=K, V=V, raw=raw, scaled=scaled, exps=[e for e, _, _ in sm], sums=[s for _, s, _ in sm], A=A, weighted=weighted, Z=Z))
    concat = [heads[0]["Z"][i] + heads[1]["Z"][i] for i in range(len(tokens))]
    MH = [row_times(c, params["WO"]) for c in concat]
    R1 = [[X[i][d] + MH[i][d] for d in range(D_MODEL)] for i in range(len(tokens))]
    LN1 = [layer_norm(r) for r in R1]
    Hpre = [[a + b for a, b in zip(row_times(x, params["W1"]), params["b1"])] for x in LN1]
    F = [[a + b for a, b in zip(row_times([max(0.0, v) for v in h], params["W2"]), params["b2"])] for h in Hpre]
    R2 = [[LN1[i][d] + F[i][d] for d in range(D_MODEL)] for i in range(len(tokens))]
    return dict(X=X, heads=heads, Concat=concat, MH=MH, R1=R1, LN1=LN1, Hpre=Hpre, F=F, LN2=[layer_norm(r) for r in R2])


def max_diff(a, b):
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((max_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if b is None:  # masked logit exported as null
        assert a == -math.inf
        return 0.0
    return abs(a - b)


worst, count = 0.0, 0
for cond, (pos_on, scale) in {"on": (True, math.sqrt(D_K)), "off": (False, math.sqrt(D_K)),
                               "onNoScale": (True, 1.0), "offNoScale": (False, 1.0)}.items():
    ref, js = encoder(pos_on, scale), data["conds"][cond]
    for key in ("X", "Concat", "MH", "R1", "LN1", "Hpre", "F", "LN2"):
        worst = max(worst, max_diff(ref[key], js[key])); count += 1
    for h in range(2):
        for key in ("Q", "K", "V", "raw", "scaled", "exps", "sums", "A", "weighted", "Z"):
            worst = max(worst, max_diff(ref["heads"][h][key], js["heads"][h][key])); count += 1

dec = params["decoder"]
for key, toks in (("g4", dec["inputs"]), ("g5", dec["inputs"] + [dec["next"]])):
    X = [[params["embeddings"][t.lower()][d] + pe(i)[d] for d in range(D_MODEL)] for i, t in enumerate(toks)]
    Q = [row_times(x, dec["WQ"]) for x in X]
    K = [row_times(x, dec["WK"]) for x in X]
    masked = [[sum(q[c] * k[c] for c in range(D_K)) / math.sqrt(D_K) if j <= i else -math.inf for j, k in enumerate(K)] for i, q in enumerate(Q)]
    A = [softmax(row)[2] for row in masked]
    for i, row in enumerate(A):
        for j, a in enumerate(row):
            if j > i:
                assert a == 0.0 and data["decoder"][key]["A"][i][j] == 0.0, "future position must get exactly 0"
    worst = max(worst, max_diff(masked, data["decoder"][key]["masked"]), max_diff(A, data["decoder"][key]["A"])); count += 2
vocab = data["decoder"]["vocab"]
worst = max(worst, max_diff(softmax([dec["nextTokenLogits"][w] for w in vocab])[2], data["decoder"]["nextP"])); count += 1

# ---- epilogue: the same operations at the paper's sizes (d_model = 512, h = 8, d_k = d_v = 64).
# The values are seeded pseudo-random (illustrative). The generator is re-implemented here from its
# definition — FNV-1a hash of "<seed>:<tag>" seeding mulberry32, uniform on [-a, a) — so every projection,
# softmax, concatenation and output projection is recomputed independently of the page's code.
M32 = 0xFFFFFFFF


def imul(a, b):
    return (a * b) & M32


def fnv1a(text):
    h = 0x811C9DC5
    for ch in text:
        h = imul(h ^ ord(ch), 0x01000193)
    return h


def mulberry32(seed):
    state = seed & M32

    def rnd():
        nonlocal state
        state = (state + 0x6D2B79F5) & M32
        t = imul(state ^ (state >> 15), 1 | state)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return rnd


paper = data["paper"]
P_MODEL, P_HEADS, P_DK, P_DV = (paper["dims"][k] for k in ("dModel", "nHeads", "dK", "dV"))
assert (P_MODEL, P_HEADS, P_DK, P_DV) == (512, 8, 64, 64), "paper §3.2.2 base-model sizes"


def uniform(tag, n, amp):
    rnd = mulberry32(fnv1a(f"{paper['seed']}:{tag}"))
    return [(2 * rnd() - 1) * amp for _ in range(n)]


def seeded_matrix(tag, rows, cols, amp):
    flat = uniform(tag, rows * cols, amp)
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]


def pe_d(pos, d):
    out = []
    for i in range(d // 2):
        angle = pos / (10000 ** (2 * i / d))
        out += [math.sin(angle), math.cos(angle)]
    return out


n_tok = len(tokens)
pE = [uniform(f"E:{t.lower()}", P_MODEL, 1.0) for t in tokens]
pP = [pe_d(i, P_MODEL) for i in range(n_tok)]
pX = [[pE[i][c] + pP[i][c] for c in range(P_MODEL)] for i in range(n_tok)]
amp_in = math.sqrt(3 / P_MODEL)
p_heads = []
for h in range(P_HEADS):
    W = {w: seeded_matrix(f"{w}{h}", P_MODEL, P_DK, amp_in) for w in ("WQ", "WK", "WV")}
    Q, K, V = ([row_times(x, W[w]) for x in pX] for w in ("WQ", "WK", "WV"))
    raw = [[sum(q[c] * k[c] for c in range(P_DK)) for k in K] for q in Q]
    A = [softmax([s / math.sqrt(P_DK) for s in row])[2] for row in raw]
    Z = [[sum(A[i][j] * V[j][c] for j in range(n_tok)) for c in range(P_DV)] for i in range(n_tok)]
    p_heads.append(dict(Q=Q, K=K, V=V, A=A, Z=Z))
p_concat = [sum((p_heads[h]["Z"][i] for h in range(P_HEADS)), []) for i in range(n_tok)]
assert all(len(c) == P_HEADS * P_DV == P_MODEL for c in p_concat), "Concat(head_1..head_8) is 512 wide"
WO = seeded_matrix("WO", P_HEADS * P_DV, P_MODEL, math.sqrt(3 / (P_HEADS * P_DV)))
p_mh = [row_times(c, WO) for c in p_concat]
worst = max(worst, max_diff(pX, paper["X"]), max_diff(p_concat, paper["Concat"]), max_diff(p_mh, paper["MH"])); count += 3
for h in range(P_HEADS):
    for key in ("Q", "K", "V", "A", "Z"):
        worst = max(worst, max_diff(p_heads[h][key], paper["heads"][h][key])); count += 1

print(f"compared {count} arrays; max |python - js| = {worst:.3e}")
if worst > 1e-12:
    sys.exit("MISMATCH between the independent reference and the page engine")

# ---- optional branch: inside the decoder, at the paper's sizes (d_model 512, h 8, d_k = d_v 64, d_ff 2048, N = 6).
# It continues from the guided sentence: the epilogue's X goes through six encoder layers into the memory the decoder
# reads. About 45 million seeded weights, so this part uses numpy; the generator is vectorised
# from its definition: mulberry32's state after k draws is seed + k·0x6D2B79F5 (mod 2^32), and each output is a
# fixed function of that state. Matrix products here sum in a different order than the page, hence a looser bound.
try:
    import numpy as np
except ImportError:
    sys.exit("numpy is needed for the decoder-dive reference (pip install numpy)")

dive = data["dive"]
U32 = np.uint64(0xFFFFFFFF)


def uniform_np(tag, n, amp):
    a = (np.uint64(fnv1a(f"{paper['seed']}:{tag}")) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x6D2B79F5)) & U32
    t = ((a ^ (a >> np.uint64(15))) * (np.uint64(1) | a)) & U32
    t = ((t + (((t ^ (t >> np.uint64(7))) * (np.uint64(61) | t)) & U32)) & U32) ^ t
    u = ((t ^ (t >> np.uint64(14))) & U32).astype(np.float64) / 4294967296.0
    return (2 * u - 1) * amp


assert np.allclose(uniform_np("WO", 5, 1.0), uniform("WO", 5, 1.0), rtol=0, atol=0), "vectorised generator = scalar generator"
D_FF, N_LAYERS = paper["dims"]["dFF"], paper["dims"]["layers"]
assert (D_FF, N_LAYERS) == (2048, 6), "paper §3.3 d_ff and §3.1 N"
LN_EPS = 1e-6


def seeded(tag, rows, cols, amp):
    return uniform_np(tag, rows * cols, amp).reshape(rows, cols)


def ln(x):
    mu = x.mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=-1, keepdims=True) + LN_EPS)


def softmax_rows(S):
    e = np.exp(S)  # exp(-inf) == 0
    return e / e.sum(axis=-1, keepdims=True)


G_BRANCH = 1 / math.sqrt(2 * N_LAYERS)  # residual-branch output projections beyond each stack's first self-attention
assert abs(dive["branchScale"] - G_BRANCH) < 1e-15


def mha(Xq, Xkv, tag, causal=False, variant=lambda h: "", out_scale=1.0, epilogue_tags=False):
    a, heads = math.sqrt(3 / P_MODEL), []
    name = (lambda w, h: f"{w}{h}") if epilogue_tags else (lambda w, h: f"{tag}:{w}{h}" + ("" if w == "WV" else variant(h)))
    for h in range(P_HEADS):
        Q, K, V = Xq @ seeded(name("WQ", h), P_MODEL, P_DK, a), Xkv @ seeded(name("WK", h), P_MODEL, P_DK, a), Xkv @ seeded(name("WV", h), P_MODEL, P_DV, a)
        raw = Q @ K.T
        scaled = raw / math.sqrt(P_DK)
        if causal:  # position i may attend to j <= i; the future gets -inf before softmax (paper section 3.2.3)
            scaled = np.where(np.tril(np.ones(scaled.shape, dtype=bool)), scaled, -np.inf)
        A = softmax_rows(scaled)
        heads.append(dict(Q=Q, K=K, V=V, raw=raw, scaled=scaled, A=A, Z=A @ V))
    concat = np.concatenate([hd["Z"] for hd in heads], axis=1)
    return heads, concat, concat @ seeded("WO" if epilogue_tags else f"{tag}:WO", P_HEADS * P_DV, P_MODEL, math.sqrt(3 / (P_HEADS * P_DV)) * out_scale)


def ffn(X, tag, out_scale):
    Hpre = X @ seeded(f"{tag}:W1", P_MODEL, D_FF, math.sqrt(3 / P_MODEL)) + uniform_np(f"{tag}:b1", D_FF, 0.1)
    H = np.maximum(0.0, Hpre)
    return Hpre, H, H @ seeded(f"{tag}:W2", D_FF, P_MODEL, math.sqrt(3 / D_FF) * out_scale) + uniform_np(f"{tag}:b2", P_MODEL, 0.1)


def npdiff(ref, js):
    js = np.array([[(-np.inf if v is None else v) for v in row] for row in js], dtype=np.float64)
    assert ref.shape == js.shape, (ref.shape, js.shape)
    fin = np.isfinite(ref)
    assert (np.isfinite(js) == fin).all() and (ref[~fin] == js[~fin]).all(), "masked entries agree (-inf)"
    return float(np.abs(ref[fin] - js[fin]).max()) if fin.any() else 0.0


worst_d, count_d = 0.0, 0
src, prefix = dive["source"], dive["prefix"]
t_len, s_len = len(prefix), len(src)
assert src == tokens and s_len == 8, "the decoder reads the guided sentence's 8 positions"
# the encoder: layer 1's attention is the epilogue's (its weights, plain W^O); its output is the memory E (8 x 512)
Xe = np.array(pX)
for layer in range(N_LAYERS):
    js = dive["enc"]["layers"][layer]
    heads, cat, out = mha(Xe, Xe, f"EN{layer}S", out_scale=1.0 if layer == 0 else G_BRANCH, epilogue_tags=(layer == 0))
    N1 = ln(Xe + out)
    _, H, F = ffn(N1, f"EN{layer}F", G_BRANCH)
    N2 = ln(N1 + F)
    for hd in heads:
        assert hd["A"].shape == (s_len, s_len), "encoder self-attention scores are 8 x 8 per head"
    worst_d = max(worst_d, npdiff(out, js["MH"]), npdiff(N1, js["N1"]), npdiff(H, js["H"]), npdiff(F, js["F"]), npdiff(N2, js["N2"])); count_d += 5
    for h, hd in enumerate(heads):
        worst_d = max(worst_d, npdiff(hd["A"], js["A"][h])); count_d += 1
    if layer == 0:
        worst_d = max(worst_d, npdiff(out, paper["MH"])); count_d += 1  # = the epilogue's MultiHead
    Xe = N2
M = Xe
assert M.shape == (8, P_MODEL), "memory E is 8 x 512"
root = math.sqrt(P_MODEL)
raw = np.array([uniform_np(f"E:{w.lower()}", P_MODEL, 1.0) / root for w in prefix])
emb = raw * root
X = emb + np.array([pe_d(i, P_MODEL) for i in range(t_len)])
for key, ref in (("M", M), ("raw", raw), ("emb", emb), ("X0", X)):
    worst_d = max(worst_d, npdiff(ref, dive[key])); count_d += 1
for layer in range(N_LAYERS):
    js = dive["layers"][layer]
    s_heads, s_cat, s_out = mha(X, X, f"D{layer}S", causal=True, out_scale=1.0 if layer == 0 else G_BRANCH)
    N1 = ln(X + s_out)
    c_heads, c_cat, c_out = mha(N1, M, f"D{layer}C", variant=lambda h: dive["crossSeed"] if (layer == 0 and h == 0) else "", out_scale=G_BRANCH)
    N2 = ln(N1 + c_out)
    Hpre, H, F = ffn(N2, f"D{layer}F", G_BRANCH)
    N3 = ln(N2 + F)
    for key, ref in (("X", X), ("N1", N1), ("N2", N2), ("Hpre", Hpre), ("H", H), ("F", F), ("N3", N3)):
        worst_d = max(worst_d, npdiff(ref, js[key])); count_d += 1
    for name, heads, cat, out, kv in (("self", s_heads, s_cat, s_out, t_len), ("cross", c_heads, c_cat, c_out, s_len)):
        worst_d = max(worst_d, npdiff(cat, js[name]["Concat"]), npdiff(out, js[name]["out"])); count_d += 2
        for h, hd in enumerate(heads):
            assert hd["Q"].shape == (t_len, P_DK) and hd["K"].shape == (kv, P_DK) and hd["A"].shape == (t_len, kv)
            for key in ("Q", "K", "V", "raw", "scaled", "A", "Z"):
                worst_d = max(worst_d, npdiff(hd[key], js[name]["heads"][h][key])); count_d += 1
        assert cat.shape == (t_len, P_HEADS * P_DV) == (t_len, P_MODEL), "8 x 64 concatenates to 512"
    for hd in c_heads:
        assert hd["Q"].shape == (t_len, P_DK) and hd["K"].shape == (8, P_DK) and hd["V"].shape == (8, P_DV) and hd["A"].shape == (t_len, 8), "cross: Q t x 64, K/V 8 x 64, scores t x 8"
    for h, hd in enumerate(s_heads):
        assert hd["A"][0, 1] == 0.0 and js["self"]["heads"][h]["A"][0][1] == 0.0, "the masked future weight is exactly 0"
    X = N3
worst_d = max(worst_d, npdiff(X, dive["out"])); count_d += 1
for k, step in enumerate(dive["next"]):
    assert step["t"] == k + 1 and step["row"] == step["t"] - 1, "the projection reads the newest position"
    logits = list(dive["logits"][k].values())
    assert logits == step["logits"]
    e = [math.exp(v) for v in logits]
    worst_d = max(worst_d, max(abs(a / sum(e) - b) for a, b in zip(e, step["probs"]))); count_d += 1
print(f"decoder dive: compared {count_d} arrays; max |numpy - js| = {worst_d:.3e}")
if worst_d > 1e-9:
    sys.exit("MISMATCH between the independent decoder reference and the page engine")
print("independent Python reference agrees with the page engine")
