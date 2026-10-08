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

print(f"compared {count} arrays; max |python - js| = {worst:.3e}")
if worst > 1e-12:
    sys.exit("MISMATCH between the independent reference and the page engine")
print("independent Python reference agrees with the page engine")
