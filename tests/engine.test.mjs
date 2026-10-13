// Numerical tests for the engine embedded in index.html.
// Run: node --test tests/
// The engine <script> is extracted from index.html and executed in a bare VM context (no DOM),
// so these tests exercise exactly the code the page runs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const root = new URL("..", import.meta.url);
const html = readFileSync(new URL("index.html", root), "utf8");
const match = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
assert.ok(match, "index.html must contain <script id=\"engine\">");
const ctx = vm.createContext({});
vm.runInContext(match[1], ctx);
const E = ctx.AttentionEngine;
const fixture = JSON.parse(readFileSync(new URL("tests/fixtures/m1-locked.json", root), "utf8"));

const TOL = 1e-12;
function close(actual, expected, path = "value") {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, `${path}: length`);
    expected.forEach((e, i) => close(actual[i], e, `${path}[${i}]`));
  } else if (expected === null) {
    assert.equal(actual, -Infinity, `${path}: masked logit`);
  } else {
    assert.ok(Math.abs(actual - expected) <= TOL, `${path}: ${actual} vs locked ${expected}`);
  }
}
const tokens = E.tokenize(E.GUIDED.sentence);
const FOCUS = tokens.indexOf("it");
const ANIMAL = tokens.indexOf("animal");
const model = (positionOn = true, scale = Math.SQRT2) => E.computeModel(E.PARAMS, { tokens, positionOn, scale });

test("engine is DOM-free and exposes the spec's functions", () => {
  for (const fn of ["dot", "vectorAdd", "vectorScale", "matVecMul", "matMul", "transpose", "softmax", "relu", "concat", "layerNorm", "positionalEncoding", "scaledDotProductAttention", "computeModel"]) {
    assert.equal(typeof E[fn], "function", fn);
  }
});

test("guided sentence tokenizes into 8 tokens with 'it' at position 4", () => {
  assert.deepEqual([...tokens], ["The", "animal", "stopped", "because", "it", "was", "tired", "."]);
  assert.equal(FOCUS, 4);
  assert.deepEqual([...E.tokenize("The animal didn't cross.")], ["The", "animal", "didn't", "cross", "."]);
});

test("parameters are exactly the approved (locked) M1 values", () => {
  assert.deepEqual(JSON.parse(JSON.stringify(E.PARAMS)), fixture.params);
});

test("matrix shapes follow the paper (d_k = d_v = d_model / h)", () => {
  const { dModel, nHeads, dK, dV, dFF } = E.CONFIG;
  assert.equal(dK, dModel / nHeads);
  assert.equal(dV, dModel / nHeads);
  for (const h of E.PARAMS.heads) {
    for (const [W, cols] of [["WQ", dK], ["WK", dK], ["WV", dV]]) {
      assert.equal(h[W].length, dModel);
      h[W].forEach((row) => assert.equal(row.length, cols));
    }
  }
  assert.equal(E.PARAMS.WO.length, nHeads * dV);
  E.PARAMS.WO.forEach((row) => assert.equal(row.length, dModel));
  assert.equal(E.PARAMS.W1.length, dModel); E.PARAMS.W1.forEach((r) => assert.equal(r.length, dFF));
  assert.equal(E.PARAMS.W2.length, dFF); E.PARAMS.W2.forEach((r) => assert.equal(r.length, dModel));
  assert.equal(E.PARAMS.b1.length, dFF); assert.equal(E.PARAMS.b2.length, dModel);
});

test("positional encoding implements PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(…)", () => {
  for (let pos = 0; pos < 12; pos++) {
    const pe = E.positionalEncoding(pos, 4);
    assert.ok(Math.abs(pe[0] - Math.sin(pos)) < 1e-15);
    assert.ok(Math.abs(pe[1] - Math.cos(pos)) < 1e-15);
    assert.ok(Math.abs(pe[2] - Math.sin(pos / 100)) < 1e-15);
    assert.ok(Math.abs(pe[3] - Math.cos(pos / 100)) < 1e-15);
  }
});

test("every computed array matches the locked M1 fixture (4 conditions)", () => {
  const conds = { on: [true, Math.SQRT2], off: [false, Math.SQRT2], onNoScale: [true, 1], offNoScale: [false, 1] };
  for (const [name, [pos, scale]] of Object.entries(conds)) {
    const m = model(pos, scale), f = fixture.conditions[name];
    for (const k of ["E", "P", "X", "Concat", "MH", "R1", "Hpre", "F"]) close(m[k], f[k], `${name}.${k}`);
    close(m.LN1.map((t) => t.out), f.LN1, `${name}.LN1`);
    close(m.LN2.map((t) => t.out), f.LN2, `${name}.LN2`);
    m.heads.forEach((H, h) => {
      for (const k of ["Q", "K", "V", "raw", "scaled", "A", "Z"]) close(H[k], f.heads[h][k], `${name}.heads[${h}].${k}`);
      close(H.soft.map((s) => s.exps), f.heads[h].exps, `${name}.heads[${h}].exps`);
      close(H.soft.map((s) => s.sum), f.heads[h].sums, `${name}.heads[${h}].sums`);
    });
  }
});

test("decoder masked grid and illustrative next-token softmax match the locked values", () => {
  const D = E.PARAMS.decoder;
  for (const [key, toks] of [["grid4", D.inputs], ["grid5", [...D.inputs, D.next]]]) {
    const g = E.maskedDecoderGrid(toks);
    close(g.masked, fixture.decoder[key].masked, `${key}.masked`);
    close(g.A, fixture.decoder[key].A, `${key}.A`);
    g.A.forEach((row, i) => row.forEach((a, j) => { if (j > i) assert.equal(a, 0, `future cell (${i},${j}) must be exactly 0`); }));
  }
  const nt = E.nextTokenDistribution();
  assert.deepEqual([...nt.vocab], fixture.decoder.vocab);
  close(nt.probs, fixture.decoder.nextTokenProbs, "nextTokenProbs");
});

test("multi-head: two heads with their own projections, Concat is h·d_v = 4, W_O maps back to d_model", () => {
  const m = model(), H = E.PARAMS.heads;
  assert.notDeepEqual(H[0].WQ, H[1].WQ);
  assert.notDeepEqual(H[0].WK, H[1].WK);
  assert.notDeepEqual(H[0].WV, H[1].WV);
  // the patterns differ because the projections differ (no styling involved): head 1 peaks on "animal", head 2 on "because"
  const argmax = (row) => row.indexOf(Math.max(...row));
  assert.equal(tokens[argmax(m.heads[0].A[FOCUS])], "animal");
  assert.equal(tokens[argmax(m.heads[1].A[FOCUS])], "because");
  for (let i = 0; i < tokens.length; i++) {
    assert.equal(m.Concat[i].length, E.CONFIG.nHeads * E.CONFIG.dV);
    assert.deepEqual([...m.Concat[i]], [...m.heads[0].Z[i], ...m.heads[1].Z[i]]);
    assert.equal(E.PARAMS.WO.length, E.CONFIG.nHeads * E.CONFIG.dV);
    assert.equal(E.PARAMS.WO[0].length, E.CONFIG.dModel);
    const direct = E.PARAMS.WO[0].map((_, c) => m.Concat[i].reduce((acc, v, r) => acc + v * E.PARAMS.WO[r][c], 0));
    m.MH[i].forEach((v, c) => assert.ok(Math.abs(v - direct[c]) < 1e-14));
    // encoder layer: residual + LayerNorm, FFN 4 → 8 → ReLU → 4, residual + LayerNorm
    m.R1[i].forEach((v, d) => assert.equal(v, m.X[i][d] + m.MH[i][d]));
    assert.equal(m.Hpre[i].length, E.CONFIG.dFF);
    m.Hid[i].forEach((v, k) => assert.equal(v, Math.max(0, m.Hpre[i][k])));
    assert.equal(m.F[i].length, E.CONFIG.dModel);
    assert.equal(m.LN2[i].out.length, E.CONFIG.dModel);
  }
});

test("softmax rows sum to 1, scaling divides by √d_k, z = Σ α v", () => {
  for (const [pos, scale] of [[true, Math.SQRT2], [false, Math.SQRT2], [true, 1], [false, 1]]) {
    const m = model(pos, scale);
    for (const H of m.heads) {
      H.A.forEach((row) => assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-15));
      H.raw.forEach((row, i) => row.forEach((s, j) => assert.equal(H.scaled[i][j], s / scale)));
      H.Z.forEach((z, i) => z.forEach((zc, c) => {
        const direct = H.A[i].reduce((acc, a, j) => acc + a * H.V[j][c], 0);
        assert.ok(Math.abs(zc - direct) < 1e-14);
      }));
    }
  }
});

test("LayerNorm: mean 0 and variance σ²/(σ²+ε) exactly (post-LN blocks)", () => {
  for (const pos of [true, false]) {
    const m = model(pos);
    for (const t of [...m.LN1, ...m.LN2]) {
      const mu = t.out.reduce((a, b) => a + b, 0) / t.out.length;
      const v = t.out.reduce((a, b) => a + (b - mu) ** 2, 0) / t.out.length;
      assert.ok(Math.abs(mu) < 1e-12);
      assert.ok(Math.abs(v - t.variance / (t.variance + E.CONFIG.lnEps)) < 1e-12);
    }
  }
});

test("position OFF removes PE from the representations (and nothing else)", () => {
  const off = model(false);
  off.X.forEach((x, i) => assert.deepEqual(x, off.E[i]));
  const same = E.computeModel(E.PARAMS, { tokens: ["the", "cat", "saw", "The", "dog"], positionOn: false });
  // identical words are indistinguishable without position: identical rows and columns
  for (const H of same.heads) {
    assert.deepEqual(H.A[0], H.A[3]);
    H.A.forEach((row) => assert.equal(row[0], row[3]));
  }
});

test("computation is deterministic; index.html never calls Math.random", () => {
  assert.deepEqual(JSON.stringify(model()), JSON.stringify(model()));
  assert.ok(!/Math\.random/.test(html));
  const a = E.embeddingFor("zebra"), b = E.embeddingFor("Zebra");
  assert.deepEqual(a, b);
  assert.equal(a.known, false);
  a.vector.forEach((v) => assert.ok(v >= -0.9 && v <= 0.9 && Math.abs(v * 10 - Math.round(v * 10)) < 1e-9));
  assert.equal(E.embeddingFor("Animal").known, true);
});

test("number formatting: proper minus sign, no negative zero, −∞", () => {
  assert.equal(E.fmt(-0.0004, 3), "0.000");
  assert.equal(E.fmt(-1.2568, 2), "−1.26");
  assert.equal(E.fmt(0.125, 2), "0.13");
  assert.equal(E.fmt(-0.125, 2), "−0.13");
  assert.equal(E.fmt(-Infinity, 2), "−∞");
});

// ---------------------------------------------------------------- display consistency (guided path)
// Each highlighted calculation must read exactly when a viewer redoes it with the displayed numbers.
function guidedChecks(p) {
  const m = model(), H = m.heads[0], fails = [];
  const sumOK = (label, terms, total) => { if (E.relationSign(terms, total, p) !== "=") fails.push(label); };
  const prodOK = (label, a, pa, b, pb, c) => { if (E.productSign(a, pa, b, pb, c, p) !== "=") fails.push(label); };
  m.X[FOCUS].forEach((x, d) => sumOK(`Scene 3: x[${d}] = e + PE`, [m.E[FOCUS][d], m.P[FOCUS][d]], x));
  H.qT[FOCUS].columns.forEach((col, c) => {
    col.terms.forEach((t, r) => prodOK(`Scene 4: x[${r}]·W_Q[${r}][${c}]`, m.X[FOCUS][r], p, E.PARAMS.heads[0].WQ[r][c], 1, t));
    sumOK(`Scene 4: q[${c}]`, col.terms, col.sum);
  });
  const st = H.scoreTraces[FOCUS][ANIMAL];
  st.terms.forEach((t, c) => prodOK(`Scene 5: q[${c}]·k_animal[${c}]`, H.Q[FOCUS][c], p, H.K[ANIMAL][c], p, t));
  sumOK("Scene 5: q·k_animal", st.terms, st.sum);
  H.raw[FOCUS].forEach((s, j) => {
    const viewer = E.roundUnits(E.roundUnits(s, p) / 10 ** p / Math.SQRT2, p);
    if (viewer !== E.roundUnits(H.scaled[FOCUS][j], p)) fails.push(`Scene 6: score[${j}] ÷ √2`);
  });
  if (p === 3) sumOK("Scene 6: Σ exp", H.soft[FOCUS].exps, H.soft[FOCUS].sum);
  sumOK("Scene 6: Σ α = 1", H.A[FOCUS], 1);
  m.heads.forEach((Hh, h) => sumOK(`Scene 6: head ${h + 1} Σ α = 1`, Hh.A[FOCUS], 1));
  [0, 1].forEach((c) => {
    prodOK(`Scene 7: α_animal·v_animal[${c}]`, H.A[FOCUS][ANIMAL], p, H.V[ANIMAL][c], p, H.weighted[FOCUS][ANIMAL][c]);
    m.heads.forEach((Hh, h) => sumOK(`Scene 7: head ${h + 1} z[${c}]`, Hh.weighted[FOCUS].map((w) => w[c]), Hh.Z[FOCUS][c]));
  });
  m.R1[FOCUS].forEach((r, d) => sumOK(`Scene 9: x[${d}] + MultiHead[${d}] = R1`, [m.X[FOCUS][d], m.MH[FOCUS][d]], r));
  return fails;
}
test("guided-path arithmetic reads exactly at 2 decimals", () => assert.deepEqual(guidedChecks(2), []));
test("guided-path arithmetic reads exactly at 3 decimals (Show calculation)", () => assert.deepEqual(guidedChecks(3), []));

test("off-path rounding is flagged with ≈ instead of being hidden", () => {
  // Head 2's q expansion (Scene 8, Show calculation only): displayed terms 0.97 − 1.15 + 0.06 + 0.28 vs 0.48
  const H2 = model().heads[1].qT[FOCUS].columns[0];
  assert.equal(E.relationSign(H2.terms, H2.sum, 2), "≈");
  // exact half-way products are ambiguous for a viewer → ≈
  assert.equal(E.productSign(1.25, 2, 0.1, 1, 0.125, 2), "≈");
  assert.equal(E.productSign(1.26, 2, 0.1, 1, 0.126, 2), "=");
});

// ---------------------------------------------------------------- epilogue: the same operations at paper scale
// Arrays created inside the VM context have that context's Array.prototype, so every deepEqual below
// compares spread copies.
const PAPER = E.paperScale(tokens);
const same = (a, b, msg) => assert.deepEqual([...a], [...b], msg);
const plainRowTimes = (v, W) => [...W[0]].map((_, c) => { let s = 0; for (let r = 0; r < v.length; r++) s += v[r] * W[r][c]; return s; });

test("paper scale uses the base model's sizes (§3.2.2): d_model 512, h 8, d_k = d_v = d_model/h = 64", () => {
  const { dModel, nHeads, dK, dV } = E.CONFIG.paper, n = tokens.length;
  assert.deepEqual([dModel, nHeads, dK, dV], [512, 8, 64, 64]);
  assert.equal(dK, dModel / nHeads);
  assert.equal(dV, dModel / nHeads);
  const shape = (M, r, c, what) => { assert.equal(M.length, r, `${what}: rows`); M.forEach((row) => assert.equal(row.length, c, `${what}: cols`)); };
  for (const [M, what] of [[PAPER.E, "E"], [PAPER.P, "PE"], [PAPER.X, "X"], [PAPER.Concat, "Concat"], [PAPER.MH, "MultiHead"]]) shape(M, n, dModel, what);
  assert.equal(PAPER.heads.length, nHeads);
  for (const H of PAPER.heads) { shape(H.Q, n, dK, "Q"); shape(H.K, n, dK, "K"); shape(H.V, n, dV, "V"); shape(H.A, n, n, "A"); shape(H.Z, n, dV, "head output"); }
  for (let h = 0; h < nHeads; h++) { const W = E.paperHeadWeights(h); shape(W.WQ, dModel, dK, "W_Q"); shape(W.WK, dModel, dK, "W_K"); shape(W.WV, dModel, dV, "W_V"); }
  shape(E.paperWO(), nHeads * dV, dModel, "W^O (h·d_v × d_model)");
});

test("paper scale: positional encoding is the paper's formula at d_model = 512, and X = E + PE", () => {
  const d = E.CONFIG.paper.dModel;
  PAPER.P.forEach((pe, pos) => {
    same(pe, E.positionalEncoding(pos, d));
    for (const i of [0, 1, 7, 100, 255]) {
      const angle = pos / Math.pow(10000, (2 * i) / d);
      assert.ok(Math.abs(pe[2 * i] - Math.sin(angle)) < 1e-15 && Math.abs(pe[2 * i + 1] - Math.cos(angle)) < 1e-15, `PE(${pos}, ${2 * i})`);
    }
  });
  PAPER.X.forEach((x, i) => x.forEach((v, c) => assert.equal(v, PAPER.E[i][c] + PAPER.P[i][c])));
});

test("paper scale: each head has its own W_Q, W_K, W_V; Q = X·W_Q …; softmax(QKᵀ/√64)V; Concat·W^O", () => {
  const { nHeads, dK } = E.CONFIG.paper, n = tokens.length;
  assert.notDeepEqual([...E.paperHeadWeights(0).WQ[0]], [...E.paperHeadWeights(1).WQ[0]]);
  PAPER.heads.forEach((H, h) => {
    const W = E.paperHeadWeights(h);
    for (const [M, w] of [[H.Q, W.WQ], [H.K, W.WK], [H.V, W.WV]]) M.forEach((row, i) => same(row, plainRowTimes(PAPER.X[i], w), `head ${h + 1}, row ${i}`));
    H.A.forEach((row, i) => {
      assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-14, "softmax rows sum to 1");
      same(row, E.softmax(H.K.map((k) => E.dot(H.Q[i], k) / Math.sqrt(dK))), "scaled by √d_k = 8");
      H.Z[i].forEach((z, c) => { let s = 0; for (let j = 0; j < n; j++) s += row[j] * H.V[j][c]; assert.ok(Math.abs(z - s) < 1e-14); });
    });
  });
  const WO = E.paperWO();
  PAPER.Concat.forEach((c, i) => same(c, PAPER.heads.flatMap((H) => [...H.Z[i]]), "Concat(head_1, …, head_8)"));
  PAPER.MH.forEach((m, i) => same(m, plainRowTimes(PAPER.Concat[i], WO), "MultiHead = Concat·W^O"));
  assert.equal(PAPER.Concat[0].length, nHeads * E.CONFIG.paper.dV);
});

test("paper scale values are deterministic, readable, illustrative — and not the toy vectors", () => {
  assert.equal(JSON.stringify(E.paperScale(tokens)), JSON.stringify(PAPER));
  same(E.paperScale(["The"]).E[0], E.paperScale(["the"]).E[0], "case-insensitive, like the toy table");
  const all = (M) => M.flat();
  assert.ok(all(PAPER.X).every((v) => Math.abs(v) < 2.5), "X stays readable");
  for (const H of PAPER.heads) {
    assert.ok(all(H.Q).every((v) => Math.abs(v) < 6) && all(H.K).every((v) => Math.abs(v) < 6), "Q, K stay readable");
    assert.ok(Math.max(...all(H.A)) < 0.99, "softmax is not saturated");
  }
  // the 4-d toy x_it is not the first four entries of the illustrative 512-d one
  const toy = model().X[FOCUS];
  assert.ok(toy.some((v, d) => Math.abs(v - PAPER.X[FOCUS][d]) > 0.05));
  // the locked toy model is untouched by the epilogue
  assert.deepEqual(JSON.parse(JSON.stringify(E.PARAMS)), fixture.params);
});

// ---------------------------------------------------------------- optional branch: inside the decoder, at paper scale
const DD = E.paperDecoder();
const PDc = E.CONFIG.paper;
const shapeOf = (M) => [M.length, M[0].length];
const ampOf = (fanIn) => Math.sqrt(3 / fanIn);
const closeRows = (A, B, tol, msg) => A.forEach((row, i) => row.forEach((v, c) => assert.ok(Math.abs(v - B[i][c]) <= tol, `${msg} [${i}][${c}]: ${v} vs ${B[i][c]}`)));
const lnRef = (x) => { const m = x.reduce((a, b) => a + b, 0) / x.length, v = x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length; return x.map((u) => (u - m) / Math.sqrt(v + E.CONFIG.lnEps)); };

test("decoder dive reuses Scene 10's example: s and t come from its token arrays", () => {
  const T = E.TRANSLATION;
  same(DD.source, T.source, "source = Scene 10's source sentence");
  assert.equal(DD.M.length, T.source.length, "s = number of source tokens");
  same(DD.prefix, ["<START>", T.target[0]], "step 2 prefix: <START> + the first generated token");
  same(DD.generated, [...T.target, E.DECODER_DIVE.end], "generation runs through the target to the end token");
  DD.next.forEach((nx, k) => {
    assert.equal(nx.vocab[0], T.target[k], `step ${k + 1}: the most probable token is the example's next word`);
    assert.equal(nx.probs.indexOf(Math.max(...nx.probs)), 0);
    assert.ok(Math.abs(nx.probs.reduce((a, b) => a + b, 0) - 1) < 1e-14);
  });
});

test("decoder dive: shapes at every stage (paper base model: 512, 8 × 64, 2048, N = 6)", () => {
  const t = DD.prefix.length, s = DD.source.length, { dModel, nHeads, dK, dV, dFF, layers } = PDc;
  assert.deepEqual([dModel, nHeads, dK, dV, dFF, layers], [512, 8, 64, 64, 2048, 6]);
  assert.deepEqual(shapeOf(DD.M), [s, dModel], "encoder output s × 512");
  for (const k of ["raw", "emb", "P", "X0", "out"]) assert.deepEqual(shapeOf(DD[k]), [t, dModel], `${k}: t × 512`);
  assert.equal(DD.layers.length, layers, "six decoder layers");
  DD.layers.forEach((Ly, l) => {
    const at = (what) => `layer ${l + 1}: ${what}`;
    for (const k of ["X", "R1", "N1", "R2", "N2", "F", "R3", "N3"]) assert.deepEqual(shapeOf(Ly[k]), [t, dModel], at(`${k} t × 512`));
    for (const k of ["Hpre", "H"]) assert.deepEqual(shapeOf(Ly[k]), [t, dFF], at(`${k} t × 2048`));
    for (const [att, kv] of [[Ly.self, t], [Ly.cross, s]]) {
      assert.equal(att.heads.length, nHeads);
      att.heads.forEach((H, h) => {
        assert.deepEqual(shapeOf(H.Q), [t, dK], at(`head ${h + 1} Q t × 64`));
        assert.deepEqual(shapeOf(H.K), [kv, dK], at(`head ${h + 1} K`));
        assert.deepEqual(shapeOf(H.V), [kv, dV], at(`head ${h + 1} V`));
        for (const k of ["raw", "scaled", "A"]) assert.deepEqual(shapeOf(H[k]), [t, kv], at(`head ${h + 1} ${k}: ${att === Ly.self ? "t × t" : "t × s"}`));
        assert.deepEqual(shapeOf(H.Z), [t, dV], at(`head ${h + 1} output t × 64`));
      });
      assert.deepEqual(shapeOf(att.Concat), [t, nHeads * dV], at("Concat: 8 × 64 = 512"));
      assert.equal(nHeads * dV, dModel);
      assert.deepEqual(shapeOf(att.out), [t, dModel], at("W^O output t × 512"));
    }
  });
});

test("decoder dive: scores are divided by √d_k = √64 = 8; softmax rows sum to 1", () => {
  assert.equal(Math.sqrt(PDc.dK), 8);
  for (const Ly of DD.layers) for (const att of [Ly.self, Ly.cross]) for (const H of att.heads) {
    H.raw.forEach((row, i) => row.forEach((r, j) => { if (H.scaled[i][j] !== -Infinity) assert.equal(H.scaled[i][j], r / 8); }));
    H.raw.forEach((row, i) => row.forEach((r, j) => assert.equal(H.unmasked[i][j], r / 8)));
    H.A.forEach((row) => assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-14));
  }
});

test("decoder dive: the causal mask — future logits are −∞, their weights exactly 0; at t = 1 there is nothing to mask", () => {
  for (const Ly of DD.layers) for (const H of Ly.self.heads) {
    assert.equal(H.scaled[0][1], -Infinity, "<START> cannot see L'");
    assert.equal(H.A[0][1], 0, "masked weight is exactly 0");
    assert.equal(H.A[0][0], 1, "so <START> keeps all of its weight");
    assert.ok(H.A[1][0] > 0 && H.A[1][1] > 0, "the second position sees both");
  }
  for (const Ly of DD.layers) for (const H of Ly.cross.heads) H.A.forEach((row) => row.forEach((a) => assert.ok(a > 0, "cross-attention is not masked")));
  const one = E.paperDecoder([DD.prefix[0]]);
  for (const Ly of one.layers) for (const H of Ly.self.heads) { assert.deepEqual(shapeOf(H.A), [1, 1]); assert.equal(H.A[0][0], 1, "softmax([s/8]) = [1]"); same(H.Z[0], H.V[0], "head output = 1 · v"); }
});

test("decoder dive: causality — row 0 of the two-token run equals the one-token run, bit for bit", () => {
  const one = E.paperDecoder([DD.prefix[0]]);
  one.layers.forEach((Ly, l) => {
    for (const k of ["N1", "N2", "Hpre", "N3"]) same(Ly[k][0], DD.layers[l][k][0], `layer ${l + 1} ${k}[0]`);
    Ly.cross.heads.forEach((H, h) => same(H.A[0], DD.layers[l].cross.heads[h].A[0], `layer ${l + 1} cross head ${h + 1}`));
  });
  same(one.out[0], DD.out[0]);
});

test("decoder dive: Q from the decoder, K and V from the encoder output; each head projects the whole 512-d row", () => {
  const Ly = DD.layers[0], a = ampOf(PDc.dModel), v = E.DECODER_DIVE.crossSeed;
  for (const h of [0, 3]) {
    const tagQ = `D0C:WQ${h}${h === 0 ? v : ""}`, tagK = `D0C:WK${h}${h === 0 ? v : ""}`, H = Ly.cross.heads[h];
    const WQ = E.paperMatrix(tagQ, 512, 64, a), WK = E.paperMatrix(tagK, 512, 64, a), WV = E.paperMatrix(`D0C:WV${h}`, 512, 64, a);
    assert.equal(WQ.length, 512, "W_Q has 512 rows: the whole vector, not a 64-d slice");
    H.Q.forEach((q, i) => same(q, plainRowTimes(Ly.N1[i], WQ), "Q = (decoder state) · W_Q"));
    H.K.forEach((k, j) => same(k, plainRowTimes(DD.M[j], WK), "K = (encoder output) · W_K"));
    H.V.forEach((vv, j) => same(vv, plainRowTimes(DD.M[j], WV), "V = (encoder output) · W_V"));
  }
  const S = Ly.self.heads[2], W = E.paperMatrix("D0S:WQ2", 512, 64, a);
  S.Q.forEach((q, i) => same(q, plainRowTimes(Ly.X[i], W), "masked self-attention: Q from the decoder input"));
});

test("decoder dive: Concat · W^O (512 → 512), post-LN Add & Norm, FFN 512 → 2048 → ReLU → 512", () => {
  const Ly = DD.layers[0];
  const WO = E.paperMatrix("D0S:WO", 512, 512, ampOf(512));
  Ly.self.Concat.forEach((c, i) => same(c, Ly.self.heads.flatMap((H) => [...H.Z[i]]), "Concat(head_1, …, head_8)"));
  Ly.self.out.forEach((o, i) => same(o, plainRowTimes(Ly.self.Concat[i], WO), "MultiHead = Concat · W^O"));
  // post-LN (§3.1): LayerNorm(x + Sublayer(x)) after each of the three sub-layers
  closeRows(Ly.N1, Ly.X.map((x, i) => lnRef(x.map((u, c) => u + Ly.self.out[i][c]))), 1e-12, "N1 = LN(X + SelfAttn)");
  closeRows(Ly.N2, Ly.N1.map((x, i) => lnRef(x.map((u, c) => u + Ly.cross.out[i][c]))), 1e-12, "N2 = LN(N1 + CrossAttn)");
  closeRows(Ly.N3, Ly.N2.map((x, i) => lnRef(x.map((u, c) => u + Ly.F[i][c]))), 1e-12, "N3 = LN(N2 + FFN)");
  const W1 = E.paperMatrix("D0F:W1", 512, 2048, ampOf(512)), b1 = E.paperUniform("D0F:b1", 2048, 0.1);
  Ly.Hpre.forEach((hp, i) => same(hp, plainRowTimes(Ly.N2[i], W1).map((u, c) => u + b1[c]), "zW₁ + b₁"));
  Ly.H.forEach((h, i) => same(h, Ly.Hpre[i].map((u) => Math.max(0, u)), "ReLU"));
  assert.ok(Ly.H[0].some((u) => u === 0) && Ly.H[0].some((u) => u > 0), "ReLU zeroes some units, not all");
  const W2 = E.paperMatrix("D0F:W2", 2048, 512, ampOf(2048)), b2 = E.paperUniform("D0F:b2", 512, 0.1);
  Ly.F.forEach((f, i) => same(f, plainRowTimes(Ly.H[i], W2).map((u, c) => u + b2[c]), "FFN(z) = ReLU(·)W₂ + b₂"));
  // the decoder input: embedding row, × √512 (§3.4), + PE (§3.5)
  DD.emb.forEach((e, i) => e.forEach((u, c) => assert.equal(u, DD.raw[i][c] * Math.sqrt(512))));
  DD.X0.forEach((x, i) => { same(DD.P[i], E.positionalEncoding(i, 512)); x.forEach((u, c) => assert.equal(u, DD.emb[i][c] + DD.P[i][c])); });
});

test("decoder dive: seededMatMul is the product with the materialised seeded matrix, bit for bit", () => {
  const X = [E.paperUniform("probe:a", 512, 1), E.paperUniform("probe:b", 512, 1)];
  for (const [tag, cols] of [["probe:W", 64], ["D0S:WV5", 64]]) same(E.seededMatMul(X, tag, 512, cols, 0.1).map((r) => [...r]).flat(), E.matMulPlain(X, E.paperMatrix(tag, 512, cols, 0.1)).flat());
});

test("decoder dive: every layer has its own parameters", () => {
  const a = ampOf(512);
  for (const w of ["S:WQ0", "C:WK3", "S:WO", "F:W1"]) {
    const rows = w === "S:WO" ? 512 : 512, cols = w === "F:W1" ? 2048 : w === "S:WO" ? 512 : 64;
    const first = (l) => [...E.paperMatrix(`D${l}${w}`, rows, cols, a)[0]];
    for (let l = 1; l < PDc.layers; l++) assert.notDeepEqual(first(l), first(0), `layer ${l + 1} ${w} differs from layer 1`);
  }
  for (let l = 1; l < PDc.layers; l++) assert.notDeepEqual([...DD.layers[l].N3[0]], [...DD.layers[l - 1].N3[0]]);
});

test("decoder dive: the vocabulary projection reads the newest decoder position only", () => {
  DD.next.forEach((nx, k) => {
    assert.equal(nx.t, k + 1, "prefix length at that step");
    assert.equal(nx.row, nx.t - 1, "newest position = last row");
    same(nx.h, DD.out[nx.row], "h_last = the last row of the decoder output");
  });
  const one = E.paperDecoder([DD.prefix[0]]);
  same(DD.next[0].h, one.out[one.out.length - 1], "step 1: the one-token run's last row");
});

test("decoder dive values are deterministic, illustrative and readable", () => {
  assert.equal(JSON.stringify(E.paperDecoder()), JSON.stringify(DD));
  const big = (M) => Math.max(...M.flat().map(Math.abs));
  assert.ok(big(DD.X0) < 3 && big(DD.out) < 5, "decoder vectors stay readable");
  const cross = DD.layers[0].cross.heads[0].A[0];
  assert.ok(Math.max(...cross) < 0.9 && Math.min(...cross) > 0.005, "illustrative cross weights neither saturated nor vanishing");
  assert.deepEqual(JSON.parse(JSON.stringify(E.PARAMS)), fixture.params, "the toy model is untouched");
});
