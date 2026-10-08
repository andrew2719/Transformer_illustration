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
