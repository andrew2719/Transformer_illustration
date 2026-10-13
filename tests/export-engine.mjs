// Writes the page engine's parameters and results to tests/fixtures/engine-export.json
// for the independent Python reference (tests/reference.py).
// Run: node tests/export-engine.mjs && python3 tests/reference.py
import { readFileSync, writeFileSync } from "node:fs";
import vm from "node:vm";

const root = new URL("..", import.meta.url);
const html = readFileSync(new URL("index.html", root), "utf8");
const ctx = vm.createContext({});
vm.runInContext(html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1], ctx);
const E = ctx.AttentionEngine;
const tokens = E.tokenize(E.GUIDED.sentence);
const conds = {};
for (const [k, positionOn, scale] of [["on", true, Math.SQRT2], ["off", false, Math.SQRT2], ["onNoScale", true, 1], ["offNoScale", false, 1]]) {
  const m = E.computeModel(E.PARAMS, { tokens, positionOn, scale });
  conds[k] = {
    X: m.X,
    heads: m.heads.map((H) => ({ Q: H.Q, K: H.K, V: H.V, raw: H.raw, scaled: H.scaled, exps: H.soft.map((s) => s.exps), sums: H.soft.map((s) => s.sum), A: H.A, weighted: H.weighted, Z: H.Z })),
    Concat: m.Concat, MH: m.MH, R1: m.R1, LN1: m.LN1.map((t) => t.out), Hpre: m.Hpre, F: m.F, LN2: m.LN2.map((t) => t.out),
  };
}
const D = E.PARAMS.decoder;
const grid = (toks) => { const g = E.maskedDecoderGrid(toks); return { masked: g.masked.map((r) => r.map((x) => (x === -Infinity ? null : x))), A: g.A }; };
const nt = E.nextTokenDistribution();
// epilogue: the paper-scale arrays (illustrative seeded values); the reference regenerates the weights itself
const P = E.paperScale(tokens);
const paper = { seed: E.PAPER_SEED, dims: E.CONFIG.paper, X: P.X, heads: P.heads.map((H) => ({ Q: H.Q, K: H.K, V: H.V, A: H.A, Z: H.Z })), Concat: P.Concat, MH: P.MH };
// optional branch: one decoder at paper scale on the Scene 10 example (seeded illustrative weights)
const DV = E.paperDecoder();
const inf = (M) => M.map((r) => r.map((x) => (x === -Infinity ? null : x)));
const att = (a) => ({ heads: a.heads.map((H) => ({ Q: H.Q, K: H.K, V: H.V, raw: H.raw, scaled: inf(H.scaled), A: H.A, Z: H.Z })), Concat: a.Concat, out: a.out });
const dive = {
  crossSeed: E.DECODER_DIVE.crossSeed, branchScale: E.BRANCH_SCALE, source: DV.source, prefix: DV.prefix, M: DV.M, raw: DV.raw, emb: DV.emb, P: DV.P, X0: DV.X0, out: DV.out,
  enc: { X0: DV.enc.X0, layers: DV.enc.layers.map((L) => ({ A: L.self.heads.map((H) => H.A), MH: L.self.out, N1: L.N1, H: L.H, F: L.F, N2: L.N2 })) },
  layers: DV.layers.map((L) => ({ X: L.X, self: att(L.self), N1: L.N1, cross: att(L.cross), N2: L.N2, Hpre: L.Hpre, H: L.H, F: L.F, N3: L.N3 })),
  next: DV.next.map((n) => ({ t: n.t, row: n.row, logits: n.logits, probs: n.probs })), logits: E.DECODER_DIVE.next,
};
writeFileSync(new URL("tests/fixtures/engine-export.json", root), JSON.stringify({
  sentence: E.GUIDED.sentence, tokens, params: E.PARAMS, conds,
  decoder: { g4: grid(D.inputs), g5: grid([...D.inputs, D.next]), vocab: nt.vocab, nextP: nt.probs },
  paper, dive,
}));
console.log("wrote tests/fixtures/engine-export.json");
