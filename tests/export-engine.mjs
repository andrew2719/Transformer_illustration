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
writeFileSync(new URL("tests/fixtures/engine-export.json", root), JSON.stringify({
  sentence: E.GUIDED.sentence, tokens, params: E.PARAMS, conds,
  decoder: { g4: grid(D.inputs), g5: grid([...D.inputs, D.next]), vocab: nt.vocab, nextP: nt.probs },
}));
console.log("wrote tests/fixtures/engine-export.json");
