// Browser checks for index.html in headless Chrome, driven over the DevTools protocol
// (Node ≥ 22 built-in WebSocket; no npm packages).
// Run: node tests/browser.mjs          (CHROME=/path/to/chrome to override the binary)
// Writes screenshots to tests/screenshots/<width>x<height>/.
import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const CHROME = process.env.CHROME ?? "google-chrome";
const root = new URL("..", import.meta.url);
const pageUrl = new URL("index.html", root).href;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const failures = [];
const fail = (msg) => { failures.push(msg); console.log(`  ✗ ${msg}`); };

// ------------------------------------------------------------------ minimal CDP client
async function launch() {
  const dir = mkdtempSync(path.join(tmpdir(), "attn-demo-"));
  const proc = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
    `--user-data-dir=${dir}`, "--remote-debugging-port=0", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
  const wsUrl = await new Promise((resolve, reject) => {
    let buf = "";
    proc.stderr.on("data", (d) => { buf += d; const m = buf.match(/DevTools listening on (ws:\/\/\S+)/); if (m) resolve(m[1]); });
    proc.on("exit", () => reject(new Error(`chrome exited:\n${buf}`)));
    setTimeout(() => reject(new Error("timed out launching chrome")), 20000);
  });
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0;
  const pending = new Map(), listeners = [];
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error))); else resolve(msg.result);
    } else listeners.forEach((l) => l(msg));
  };
  const send = (method, params = {}, sessionId) => {
    const mid = ++id;
    ws.send(JSON.stringify({ id: mid, method, params, sessionId }));
    return new Promise((resolve, reject) => pending.set(mid, { resolve, reject }));
  };
  return { proc, dir, ws, send, on: (l) => listeners.push(l) };
}

const browser = await launch();
const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await browser.send("Target.attachToTarget", { targetId, flatten: true });
const S = (m, p) => browser.send(m, p, sessionId);
const consoleErrors = [];
browser.on((msg) => {
  if (msg.sessionId !== sessionId) return;
  if (msg.method === "Runtime.exceptionThrown") consoleErrors.push(`exception: ${msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text}`);
  if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(msg.params.type)) consoleErrors.push(`console.${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(" ")}`);
  if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") consoleErrors.push(`log: ${msg.params.entry.text}`);
});
await S("Page.enable"); await S("Runtime.enable"); await S("Log.enable");

async function evaluate(expression) {
  const r = await S("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`evaluate failed: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
  return r.result.value;
}
async function waitFor(expr, timeout = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    try { if (await evaluate(`!!(${expr})`)) return; } catch { /* page still loading */ }
    await sleep(40);
  }
  throw new Error(`timed out waiting for ${expr}`);
}
const idle = async () => { await waitFor("__demo.idle()", 90000); await sleep(50); };
async function load(width, height, { reducedMotion = false, speed = 6 } = {}) {
  await S("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  await S("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: reducedMotion ? "reduce" : "no-preference" }] });
  await S("Page.navigate", { url: pageUrl });
  await waitFor("globalThis.__demo && __demo.ready");
  await evaluate(`__demo.speed(${speed})`);
  await idle();
}
async function shot(dir, name) {
  const { data } = await S("Page.captureScreenshot", { format: "png" });
  const d = new URL(`tests/screenshots/${dir}/`, root);
  mkdirSync(d, { recursive: true });
  writeFileSync(new URL(name, d), Buffer.from(data, "base64"));
}

// Layout regressions, measured in screen space on what is actually visible:
//  - text overlapping other text;
//  - at rest: numbers crowding numbers or operators on the same line (they would read as one number),
//    numbers touching a vector/matrix bracket, and α labels touching their bars.
const LAYOUT = (atRest) => `((atRest) => {
  const vis = (el) => { let o = 1; for (let n = el; n && n.id !== "stage"; n = n.parentNode) { if (!n.style) continue; if (n.style.visibility === "hidden") return 0; if (n.style.opacity !== "") o *= Number(n.style.opacity); } return o; };
  const stage = document.getElementById("stage").getBoundingClientRect();
  const k = Math.min(stage.width / 1920, stage.height / 1080); // screen px per stage unit
  const items = [...document.querySelectorAll("#stage text")].map((t) => ({ t, s: t.textContent.trim(), o: vis(t), r: t.getBoundingClientRect() })).filter((x) => x.o > 0.45 && x.r.width > 0 && x.s);
  const isNum = (s) => /[0-9]/.test(s), isOp = (s) => /^[=≈+×÷·\\u2212]$/.test(s);
  const vOver = (a, b) => Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
  const hGap = (a, b) => Math.max(a.left, b.left) - Math.min(a.right, b.right);
  const name = (x) => x.s.slice(0, 24);
  const overlaps = [], crowded = [];
  for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
    const A = items[i], B = items[j], a = A.r, b = B.r;
    const w = Math.min(a.right, b.right) - Math.max(a.left, b.left), h = vOver(a, b);
    if (w > 2 && h > 2 && w * h > 0.15 * Math.min(a.width * a.height, b.width * b.height)) { overlaps.push(name(A) + "  ×  " + name(B)); continue; }
    if (!atRest) continue;
    const numA = isNum(A.s), numB = isNum(B.s);
    if (!((numA && numB) || (numA && isOp(B.s)) || (isOp(A.s) && numB))) continue;
    const hmin = Math.min(a.height, b.height);
    if (h > 0.5 * hmin && hGap(a, b) < 0.22 * hmin) crowded.push(name(A) + "  |  " + name(B) + "  gap " + hGap(a, b).toFixed(1) + "px");
  }
  if (atRest) {
    const nums = items.filter((x) => isNum(x.s));
    for (const el of document.querySelectorAll("#stage path.bracket")) {
      if (vis(el) < 0.45) continue;
      const r = el.getBoundingClientRect(), pr = el.parentNode.getBoundingClientRect();
      if (!r.height) continue;
      const left = r.left + r.width / 2 < pr.left + pr.width / 2, line = left ? r.left : r.right;
      for (const x of nums) {
        if (vOver(x.r, r) < 0.3 * x.r.height) continue;
        // clearance between the number and the bracket's vertical stroke, on whichever side it sits
        const clear = left ? (x.r.left >= line - 1 ? x.r.left - line : line - x.r.right) : (x.r.right <= line + 1 ? line - x.r.right : x.r.left - line);
        if (clear < 4 * k) crowded.push(name(x) + "  |  bracket  gap " + clear.toFixed(1) + "px");
      }
    }
    for (const el of document.querySelectorAll("#stage rect.bar")) {
      if (vis(el) < 0.45) continue;
      const r = el.getBoundingClientRect();
      if (!r.width) continue;
      for (const x of nums) if (vOver(x.r, r) > 0 && x.r.right > r.left && x.r.left - r.right < 6 * k) crowded.push(name(x) + "  |  bar  gap " + (x.r.left - r.right).toFixed(1) + "px");
    }
  }
  const outside = items.filter((x) => x.r.left < stage.left - 1 || x.r.right > stage.right + 1 || x.r.top < stage.top - 1 || x.r.bottom > stage.bottom + 1).map(name);
  // wires, leaders and blocks must not be cropped by the stage edge
  for (const el of document.querySelectorAll("#stage path.wire, #stage path.lead, #stage rect.box-r")) {
    if (vis(el) < 0.05) continue;
    const r = el.getBoundingClientRect();
    if (r.width + r.height > 0 && (r.left < stage.left - 2 || r.right > stage.right + 2 || r.top < stage.top - 2 || r.bottom > stage.bottom + 2)) outside.push((el.closest("[id]")?.id || el.getAttribute("class")) + " (shape)");
  }
  const approx = items.filter((x) => x.t.textContent.includes("\\u2248")).map((x) => x.s);
  return { overlaps, crowded, outside, approx };
})(${atRest})`;

// A beat must look the same however it was reached (forwards, backwards, by a jump or after a skip):
// otherwise some layout is relying on state left over from another scene.
const snaps = [];
// Positions may differ by sub-pixel text-measurement noise between viewport sizes; anything else must match.
const nearlySame = (a, b) => {
  if (a === undefined || b === undefined) return a === b;
  for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
    const x = a[key], y = b[key];
    if (typeof x === "number" && typeof y === "number") { if (Math.abs(x - y) > (["o", "s", "draw", "glow", "lo", "ts", "pl", "pill", "c", "so", "nu", "ix", "hla"].includes(key) ? 0.02 : 1.5)) return false; }
    else if (key === "geo" && typeof x === "string" && typeof y === "string") {
      const u = x.split(",").map(Number), v = y.split(",").map(Number);
      if (u.length !== v.length || u.some((n, i) => Math.abs(n - v[i]) > 1.5)) return false;
    } else if (x !== y) return false;
  }
  return true;
};
async function sameAsForward(label, i) {
  if (!snaps[i]) return;
  const now = await evaluate("__demo.snapshot()"), ref = snaps[i], diffs = [];
  for (const k of new Set([...Object.keys(ref), ...Object.keys(now)])) {
    if (!nearlySame(ref[k], now[k])) diffs.push(`${k}: ${JSON.stringify(ref[k]) ?? "hidden"} → ${JSON.stringify(now[k]) ?? "hidden"}`);
  }
  if (diffs.length) fail(`${label}: ${diffs.length} object(s) differ from the forward pass, e.g. ${diffs.slice(0, 2).join(" | ")}`);
}

async function check(label, { allowApprox = false, geometry = true } = {}) {
  const a = await evaluate("__demo.audit()");
  if (a.mismatches.length) fail(`${label}: ${a.mismatches.length} displayed numbers differ from the engine, e.g. ${JSON.stringify(a.mismatches[0])}`);
  if (a.badText.length) fail(`${label}: invalid text ${JSON.stringify(a.badText.slice(0, 3))}`);
  if (geometry && a.geometry.mismatches.length) fail(`${label}: ${a.geometry.mismatches.length} bars/wires not drawn from their α, e.g. ${JSON.stringify(a.geometry.mismatches[0])}`);
  const v = await evaluate(LAYOUT(geometry));
  if (v.overlaps.length) fail(`${label}: overlapping text ${JSON.stringify(v.overlaps.slice(0, 4))}`);
  if (v.crowded.length) fail(`${label}: crowded numbers ${JSON.stringify(v.crowded.slice(0, 4))}`);
  if (v.outside.length) fail(`${label}: text or shapes outside the stage ${JSON.stringify(v.outside.slice(0, 4))}`);
  if (!allowApprox && v.approx.length) fail(`${label}: "≈" shown on the guided path ${JSON.stringify(v.approx)}`);
  return a.checked;
}

// The presentation has a main path (Scenes 1–10, the epilogue, Explore) and an optional branch, "Inside the decoder",
// between the epilogue and Explore: Next at the end of the epilogue steps over it; I (or its button) enters it.
let beats, BR0, BR1, EXI, PUI;
const isBranch = (i) => beats[i].branch;
// Walk every beat the way a presenter would: Next through the main path, then back to the end of the epilogue,
// I into the branch and Next through it to Explore. `visit(i)` runs at each beat once it is at rest.
async function walk(visit) {
  for (let i = 0; i < beats.length; i++) {
    if (isBranch(i)) continue;
    if (i > 0) { await evaluate("__demo.next()"); await idle(); }
    const at = await evaluate("__demo.index");
    if (at !== i) fail(`main path: expected beat ${i + 1} (${beats[i].id}), got ${at + 1}`);
    await visit(i);
  }
  await evaluate(`__demo.goto(${PUI}, 0)`); await idle();
  await evaluate("__demo.decoder()"); await idle();
  for (let i = BR0; i <= BR1 + 1; i++) {
    if (i > BR0) { await evaluate("__demo.next()"); await idle(); }
    const at = await evaluate("__demo.index");
    if (at !== i) fail(`branch: expected beat ${i + 1} (${beats[i].id}), got ${at + 1}`);
    await visit(i, i === EXI ? "explore, entered from the branch" : null);
  }
}
const pad = (i) => String(i + 1).padStart(2, "0");

try {
  // ---------------------------------------------------------------- 1920×1080: forward through every beat
  console.log("1920×1080 — forward pass (main path, then the optional branch)");
  await load(1920, 1080);
  beats = await evaluate("__demo.beats");
  BR0 = beats.findIndex((b) => b.branch); BR1 = beats.findLastIndex((b) => b.branch);
  EXI = beats.findIndex((b) => b.id === "explore"); PUI = beats.findIndex((b) => b.id === "p-uses");
  if (BR0 !== PUI + 1 || EXI !== BR1 + 1) fail(`the branch must sit between the epilogue and Explore (p-uses ${PUI}, branch ${BR0}–${BR1}, explore ${EXI})`);
  let checkedNumbers = 0;
  await walk(async (i, again) => {
    const label = `beat ${i + 1} (${beats[i].id})${again ? ` — ${again}` : ""}`;
    checkedNumbers += await check(label);
    if (again) { await sameAsForward(label, i); return; }
    snaps[i] = await evaluate("__demo.snapshot()");
    await shot("1920x1080", `${pad(i)}-${beats[i].id}.png`);
  });
  console.log(`  ${beats.length} beats (${BR1 - BR0 + 1} in the branch), ${checkedNumbers} displayed numbers audited`);

  // past the last built beat: stays put
  await evaluate("__demo.next()"); await idle();
  if ((await evaluate("__demo.index")) !== beats.length - 1) fail("Next past the last built beat moved the index");

  // ---------------------------------------------------------------- backward: Explore (entered from the branch) → branch → main path
  console.log("backward pass");
  const backOrder = [...[...Array(BR1 - BR0 + 1).keys()].map((k) => BR1 - k), ...[...Array(PUI + 1).keys()].map((k) => PUI - k)];
  for (const i of backOrder) {
    await evaluate("__demo.prev()"); await idle();
    const at = await evaluate("__demo.index");
    if (at !== i) fail(`Previous did not land on beat ${i + 1} (${beats[i].id}), got ${at + 1}`);
    await check(`back to beat ${i + 1} (${beats[i].id})`);
    await sameAsForward(`back to beat ${i + 1} (${beats[i].id})`, i);
  }

  // ---------------------------------------------------------------- interrupts: Next during animation, Prev during animation
  console.log("interrupts");
  await evaluate("__demo.speed(1)");
  for (let i = 0; i < beats.length - 1; i++) {
    const j = await evaluate(`__demo.nextOf(${i})`);
    await evaluate("__demo.goto(" + i + ", 0)"); await idle();
    await evaluate("__demo.next()");
    await sleep(120 + (i % 4) * 150);
    await evaluate("__demo.next()"); // skip to the end of the running beat
    await idle();
    if ((await evaluate("__demo.index")) !== j) fail(`skip during beat ${j + 1} landed on ${(await evaluate("__demo.index")) + 1}`);
    await check(`skip during beat ${j + 1}`);
    await sameAsForward(`skip during beat ${j + 1}`, j);
    await evaluate("__demo.next()");
    await sleep(200);
    await evaluate("__demo.prev()"); // reverse while animating
    await idle();
    await check(`prev during the beat after ${j + 1}`);
    await sameAsForward(`prev during the beat after ${j + 1}`, await evaluate("__demo.index"));
  }
  // entering the branch, interrupted: skip it, or go back while it is still zooming in
  for (const [label, act, want] of [["skip", "__demo.next()", () => BR0], ["previous", "__demo.prev()", () => PUI]]) {
    await evaluate(`__demo.goto(${PUI}, 0)`); await idle();
    await evaluate("__demo.decoder()");
    await sleep(700);
    await evaluate(act); await idle();
    if ((await evaluate("__demo.index")) !== want()) fail(`${label} while entering the branch landed on ${(await evaluate("__demo.index")) + 1}`);
    await check(`${label} while entering the branch`);
    await sameAsForward(`${label} while entering the branch`, want());
  }
  // rapid-fire presses
  await evaluate("__demo.goto(0, 0)"); await idle();
  for (let k = 0; k < 25; k++) { await evaluate(k % 5 === 4 ? "__demo.prev()" : "__demo.next()"); await sleep(35); }
  await idle();
  await check("after 25 rapid presses");
  // restart from the middle of an animation
  await evaluate(`__demo.goto(${beats.length - 3}, 0)`); await idle();
  await evaluate("__demo.next()"); await sleep(300);
  await evaluate("__demo.restart()"); await idle();
  if ((await evaluate("__demo.index")) !== 0) fail("restart did not return to beat 1");
  await check("after restart");

  // ---------------------------------------------------------------- random jumps
  console.log("jumps");
  await evaluate("__demo.speed(6)");
  const B = (k) => BR0 + k; // into the branch and out again, in no particular order
  for (const i of [7, 2, 13, 0, 10, 5, 14, 3, 11, 1, 18, 26, 16, 22, 24, 20, 15, 25, 33, 41, 28, 43, 36, 30, 38, 27, 42, 31, 35, 40, 29, 34, 37, 39, 32, 46, 44, 48, 45, 42, 47, 43,
    B(4), 9, B(0), B(12), B(6), 44, B(2), B(9), EXI, B(11), B(1), PUI, B(7), B(3), 21, B(10), B(5), B(8), EXI, B(12), 0]) {
    if (i >= beats.length) continue;
    await evaluate(`__demo.goto(${i}, 400)`); await idle();
    await check(`jump to beat ${i + 1}`);
    await sameAsForward(`jump to beat ${i + 1}`, i);
  }

  // ---------------------------------------------------------------- Show calculation (3 decimals) on the detailed beats
  console.log("show calculation");
  const qi = beats.findIndex((b) => b.id === "query"), di = beats.findIndex((b) => b.id === "detail"), kvi = beats.findIndex((b) => b.id === "keyvalue");
  await evaluate("__demo.speed(1)");
  await evaluate(`__demo.goto(${qi - 1}, 0)`); await idle();
  await evaluate("void __demo.play(" + qi + ")");
  await sleep(4200);
  await check("mid detailed product (2 decimals)", { geometry: false });
  await shot("1920x1080", "mid-query-2dp.png");
  await idle();
  await evaluate("__demo.toggleCalc()"); // replays the current beat with 3 decimals
  await sleep(5200);
  await check("mid detailed product (3 decimals)", { geometry: false });
  await shot("1920x1080", "mid-query-3dp.png");
  await idle();
  await check("query, show calculation");
  await evaluate("__demo.speed(6)");
  await evaluate(`__demo.goto(${kvi - 1}, 0)`); await idle();
  await evaluate("void __demo.play(" + kvi + ")");
  await sleep(2000);
  await check("mid key/value detailed products (3 decimals)", { geometry: false });
  await shot("1920x1080", "mid-keyvalue-3dp.png");
  await idle();
  await check("key/value, show calculation");
  await shot("1920x1080", "keyvalue-3dp.png");
  await evaluate(`__demo.goto(${di - 1}, 0)`); await idle();
  await evaluate("void __demo.play(" + di + ")"); await idle();
  await check("PE detail, show calculation");
  await shot("1920x1080", "detail-3dp.png");
  await evaluate("__demo.toggleCalc()"); await idle();
  if (await evaluate("__demo.showCalc")) fail("show calculation did not toggle off");

  // ---------------------------------------------------------------- Scenes 5–7: mid-animation states (2 and 3 decimals)
  console.log("scenes 5–7 mid-animation");
  const capture = async (id, ms, name) => {
    const i = beats.findIndex((b) => b.id === id);
    await evaluate(`__demo.goto(${i - 1}, 0)`); await idle();
    await evaluate(`void __demo.play(${i})`);
    await sleep(ms);
    await check(`mid ${id}${name.includes("3dp") ? " (3 decimals)" : ""}`, { geometry: false });
    await shot("1920x1080", name);
    await idle();
    await check(`end ${id}`);
  };
  await evaluate("__demo.speed(1)");
  for (const [id, ms] of [["dot", 4200], ["allscores", 1300], ["scale", 900], ["softmax", 1300], ["product", 3300], ["products", 1200], ["flow", 2600]]) await capture(id, ms, `mid-${id}.png`);
  console.log("scenes 8–10 mid-animation");
  for (const [id, ms] of [["heads", 1400], ["head2", 2600], ["concat", 1500], ["wo", 2000], ["block", 1600], ["residual", 2600], ["ffn", 3800], ["stack", 1200],
    ["zoom", 1800], ["example", 1500], ["decoder", 2200], ["mask", 3600], ["cross", 3200], ["output", 2600], ["loop", 2600], ["final", 2600], ["explore", 900]]) await capture(id, ms, `mid-${id}.png`);
  console.log("decoder branch mid-animation");
  for (const [id, ms] of [["d-start", 1500], ["d-start", 3200], ["d-heads", 1800], ["d-t1", 2100], ["d-concat", 1300], ["d-cross", 2300], ["d-scores", 2600], ["d-values", 700],
    ["d-values", 3600], ["d-ffn", 2200], ["d-layers", 900], ["d-layers", 3000], ["d-token", 4600], ["d-token", 6300], ["d-mask", 3300], ["d-mask", 5600], ["d-next", 5200], ["d-next", 7400], ["d-summary", 1600]])
    await capture(id, ms, `mid-${id}-${ms}.png`);
  console.log("epilogue mid-animation");
  for (const [id, ms] of [["p-vector", 2300], ["p-vector", 3450], ["p-matrix", 950], ["p-qkv", 1050], ["p-qkv", 4700], ["p-heads", 2400], ["p-heads", 4900], ["p-uses", 1200], ["p-uses", 3450]])
    await capture(id, ms, `mid-${id}-${ms}.png`);
  await evaluate("__demo.toggleCalc()"); await sleep(100); await idle();
  for (const [id, ms] of [["dot", 5600], ["softmax", 2600], ["product", 4300], ["head2", 2600], ["wo", 2400], ["residual", 3600], ["ffn", 4500], ["mask", 3600], ["output", 2600]]) await capture(id, ms, `mid-${id}-3dp.png`);
  await shot("1920x1080", "product-3dp-end.png");
  const si = beats.findIndex((b) => b.id === "softmax");
  await evaluate(`__demo.goto(${si}, 0)`); await idle();
  await check("softmax end (3 decimals)");
  await shot("1920x1080", "softmax-3dp-end.png");
  await evaluate("__demo.toggleCalc()"); await sleep(100); await idle();
  await evaluate("__demo.speed(6)");

  // ---------------------------------------------------------------- Scenes 8–10: what is shown is what the brief requires
  console.log("scenes 8–10 content");
  const at = async (id) => { await evaluate(`__demo.goto(${beats.findIndex((b) => b.id === id)}, 0)`); await idle(); };
  const shownSrc = (re) => evaluate(`[...document.querySelectorAll("#stage [data-src]")].filter((e) => { for (let n = e; n && n.id !== "stage"; n = n.parentNode) if (n.style && (n.style.visibility === "hidden" || Number(n.style.opacity || 1) < 0.01)) return false; return ${re}.test(e.dataset.src); }).map((e) => [e.dataset.src, e.textContent])`);
  await at("head2");
  const h1 = await shownSrc("/^heads\\.0\\.A\\.4\\.\\d$/"), h2 = await shownSrc("/^heads\\.1\\.A\\.4\\.\\d$/");
  if (h1.length !== 8 || h2.length !== 8) fail(`Scene 8: expected the 8 head-1 and 8 head-2 weights of 'it', got ${h1.length} and ${h2.length}`);
  if ((await shownSrc("/^heads\\.[01]\\.Z\\.4\\.\\d$/")).length !== 4) fail("Scene 8: both head outputs (2 numbers each) should be shown");
  const wireSrcs = await evaluate(`[...document.querySelectorAll("#stage [data-kind=wire]")].filter((e) => e.closest("g").style.visibility !== "hidden").map((e) => e.dataset.alpha)`);
  if (wireSrcs.filter((a) => a.startsWith("heads.0.A.4.")).length !== 8 || wireSrcs.filter((a) => a.startsWith("heads.1.A.4.")).length !== 8) fail("Scene 8: each head's arcs must be drawn from that head's own α");
  await at("wo");
  if ((await shownSrc("/^Concat\\.4\\.\\d$/")).length !== 4) fail("Scene 8: Concat must show 4 numbers (h·d_v = 2·2)");
  if ((await shownSrc("/^params\\.WO\\.\\d\\.\\d$/")).length !== 16) fail("Scene 8: W_O must show its 4×4 entries");
  if ((await shownSrc("/^MH\\.4\\.\\d$/")).length !== 4) fail("Scene 8: the multi-head output must show 4 numbers (d_model)");
  await at("residual");
  for (const re of ["/^X\\.4\\.\\d$/", "/^MH\\.4\\.\\d$/", "/^R1\\.4\\.\\d$/", "/^LN1\\.4\\.out\\.\\d$/"]) if ((await shownSrc(re)).length < 4) fail(`Scene 9: residual step should show ${re}`);
  await at("ffn");
  if ((await shownSrc("/^Hid\\.4\\.\\d$/")).length !== 8) fail("Scene 9: the FFN's 8 hidden values should be shown");
  await at("mask");
  const cells = await shownSrc("/^dec\\.g4\\.A\\.\\d\\.\\d$/");
  const future = cells.filter(([src]) => { const [r, c] = src.split(".").slice(-2).map(Number); return c > r; });
  if (cells.length !== 16 || future.length !== 6 || future.some(([, t]) => t !== "0")) fail(`Scene 10: masked cells must read exactly 0 after softmax, got ${JSON.stringify(future)}`);
  await at("output");
  if (!(await evaluate(`[...document.querySelectorAll("#stage text")].some((t) => t.textContent.includes("illustrative — not from a trained translation model") && getComputedStyle(t).visibility !== "hidden" && Number(t.closest("g").style.opacity) > 0.5)`))) fail("Scene 10: the next-token probabilities must be labelled illustrative");
  if ((await shownSrc("/^dec\\.next\\.probs\\.\\d$/")).length !== 6) fail("Scene 10: all six illustrative probabilities should be shown");
  await at("final");
  if (!(await evaluate(`[...document.querySelectorAll("#stage text")].some((t) => t.textContent === "Everything we just calculated lives here." && Number(t.closest("g").style.opacity) > 0.9)`))) fail("Scene 10: the final caption is missing");

  // ---------------------------------------------------------------- epilogue (paper scale): compressed, honest, bound to the engine
  console.log("epilogue content");
  // whitespace-insensitive: labels space their operators generously
  const shownText = (needle) => evaluate(`[...document.querySelectorAll("#stage text")].some((t) => {
    if (!t.textContent.replace(/\\s+/g, " ").includes(${JSON.stringify(needle.replace(/\s+/g, " "))})) return false;
    let o = 1; for (let n = t; n && n.id !== "stage"; n = n.parentNode) { if (!n.style) continue; if (n.style.visibility === "hidden") return false; if (n.style.opacity !== "") o *= Number(n.style.opacity); }
    return o > 0.5; })`);
  for (const id of ["p-vector", "p-matrix", "p-qkv", "p-heads", "p-uses"]) {
    await at(id);
    if (!(await shownText("illustrative values, not trained weights"))) fail(`${id}: the paper-scale values must be labelled illustrative`);
    if (await shownText("Tiny teaching model")) fail(`${id}: the tiny-model tag should give way to the paper-scale tag`);
    if ((await evaluate("document.querySelector('#l-hud').textContent")).includes("/ 11")) fail(`${id}: the epilogue must not renumber the ten guided scenes`);
  }
  await at("p-vector");
  if ((await shownSrc("/^X\\.4\\.\\d$/")).length !== 4) fail("epilogue: the toy x_it (its 4 numbers) should be shown");
  const pv = await shownSrc("/^paper\\.X\\.4\\.\\d+$/");
  if (pv.map(([s]) => s.split(".").pop()).join() !== "0,1,2,511") fail(`epilogue: paper-scale x_it should show entries 1, 2, 3 and 512, got ${JSON.stringify(pv)}`);
  for (const t of ["Same computation — larger vectors.", "illustrative values — not trained weights", "PAPER", "DEMO"]) if (!(await shownText(t))) fail(`epilogue: "${t}" should be on screen`);
  await at("p-matrix");
  const px = await shownSrc("/^paper\\.X\\.\\d\\.\\d+$/");
  if (px.length !== 32) fail(`epilogue: X ∈ R^{8×512} should show 4 entries of each of its 8 rows, got ${px.length}`);
  if (!(await shownText("8×512"))) fail("epilogue: X must be labelled 8×512");
  await at("p-qkv");
  if ((await shownSrc("/^paper\\.heads\\.0\\.Q\\.4\\.\\d+$/")).map(([s]) => s.split(".").pop()).join() !== "0,1,63") fail("epilogue: q_it should show entries 1, 2 and 64");
  for (const t of ["512×64", "8×64"]) if (!(await shownText(t))) fail(`epilogue: projection shapes should show ${t}`);
  await at("p-heads");
  for (const t of ["HEAD 8", "Concat(head", "512×512", "√64"]) if (!(await shownText(t))) fail(`epilogue: eight heads → Concat → W^O should show "${t}"`);
  await at("p-uses");
  const g = await evaluate("__demo.snapshot()");
  for (const id of ["enc:mhsa", "dec:mm", "dec:cross"]) if (!(g[id] && g[id].glow > 0.9)) fail(`epilogue: ${id} should be lit as one of the three uses of attention`);
  for (const t of ["The math did not change.", "per head", "ENCODER–DECODER ATTENTION"]) if (!(await shownText(t))) fail(`epilogue: "${t}" should be on screen`);
  // Next at the end of the epilogue enters Explore
  await evaluate("__demo.next()"); await idle();
  if ((await evaluate("__demo.beats[__demo.index].id")) !== "explore") fail("Next at the end of the epilogue did not enter Explore");
  await check("explore entered from the epilogue");

  // ---------------------------------------------------------------- Explore: every interaction recomputes, the audit stays clean
  console.log("explore");
  await evaluate("__demo.speed(4)");
  await evaluate("__demo.explore.enter()"); await idle();
  for (const [label, expr] of [["query animal", "__demo.explore.select(1)"], ["query was", "__demo.explore.select(5)"], ["head 2", "__demo.explore.head(1)"],
    ["position off", "__demo.explore.pos(false)"], ["no scaling", "__demo.explore.scale('none')"], ["divisor 0.25", "__demo.explore.scale('custom', 0.25)"],
    ["divisor 8", "__demo.explore.scale('custom', 8)"], ["paper scaling", "__demo.explore.scale('paper')"], ["position on", "__demo.explore.pos(true)"], ["head 1", "__demo.explore.head(0)"]]) {
    await evaluate(expr); await idle();
    await check(`explore: ${label}`);
    await shot("1920x1080", `explore-${label.replace(/\s+/g, "-")}.png`);
  }
  if (!(await evaluate("document.getElementById('ex-whatif').hidden"))) fail("Explore: the what-if label should be hidden with paper scaling");
  await evaluate("__demo.explore.scale('custom', 3)"); await idle();
  if (await evaluate("document.getElementById('ex-whatif').hidden")) fail("Explore: a custom divisor must be labelled 'What-if experiment'");
  await evaluate("__demo.explore.scale('paper')"); await idle();
  // a real click on a word, and the arrow keys
  const row = await evaluate("(() => { const r = document.querySelectorAll('.ex-hit')[6].getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()");
  await S("Input.dispatchMouseEvent", { type: "mousePressed", x: row.x, y: row.y, button: "left", clickCount: 1 });
  await S("Input.dispatchMouseEvent", { type: "mouseReleased", x: row.x, y: row.y, button: "left", clickCount: 1 });
  await idle();
  if ((await evaluate("__demo.explore.state.q")) !== 6) fail("Explore: clicking a word did not make it the query");
  await check("explore: after a click");
  // leaving Explore restores the guided scene exactly
  await evaluate("__demo.prev()"); await idle();
  if ((await evaluate("__demo.beats[__demo.index].id")) !== "p-uses") fail("Previous from Explore should land on the epilogue's last beat");
  await check("back from explore to the epilogue");
  await at("softmax");
  await check("guided softmax after exploring");

  // ---------------------------------------------------------------- optional branch: inside the decoder — paper sizes, honest labels
  console.log("decoder branch content");
  const ex = await evaluate("__demo.decoderExample"), sLen = ex.source.length;
  const srcs = async (re) => (await shownSrc(re)).map(([x]) => x);
  const lastIdx = (list) => list.map((x) => x.split(".").pop()).join();
  const need = async (where, texts) => { for (const t of texts) if (!(await shownText(t))) fail(`${where}: "${t}" should be on screen`); };
  const num = (t) => Number(String(t).replace("−", "-"));
  for (let i = BR0; i <= BR1; i++) {
    await at(beats[i].id);
    await need(beats[i].id, ["illustrative values, not trained weights", "Inside the decoder"]);
    if ((await evaluate("document.querySelector('#l-hud').textContent")).includes("/ 1")) fail(`${beats[i].id}: the optional branch must not be numbered as a guided scene`);
    if (!(await evaluate("document.getElementById('dd-btn').hidden"))) fail(`${beats[i].id}: the branch button belongs to the end of the epilogue only`);
  }
  if (ex.source.join(" ") !== "The animal is tired ." || ex.prefix.join(" ") !== "<START> L'") fail(`the branch must reuse Scene 10's example, got ${JSON.stringify(ex)}`);
  await at("d-start");
  await need("d-start", ["At the first generation step, the decoder only has <START>.", "1 × 512", "√512 = 22.63", "embedding(<START>)", "PE(pos = 0)"]);
  if (lastIdx(await srcs("/^dd\\.X0\\.0\\.\\d+$/")) !== "0,1,2,511") fail("d-start: x_START should show entries 1, 2, 3 and 512");
  if ((await srcs("/^dd\\.(raw|emb|P)\\.0\\.\\d+$/")).length !== 12) fail("d-start: the embedding, × √512 and PE rows should each show 4 entries");
  let snap = await evaluate("__demo.snapshot()");
  if (!snap["dtok:0"] || ["dtok:1", "dtok:2", "dtok:3", "dtok:4"].some((id) => snap[id])) fail("d-start: the decoder input must be <START> alone");
  await at("d-heads");
  snap = await evaluate("__demo.snapshot()");
  const copies = [...Array(8).keys()].map((h) => snap[`dd:mx${h}`]);
  if (copies.some((c) => !c)) fail("d-heads: each of the 8 heads must receive a copy of x_START");
  else if (copies.some((c) => c.w < 150 || c.w !== copies[1].w * (c === copies[0] ? 2 : 1))) fail(`d-heads: every head gets the whole 512-d vector, never a 64-d slice (widths ${copies.map((c) => c.w)})`);
  for (const b of ["Q", "K", "V"]) if (lastIdx(await srcs(`/^dd\\.layers\\.0\\.self\\.heads\\.0\\.${b}\\.0\\.\\d+$/`)) !== "0,1,63") fail(`d-heads: head 1's ${b} should show entries 1, 2 and 64`);
  await need("d-heads", ["(1×512)(512×64) → 1×64", "HEAD 8", "512 → 64", "the whole xSTART", "own W"]);
  await at("d-t1");
  const t1 = Object.fromEntries(await shownSrc("/^dd\\.layers\\.0\\.self\\.heads\\.\\d\\.(raw|unmasked|A)\\.0\\.0$/"));
  for (let h = 0; h < 8; h++) if (t1[`dd.layers.0.self.heads.${h}.A.0.0`] !== "1.00") fail(`d-t1: head ${h + 1}: softmax over the one available position must read 1.00`);
  if (!t1["dd.layers.0.self.heads.0.raw.0.0"] || !t1["dd.layers.0.self.heads.0.unmasked.0.0"]) fail("d-t1: the 1 × 1 score and score ÷ 8 should be shown");
  else if (Math.abs(num(t1["dd.layers.0.self.heads.0.raw.0.0"]) / 8 - num(t1["dd.layers.0.self.heads.0.unmasked.0.0"])) > 0.006) fail("d-t1: the shown score ÷ 8 must match the shown score");
  await need("d-t1", ["(1×64)(64×1) → 1×1", "÷ √64 = 8", "softmax", "nothing to block", "only one available decoder position"]);
  const z1 = await shownSrc("/^dd\\.layers\\.0\\.self\\.heads\\.0\\.Z\\.0\\.\\d+$/"), v1 = await shownSrc("/^dd\\.layers\\.0\\.self\\.heads\\.0\\.V\\.0\\.\\d+$/");
  if (!z1.length || z1.map(([, t]) => t).join() !== v1.map(([, t]) => t).join()) fail("d-t1: head 1's output must equal 1 · v₁");
  await at("d-concat");
  await need("d-concat", ["8 × 64 = 512", "512×512", "(1×512)(512×512) → 1×512", "LayerNorm(x + MultiHead)", "It has not generated a word yet."]);
  if (lastIdx(await srcs("/^dd\\.layers\\.0\\.N1\\.0\\.\\d+$/")) !== "0,1,2,511") fail("d-concat: y should show entries 1, 2, 3 and 512");
  await at("d-cross");
  const rowsOf = async (re) => new Set((await srcs(re)).map((x) => x.split(".").slice(-2, -1)[0])).size;
  if ((await rowsOf("/^dd\\.M\\.\\d\\.\\d+$/")) !== sLen) fail(`d-cross: the encoder output should show s = ${sLen} rows`);
  for (const b of ["K", "V"]) if ((await rowsOf(`/^dd\\.layers\\.0\\.cross\\.heads\\.0\\.${b}\\.\\d\\.\\d+$/`)) !== sLen) fail(`d-cross: ${b} should have one row per source token (s × 64)`);
  if ((await srcs("/^dd\\.layers\\.0\\.cross\\.heads\\.0\\.Q\\.0\\.\\d+$/")).length !== 3) fail("d-cross: q (1 × 64) should be shown");
  await need("d-cross", ["Q from decoder · K,V from encoder", "ENCODER OUTPUT", "DECODER STATE", `s = ${sLen}`, "(s×512)(512×64) → s×64", "(1×512)(512×64) → 1×64", "K = E WK", "V = E WV", "q = y WQ", ...ex.source.filter((w) => w !== ".")]);
  await at("d-scores");
  const al = await shownSrc("/^dd\\.layers\\.0\\.cross\\.heads\\.0\\.A\\.0\\.\\d$/");
  if (al.length !== sLen) fail(`d-scores: one weight per source token (1 × s), got ${al.length}`);
  if (Math.abs(al.reduce((a, [, t]) => a + num(t), 0) - 1) > 0.03) fail("d-scores: the shown weights should sum to 1");
  await need("d-scores", ["illustrative cross-attention weights", "(1×64)(64×s) → 1×s", "Σ αj = 1.00", "softmax(qKT / √64)"]);
  if ((await evaluate(`[...document.querySelectorAll("#stage [data-kind=cbar]")].filter((e) => e.closest("g").style.visibility !== "hidden").length`)) !== sLen) fail("d-scores: one bar per source token, drawn from its α");
  await at("d-values");
  const ws = await evaluate(`[...document.querySelectorAll("#stage [data-kind=wire]")].filter((e) => e.closest("g").style.visibility !== "hidden").map((e) => e.dataset.alpha)`);
  if (ws.filter((a) => /^dd\.layers\.0\.cross\.heads\.0\.A\.0\.\d$/.test(a)).length !== sLen) fail("d-values: one weighted-value stream per source token, its width drawn from its α");
  if (lastIdx(await srcs("/^dd\\.layers\\.0\\.cross\\.heads\\.0\\.Z\\.0\\.\\d+$/")) !== "0,1,63") fail("d-values: head 1's cross-attention output (1 × 64) should be shown");
  if ((await srcs("/^dd\\.layers\\.0\\.N2\\.0\\.\\d+$/")).length !== 3) fail("d-values: z (after Add & Norm) should be shown");
  await need("d-values", ["head1 = Σj αjvj", "8 heads → Concat ∈ ℝ1×512 → WO", "LayerNorm", "Cross-attention lets the current decoder representation gather information from the encoded source sentence."]);
  await at("d-ffn");
  await need("d-ffn", ["(512 × 2048)", "(2048 × 512)", "ReLU", "ℝ1×2048", "This completes one decoder layer.", "the output of decoder layer 1"]);
  const zeros = await shownSrc("/^derived\\.ddZeros$/");
  if (zeros.length !== 1 || !(num(zeros[0][1]) > 0 && num(zeros[0][1]) < 2048)) fail(`d-ffn: the count of ReLU zeros should be shown, got ${JSON.stringify(zeros)}`);
  await at("d-layers");
  await need("d-layers", [...[1, 2, 3, 4, 5, 6].map((k) => `Decoder layer ${k}`), "Paper: N = 6", "Each decoder layer has its own learned parameters.", "K, V of every layer"]);
  snap = await evaluate("__demo.snapshot()");
  if ([1, 2, 3, 4, 5, 6].some((k) => !(snap[`dd:tap${k}`]?.draw > 0.99))) fail("d-layers: the encoder output should feed the cross-attention of every layer");
  await at("d-token");
  const hl = await srcs("/^dd\\.out\\.\\d\\.\\d+$/");
  if (!hl.length || hl.some((x) => !x.startsWith("dd.out.0."))) fail("d-token: at t = 1 the projection reads row 0, the only (newest) position");
  if ((await srcs("/^dd\\.next\\.0\\.probs\\.\\d$/")).length !== 5) fail("d-token: four entries and the 'thousands more' bucket should be shown");
  await need("d-token", ["These 512 numbers are not themselves a word.", "|Vocab|", "not an attention head", "illustrative next-token probabilities", "thousands more", "predict → append → feed back → predict again", "OUTPUT SO FAR", "DECODER INPUT"]);
  if (await evaluate(`[...document.querySelectorAll("#stage text")].some((t) => /\\b3[27][ ,.]?000\\b/.test(t.textContent))`)) fail("no vocabulary size may be shown as if it were the paper's");
  await at("d-mask");
  const gm = Object.fromEntries(await shownSrc("/^dd\\.layers\\.0\\.self\\.heads\\.0\\.A\\.\\d\\.\\d$/"));
  if (gm["dd.layers.0.self.heads.0.A.0.1"] !== "0") fail(`d-mask: the masked future cell must read exactly 0, got ${gm["dd.layers.0.self.heads.0.A.0.1"]}`);
  if (gm["dd.layers.0.self.heads.0.A.0.0"] !== "1.00") fail("d-mask: the first position keeps all of its weight");
  if (Math.abs(num(gm["dd.layers.0.self.heads.0.A.1.0"]) + num(gm["dd.layers.0.self.heads.0.A.1.1"]) - 1) > 0.011) fail("d-mask: the second row sums to 1");
  if (!(await evaluate(`[...document.querySelectorAll("#stage [data-src='dd.layers.0.self.heads.0.A.0.1']")].some((e) => e.classList.contains("masked"))`))) fail("d-mask: the masked cell should be styled as masked");
  if ((await shownSrc("/^dd\\.layers\\.0\\.cross\\.heads\\.0\\.A\\.[01]\\.\\d$/")).length !== 2 * sLen) fail("d-mask: the cross-attention weights should be 2 × s");
  await need("d-mask", ["Now the causal mask matters.", "ℝ2×512", "ℝ2×64", "(2×64)(64×2) → 2×2", "1 × s → 2 × s", "(2×64)(64×s) → 2×s"]);
  await at("d-next");
  if (!(await srcs("/^dd\\.out\\.1\\.\\d+$/")).length || !(await srcs("/^dd\\.out\\.0\\.\\d+$/")).length) fail("d-next: both rows of the 2 × 512 output should be shown");
  if ((await srcs("/^dd\\.next\\.1\\.probs\\.\\d$/")).length !== 5) fail("d-next: the step-2 distribution should be shown");
  snap = await evaluate("__demo.snapshot()");
  const rowWords = (ids) => ids.map((id) => snap[id]).filter(Boolean).sort((a, b) => a.x - b.x).map((t) => t.text).join(" ");
  if (rowWords(["outl:0", "outl:1", "outl:2", "dd:o3", "dd:o4", "dd:o5"]) !== ex.generated.join(" ")) fail(`d-next: the output should read "${ex.generated.join(" ")}"`);
  if (rowWords(["dtok:0", "dtok:1", "dtok:2", "dtok:3", "dtok:4", "dd:i5"]) !== ["<START>", ...ex.generated.slice(0, -1)].join(" ")) fail("d-next: the decoder input is the output shifted right by one");
  await need("d-next", ["Generation uses the representation at the newest decoder position.", "after 6 layers", "generation stops"]);
  await at("d-summary");
  await need("d-summary", ["t × 512", "scores t × t", "scores t × s", "s × 64 (encoder)", "t × 2048", "× 6, each with its own weights", "1 × |Vocab|", `s = ${sLen}`,
    "Masked self-attention mixes information across the output generated so far.", "Cross-attention brings information in from the encoded input."]);

  // ---------------------------------------------------------------- fullscreen (F) keeps the stage intact
  console.log("fullscreen");
  await at("final");
  const fsKey = async () => {
    await S("Input.dispatchKeyEvent", { type: "rawKeyDown", key: "f", code: "KeyF", windowsVirtualKeyCode: 70 });
    await S("Input.dispatchKeyEvent", { type: "char", text: "f", key: "f", code: "KeyF", windowsVirtualKeyCode: 70 });
    await S("Input.dispatchKeyEvent", { type: "keyUp", key: "f", code: "KeyF", windowsVirtualKeyCode: 70 });
    await sleep(600);
  };
  await fsKey();
  const fsOn = await evaluate("!!document.fullscreenElement");
  const fsMsg = await evaluate("document.getElementById('toast').textContent");
  if (!fsOn && !fsMsg.includes("Fullscreen")) fail("F neither entered fullscreen nor explained why not");
  await check(`fullscreen ${fsOn ? "on" : "unavailable"}`);
  for (const id of ["p-vector", "p-heads", "p-uses"]) { await at(id); await check(`fullscreen ${fsOn ? "on" : "unavailable"}: ${id}`); }
  await at("final");
  if (fsOn) { await fsKey(); if (await evaluate("!!document.fullscreenElement")) fail("F did not leave fullscreen"); }

  // ---------------------------------------------------------------- real keyboard input (presenter clickers send PageUp/PageDown)
  console.log("keyboard & panels");
  const key = async (k, code = k, keyCode = 0) => {
    const base = { key: k, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode };
    await S("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base });
    if (k.length === 1) await S("Input.dispatchKeyEvent", { type: "char", text: k, ...base });
    await S("Input.dispatchKeyEvent", { type: "keyUp", ...base });
  };
  await evaluate("__demo.speed(8)");
  await evaluate("__demo.goto(0, 0)"); await idle();
  const expectIndex = async (want, label) => { await idle(); const got = await evaluate("__demo.index"); if (got !== want) fail(`${label}: expected beat ${want + 1}, got ${got + 1}`); };
  await key("ArrowRight", "ArrowRight", 39); await expectIndex(1, "ArrowRight");
  await key("PageDown", "PageDown", 34); await expectIndex(2, "PageDown");
  await key("ArrowLeft", "ArrowLeft", 37); await expectIndex(1, "ArrowLeft");
  await key("PageUp", "PageUp", 33); await expectIndex(0, "PageUp");
  // Space on a focused button toggles play/pause only — it must not also click the button
  await evaluate("document.getElementById('btn-next').focus()");
  await key(" ", "Space", 32);
  await sleep(150);
  if ((await evaluate("__demo.index")) > 1) fail("Space on a focused Next button also activated it");
  await key(" ", "Space", 32); // pause
  if (!(await evaluate("document.getElementById('btn-play').getAttribute('aria-label')")).startsWith("Play")) fail("Space did not pause");
  await key(" ", "Space", 32); // resume
  await evaluate("document.activeElement.blur()");
  await evaluate("__demo.speed(8)");
  await sleep(300); await idle();
  // panels
  await key("d", "KeyD", 68);
  if (await evaluate("document.getElementById('debug').hidden")) fail("D did not open the verification panel");
  if (!(await evaluate("document.getElementById('debug').textContent.includes('0 mismatches')"))) fail("verification panel does not report a clean audit");
  await shot("1920x1080", "panel-debug.png");
  await key("d", "KeyD", 68);
  await key("n", "KeyN", 78);
  if (await evaluate("document.getElementById('notes').hidden")) fail("N did not open presenter notes");
  await shot("1920x1080", "panel-notes.png");
  await key("n", "KeyN", 78);
  await evaluate("document.getElementById('btn-about').click()");
  if (await evaluate("document.getElementById('about').hidden")) fail("About did not open");
  if (!(await evaluate("document.getElementById('about').textContent.includes('Built with LLM coding assistance.')"))) fail("About lacks the disclosure line");
  await shot("1920x1080", "panel-about.png");
  await key("Escape", "Escape", 27);
  if (!(await evaluate("document.getElementById('about').hidden"))) fail("Escape did not close About");
  // Reset: R from Scene 10, the Restart button from Explore
  await evaluate(`__demo.goto(${beats.findIndex((b) => b.id === "cross")}, 0)`); await idle();
  await key("r", "KeyR", 82); await sleep(250);
  if ((await evaluate("__demo.index")) !== 0) fail("R did not restart from Scene 10");
  await idle();
  await check("after R from Scene 10");
  await evaluate("__demo.explore.enter()"); await idle();
  await evaluate("document.getElementById('btn-restart').click()"); await sleep(250);
  if ((await evaluate("__demo.index")) !== 0) fail("Restart button did not restart from Explore");
  if (!(await evaluate("document.getElementById('explore-ui').hidden"))) fail("Explore controls still visible after restart");
  await idle();
  await check("after restart from Explore");
  // E enters Explore from anywhere
  await key("e", "KeyE", 69); await idle();
  if ((await evaluate("__demo.beats[__demo.index].id")) !== "explore") fail("E did not enter Explore");
  const q0 = await evaluate("__demo.explore.state.q");
  await key("ArrowDown", "ArrowDown", 40); await idle();
  if ((await evaluate("__demo.explore.state.q")) !== (q0 + 1) % 8) fail("ArrowDown in Explore did not move the query to the next word");
  await check("explore after keyboard");
  // Show calculation by keyboard, then PE formula in Scene 3
  await key("c", "KeyC", 67); await sleep(200); await idle();
  if (!(await evaluate("__demo.showCalc"))) fail("C did not enable show calculation");
  await key("c", "KeyC", 67); await sleep(200); await idle();
  const wi = beats.findIndex((b) => b.id === "waves");
  await evaluate(`__demo.goto(${wi}, 0)`); await idle();
  await key("m", "KeyM", 77);
  if (await evaluate("document.getElementById('pe-math').hidden")) fail("M did not open the PE formula in Scene 3");
  await shot("1920x1080", "panel-pe-formula.png");
  await key("m", "KeyM", 77);

  // ---------------------------------------------------------------- the branch is optional: skipping it, entering it, leaving it
  console.log("decoder branch navigation");
  const idx = () => evaluate("__demo.index");
  const clickEl = async (expr) => {
    const r = await evaluate(`(() => { const e = ${expr}; if (!e) return null; const b = e.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; })()`);
    if (!r) return false;
    await S("Input.dispatchMouseEvent", { type: "mousePressed", x: r.x, y: r.y, button: "left", clickCount: 1 });
    await S("Input.dispatchMouseEvent", { type: "mouseReleased", x: r.x, y: r.y, button: "left", clickCount: 1 });
    return true;
  };
  await evaluate("__demo.speed(8)");
  await at("p-uses");
  if (await evaluate("document.getElementById('dd-btn').hidden")) fail("the 'Inside the decoder' button should be offered at the end of the epilogue");
  await evaluate("__demo.next()"); await idle();
  if ((await idx()) !== EXI) fail("Next at the end of the epilogue must skip the optional branch and enter Explore");
  await evaluate("__demo.prev()"); await idle();
  if ((await idx()) !== PUI) fail("Previous from Explore (entered from the epilogue) should return to the epilogue");
  await sameAsForward("back from Explore to the epilogue", PUI);
  if (!(await clickEl("document.getElementById('dd-btn')"))) fail("the branch button could not be clicked");
  await idle();
  if ((await idx()) !== BR0) fail("the button should open the branch");
  await check("branch opened by its button"); await sameAsForward("branch opened by its button", BR0);
  await key("i", "KeyI", 73); await idle();
  if ((await idx()) !== BR0) fail("I inside the branch should not restart it");
  await evaluate("__demo.prev()"); await idle();
  if ((await idx()) !== PUI) fail("Previous from the first beat of the branch should return to the epilogue");
  await sameAsForward("back out of the branch", PUI);
  await key("i", "KeyI", 73); await idle();
  if ((await idx()) !== BR0) fail("I at the end of the epilogue should enter the branch");
  await at("waves");
  await key("i", "KeyI", 73); await idle();
  if ((await idx()) !== BR0) fail("I should open the branch from anywhere");
  await sameAsForward("branch opened from Scene 3", BR0);
  await evaluate(`__demo.goto(${BR0 + 4}, 0)`); await idle();
  await key("e", "KeyE", 69); await idle();
  if ((await idx()) !== EXI) fail("E inside the branch should enter Explore");
  await check("explore entered from inside the branch");
  const tlb = await evaluate(`(() => { const b = [...document.querySelectorAll("#timeline button")].find((x) => x.textContent === "Decoder"); return b ? { cls: b.className, title: b.title } : null; })()`);
  if (!tlb || !tlb.cls.includes("branch") || tlb.cls.includes("past") || !tlb.title.startsWith("Optional")) fail(`timeline: the branch entry should be marked optional and never as passed, got ${JSON.stringify(tlb)}`);
  if (!(await clickEl(`[...document.querySelectorAll("#timeline button")].find((x) => x.textContent === "Decoder")`))) fail("the timeline's branch entry could not be clicked");
  await idle();
  if ((await idx()) !== BR0) fail("the timeline's branch entry should start the branch");
  await evaluate(`__demo.goto(${BR1}, 0)`); await idle();
  await evaluate("__demo.next()"); await idle();
  if ((await idx()) !== EXI) fail("Next at the end of the branch should enter Explore");
  await check("explore after the branch"); // (Explore keeps the query chosen earlier, so no forward-pass comparison here)
  await evaluate("__demo.prev()"); await idle();
  if ((await idx()) !== BR1) fail("Previous from Explore (entered from the branch) should return to the end of the branch");
  // autoplay inside the branch runs to its end and stops before Explore
  await evaluate(`__demo.goto(${BR1 - 2}, 0)`); await idle();
  await key(" ", "Space", 32); await sleep(300); await idle(); await sleep(200);
  for (let k = 0; k < 40 && (await idx()) !== BR1; k++) { await sleep(250); await idle(); }
  await sleep(1200); await idle();
  if ((await idx()) !== BR1) fail(`autoplay in the branch should stop at its last beat, got ${(await idx()) + 1}`);
  await check("autoplay through the end of the branch");
  // R restarts the whole presentation from inside the branch
  await evaluate(`__demo.goto(${BR0 + 6}, 0)`); await idle();
  await key("r", "KeyR", 82); await sleep(250);
  if ((await idx()) !== 0) fail("R did not restart from inside the branch");
  await idle();
  await check("after R from the branch");

  // ---------------------------------------------------------------- reduced motion
  console.log("reduced motion");
  await load(1920, 1080, { reducedMotion: true, speed: 1 });
  await walk(async (i, again) => {
    const label = `reduced motion beat ${i + 1} (${beats[i].id})${again ? ` — ${again}` : ""}`;
    await check(label);
    await sameAsForward(label, i);
  });

  // ---------------------------------------------------------------- 1366×768
  console.log("1366×768");
  await load(1366, 768);
  await walk(async (i, again) => {
    const label = `1366 beat ${i + 1} (${beats[i].id})${again ? ` — ${again}` : ""}`;
    await check(label);
    await sameAsForward(label, i);
    if (!again) await shot("1366x768", `${pad(i)}-${beats[i].id}.png`);
  });
} catch (e) {
  fail(`harness: ${e.message}`);
} finally {
  if (consoleErrors.length) consoleErrors.forEach((m) => fail(m));
  try { browser.ws.close(); } catch {}
  browser.proc.kill("SIGKILL");
  await sleep(200);
  try { rmSync(browser.dir, { recursive: true, force: true }); } catch {}
}
if (failures.length) { console.log(`\nFAILED: ${failures.length} problem(s)`); process.exit(1); }
console.log("\nall browser checks passed");
