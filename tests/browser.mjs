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

// Visible text overlapping other visible text (layout regressions), measured in screen space.
const OVERLAPS = `(() => {
  const vis = (el) => { let o = 1; for (let n = el; n && n.id !== "stage"; n = n.parentNode) { if (!n.style) continue; if (n.style.visibility === "hidden") return 0; if (n.style.opacity !== "") o *= Number(n.style.opacity); } return o; };
  const items = [...document.querySelectorAll("#stage text")].map((t) => ({ t, o: vis(t), r: t.getBoundingClientRect() })).filter((x) => x.o > 0.45 && x.r.width > 0 && x.t.textContent.trim());
  const out = [];
  for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
    const a = items[i].r, b = items[j].r;
    const w = Math.min(a.right, b.right) - Math.max(a.left, b.left), h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    if (w > 2 && h > 2 && w * h > 0.15 * Math.min(a.width * a.height, b.width * b.height)) out.push(items[i].t.textContent.trim().slice(0, 24) + "  ×  " + items[j].t.textContent.trim().slice(0, 24));
  }
  const box = document.getElementById("stage").getBoundingClientRect();
  const outside = items.filter((x) => x.r.left < box.left - 1 || x.r.right > box.right + 1 || x.r.top < box.top - 1 || x.r.bottom > box.bottom + 1).map((x) => x.t.textContent.trim().slice(0, 24));
  const approx = items.filter((x) => x.t.textContent.includes("\\u2248")).map((x) => x.t.textContent.trim());
  return { overlaps: out, outside, approx };
})()`;

async function check(label, { allowApprox = false } = {}) {
  const a = await evaluate("__demo.audit()");
  if (a.mismatches.length) fail(`${label}: ${a.mismatches.length} displayed numbers differ from the engine, e.g. ${JSON.stringify(a.mismatches[0])}`);
  if (a.badText.length) fail(`${label}: invalid text ${JSON.stringify(a.badText.slice(0, 3))}`);
  const v = await evaluate(OVERLAPS);
  if (v.overlaps.length) fail(`${label}: overlapping text ${JSON.stringify(v.overlaps.slice(0, 4))}`);
  if (v.outside.length) fail(`${label}: text outside the stage ${JSON.stringify(v.outside.slice(0, 4))}`);
  if (!allowApprox && v.approx.length) fail(`${label}: "≈" shown on the guided path ${JSON.stringify(v.approx)}`);
  return a.checked;
}

try {
  // ---------------------------------------------------------------- 1920×1080: forward through every beat
  console.log("1920×1080 — forward pass");
  await load(1920, 1080);
  const beats = await evaluate("__demo.beats");
  let checkedNumbers = 0;
  for (let i = 0; i < beats.length; i++) {
    if (i > 0) { await evaluate("__demo.next()"); await idle(); }
    const label = `beat ${i + 1} (${beats[i].id})`;
    checkedNumbers += await check(label);
    await shot("1920x1080", `${String(i + 1).padStart(2, "0")}-${beats[i].id}.png`);
  }
  console.log(`  ${beats.length} beats, ${checkedNumbers} displayed numbers audited`);

  // past the last built beat: stays put
  await evaluate("__demo.next()"); await idle();
  if ((await evaluate("__demo.index")) !== beats.length - 1) fail("Next past the last built beat moved the index");

  // ---------------------------------------------------------------- backward
  console.log("backward pass");
  for (let i = beats.length - 2; i >= 0; i--) {
    await evaluate("__demo.prev()"); await idle();
    if ((await evaluate("__demo.index")) !== i) fail(`Previous from beat ${i + 2} did not land on beat ${i + 1}`);
    await check(`back to beat ${i + 1} (${beats[i].id})`);
  }

  // ---------------------------------------------------------------- interrupts: Next during animation, Prev during animation
  console.log("interrupts");
  await evaluate("__demo.speed(1)");
  for (let i = 0; i < beats.length - 1; i++) {
    await evaluate("__demo.goto(" + i + ", 0)"); await idle();
    await evaluate("__demo.next()");
    await sleep(120 + (i % 4) * 150);
    await evaluate("__demo.next()"); // skip to the end of the running beat
    await idle();
    if ((await evaluate("__demo.index")) !== i + 1) fail(`skip during beat ${i + 2} landed on ${await evaluate("__demo.index")}`);
    await check(`skip during beat ${i + 2}`);
    await evaluate("__demo.next()");
    await sleep(200);
    await evaluate("__demo.prev()"); // reverse while animating
    await idle();
    await check(`prev during beat ${i + 3 <= beats.length ? i + 3 : i + 2}`);
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
  for (const i of [7, 2, 13, 0, 10, 5, 14, 3, 11, 1]) {
    if (i >= beats.length) continue;
    await evaluate(`__demo.goto(${i}, 400)`); await idle();
    await check(`jump to beat ${i + 1}`);
  }

  // ---------------------------------------------------------------- Show calculation (3 decimals) on the detailed beats
  console.log("show calculation");
  const qi = beats.findIndex((b) => b.id === "query"), di = beats.findIndex((b) => b.id === "detail"), kvi = beats.findIndex((b) => b.id === "keyvalue");
  await evaluate("__demo.speed(1)");
  await evaluate(`__demo.goto(${qi - 1}, 0)`); await idle();
  await evaluate("void __demo.play(" + qi + ")");
  await sleep(4200);
  await check("mid detailed product (2 decimals)");
  await shot("1920x1080", "mid-query-2dp.png");
  await idle();
  await evaluate("__demo.toggleCalc()"); // replays the current beat with 3 decimals
  await sleep(5200);
  await check("mid detailed product (3 decimals)");
  await shot("1920x1080", "mid-query-3dp.png");
  await idle();
  await check("query, show calculation");
  await evaluate("__demo.speed(6)");
  await evaluate(`__demo.goto(${kvi - 1}, 0)`); await idle();
  await evaluate("void __demo.play(" + kvi + ")");
  await sleep(2000);
  await check("mid key/value detailed products (3 decimals)");
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

  // ---------------------------------------------------------------- reduced motion
  console.log("reduced motion");
  await load(1920, 1080, { reducedMotion: true, speed: 1 });
  for (let i = 1; i < beats.length; i++) {
    await evaluate("__demo.next()");
    await idle();
    await check(`reduced motion beat ${i + 1}`);
  }

  // ---------------------------------------------------------------- 1366×768
  console.log("1366×768");
  await load(1366, 768);
  for (let i = 0; i < beats.length; i++) {
    if (i > 0) { await evaluate("__demo.next()"); await idle(); }
    await check(`1366 beat ${i + 1}`);
    await shot("1366x768", `${String(i + 1).padStart(2, "0")}-${beats[i].id}.png`);
  }
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
