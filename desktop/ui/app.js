// Vergence's window. Everything goes through the bounded commands in src/commands.rs: keys go
// in and never come back, files are picked and dropped through Rust, and nothing is loaded
// from anywhere but this folder.
"use strict";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fileName = (p) => String(p ?? "").split(/[\\/]/).pop();

function toast(text, ms = 3500) {
  const t = $("toast");
  t.textContent = text;
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), ms);
}

async function call(command, args) {
  try {
    return await invoke(command, args);
  } catch (e) {
    toast(String(e), 6000);
    throw e;
  }
}

function bytes(n) {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

// A <progress>, not a styled div: the CSP allows no inline style attributes.
function bar(fraction) {
  return `<progress class="bar" max="1" value="${Math.max(0, Math.min(1, Number(fraction) || 0))}"></progress>`;
}

async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    if (button) { const label = button.textContent; button.textContent = "Copied"; setTimeout(() => (button.textContent = label), 1200); }
  } catch {
    toast("Select the text and copy it.");
  }
}

// MARK: Tabs

let currentTab = "library";
function showTab(name) {
  currentTab = name;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
  refresh();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function refresh() {
  refreshStatus();
  if (currentTab === "library") { refreshQueue(); refreshWorker(); refreshTokens(); }
  if (currentTab === "convert") refreshConvert();
  if (currentTab === "headsets") refreshPairing();
  if (currentTab === "settings") loadSettings();
}

// MARK: Status

async function refreshStatus() {
  const s = await invoke("status").catch(() => null);
  if (!s) return;
  const pill = $("status-pill");
  const c = s.coordinator;
  if (c.running) {
    pill.className = "pill ok";
    pill.textContent = s.lan ? `Coordinator on port ${s.port} (network)` : `Coordinator on port ${s.port}`;
  } else {
    pill.className = "pill bad";
    pill.textContent = c.error ? "Coordinator stopped" : "Starting…";
  }
  const addresses = s.lan && s.addresses.length ? ` · reachable at ${s.addresses.map((a) => `http://${a}:${s.port}`).join(", ")}` : "";
  $("coordinator-line").innerHTML = c.running
    ? `Serving at <span class="mono">${esc(s.localUrl)}</span>${esc(addresses)}`
    : `<span class="bad-text">${esc(c.error || "Not running.")}</span>`;
  if (c.pairedListen) $("paired-port").textContent = c.pairedListen.split(":").pop();
  $("paired-error").textContent = c.pairedError || "";
  const scan = c.lastScan;
  $("scan-line").textContent = scan ? `Last scan: ${scan.titles} titles${scan.problems?.length ? `, ${scan.problems.length} problems (${scan.problems[0]})` : ""}.` : "";
}

// MARK: Queue

async function refreshQueue() {
  let q;
  try { q = await invoke("queue_snapshot"); } catch { q = { titles: [], leases: [], workers: [], playheads: [] }; }
  const titles = [...q.titles].sort((a, b) => (a.key < b.key ? -1 : 1));
  $("titles").querySelector("tbody").innerHTML = titles.map((t) => {
    const [server, id, workload] = t.key.split("/");
    const s = t.status || {};
    return `<tr>
      <td><div>${esc(server)} <span class="mono">${esc(id)}</span></div><div class="small muted">${esc(workload)} · ${esc(t.stereo)} · ${Math.round(t.durationSeconds / 60)} min</div></td>
      <td><span class="state ${esc(s.state)}">${esc(s.state)}</span></td>
      <td>${bar(s.progress)}</td>
      <td class="small">${s.piecesDone ?? 0}/${s.pieces ?? 0}${s.piecesLeased ? ` (${s.piecesLeased} leased)` : ""}</td>
      <td class="small">${bytes(t.bytes)}</td>
      <td><button class="link danger" data-remove="${esc(t.key)}">Remove</button></td>
    </tr>`;
  }).join("");
  $("titles-empty").classList.toggle("hidden", titles.length > 0);
  $("titles").classList.toggle("hidden", titles.length === 0);
  const list = (el, items, render, none) => {
    $(el).innerHTML = items.length ? items.map((i) => `<li>${render(i)}</li>`).join("") : `<li class="none">${none}</li>`;
  };
  list("workers", q.workers, (w) => `<b>${esc(w.worker)}</b> · ${w.pieces} pieces${w.pieceSeconds ? ` · ${w.pieceSeconds.toFixed(1)} s each` : ""}${w.remote ? " · remote" : ""}${w.idleSeconds != null ? ` · seen ${Math.round(w.idleSeconds)} s ago` : ""}`, "No worker has asked for work.");
  list("leases", q.leases, (l) => `<span class="mono">${esc(l.title)}</span> piece ${l.piece} · ${esc(l.worker)} · ${Math.round(l.secondsLeft)} s left`, "None.");
  list("playheads", q.playheads, (p) => `<b>${esc(p.player)}</b> · <span class="mono">${esc(p.title)}</span> at ${Math.floor(p.seconds / 60)}:${String(Math.floor(p.seconds % 60)).padStart(2, "0")}${p.rate !== 1 ? ` ×${p.rate}` : ""}`, "Nothing playing.");
}

$("titles").addEventListener("click", async (e) => {
  const key = e.target.dataset.remove;
  if (!key || !confirm(`Remove ${key} and its results?`)) return;
  await call("remove_title", { key });
  refreshQueue();
});

$("scan-now").addEventListener("click", async (e) => {
  e.target.disabled = true;
  try {
    const summary = await call("scan_now");
    toast(`Scanned: ${summary.titles} titles${summary.problems.length ? `, ${summary.problems.length} problems` : ""}.`);
  } finally {
    e.target.disabled = false;
    refresh();
  }
});

// MARK: Worker

async function refreshWorker() {
  const w = await invoke("worker_status").catch(() => null);
  if (!w) return;
  $("worker-start").disabled = w.running || !w.available;
  $("worker-stop").disabled = !w.running;
  const parts = [];
  if (!w.available) parts.push("This computer's worker runs on a Mac or Windows for now.");
  else {
    parts.push(w.running ? `Running (pid ${w.pid}).` : "Stopped.");
    parts.push(w.cli ? `Runs ${w.cli}` : "The worker program wasn't found: give its path in Settings.");
    parts.push(w.modelsFolder ? `Models: ${w.modelsFolder}` : "Choose the models folder in Settings.");
    if (w.lastExit && !w.running) parts.push(w.lastExit);
  }
  $("worker-line").textContent = parts.join(" · ");
  const log = $("worker-log");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = w.log.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
}
$("worker-start").addEventListener("click", async () => { await call("worker_start").catch(() => {}); refreshWorker(); refreshTray(); });
$("worker-stop").addEventListener("click", async () => { await call("worker_stop").catch(() => {}); refreshWorker(); });
function refreshTray() { refreshStatus(); }

// MARK: Tokens

async function refreshTokens() {
  const tokens = await invoke("tokens_list").catch(() => []);
  $("tokens").querySelector("tbody").innerHTML = tokens.map((t) => `<tr>
    <td>${esc(t.name)}</td><td>${esc(t.grants.join(", "))}</td>
    <td><button class="link danger" data-token="${esc(t.name)}">Remove</button></td></tr>`).join("");
  $("tokens").classList.toggle("hidden", tokens.length === 0);
}
$("token-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const made = await call("token_create", { name: $("token-name").value, grant: $("token-grant").value }).catch(() => null);
  if (!made) return;
  $("token-new-name").textContent = made.name;
  $("token-new-value").value = made.token;
  $("token-new").classList.remove("hidden");
  $("token-name").value = "";
  refreshTokens();
});
$("token-copy").addEventListener("click", (e) => copy($("token-new-value").value, e.target));
$("token-hide").addEventListener("click", () => { $("token-new-value").value = ""; $("token-new").classList.add("hidden"); });
$("tokens").addEventListener("click", async (e) => {
  const name = e.target.dataset.token;
  if (!name || !confirm(`Remove the token ${name}? Whatever uses it stops working.`)) return;
  await call("token_remove", { name });
  refreshTokens();
});

// MARK: Convert

let convertState = null;
let optionsDirty = false;

function options() {
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : Number(v); };
  return {
    layout: $("opt-layout").value,
    strength: num("opt-strength"),
    convergence: num("opt-convergence"),
    sample: $("opt-sample").checked,
    force: $("opt-force").checked,
    upscale: $("opt-upscale").checked,
  };
}

const describeLayout = { fsbs: "full SBS", hsbs: "half SBS", tab: "top-bottom", htab: "half top-bottom", spatial: "spatial video", "2d": "2D" };

async function refreshConvert() {
  const c = await invoke("convert_status").catch(() => null);
  if (!c) return;
  convertState = c;
  $("convert-unavailable").classList.toggle("hidden", c.available);
  $("convert-unavailable").textContent = c.unavailableReason || "";
  for (const id of ["add-files", "add-folder", "out-pick", "out-beside"]) $(id).disabled = !c.available;
  if (!optionsDirty) {
    const d = c.defaults;
    $("opt-layout").value = d.layout;
    $("opt-strength").value = d.strength ?? "";
    $("opt-convergence").value = d.convergence ?? "";
    $("opt-sample").checked = d.sample;
    $("opt-force").checked = d.force;
    $("opt-upscale").checked = d.upscale;
  }
  $("sample-seconds").textContent = c.sampleSeconds;
  $("out-dir").textContent = c.outputDir || "Beside the original";
  $("cli-line").textContent = c.available ? (c.cli ? `Runs ${c.cli}` : "The converter wasn't found: give its path in Settings.") : "";
  const jobs = [...c.jobs].reverse();
  $("jobs").innerHTML = jobs.map((j) => {
    const o = j.options;
    const what = [describeLayout[o.layout], o.upscale ? "upscaled 2×" : null, o.sample ? `${c.sampleSeconds} s sample` : null,
      o.strength != null ? `strength ${o.strength}` : null, o.convergence != null ? `convergence ${o.convergence}` : null].filter(Boolean).join(" · ");
    const active = j.state === "running" || j.state === "queued";
    const detail = j.state === "running"
      ? `${j.phase || "starting"} · ${Math.round(j.progress * 100)}%`
      : j.state === "done" ? `→ ${fileName(j.output)}` : (j.message || "");
    return `<li>
      <div><div class="job-name">${esc(fileName(j.input))} <span class="state ${esc(j.state)}">${esc(j.state)}</span></div>
      <div class="job-sub">${esc(what)}${detail ? ` · ${esc(detail)}` : ""}</div></div>
      <div class="job-actions">
        ${active ? `<button class="link danger" data-act="convert_cancel" data-id="${j.id}">Cancel</button>` : ""}
        ${["failed", "cancelled", "done", "skipped"].includes(j.state) ? `<button class="link" data-act="convert_retry" data-id="${j.id}">${j.state === "cancelled" || j.state === "failed" ? "Retry" : "Again"}</button>` : ""}
        ${j.state === "done" && j.output ? `<button class="link" data-act="convert_reveal" data-id="${j.id}">Show</button>` : ""}
        ${!active ? `<button class="link" data-act="convert_remove" data-id="${j.id}">Remove</button>` : ""}
      </div>
      ${active ? `<div class="job-bar">${bar(j.progress)}</div>` : ""}
    </li>`;
  }).join("");
  $("jobs-empty").classList.toggle("hidden", jobs.length > 0);
}

let optionsTimer = null;
for (const id of ["opt-layout", "opt-strength", "opt-convergence", "opt-sample", "opt-force", "opt-upscale"]) {
  $(id).addEventListener("input", () => {
    optionsDirty = true;
    clearTimeout(optionsTimer);
    optionsTimer = setTimeout(async () => {
      try { await invoke("convert_set_options", { options: options() }); optionsDirty = false; } catch (e) { toast(String(e)); }
    }, 400);
  });
}
$("add-files").addEventListener("click", async () => { await flushOptions(); await call("convert_pick", { folder: false }).catch(() => {}); refreshConvert(); });
$("add-folder").addEventListener("click", async () => { await flushOptions(); await call("convert_pick", { folder: true }).catch(() => {}); refreshConvert(); });
$("out-pick").addEventListener("click", async () => { await call("convert_pick_output", { beside: false }).catch(() => {}); refreshConvert(); });
$("out-beside").addEventListener("click", async () => { await call("convert_pick_output", { beside: true }).catch(() => {}); refreshConvert(); });
$("clear-finished").addEventListener("click", async () => { await call("convert_clear_finished").catch(() => {}); refreshConvert(); });
$("jobs").addEventListener("click", async (e) => {
  const { act, id } = e.target.dataset;
  if (!act) return;
  await call(act, { id: Number(id) }).catch(() => {});
  refreshConvert();
});

async function flushOptions() {
  if (!optionsDirty) return;
  clearTimeout(optionsTimer);
  await call("convert_set_options", { options: options() });
  optionsDirty = false;
}

// Drops are handled in Rust; the page only shows where they land.
listen("tauri://drag-enter", () => $("drop").classList.add("over"));
listen("tauri://drag-leave", () => $("drop").classList.remove("over"));
listen("tauri://drag-drop", () => $("drop").classList.remove("over"));

// MARK: Headsets

function renderPairing(p) {
  const open = p.open;
  $("pair-open").classList.toggle("hidden", !open);
  $("pair-start").disabled = !!open;
  $("pair-stop").disabled = !open;
  if (open) {
    $("pair-port").textContent = open.port;
    $("pair-addresses").textContent = open.addresses.length ? ` (this computer: ${open.addresses.join(", ")})` : "";
    $("pair-link").value = open.link;
  }
  const w = p.waiting;
  $("pair-waiting").classList.toggle("hidden", !w);
  if (w) {
    $("pair-device").textContent = w.deviceName;
    $("pair-code").textContent = w.code;
    $("pair-yes").dataset.id = w.id;
    $("pair-no").dataset.id = w.id;
  }
  $("pair-last").textContent = p.last || "";
  const devices = p.devices || [];
  $("devices").querySelector("tbody").innerHTML = devices.map((d) => `<tr>
    <td>${esc(d.name)}<div class="small muted mono">${esc(d.id)}</div></td>
    <td>${esc(d.grants.join(", "))}</td>
    <td class="small">${d.pairedAt ? new Date(d.pairedAt * 1000).toLocaleString() : ""}</td>
    <td><button class="link danger" data-forget="${esc(d.id)}">Forget</button></td></tr>`).join("");
  $("devices").classList.toggle("hidden", devices.length === 0);
  $("devices-empty").classList.toggle("hidden", devices.length > 0);
}
async function refreshPairing() { renderPairing(await invoke("pairing_status")); }
$("pair-start").addEventListener("click", async () => renderPairing(await call("pairing_start")));
$("pair-stop").addEventListener("click", async () => renderPairing(await call("pairing_stop")));
$("pair-yes").addEventListener("click", async (e) => renderPairing(await call("pairing_decide", { id: Number(e.target.dataset.id), pair: true })));
$("pair-no").addEventListener("click", async (e) => renderPairing(await call("pairing_decide", { id: Number(e.target.dataset.id), pair: false })));
$("pair-link-copy").addEventListener("click", (e) => copy($("pair-link").value, e.target));
$("devices").addEventListener("click", async (e) => {
  const id = e.target.dataset.forget;
  if (!id || !confirm("Forget this headset? It has to pair again.")) return;
  renderPairing(await call("device_forget", { id }));
});

// MARK: Settings

let loaded = null;

function keyPlaceholder(hint) {
  if (!hint.set) return "Not set";
  return hint.endsWith ? `Set, ends in …${hint.endsWith} (type to replace)` : "Set (type to replace)";
}

async function loadSettings() {
  const v = await invoke("get_settings");
  loaded = v;
  const s = v.settings;
  $("jf-url").value = s.jellyfinUrl || "";
  $("st-url").value = s.stashUrl || "";
  $("jf-key").value = "";
  $("st-key").value = "";
  $("jf-key").placeholder = keyPlaceholder(v.jellyfinKey);
  $("st-key").placeholder = keyPlaceholder(v.stashKey);
  $("scan").checked = s.scan;
  $("scan-hours").value = s.scanEveryHours;
  $("port").value = s.port;
  $("lan").checked = s.lan;
  $("models-folder").textContent = s.modelsFolder || "Not chosen";
  $("worker-name").value = s.workerName || "";
  $("worker-at-start").checked = s.workerAtStart;
  $("cli-path").value = s.vergenceCli || "";
  $("cli-path").placeholder = v.cli ? `Found: ${v.cli}` : "Not found: give its path";
  $("depth-model").value = s.depthModel || "";
  await loadDepthVariants();
  $("login").checked = v.launchAtLogin;
  $("updates").checked = s.checkUpdates;
  $("private-updates").checked = s.privateUpdates;
  $("data-dir").textContent = `Data: ${v.dataDir}`;
  const u = await invoke("update_status");
  $("update-line").textContent = u.message || "";
}

// The models a conversion can download: only Small, unless this build offers the larger,
// non-commercial ones; each of those is chosen only after its licence was shown and accepted.
let depthVariants = { variants: [], chosen: "small" };

async function loadDepthVariants() {
  depthVariants = await invoke("depth_variants");
  const select = $("depth-variant");
  select.replaceChildren(...depthVariants.variants.map((v) => {
    const option = document.createElement("option");
    option.value = v.variant;
    option.textContent = `${v.display} (${v.licence}, about ${v.megabytes} MB)`;
    return option;
  }));
  select.value = depthVariants.chosen;
  $("depth-variant-row").hidden = depthVariants.variants.length < 2;
}

$("depth-variant").addEventListener("change", async () => {
  const select = $("depth-variant");
  const v = depthVariants.variants.find((x) => x.variant === select.value);
  let accept = false;
  if (v && v.restricted && !depthVariants.accepted.includes(v.variant)) {
    accept = confirm(`${v.display}'s weights are licensed ${v.licence} (${v.licenceUrl}).\n\n${v.note}\n\n` +
                     `Accept this licence and download about ${v.megabytes} MB at the next conversion?`);
    if (!accept) {
      select.value = depthVariants.chosen;
      return;
    }
  }
  try {
    depthVariants = await call("choose_depth_variant", { variant: select.value, acceptLicence: accept });
  } catch {
    select.value = depthVariants.chosen;
  }
});

function settingsFromForm() {
  const s = { ...loaded.settings };
  s.jellyfinUrl = $("jf-url").value;
  s.stashUrl = $("st-url").value;
  s.scan = $("scan").checked;
  s.scanEveryHours = Number($("scan-hours").value) || 6;
  s.port = Number($("port").value) || 8420;
  s.lan = $("lan").checked;
  s.workerName = $("worker-name").value;
  s.workerAtStart = $("worker-at-start").checked;
  s.vergenceCli = $("cli-path").value;
  s.depthModel = $("depth-model").value;
  s.checkUpdates = $("updates").checked;
  s.privateUpdates = $("private-updates").checked;
  return s;
}

async function saveSettings(keys = {}) {
  const result = $("settings-result");
  result.textContent = "Saving…";
  result.className = "small muted";
  try {
    const v = await invoke("save_settings", {
      settings: settingsFromForm(),
      jellyfinKey: keys.jellyfin ?? ($("jf-key").value || null),
      stashKey: keys.stash ?? ($("st-key").value || null),
    });
    result.textContent = v.coordinatorError ? `Saved, but: ${v.coordinatorError}` : "Saved.";
    result.className = v.coordinatorError ? "small bad-text" : "small ok-text";
    await loadSettings();
    refreshStatus();
  } catch (e) {
    result.textContent = String(e);
    result.className = "small bad-text";
    throw e;
  }
}

$("settings-form").addEventListener("submit", (e) => { e.preventDefault(); saveSettings().catch(() => {}); });

for (const [prefix, server] of [["jf", "jellyfin"], ["st", "stash"]]) {
  $(`${prefix}-test`).addEventListener("click", async () => {
    const out = $(`${prefix}-result`);
    out.textContent = "Checking…";
    out.className = "small muted";
    try {
      // What's typed is saved first, so the test is of what will be used.
      await saveSettings();
      const r = await invoke("test_server", { server });
      out.textContent = `✓ ${r.detail}`;
      out.className = "small ok-text";
    } catch (e) {
      out.textContent = String(e);
      out.className = "small bad-text";
    }
  });
  $(`${prefix}-forget`).addEventListener("click", async () => {
    if (!confirm(`Remove the ${server} key?`)) return;
    await saveSettings({ [server]: "" }).catch(() => {});
  });
}

$("models-pick").addEventListener("click", async () => {
  const v = await call("pick_models_folder");
  $("models-folder").textContent = v.settings.modelsFolder || "Not chosen";
  loaded = v;
});
$("login").addEventListener("change", async (e) => {
  e.target.checked = await call("set_launch_at_login", { on: e.target.checked }).catch(() => !e.target.checked);
});
$("update-check").addEventListener("click", async () => {
  try {
    const s = await invoke("check_update");
    $("update-line").textContent = s.message || (s.version ? `Vergence ${s.version} is available.` : "Up to date.");
  } catch (e) {
    $("update-line").textContent = String(e);
  }
});

$("quit").addEventListener("click", () => invoke("quit_app"));

// MARK: Events

listen("convert", () => { if (currentTab === "convert") refreshConvert(); });
listen("convert-added", (e) => {
  const { added, error } = e.payload;
  if (error) toast(error, 6000);
  else toast(`${added} ${added === 1 ? "file" : "files"} added.`);
  refreshConvert();
});
listen("pairing", (e) => { if (currentTab === "headsets") renderPairing(e.payload); });
listen("show-tab", (e) => showTab(e.payload));
listen("status", () => refreshStatus());

showTab("library");
setInterval(() => {
  refreshStatus();
  if (currentTab === "library") { refreshQueue(); refreshWorker(); }
  if (currentTab === "convert") refreshConvert();
  if (currentTab === "headsets") refreshPairing();
}, 2000);
