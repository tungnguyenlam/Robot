// egograph dataset viewer: plain ES module, no build step.
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) n.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined) n.append(c.nodeType ? c : document.createTextNode(c));
  return n;
};
const api = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const fmt = (n, d = 0) => (n === null || n === undefined ? "–" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
const hue = (s) => { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360; return h; };
const segColor = (s) => `hsl(${hue(s.toLowerCase())} 55% 52%)`;
const CHUNK = 300;

const LAYERS = [
  { key: "hands", label: "hand kps (GT 2D)", color: "var(--right)", on: true },
  { key: "mano", label: "MANO → 2D", color: "var(--mano)", on: false },
  { key: "box3d", label: "object 3D box", color: "var(--obj)", on: true },
  { key: "bbox", label: "object 2D box", color: "#d6d6d6", on: false },
  { key: "axes", label: "object axes", color: "#ff5d5d", on: false },
];

const S = {
  datasets: [], ds: null, q: "", require: new Set(), tags: {}, offset: 0, total: 0,
  seq: null, frame: 0, chunks: new Map(), pending: new Set(), layers: Object.fromEntries(LAYERS.map((l) => [l.key, l.on])),
};

// ─────────────────────────────── datasets ───────────────────────────────
async function init() {
  S.datasets = await api("/api/datasets");
  const [hDs, hSeq, hFrame] = decodeURIComponent(location.hash.slice(1)).split("/");
  const first = S.datasets.find((d) => d.name === hDs) || S.datasets.find((d) => d.indexed) || S.datasets[0];
  renderTabs();
  await selectDataset(first.name, false);
  if (hSeq) openSequence(first.name, hSeq, Number(hFrame) || 0);
  buildLayers();
  bindPlayer();
}

function renderTabs() {
  const nav = $("#dataset-tabs");
  nav.replaceChildren(...S.datasets.map((d) =>
    el("button", { class: `tab ${d.name === S.ds?.name ? "active" : ""} ${d.indexed ? "" : "off"}`, onclick: () => selectDataset(d.name) },
      d.display_name, el("span", { class: "n" }, d.indexed ? fmt(d.sequences) : "—"))));
}

async function selectDataset(name, resetHash = true) {
  S.ds = S.datasets.find((d) => d.name === name);
  S.q = ""; S.require.clear(); S.tags = {}; $("#search").value = "";
  renderTabs(); renderDatasetCard(); renderFilters();
  if (resetHash && S.seq?.meta.dataset !== name) history.replaceState(null, "", `#${name}`);
  await loadSequences(true);
}

function renderDatasetCard() {
  const d = S.ds, st = d.status || {};
  const stats = d.indexed ? [
    ["sequences", fmt(d.sequences)], ["hours", fmt(d.hours, 1)], ["frames", fmt(d.frames / 1e6, 2) + "M"],
    ["hand pose", fmt(d.with_hand_pose)], ["obj pose", fmt(d.with_object_pose)], ["actions", fmt(d.with_actions)],
  ] : [];
  let note = st.note || "";
  if (st.available && !d.indexed) note = (note ? note + " " : "") + `Not indexed yet: run "egograph index --dataset ${d.name}".`;
  $("#dataset-card").replaceChildren(...[
    el("div", { class: "ds-title" }, el("h2", {}, d.display_name), el("span", { class: "dim small mono" }, d.name)),
    el("div", { class: "ds-desc" }, d.description),
    stats.length ? el("div", { class: "stats" }, stats.map(([k, v]) => el("div", { class: "stat" }, el("div", { class: "v" }, v), el("div", { class: "k" }, k)))) : null,
    note ? el("div", { class: "note" }, note) : null,
    el("div", { class: "dim small mono", style: "margin-top:8px;word-break:break-all" }, st.root || "")].filter(Boolean));
}

function renderFilters() {
  const box = $("#filters");
  box.replaceChildren();
  if (!S.ds.indexed) return;
  const flags = [["has_hand_pose", "hands"], ["has_object_pose", "objects"], ["has_camera_pose", "camera"], ["has_actions", "actions"]]
    .filter(([k]) => S.ds[`with_${k.slice(4)}`] > 0 && S.ds[`with_${k.slice(4)}`] < S.ds.sequences);
  for (const [k, label] of flags) {
    box.append(el("button", {
      class: `chip ${S.require.has(k) ? "on" : ""}`,
      onclick: (e) => { S.require.has(k) ? S.require.delete(k) : S.require.add(k); e.target.classList.toggle("on"); loadSequences(true); },
    }, label));
  }
  for (const [tag, values] of Object.entries(S.ds.facets || {})) {
    if (values.length < 2) continue;
    const sel = el("select", { onchange: (e) => { e.target.value ? (S.tags[tag] = e.target.value) : delete S.tags[tag]; loadSequences(true); } },
      el("option", { value: "" }, `${tag.replace("_name", "")}: all`),
      values.sort((a, b) => String(a[0]).localeCompare(String(b[0]), undefined, { numeric: true }))
        .map(([v, n]) => el("option", { value: v }, `${v} (${n})`)));
    box.append(sel);
  }
}

let searchTimer;
$("#search").addEventListener("input", (e) => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { S.q = e.target.value; loadSequences(true); }, 200); });
$("#load-more").addEventListener("click", () => loadSequences(false));

async function loadSequences(reset) {
  if (reset) { S.offset = 0; $("#seq-list").replaceChildren(); }
  if (!S.ds.indexed) { $("#seq-count").textContent = ""; $("#load-more").hidden = true; return; }
  const p = new URLSearchParams({ dataset: S.ds.name, offset: S.offset, limit: 150 });
  if (S.q) p.set("q", S.q);
  S.require.forEach((r) => p.append("require", r));
  Object.entries(S.tags).forEach(([k, v]) => p.append("tag", `${k}:${v}`));
  const res = await api(`/api/sequences?${p}`);
  S.total = res.total; S.offset += res.items.length;
  $("#seq-count").textContent = `${fmt(res.total)} sequences`;
  $("#seq-list").append(...res.items.map(seqItem));
  $("#load-more").hidden = S.offset >= S.total;
  markActive();
}

function seqItem(m) {
  const t = m.tags || {};
  const sub = [t.category_name || t.participant, `${fmt(m.duration_s, 1)}s`, t.num_actions !== undefined ? `${t.num_actions} acts` : null, t.hands?.length ? t.hands.join("+") : null].filter(Boolean);
  return el("li", { "data-id": m.sequence_id, onclick: () => openSequence(m.dataset, m.sequence_id) },
    el("div", { class: "sid" }, m.sequence_id),
    el("div", { class: "sub" }, sub.join(" · "),
      el("span", { class: "dots" },
        el("i", { class: `dot ${m.has_hand_pose ? "h" : ""}`, title: "hand pose" }),
        el("i", { class: `dot ${m.has_object_pose ? "o" : ""}`, title: "object pose" }),
        el("i", { class: `dot ${m.has_camera_pose ? "c" : ""}`, title: "camera pose" }),
        el("i", { class: `dot ${m.has_actions ? "a" : ""}`, title: "actions" }))));
}
function markActive() {
  document.querySelectorAll("#seq-list li").forEach((li) => li.classList.toggle("active", li.dataset.id === S.seq?.meta.sequence_id));
}

// ─────────────────────────────── sequence ───────────────────────────────
async function openSequence(dataset, sid, startFrame = 0) {
  const seq = await api(`/api/sequence/${dataset}/${encodeURIComponent(sid)}`);
  S.seq = seq; S.chunks = new Map(); S.pending = new Set(); S.frame = 0;
  history.replaceState(null, "", `#${dataset}/${sid}`);
  const m = seq.meta;
  $("#empty").hidden = true; $("#player").hidden = false;
  $("#seq-id").textContent = m.sequence_id;
  $("#seq-badges").replaceChildren(...[
    ["hand pose", m.has_hand_pose], ["object pose", m.has_object_pose], ["camera pose", m.has_camera_pose],
    ["actions", m.has_actions], ["intrinsics", !!m.intrinsics]].map(([k, v]) => el("span", { class: `badge ${v ? "yes" : ""}` }, `${v ? "✓" : "✗"} ${k}`)));
  $("#stage").style.aspectRatio = `${m.width} / ${m.height}`;
  const v = $("#video");
  v.src = seq.video_url;
  if (startFrame) v.addEventListener("loadedmetadata", () => seekFrame(startFrame), { once: true });
  v.playbackRate = Number($("#rate").value);
  $("#frame-total").textContent = m.num_frames - 1;
  renderMeta(m); renderTimeline(); renderActions(); renderTraj();
  $("#clip-strip").replaceChildren(); $("#clip-info").replaceChildren();
  markActive();
  ensureChunk(0);
}

function renderMeta(m) {
  const rows = [
    ["dataset", m.dataset], ["resolution", `${m.width}×${m.height}`], ["fps", fmt(m.fps, 3)],
    ["frames", fmt(m.num_frames)], ["duration", `${fmt(m.duration_s, 2)} s`], ["video", m.video_path],
    ["K", m.intrinsics ? `fx ${fmt(m.intrinsics[0][0], 1)} fy ${fmt(m.intrinsics[1][1], 1)} cx ${fmt(m.intrinsics[0][2], 1)} cy ${fmt(m.intrinsics[1][2], 1)}` : "unknown (uncalibrated)"],
    ...Object.entries(m.tags || {}).filter(([k]) => k !== "rel").map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") || "–" : String(v)]),
  ];
  $("#meta-table").replaceChildren(...rows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, v))));
}

function renderTimeline() {
  const { meta, actions } = S.seq, dur = meta.duration_s || meta.num_frames / meta.fps;
  $("#tl-segments").replaceChildren(...actions.map((a) => el("div", {
    class: "tl-seg", title: `${a.label}  ${fmt(a.start_s, 2)}–${fmt(a.end_s, 2)}s`,
    style: `left:${(a.start_s / dur) * 100}%;width:${Math.max(((a.end_s - a.start_s) / dur) * 100, 0.15)}%;background:${segColor(a.verb || a.label)}`,
  })));
  $("#tl-valid").replaceChildren();
}

function renderValidity() {
  // Coverage strip: hands (orange or cyan) over objects (green), from the loaded chunks.
  const { meta } = S.seq, n = meta.num_frames;
  const c = el("canvas", { width: 1200, height: 7, style: "width:100%;height:7px;display:block" });
  const ctx = c.getContext("2d");
  for (const ch of S.chunks.values()) {
    for (let i = 0; i < ch.end - ch.start; i++) {
      const x = ((ch.start + i) / n) * 1200, w = Math.max(1200 / n, 1);
      if (ch.objects.some((o) => o.valid[i])) { ctx.fillStyle = "#a6e35a"; ctx.fillRect(x, 4, w, 3); }
      const l = ch.hands.left?.valid[i], r = ch.hands.right?.valid[i];
      if (l || r) { ctx.fillStyle = r ? "#ff8a4c" : "#4cc3ff"; ctx.fillRect(x, 0, w, 3); }
    }
  }
  $("#tl-valid").replaceChildren(c);
}

function renderActions() {
  const acts = S.seq.actions;
  $("#act-count").textContent = acts.length ? `${acts.length} segments` : "none";
  $("#act-list").replaceChildren(...acts.map((a, i) => el("li", { "data-i": i, onclick: () => seekTime(a.start_s + 0.001) },
    el("span", { class: "t" }, `${fmt(a.start_s, 2)}s`),
    el("span", { style: `color:${segColor(a.verb || a.label)}` }, "■"), a.label)));
}

function renderTraj() {
  const svg = $("#traj"), traj = S.seq.camera_trajectory;
  svg.replaceChildren();
  if (!traj || !traj.length) { $("#traj-note").textContent = "No camera poses for this sequence (run SLAM in Month 2)."; return; }
  const xs = traj.map((p) => p[1]), zs = traj.map((p) => p[3]);
  const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
  const span = Math.max(x1 - x0, z1 - z0, 0.05), W = 300, H = 220, pad = 18, s = (Math.min(W, H) - 2 * pad) / span;
  const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
  S.trajXY = traj.map((p) => [W / 2 + (p[1] - cx) * s, H / 2 - (p[3] - cz) * s]);
  const ns = "http://www.w3.org/2000/svg";
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", S.trajXY.map((p) => p.join(",")).join(" "));
  line.setAttribute("fill", "none"); line.setAttribute("stroke", "#8b9cff"); line.setAttribute("stroke-width", "1.5"); line.setAttribute("opacity", ".8");
  const dot = document.createElementNS(ns, "circle");
  dot.setAttribute("r", "4.5"); dot.setAttribute("fill", "#e8b04a"); dot.id = "traj-dot";
  const scale = document.createElementNS(ns, "text");
  scale.setAttribute("x", 8); scale.setAttribute("y", H - 8); scale.setAttribute("fill", "#7d8794"); scale.setAttribute("font-size", "10");
  scale.textContent = `extent ${span.toFixed(2)} m · ${traj.length} poses`;
  svg.append(line, dot, scale);
  $("#traj-note").textContent = "World frame = first camera (HOI4D reconstruction).";
}

// ─────────────────────────────── annotations ───────────────────────────────
async function ensureChunk(frame) {
  if (!S.seq) return;
  const m = S.seq.meta;
  if (!(m.has_hand_pose || m.has_object_pose)) return;
  const start = Math.floor(frame / CHUNK) * CHUNK, key = `${m.sequence_id}:${start}`;
  if (S.chunks.has(start) || S.pending.has(key)) return;
  S.pending.add(key);
  try {
    const ch = await api(`/api/sequence/${m.dataset}/${encodeURIComponent(m.sequence_id)}/annotations?start=${start}&end=${start + CHUNK}`);
    if (S.seq?.meta.sequence_id === m.sequence_id) { S.chunks.set(start, ch); renderValidity(); draw(); }
  } finally { S.pending.delete(key); }
}

// ─────────────────────────────── drawing ───────────────────────────────
function buildLayers() {
  $("#layers").replaceChildren(...LAYERS.map((l) => el("button", {
    class: `layer ${S.layers[l.key] ? "" : "off"}`,
    onclick: (e) => { S.layers[l.key] = !S.layers[l.key]; e.currentTarget.classList.toggle("off"); draw(); },
  }, el("i", { style: `background:${l.color}` }), l.label)));
}

function draw() {
  const cv = $("#overlay"), v = $("#video");
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(r.width * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!S.seq) return;
  const m = S.seq.meta, f = S.frame;
  // Readouts
  $("#frame-no").textContent = f;
  $("#time-no").textContent = (v.currentTime || 0).toFixed(2);
  const dur = m.duration_s || m.num_frames / m.fps;
  $("#tl-head").style.left = `${(f / m.fps / dur) * 100}%`;
  const t = f / m.fps;
  const cur = S.seq.actions.filter((a) => a.start_s <= t && t <= a.end_s);
  $("#action-now").textContent = cur.map((a) => a.label).join(" · ");
  document.querySelectorAll("#act-list li").forEach((li) => {
    const a = S.seq.actions[li.dataset.i]; li.classList.toggle("cur", a.start_s <= t && t <= a.end_s);
  });
  if (S.trajXY) {
    const tr = S.seq.camera_trajectory, k = tr.findIndex((p) => p[0] >= f);
    const p = S.trajXY[k < 0 ? tr.length - 1 : k]; const d = $("#traj-dot");
    if (p && d) { d.setAttribute("cx", p[0]); d.setAttribute("cy", p[1]); }
  }
  // Overlays
  const start = Math.floor(f / CHUNK) * CHUNK, ch = S.chunks.get(start);
  if (!ch) { ensureChunk(f); return; }
  if (f + 60 >= start + CHUNK && f + 60 < m.num_frames) ensureChunk(start + CHUNK);
  const i = f - ch.start, s = cv.width / m.width;
  const P = (p) => [p[0] * s, p[1] * s];
  const lw = Math.max(1.5, 2 * s * dpr);
  ctx.lineCap = "round"; ctx.lineJoin = "round";

  for (const o of ch.objects) {
    if (!o.valid[i]) continue;
    if (S.layers.box3d && o.corners_2d[i][0][0] !== null) {
      ctx.strokeStyle = "#a6e35a"; ctx.lineWidth = lw;
      ctx.beginPath();
      for (const [a, b] of S.seq.box_edges) { const pa = P(o.corners_2d[i][a]), pb = P(o.corners_2d[i][b]); ctx.moveTo(...pa); ctx.lineTo(...pb); }
      ctx.stroke();
    }
    if (S.layers.axes && o.axes_2d[i][0][0] !== null) {
      const ax = o.axes_2d[i].map(P);
      [["#ff5d5d", 1], ["#5dff7a", 2], ["#5d8bff", 3]].forEach(([c, k]) => { ctx.strokeStyle = c; ctx.lineWidth = lw * 1.3; ctx.beginPath(); ctx.moveTo(...ax[0]); ctx.lineTo(...ax[k]); ctx.stroke(); });
    }
    if (S.layers.bbox && o.bbox_2d[i][0] !== null) {
      const [x0, y0, x1, y1] = o.bbox_2d[i].map((x) => x * s);
      ctx.setLineDash([6 * dpr, 4 * dpr]); ctx.strokeStyle = "#d6d6d6"; ctx.lineWidth = lw * 0.8;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.setLineDash([]);
    }
    if (S.layers.box3d || S.layers.bbox) {
      const anchor = o.bbox_2d[i][0] !== null ? [o.bbox_2d[i][0] * s, o.bbox_2d[i][1] * s - 6 * dpr] : null;
      if (anchor) label(ctx, `${o.label}${o.depth[i] ? ` · ${o.depth[i].toFixed(2)} m` : ""}`, anchor, "#a6e35a", dpr);
    }
  }
  for (const [side, h] of Object.entries(ch.hands)) {
    if (!h.valid[i]) continue;
    const color = side === "left" ? "#4cc3ff" : "#ff8a4c";
    if (S.layers.hands) skeleton(ctx, h.joints_2d[i].map(P), color, lw, dpr);
    if (S.layers.mano && h.mano_2d && h.mano_2d[i][0][0] !== null) skeleton(ctx, h.mano_2d[i].map(P), "#f05bd8", lw * 0.7, dpr * 0.7);
  }
}

function skeleton(ctx, pts, color, lw, dpr) {
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
  S.seq.hand_parents.forEach((p, j) => { if (p >= 0 && pts[j][0] !== null && pts[p][0] !== null) { ctx.moveTo(...pts[p]); ctx.lineTo(...pts[j]); } });
  ctx.stroke();
  ctx.fillStyle = color;
  pts.forEach((p, j) => { if (p[0] === null) return; ctx.beginPath(); ctx.arc(p[0], p[1], (j % 4 === 0 ? 3.2 : 2.2) * dpr, 0, 7); ctx.fill(); });
}
function label(ctx, text, [x, y], color, dpr) {
  ctx.font = `${11 * dpr}px ui-monospace, Menlo, monospace`;
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(10,12,15,.75)"; ctx.fillRect(x, y - 12 * dpr, w + 8 * dpr, 15 * dpr);
  ctx.fillStyle = color; ctx.fillText(text, x + 4 * dpr, y);
}

// ─────────────────────────────── playback ───────────────────────────────
function bindPlayer() {
  const v = $("#video");
  const fps = () => S.seq?.meta.fps || 30;
  const onFrame = (_, md) => { S.frame = Math.min(Math.round(md.mediaTime * fps()), (S.seq?.meta.num_frames || 1) - 1); draw(); v.requestVideoFrameCallback(onFrame); };
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) v.requestVideoFrameCallback(onFrame);
  else { const loop = () => { S.frame = Math.round(v.currentTime * fps()); draw(); requestAnimationFrame(loop); }; loop(); }
  v.addEventListener("seeked", () => { S.frame = Math.round(v.currentTime * fps() - 0.5); draw(); });
  v.addEventListener("play", () => ($("#play").textContent = "❚❚"));
  v.addEventListener("pause", () => ($("#play").textContent = "▶"));
  $("#play").onclick = () => (v.paused ? v.play() : v.pause());
  $("#prev").onclick = () => step(-1);
  $("#next").onclick = () => step(1);
  $("#rate").onchange = (e) => (v.playbackRate = Number(e.target.value));
  $("#timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect(), m = S.seq.meta;
    seekTime(((e.clientX - r.left) / r.width) * (m.duration_s || m.num_frames / m.fps));
  });
  window.addEventListener("resize", draw);
  document.addEventListener("keydown", (e) => {
    if (!S.seq || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); e.shiftKey ? seekTime(v.currentTime + 1) : step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); e.shiftKey ? seekTime(v.currentTime - 1) : step(-1); }
    else if (e.key === "]" || e.key === "[") {
      const t = v.currentTime, acts = S.seq.actions;
      const a = e.key === "]" ? acts.find((x) => x.start_s > t + 0.01) : [...acts].reverse().find((x) => x.start_s < t - 0.3);
      if (a) seekTime(a.start_s + 0.001);
    }
  });
  $("#clip-run").onclick = runClip;
}
function seekFrame(f) {
  const m = S.seq.meta; f = Math.max(0, Math.min(m.num_frames - 1, f));
  history.replaceState(null, "", `#${m.dataset}/${m.sequence_id}/${f}`);
  $("#video").pause(); S.frame = f; $("#video").currentTime = (f + 0.5) / m.fps; draw();
}
const step = (d) => seekFrame(S.frame + d);
const seekTime = (t) => seekFrame(Math.round(t * S.seq.meta.fps));

// ─────────────────────────────── standard clip panel ───────────────────────────────
async function runClip() {
  const m = S.seq.meta, num = Number($("#clip-num").value) || 8, fps = $("#clip-fps").value;
  const p = new URLSearchParams({ start_s: (S.frame / m.fps).toFixed(3), num });
  if (fps) p.set("fps", fps);
  $("#clip-info").textContent = "loading…";
  const c = await api(`/api/sequence/${m.dataset}/${encodeURIComponent(m.sequence_id)}/clip?${p}`);
  $("#clip-strip").replaceChildren(...c.frame_indices.map((fi, k) => el("figure", {},
    el("img", { src: `/api/frame/${m.dataset}/${encodeURIComponent(m.sequence_id)}/${fi}.jpg?w=240`, loading: "lazy", onclick: () => seekFrame(fi), style: "cursor:pointer" }),
    el("figcaption", {}, `#${fi} · ${c.timestamps[k].toFixed(2)}s`))));
  const rows = Object.entries(c.arrays).map(([k, a]) => {
    const total = a.shape.reduce((x, y) => x * y, 1), cls = a.finite === null ? "" : a.finite > 0 ? "ok" : "nan";
    return el("div", {}, el("span", { class: cls }, k), ` [${a.shape.join("×")}] ${a.dtype}`, a.finite !== null ? el("span", { class: "dim" }, ` ${Math.round((100 * a.finite) / Math.max(total, 1))}% finite`) : null);
  });
  $("#clip-info").replaceChildren(
    el("div", { style: "grid-column:1/-1;margin-bottom:6px" },
      `valid frames → ${Object.entries(c.valid).map(([k, v]) => `${k}: ${v}/${c.frame_indices.length}`).join(" · ")}`,
      c.actions.length ? ` · actions: ${c.actions.map((a) => a.label).join(", ")}` : ""),
    ...rows);
}

init().catch((e) => { document.body.prepend(el("pre", { class: "note" }, String(e))); console.error(e); });
