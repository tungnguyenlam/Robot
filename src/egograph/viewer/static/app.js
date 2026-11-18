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
const MANO_V = 778;
const KEYFRAME_KINDS = { contact_frame: { name: "contact onset", color: "#ffe14c" }, pnr_frame: { name: "PNR (state change)", color: "#ff5d5d" } };
const CONTACT = "#ff3b6b";

// One definition of each annotation kind, used by the list flags, legend and badges.
const KINDS = [
  { key: "has_hand_pose", letter: "H", label: "hand pose", short: "hand", cls: "h" },
  { key: "has_object_pose", letter: "O", label: "object pose", short: "object", cls: "o" },
  { key: "has_camera_pose", letter: "C", label: "camera pose", short: "camera", cls: "c" },
  { key: "has_actions", letter: "A", label: "actions", short: "actions", cls: "a" },
];

// `needs` = sequence flag the overlay draws from; layers without that data are disabled.
const LAYERS = [
  { key: "hands", label: "hand kps (GT 2D)", color: "var(--right)", on: true, needs: "has_hand_pose" },
  { key: "mano", label: "MANO → 2D", color: "var(--mano)", on: false, needs: "has_hand_pose" },
  { key: "box3d", label: "object 3D box", color: "var(--obj)", on: true, needs: "has_object_pose" },
  { key: "bbox", label: "object 2D box", color: "#d6d6d6", on: true, mod: "object_box" },
  { key: "handbox", label: "hand 2D box", color: "#ffd24c", on: true, mod: "hand_box" },
  { key: "gaze", label: "gaze", color: "#ff4cf0", on: true, mod: "gaze" },
  { key: "mesh", label: "object mesh (CAD)", color: "#7fd1ff", on: true, mod: "mesh" },
  { key: "handmesh", label: "hand mesh (MANO)", color: "#ffb38a", on: true, needs: "has_hand_pose" },
  { key: "contact", label: "contact (derived)", color: "#ff3b6b", on: true, mod: "contact" },
  { key: "keyframes", label: "contact / PNR frames", color: "#ffe14c", on: true, mod: "keyframes" },
  { key: "scene", label: "scene cloud", color: "#9a9aff", on: false, mod: "scene" },
  { key: "axes", label: "object axes", color: "#ff5d5d", on: false, needs: "has_object_pose" },
  { key: "depth", label: "depth (paused)", color: "#3be0c0", on: false, mod: "depth" },
  { key: "seg", label: "masks (paused)", color: "#b04cff", on: false, mod: "seg" },
];
const hasMod = (m, k) => (m.modalities || []).includes(k);
const HAND_SIDES = ["left", "right"];

const S = {
  datasets: [], ds: null, q: "", require: new Set(), mods: new Set(), tags: {}, offset: 0, total: 0,
  seq: null, frame: 0, chunks: new Map(), pending: new Set(), meshes: new Map(), handMesh: new Map(), handFaces: null, scene: null, keyframes: [], layers: Object.fromEntries(LAYERS.map((l) => [l.key, l.on])),
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
  S.q = ""; S.require.clear(); S.mods.clear(); S.tags = {}; $("#search").value = "";
  renderTabs(); renderDatasetCard(); renderFilters(); renderLegend();
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
  // Modality chips: only the ones that actually split this dataset.
  for (const [k, n] of S.ds.modalities || []) {
    if (n <= 0 || n >= S.ds.sequences || ["hand_pose", "object_pose", "camera_pose", "actions"].includes(k)) continue;
    box.append(el("button", {
      class: `chip mod ${S.mods.has(k) ? "on" : ""}`, title: `${n} sequences have ${k}`,
      onclick: (e) => { S.mods.has(k) ? S.mods.delete(k) : S.mods.add(k); e.target.classList.toggle("on"); loadSequences(true); },
    }, k));
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
  S.mods.forEach((r) => p.append("modality", r));
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
  const sub = [t.category_name || t.participant || t.scenario, `${fmt(m.duration_s, 1)}s`, t.num_actions !== undefined ? `${t.num_actions} acts` : null, t.hands?.length ? t.hands.join("+") : null].filter(Boolean);
  return el("li", { "data-id": m.sequence_id, onclick: () => openSequence(m.dataset, m.sequence_id) },
    el("div", { class: "sid" }, m.sequence_id),
    el("div", { class: "sub" }, sub.join(" · "),
      flagRow(m)));
}
function flagRow(m) {
  return el("span", { class: "flags", title: KINDS.map((k) => `${k.label}: ${m[k.key] ? "yes" : "no"}`).join("\n") },
    KINDS.map((k) => el("i", { class: `flag ${m[k.key] ? k.cls : ""}` }, k.letter)));
}
function renderLegend() {
  $("#flag-legend").replaceChildren(...KINDS.map((k) =>
    el("span", { title: k.label }, el("i", { class: `flag ${k.cls}` }, k.letter), k.short)));
}
function markActive() {
  document.querySelectorAll("#seq-list li").forEach((li) => li.classList.toggle("active", li.dataset.id === S.seq?.meta.sequence_id));
}

// ─────────────────────────────── sequence ───────────────────────────────
async function openSequence(dataset, sid, startFrame = 0) {
  const seq = await api(`/api/sequence/${dataset}/${encodeURIComponent(sid)}`);
  S.seq = seq; S.chunks = new Map(); S.pending = new Set(); S.frame = 0; S.handMesh = new Map(); S.scene = null;
  // FHO critical frames (Ego4D): contact onset and point of no return, from the actions.
  S.keyframes = seq.actions.flatMap((a) => Object.entries(KEYFRAME_KINDS).flatMap(([k, d]) => {
    const f = a.extra?.critical_frames?.[k];
    return f === undefined || f === null ? [] : [{ f, kind: k, ...d, label: a.label }];
  })).sort((a, b) => a.f - b.f);
  history.replaceState(null, "", `#${dataset}/${sid}`);
  const m = seq.meta;
  $("#empty").hidden = true; $("#player").hidden = false;
  $("#seq-id").textContent = m.sequence_id;
  const ALL = ["rgb", "depth", "seg", "audio", "imu", "gaze", "scene", "intrinsics", "camera_pose",
    "hand_pose", "hand_box", "object_pose", "object_box", "mesh", "contact", "keyframes", "actions"];
  const mods = new Set(m.modalities?.length ? m.modalities : [
    "rgb", m.has_hand_pose && "hand_pose", m.has_object_pose && "object_pose", m.has_camera_pose && "camera_pose",
    m.has_actions && "actions", m.intrinsics && "intrinsics"].filter(Boolean));
  $("#seq-badges").replaceChildren(...ALL.map((k) => el("span", { class: `badge ${mods.has(k) ? "yes" : ""}` }, `${mods.has(k) ? "✓" : "✗"} ${k.replace("_", " ")}`)));
  S.overlayKey = S.sideKey = null; $("#mod-img").hidden = true; $("#side-img").hidden = true;
  renderImu();
  updateLayers(m);
  $("#stage").style.aspectRatio = $("#side").style.aspectRatio = `${m.width} / ${m.height}`;
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
  })), ...S.keyframes.map((k) => el("div", {
    class: "tl-key", title: `${k.name} · frame ${k.f} · ${k.label}`,
    style: `left:${(k.f / meta.fps / dur) * 100}%;background:${k.color}`,
  })));
  $("#tl-valid").replaceChildren();
}

function renderValidity() {
  // Coverage strip: hands (orange or cyan) over objects (green), from the loaded chunks.
  const { meta } = S.seq, n = meta.num_frames;
  const hasC = [...S.chunks.values()].some((ch) => ch.contact);
  const H = hasC ? 10 : 7;
  const c = el("canvas", { width: 1200, height: H, style: `width:100%;height:${H}px;display:block` });
  $("#tl-valid").style.height = `${H}px`;
  const ctx = c.getContext("2d");
  for (const ch of S.chunks.values()) {
    for (let i = 0; i < ch.end - ch.start; i++) {
      const x = ((ch.start + i) / n) * 1200, w = Math.max(1200 / n, 1);
      if (hasC && HAND_SIDES.some((sd) => contactAt(ch, sd, i))) { ctx.fillStyle = CONTACT; ctx.fillRect(x, 7, w, 3); }
      if (ch.objects.some((o) => o.valid[i])) { ctx.fillStyle = "#a6e35a"; ctx.fillRect(x, 4, w, 3); }
      const l = ch.hands.left?.valid[i] || ch.hands.left?.bbox_2d?.[i]?.[0] != null,
        r = ch.hands.right?.valid[i] || ch.hands.right?.bbox_2d?.[i]?.[0] != null;
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
  svg.replaceChildren(); S.trajXY = null;
  svg.hidden = !traj || !traj.length;
  if (svg.hidden) { $("#traj-note").textContent = "No camera poses for this sequence (run SLAM in Month 2)."; return; }
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
  if (!(m.has_hand_pose || m.has_object_pose || ["hand_box", "object_box", "gaze"].some((k) => hasMod(m, k)))) return;
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
  // depth/seg toggles redraw the paused overlay immediately.
  $("#layers").replaceChildren(...LAYERS.map((l) => el("button", {
    class: `layer ${S.layers[l.key] ? "" : "off"}`, "data-key": l.key,
    onclick: (e) => { S.layers[l.key] = !S.layers[l.key]; e.currentTarget.classList.toggle("off"); draw(); },
  }, el("i", { style: `background:${l.color}` }), l.label)));
  if (S.seq) updateLayers(S.seq.meta);
}
function updateLayers(m) {
  // Grey out overlays this sequence has no data for, and say what *is* shown.
  for (const l of LAYERS) {
    const b = document.querySelector(`#layers [data-key="${l.key}"]`);
    if (!b) continue;
    // `needs` = a has_* flag; `mod` = a modality name from the index (v1.1).
    const ok = l.needs ? !!m[l.needs] : l.mod ? (m.modalities || []).includes(l.mod) : true;
    b.disabled = !ok; b.classList.toggle("na", !ok);
    b.title = ok ? "" : `No ${(l.needs ? l.needs.slice(4) : l.mod).replace("_", " ")} in this sequence`;
  }
  const pose = m.has_hand_pose || m.has_object_pose || (m.modalities || []).some((k) => ["hand_box", "object_box", "gaze"].includes(k));
  $("#overlay-note").textContent = pose ? "" : m.has_actions
    ? "No pose annotations: only action labels (top-left + timeline)"
    : "No annotations for this sequence";
  $("#timeline").classList.toggle("no-pose", !pose);
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
  drawImuHead(t);
  updateModImage(f);
  $("#timeline").classList.toggle("hide-keys", !S.layers.keyframes);
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
  const touch = Object.fromEntries(HAND_SIDES.map((sd) => [sd, S.layers.contact ? contactAt(ch, sd, i) : null]));
  const touched = new Set(Object.values(touch).filter(Boolean).map((c) => c.o));

  if (S.layers.scene && hasMod(m, "scene") && m.intrinsics) drawScene(ctx, ch, i, m, s);
  if ((S.layers.mesh || S.layers.handmesh) && m.intrinsics) {
    // Object and hand triangles share one painter's sort, so hands occlude objects correctly.
    // Opaque faces on an offscreen canvas, then one translucent blit (no alpha build-up).
    const tris = [];
    if (S.layers.mesh) for (const o of ch.objects) if (o.valid[i] && o.mesh_path && o.T_cam_obj?.[i]?.[0]?.[0] != null) meshTris(tris, o, i, m, s);
    if (S.layers.handmesh && m.has_hand_pose) handTris(tris, ch, i, m, s, touch);
    if (tris.length) {
      const mc = S.meshCv || (S.meshCv = document.createElement("canvas"));
      if (mc.width !== cv.width || mc.height !== cv.height) { mc.width = cv.width; mc.height = cv.height; }
      const mctx = mc.getContext("2d"); mctx.clearRect(0, 0, mc.width, mc.height);
      paintTris(mctx, tris);
      ctx.globalAlpha = 0.6; ctx.drawImage(mc, 0, 0); ctx.globalAlpha = 1;
    }
  }
  ch.objects.forEach((o, oi) => {
    if (!o.valid[i]) return;
    if (S.layers.box3d && o.corners_2d[i][0][0] !== null) {
      ctx.strokeStyle = touched.has(oi) ? CONTACT : "#a6e35a"; ctx.lineWidth = touched.has(oi) ? lw * 1.6 : lw;
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
  });
  for (const [side, h] of Object.entries(ch.hands)) {
    const color = side === "left" ? "#4cc3ff" : "#ff8a4c";
    const hb = h.bbox_2d?.[i];
    if (S.layers.handbox && hb && hb[0] !== null) {
      const [x0, y0, x1, y1] = hb.map((x) => x * s);
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      label(ctx, `${side} hand`, [x0, y0 - 6 * dpr], color, dpr);
    }
    if (!h.valid[i]) continue;
    if (S.layers.hands) skeleton(ctx, h.joints_2d[i].map(P), color, lw, dpr);
    if (S.layers.mano && h.mano_2d && h.mano_2d[i][0][0] !== null) skeleton(ctx, h.mano_2d[i].map(P), "#f05bd8", lw * 0.7, dpr * 0.7);
    const c = touch[side], w = h.joints_2d[i][0];
    if (c && w[0] !== null) label(ctx, `${side} ● ${ch.objects[c.o].label} ${(c.d * 100).toFixed(1)} cm`, [w[0] * s + 10 * dpr, w[1] * s + 18 * dpr], CONTACT, dpr);
  }
  drawGaze(ctx, ch, i, s, dpr);
  drawKeyframes(ctx, f, m, dpr);
}

// Smallest hand-object distance of one hand at chunk frame i, if below the contact threshold.
function contactAt(ch, side, i) {
  const D = ch.contact?.dist?.[side];
  if (!D) return null;
  let best = null;
  D.forEach((row, o) => { const d = row[i]; if (d !== null && d < ch.contact.threshold && (!best || d < best.d)) best = { o, d }; });
  return best;
}

// Ego4D FHO key frames: a banner on the frames within ±2 of a contact onset or PNR label.
function drawKeyframes(ctx, f, m, dpr) {
  if (!S.layers.keyframes || !S.keyframes.length) return;
  const near = S.keyframes.filter((k) => Math.abs(k.f - f) <= 2);
  const when = (k) => (k.f === f ? "" : k.f > f ? ` (in ${k.f - f} f)` : ` (${f - k.f} f ago)`);
  near.forEach((k, j) => label(ctx, `◆ ${k.name}${when(k)} · ${k.label}`,
    [12 * dpr, (48 + 18 * j) * dpr], k.color, dpr));
}

// Meshes: vertices in the camera frame -> projected, flat-shaded triangles collected into one
// list, then painted back to front. Faces closer than 2 cm to the camera are dropped.
function pushTris(tris, X, faces, K, s, h, sat) {
  const n = X.length / 3, U = new Float32Array(n * 2);
  for (let k = 0; k < n; k++) {
    const z = X[3 * k + 2];
    U[2 * k] = (K[0][0] * X[3 * k] / z + K[0][2]) * s; U[2 * k + 1] = (K[1][1] * X[3 * k + 1] / z + K[1][2]) * s;
  }
  for (let j = 0; j < faces.length / 3; j++) {
    const a = faces[3 * j], b = faces[3 * j + 1], c = faces[3 * j + 2], p = 3 * a, q = 3 * b, r = 3 * c;
    if (X[p + 2] < 0.02 || X[q + 2] < 0.02 || X[r + 2] < 0.02) continue;
    const ux = X[q] - X[p], uy = X[q + 1] - X[p + 1], uz = X[q + 2] - X[p + 2];
    const vx = X[r] - X[p], vy = X[r + 1] - X[p + 1], vz = X[r + 2] - X[p + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const cx = (X[p] + X[q] + X[r]) / 3, cy = (X[p + 1] + X[q + 1] + X[r + 1]) / 3, cz = (X[p + 2] + X[q + 2] + X[r + 2]) / 3;
    const nn = Math.hypot(nx, ny, nz) * Math.hypot(cx, cy, cz) || 1;
    tris.push({ d: cz, sh: Math.abs(nx * cx + ny * cy + nz * cz) / nn, U, a, b, c, h, sat });
  }
}
function paintTris(ctx, tris) {
  ctx.lineWidth = 0.6;
  tris.sort((x, y) => y.d - x.d);
  for (const t of tris) {
    const U = t.U;
    ctx.fillStyle = ctx.strokeStyle = `hsl(${t.h} ${t.sat}% ${Math.round(25 + 50 * t.sh)}%)`;
    ctx.beginPath(); ctx.moveTo(U[2 * t.a], U[2 * t.a + 1]); ctx.lineTo(U[2 * t.b], U[2 * t.b + 1]); ctx.lineTo(U[2 * t.c], U[2 * t.c + 1]); ctx.closePath();
    ctx.fill(); ctx.stroke();  // stroke hides anti-aliasing seams between faces
  }
}

// CAD mesh (object frame, fetched once per mesh) placed by T_cam_obj.
function meshTris(tris, o, i, m, s) {
  const key = `${m.dataset}:${o.mesh_path}`, mesh = S.meshes.get(key);
  if (!mesh) {
    S.meshes.set(key, "pending");
    api(`/api/mesh/${m.dataset}?path=${encodeURIComponent(o.mesh_path)}`)
      .then((r) => { S.meshes.set(key, { v: Float32Array.from(r.vertices), f: Uint32Array.from(r.faces) }); draw(); })
      .catch(() => S.meshes.set(key, "missing"));
    return;
  }
  if (typeof mesh === "string") return;
  const T = o.T_cam_obj[i], n = mesh.v.length / 3, X = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const a = mesh.v[3 * k], b = mesh.v[3 * k + 1], c = mesh.v[3 * k + 2];
    for (let r = 0; r < 3; r++) X[3 * k + r] = T[r][0] * a + T[r][1] * b + T[r][2] * c + T[r][3];
  }
  pushTris(tris, X, mesh.f, m.intrinsics, s, hue(o.label), 70);
}

// MANO hand mesh: per chunk, int16 (2, T, 778, 3) camera-frame vertices in 0.1 mm from the server.
function handMeshChunk(ch) {
  const m = S.seq.meta, hm = S.handMesh.get(ch.start);
  if (hm !== undefined) return typeof hm === "string" ? null : hm;
  S.handMesh.set(ch.start, "pending");
  const sid = m.sequence_id, mine = S.handMesh;
  (S.handFaces ? Promise.resolve() : api("/api/hand_faces").then((r) => { S.handFaces = { left: Uint32Array.from(r.left), right: Uint32Array.from(r.right) }; }))
    .then(() => fetch(`/api/sequence/${m.dataset}/${encodeURIComponent(sid)}/handmesh?start=${ch.start}&end=${ch.end}`))
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then((b) => { mine.set(ch.start, new Int16Array(b)); if (S.handMesh === mine) draw(); })
    .catch(() => mine.set(ch.start, "missing"));
  return null;
}
function handTris(tris, ch, i, m, s, touch) {
  const V = handMeshChunk(ch);
  if (!V || !S.handFaces) return;
  const T = V.length / (2 * MANO_V * 3);
  HAND_SIDES.forEach((side, si) => {
    const off = (si * T + i) * MANO_V * 3;
    if (V[off] === -32768) return;
    const X = new Float32Array(MANO_V * 3);
    for (let k = 0; k < X.length; k++) X[k] = V[off + k] * 1e-4;
    const [h, sat] = touch[side] ? [340, 90] : side === "left" ? [197, 85] : [22, 85];
    pushTris(tris, X, S.handFaces[side], m.intrinsics, s, h, sat);
  });
}

// Static scene cloud (HOI4D 3Dseg, world frame) projected with the frame's camera pose into a
// half-resolution z-buffered image; coloured by semantic label.
function ensureScene() {
  const m = S.seq.meta;
  if (S.scene) return;
  S.scene = "pending";
  const sid = m.sequence_id;
  fetch(`/api/scene/${m.dataset}/${encodeURIComponent(sid)}?n=60000`)
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then((b) => {
      const N = new Uint32Array(b, 0, 1)[0];
      const sc = { N, xyz: new Float32Array(b, 4, 3 * N), label: new Int16Array(b, 4 + 12 * N, N), rgb: new Uint8Array(b, 4 + 14 * N, 3 * N) };
      sc.color = new Uint8Array(3 * N);
      for (let k = 0; k < N; k++) {
        const [r, g, bl] = hslRgb((sc.label[k] * 137.508) % 360, 0.75, 0.55);
        sc.color[3 * k] = r; sc.color[3 * k + 1] = g; sc.color[3 * k + 2] = bl;
      }
      if (S.seq?.meta.sequence_id === sid) { S.scene = sc; draw(); }
    })
    .catch(() => { if (S.seq?.meta.sequence_id === sid) S.scene = "missing"; });
}
function hslRgb(h, sat, l) {
  const k = (n) => (n + h / 30) % 12, a = sat * Math.min(l, 1 - l);
  const f = (n) => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))));
  return [f(0), f(8), f(4)];
}
function drawScene(ctx, ch, i, m, s) {
  const Tc = ch.T_cam_world?.[i];
  if (!Tc || Tc[0][0] === null) return;
  ensureScene();
  const sc = S.scene;
  if (!sc || typeof sc === "string") return;
  const cv = ctx.canvas, W = Math.ceil(cv.width / 2), H = Math.ceil(cv.height / 2);
  let B = S.sceneBuf;
  if (!B || B.W !== W || B.H !== H) {
    const off = document.createElement("canvas"); off.width = W; off.height = H;
    B = S.sceneBuf = { W, H, off, img: new ImageData(W, H), z: new Float32Array(W * H) };
  }
  B.z.fill(Infinity); B.img.data.fill(0);
  const K = m.intrinsics, f = s / 2, X = sc.xyz, D = B.img.data, Z = B.z;
  for (let k = 0; k < sc.N; k++) {
    const a = X[3 * k], b = X[3 * k + 1], c = X[3 * k + 2];
    const z = Tc[2][0] * a + Tc[2][1] * b + Tc[2][2] * c + Tc[2][3];
    if (z < 0.05) continue;
    const x = Tc[0][0] * a + Tc[0][1] * b + Tc[0][2] * c + Tc[0][3], y = Tc[1][0] * a + Tc[1][1] * b + Tc[1][2] * c + Tc[1][3];
    const u = Math.floor((K[0][0] * x / z + K[0][2]) * f), v = Math.floor((K[1][1] * y / z + K[1][2]) * f);
    for (let dv = 0; dv < 2; dv++) for (let du = 0; du < 2; du++) {
      const uu = u + du, vv = v + dv;
      if (uu < 0 || vv < 0 || uu >= W || vv >= H) continue;
      const p = vv * W + uu;
      if (z >= Z[p]) continue;
      Z[p] = z; D[4 * p] = sc.color[3 * k]; D[4 * p + 1] = sc.color[3 * k + 1]; D[4 * p + 2] = sc.color[3 * k + 2]; D[4 * p + 3] = 255;
    }
  }
  B.off.getContext("2d").putImageData(B.img, 0, 0);
  ctx.globalAlpha = 0.5; ctx.imageSmoothingEnabled = false; ctx.drawImage(B.off, 0, 0, cv.width, cv.height); ctx.globalAlpha = 1;
}

function drawGaze(ctx, ch, i, s, dpr) {
  const g = ch.gaze_2d?.[i];
  if (!S.layers.gaze || !g || g[0] === null) return;
  ctx.strokeStyle = "#ff4cf0"; ctx.lineWidth = 2.5 * dpr;
  ctx.beginPath(); ctx.arc(g[0] * s, g[1] * s, 14 * dpr, 0, 7); ctx.stroke();
  ctx.beginPath(); ctx.arc(g[0] * s, g[1] * s, 3 * dpr, 0, 7); ctx.fillStyle = "#ff4cf0"; ctx.fill();
}

// Depth / mask images: one PNG per frame from the server, only while paused (decoding a
// 1080p 16-bit depth frame per video frame would not keep up with playback). Normally drawn
// over the video; in split view the side pane shows depth (or masks) at full opacity, and if
// both layers are on, masks stay on the video.
function updateModImage(f) {
  const m = S.seq.meta, paused = $("#video").paused;
  const want = (k) => S.layers[k] && hasMod(m, k);
  let over = want("seg") ? "seg" : want("depth") ? "depth" : null, side = null;
  if (S.split) { side = want("depth") ? "depth" : over; if (over === side) over = null; }
  loadModImage($("#mod-img"), "overlay", over, f, paused);
  loadModImage($("#side-img"), "side", side, f, paused);
  if (!S.split) return;
  const avail = ["depth", "seg"].filter((k) => hasMod(m, k));
  $("#side-note").textContent = !avail.length ? "No depth or masks in this sequence"
    : !side ? `Turn on the ${avail.map((k) => (k === "depth" ? "“depth (paused)”" : "“masks (paused)”")).join(" or ")} layer`
    : !paused ? `Pause to show ${side === "depth" ? "depth" : "masks"}` : "";
  const leg = $("#side-legend");
  if (side === "depth" && paused) { if (!leg.firstChild) leg.replaceChildren("0.2 m", el("span", { class: "bar" }), "2.0 m"); }
  else leg.replaceChildren();
}
function loadModImage(img, slot, kind, f, paused) {
  const k = `${slot}Key`, t = `${slot}Timer`, m = S.seq.meta;
  if (!kind || !paused) { img.hidden = true; S[k] = null; return; }
  const key = `${m.sequence_id}:${kind}:${f}`;
  if (S[k] === key) return;
  S[k] = key;
  clearTimeout(S[t]);
  S[t] = setTimeout(() => {
    img.onload = () => { if (S[k] === key) img.hidden = false; };
    img.src = `/api/modality/${m.dataset}/${encodeURIComponent(m.sequence_id)}/${kind}/${f}.png?w=${slot === "side" ? 1280 : 960}`;
  }, 120);
}
function setSplit(on) {
  S.split = on;
  $("#side").hidden = !on;
  $(".stage-wrap").classList.toggle("split", on);
  $("#split").classList.toggle("on", on);
  // Turning split on with no depth/mask layer active: switch depth on so the pane isn't empty.
  if (on && S.seq && !S.layers.depth && !S.layers.seg && hasMod(S.seq.meta, "depth")) {
    S.layers.depth = true; document.querySelector('#layers [data-key="depth"]')?.classList.remove("off");
  }
  draw();
}

// IMU panel: accelerometer (top) and gyroscope (bottom), bin-averaged over the sequence.
async function renderImu() {
  const card = $("#imu-card"), m = S.seq.meta;
  S.imu = null;
  if (!hasMod(m, "imu")) { card.hidden = true; return; }
  card.hidden = false;
  $("#imu-note").textContent = "loading (the first open of an EPIC video parses its GoPro telemetry)…";
  $("#imu-plot").getContext("2d").clearRect(0, 0, 1e4, 1e4);
  try {
    const d = await api(`/api/sequence/${m.dataset}/${encodeURIComponent(m.sequence_id)}/imu?points=1200`);
    if (S.seq?.meta.sequence_id !== m.sequence_id) return;
    S.imu = d;
    $("#imu-note").textContent = d.t.length ? `frame: ${d.frame} · accel m/s² (top) · gyro rad/s (bottom) · x red, y green, z blue` : "empty";
    drawImuHead(S.frame / m.fps);
  } catch (e) { $("#imu-note").textContent = String(e); }
}
function drawImuHead(t) {
  const d = S.imu, cv = $("#imu-plot");
  if (!d || !d.t.length || cv.closest("[hidden]")) return;
  const W = (cv.width = cv.clientWidth * (window.devicePixelRatio || 1)), H = (cv.height = 160 * (window.devicePixelRatio || 1));
  const ctx = cv.getContext("2d"), dur = S.seq.meta.duration_s || d.t[d.t.length - 1];
  const X = (tt) => (tt / dur) * W;
  const plot = (arr, y0, h) => {
    const vals = arr.flat().filter((x) => x !== null), lo = Math.min(...vals), hi = Math.max(...vals), sp = hi - lo || 1;
    ["#ff5d5d", "#5dff7a", "#5d8bff"].forEach((c, k) => {
      ctx.strokeStyle = c; ctx.lineWidth = 1; ctx.beginPath(); let pen = false;
      arr.forEach((v, j) => { if (v[k] === null) { pen = false; return; } const y = y0 + h - ((v[k] - lo) / sp) * h; pen ? ctx.lineTo(X(d.t[j]), y) : ctx.moveTo(X(d.t[j]), y); pen = true; });
      ctx.stroke();
    });
  };
  plot(d.accel, 2, H / 2 - 6); plot(d.gyro, H / 2 + 4, H / 2 - 6);
  ctx.strokeStyle = "#e8b04a"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(t), 0); ctx.lineTo(X(t), H); ctx.stroke();
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
  v.addEventListener("pause", () => { $("#play").textContent = "▶"; draw(); });
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
    } else if (e.key === "v" && !e.ctrlKey && !e.metaKey) setSplit(!S.split);
  });
  $("#clip-run").onclick = runClip;
  $("#split").onclick = () => setSplit(!S.split);
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
