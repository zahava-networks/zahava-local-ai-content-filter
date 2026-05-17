const ATTR_KEYS = [
  "gender", "age_group", "sleeve_length", "neckline",
  "lower_garment", "lower_length", "fit", "visible_nudity", "shirtless_male",
];

let current = null;
let correctedLabel = null;

const $ = (id) => document.getElementById(id);

async function loadStats() {
  const s = await fetch("/api/stats").then((r) => r.json());
  $("stat-reviewed").textContent = s.reviewed;
  $("stat-pending").textContent = s.pending;
  const rate = s.reviewed > 0 ? ((s.accepted / s.reviewed) * 100).toFixed(0) + "%" : "—";
  $("stat-rate").textContent = rate;
}

async function loadNext() {
  const data = await fetch("/api/next").then((r) => r.json());
  if (data.empty) {
    renderEmpty();
    return;
  }
  current = data;
  correctedLabel = JSON.parse(JSON.stringify(data.ai_label || {}));
  renderItem(data);
  loadStats();
}

function renderEmpty() {
  current = null;
  $("viewer").innerHTML = `
    <div class="empty">
      <div class="big">Queue empty</div>
      <div>Populate via <code>POST /api/populate</code> or push reviewed batch with the button.</div>
    </div>`;
}

function renderItem(d) {
  $("viewer").innerHTML = `
    <div class="image-wrap">
      <img id="image" alt="" />
      <div class="overlay">
        <div class="verdict" id="verdict">…</div>
        <div class="reasoning" id="reasoning"></div>
      </div>
    </div>
    <div class="meta">
      <div class="row"><span class="key">image_id</span><span class="val" id="meta-id">—</span></div>
      <div class="row"><span class="key">confidence</span><span class="val" id="meta-conf">—</span></div>
      <div class="row"><span class="key">flagged</span><span class="val" id="meta-flag">—</span></div>
    </div>`;
  const img = $("image");
  img.src = d.image_url;

  const block = d.ai_label.block === true;
  $("verdict").textContent = block ? "BLOCK" : "ALLOW";
  $("verdict").className = "verdict " + (block ? "block" : "allow");
  $("reasoning").textContent = d.ai_label.reasoning || "—";

  $("meta-id").textContent = d.image_id;
  $("meta-conf").textContent = (d.ai_confidence ?? 0).toFixed(3);
  $("meta-flag").textContent = d.flag_reason || "—";

  renderAttrs(d.ai_label);
}

function renderAttrs(label) {
  const p = label.primary_person || {};
  const wrap = $("attrs");
  wrap.innerHTML = "";
  ATTR_KEYS.forEach((k, i) => {
    const v = p[k];
    if (v === undefined) return;
    const row = document.createElement("div");
    row.className = "attr";
    row.dataset.key = k;
    const violating = isViolating(k, v, p.gender);
    if (violating) row.classList.add("violating");
    row.innerHTML = `
      <span class="name">${k}</span>
      <span class="value" data-role="value">${v}</span>
      <span class="conf">${i + 1}</span>`;
    row.addEventListener("click", () => cycleAttribute(k));
    wrap.appendChild(row);
  });
  const flags = [
    ["romantic_contact", label.romantic_contact],
    ["suggestive_pose", label.suggestive_pose],
  ];
  flags.forEach(([k, v]) => {
    const row = document.createElement("div");
    row.className = "attr" + (v ? " violating" : "");
    row.dataset.key = k;
    row.innerHTML = `
      <span class="name">${k}</span>
      <span class="value">${v ? "true" : "false"}</span>
      <span class="conf">—</span>`;
    row.addEventListener("click", () => toggleBool(k));
    wrap.appendChild(row);
  });
}

const CYCLES = {
  sleeve_length: ["none", "short", "elbow", "three_quarter", "long", "not_visible"],
  neckline: ["modest", "cleavage_visible", "no_top", "not_visible"],
  lower_garment: ["skirt", "pants", "shorts", "swimwear", "underwear", "none", "not_visible"],
  lower_length: ["above_knee", "at_knee", "below_knee", "full", "not_visible"],
  fit: ["loose", "fitted", "tight", "not_visible"],
  visible_nudity: ["none", "partial", "full"],
  gender: ["female", "male", "unknown"],
  age_group: ["adult", "child", "unknown"],
  shirtless_male: [false, true],
};

function cycleAttribute(key) {
  const cycle = CYCLES[key];
  if (!cycle) return;
  const p = correctedLabel.primary_person = correctedLabel.primary_person || {};
  const current = p[key];
  const idx = cycle.indexOf(current);
  p[key] = cycle[(idx + 1) % cycle.length];
  renderAttrs(correctedLabel);
}

function toggleBool(key) {
  correctedLabel[key] = !correctedLabel[key];
  renderAttrs(correctedLabel);
}

function isViolating(key, value, gender) {
  if (key === "visible_nudity" && (value === "partial" || value === "full")) return true;
  if (key === "shirtless_male" && value === true) return true;
  const female = gender === "female" || gender === "unknown";
  if (!female) return false;
  if (key === "sleeve_length" && (value === "none" || value === "short" || value === "elbow")) return true;
  if (key === "neckline" && value === "cleavage_visible") return true;
  if (key === "lower_garment" && ["pants","shorts","swimwear","underwear","none"].includes(value)) return true;
  if (key === "lower_length" && (value === "above_knee" || value === "at_knee")) return true;
  if (key === "fit" && value === "tight") return true;
  return false;
}

async function decide(decision) {
  if (!current) return;
  const notes = $("notes").value || null;
  const payload = {
    image_id: current.image_id,
    decision,
    corrected_label: decision === "correct" ? correctedLabel : null,
    notes,
  };
  await fetch("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  $("notes").value = "";
  loadNext();
}

document.addEventListener("DOMContentLoaded", () => {
  $("btn-accept").addEventListener("click", () => decide("accept"));
  $("btn-reject").addEventListener("click", () => {
    if (correctedLabel) correctedLabel.block = true;
    decide("correct");
  });
  $("btn-skip").addEventListener("click", () => decide("skip"));
  $("btn-bad").addEventListener("click", () => decide("bad_image"));
  $("btn-push").addEventListener("click", async () => {
    const r = await fetch("/api/push", { method: "POST" }).then((x) => x.json());
    alert(`Pushed ${r.pushed} reviews to HF.`);
    loadStats();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA") return;
    if (e.key === "a" || e.key === "A") decide("accept");
    else if (e.key === "r" || e.key === "R") {
      if (correctedLabel) correctedLabel.block = true;
      decide("correct");
    }
    else if (e.key === "b" || e.key === "B") decide("bad_image");
    else if (e.key === " ") { e.preventDefault(); decide("skip"); }
    else if (e.key >= "1" && e.key <= "9") {
      const idx = parseInt(e.key, 10) - 1;
      const keys = ATTR_KEYS;
      const k = keys[idx];
      if (k) cycleAttribute(k);
    }
  });

  loadNext();
});
