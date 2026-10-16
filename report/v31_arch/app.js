const D = JSON.parse(document.getElementById("data").textContent);
const QD = JSON.parse(document.getElementById("qdata").textContent);
const A = D.attrs;
const COL = {Black:"đen",Blue:"xanh dương",Brown:"nâu",Green:"xanh lá",Grey:"xám",Orange:"cam",Pink:"hồng",Purple:"tím",Red:"đỏ",White:"trắng",Yellow:"vàng",Other:"màu khác"};
const VN = {"Age-Young":"Trẻ","Age-Adult":"Người lớn","Age-Old":"Người già","Gender-Female":"Nữ","Hair-Length-Short":"Tóc ngắn",
  "Hair-Length-Long":"Tóc dài","Hair-Length-Bald":"Hói","UpperBody-Length-Short":"Áo ngắn tay","LowerBody-Length-Short":"Quần/váy ngắn",
  "LowerBody-Type-Trousers&Shorts":"Quần","LowerBody-Type-Skirt&Dress":"Váy","Accessory-Backpack":"Ba lô","Accessory-Bag":"Túi",
  "Accessory-Glasses-Normal":"Kính","Accessory-Glasses-Sun":"Kính râm","Accessory-Hat":"Mũ"};
const vn = a => VN[a] || (a.startsWith("UpperBody-Color-") ? "Áo " + COL[a.slice(16)] : a.startsWith("LowerBody-Color-") ? "Quần " + COL[a.slice(16)] : a);
const fmt = (x, k = 2) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(k).replace(".", ",");
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("dg");
function el(tag, at = {}, parent = svg, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(at)) {
    if (k === "style") Object.assign(e.style, v); else e.setAttribute(k, v);
  }
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e); return e;
}
const g = (id, cls = "") => el("g", {id, class: cls});
let state = {mode: "inf", step: 0, attr: A.indexOf("Accessory-Hat"), playing: false};

/* ---------- static diagram ---------- */
const P = {raw: [22, 232, 54, 158], inp: [108, 238, 72, 144], gh: [226, 46, 84, 168], gc: [226, 404, 84, 168],
  sh: [346, 48, 164], sc: [346, 406, 164], ph: [584, 46, 84, 168], pc: [584, 404, 84, 168],
  hH: [862, 28, 344, 176], hJ: [862, 222, 344, 176], hC: [862, 416, 344, 176]};
const defs = el("defs");
const pat = el("pattern", {id: "hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"}, defs);
el("rect", {width: 5, height: 5, style: {fill: "var(--soft)"}}, pat);
el("line", {x1: 0, y1: 0, x2: 0, y2: 5, style: {stroke: "var(--frozen)", strokeWidth: 2}}, pat);
const mk = el("marker", {id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"}, defs);
el("path", {d: "M0,1 L9,5 L0,9 z", style: {fill: "var(--muted)"}}, mk);

// flow paths (drawn first, under everything)
const FLOW = {
  raw_in: "M78,311 L106,311", in_gh: "M180,292 C204,292 200,130 224,130", in_gc: "M180,330 C204,330 200,488 224,488",
  gh_sh: "M310,130 L344,130", gc_sc: "M310,488 L344,488", sh_ph: "M558,130 L582,130", sc_pc: "M558,488 L582,488",
  ph_hh: "M668,116 L860,116", ph_pj: "M668,170 C688,170 672,288 692,288", pj_j: "M832,288 L860,288",
  pc_pj: "M668,446 C688,446 672,336 692,336", qj_j: "M832,336 L860,336", pc_hc: "M668,504 L860,504",
  hh_v: "M1206,116 L1220,116", hj_v: "M1206,310 L1220,310", hc_v: "M1206,504 L1220,504",
  v1_m: "M1240,118 C1266,118 1268,236 1290,240", v2_m: "M1240,310 L1288,310", v3_m: "M1240,502 C1266,502 1268,384 1290,380",
  m_o: "M1312,310 L1338,310", o_l: "M1440,310 L1458,310"};
const flowG = g("flows");
const flowEl = {};
for (const [k, d] of Object.entries(FLOW)) flowEl[k] = el("path", {d, class: "flowpath" + (["v1_m", "v2_m", "v3_m", "m_o", "o_l"].includes(k) ? " infonly" : ""), "marker-end": "url(#arr)"}, flowG);

// raw + input
const gRaw = g("g-raw");
el("image", {href: D.raw, x: P.raw[0], y: P.raw[1], width: P.raw[2], height: P.raw[3], preserveAspectRatio: "none"}, gRaw);
el("rect", {x: P.raw[0], y: P.raw[1], width: P.raw[2], height: P.raw[3], rx: 3, style: {fill: "none", stroke: "var(--line)"}}, gRaw);
el("text", {x: 49, y: 214, "text-anchor": "middle", class: "t-lbl"}, gRaw, "Ảnh gốc");
el("text", {x: 49, y: 408, "text-anchor": "middle", class: "t-sm"}, gRaw, `${D.raw_size[0]}×${D.raw_size[1]} px`);
const gIn = g("g-input");
el("image", {href: D.input, x: P.inp[0], y: P.inp[1], width: P.inp[2], height: P.inp[3], preserveAspectRatio: "none"}, gIn);
el("rect", {x: P.inp[0], y: P.inp[1], width: P.inp[2], height: P.inp[3], rx: 3, style: {fill: "none", stroke: "var(--ink)"}}, gIn);
el("text", {x: 144, y: 400, "text-anchor": "middle", class: "t-lbl"}, gIn, "Đầu vào");
el("text", {x: 144, y: 415, "text-anchor": "middle", class: "t-sm"}, gIn, "3×224×112, [0, 1]");
const gAug = g("g-aug", "trainonly");
el("rect", {x: 92, y: 168, width: 108, height: 50, rx: 8, style: {fill: "var(--grad-soft)", stroke: "var(--grad)"}}, gAug);
el("text", {x: 146, y: 188, "text-anchor": "middle", class: "t-lbl"}, gAug, "Tăng cường");
el("text", {x: 146, y: 205, "text-anchor": "middle", class: "t-sm"}, gAug, "lật · cắt · màu · xoá");

// patch grids
function gridBlock(id, box, cols, rows, color, label1, label2, ly) {
  const gg = g(id);
  el("image", {href: D.input, x: box[0], y: box[1], width: box[2], height: box[3], preserveAspectRatio: "none"}, gg);
  const cw = box[2] / cols, ch = box[3] / rows;
  for (let i = 0; i <= cols; i++) el("line", {x1: box[0] + i * cw, y1: box[1], x2: box[0] + i * cw, y2: box[1] + box[3], style: {stroke: color, strokeWidth: .8, opacity: .9}}, gg);
  for (let j = 0; j <= rows; j++) el("line", {x1: box[0], y1: box[1] + j * ch, x2: box[0] + box[2], y2: box[1] + j * ch, style: {stroke: color, strokeWidth: .8, opacity: .9}}, gg);
  el("text", {x: box[0] + box[2] / 2, y: ly, "text-anchor": "middle", class: "t-lbl", style: {fill: color}}, gg, label1);
  el("text", {x: box[0] + box[2] / 2, y: ly + 15, "text-anchor": "middle", class: "t-sm"}, gg, label2);
  return gg;
}
gridBlock("g-gh", P.gh, 7, 14, "var(--hplus)", "Mảnh 16×16", "14×7 = 98 mảnh", 34 - 4).querySelector("text").setAttribute("y", 26);
document.querySelector("#g-gh .t-sm").setAttribute("y", 40);
gridBlock("g-gc", P.gc, 8, 16, "var(--clip)", "Mảnh 14×14", "16×8 = 128 mảnh", 588);
el("text", {x: 268, y: 232, "text-anchor": "middle", class: "t-sm", style: {fill: "var(--hplus)"}}, document.getElementById("g-gh"), "+ CLS + 4 register = 103 token");
el("text", {x: 268, y: 393, "text-anchor": "middle", class: "t-sm", style: {fill: "var(--clip)"}}, document.getElementById("g-gc"), "+ CLS = 129 token");

// backbone stacks
function stack(id, x0, y, h, n, nFrozen, color, title, sub, ty) {
  const gg = g(id);
  const W = 211.2, gap = 1.6, sw = W / n - gap;
  const slabs = [];
  for (let i = 0; i < n; i++) {
    const frozen = i < nFrozen;
    slabs.push(el("rect", {x: x0 + i * (sw + gap), y, width: sw, height: h, rx: 1.5,
      style: {fill: frozen ? "url(#hatch)" : color, opacity: frozen ? 1 : .82}, "data-frozen": frozen ? 1 : 0}, gg));
  }
  el("text", {x: x0, y: ty, class: "t-h"}, gg, title);
  el("text", {x: x0, y: ty + 15, class: "t-sm"}, gg, sub);
  const scan = el("rect", {x: x0 - 2, y: y - 3, width: sw + 4, height: h + 6, rx: 2, style: {fill: "none", stroke: "var(--ink)", strokeWidth: 1.5, opacity: 0}}, gg);
  const ctr = el("text", {x: x0 + W, y: ty + 15, "text-anchor": "end", class: "t-val", style: {opacity: 0}}, gg, "");
  return {gg, slabs, scan, ctr, x0, sw, gap, n};
}
const SH = stack("g-sh", P.sh[0], P.sh[1], P.sh[2], D.blocks.h, 8, "var(--hplus)", "ViT-H+/16 · DINOv3", "32 khối · 840M · 1280 chiều", 26);
const SC = stack("g-sc", P.sc[0], P.sc[1], P.sc[2], D.blocks.c, 8, "var(--clip)", "ViT-L/14 · CLIP OpenAI", "24 khối · 303M · 1024 chiều", 588);
SC.gg.querySelector(".t-h").setAttribute("y", 600); SC.gg.querySelector(".t-sm").setAttribute("y", 614);
SH.ctr.setAttribute("y", 228); SC.ctr.setAttribute("y", 398);
const mid = g("g-mid");
el("text", {x: 452, y: 300, "text-anchor": "middle", class: "t-sm infonly"}, mid, "trọng số EMA · bf16");
el("text", {x: 452, y: 316, "text-anchor": "middle", class: "t-sm infonly"}, mid, "hai backbone chạy trên cùng ảnh");
el("text", {x: 452, y: 288, "text-anchor": "middle", class: "t-lbl trainonly", style: {fill: "var(--frozen)"}}, mid, "8 khối đầu đóng băng (312M)");
el("text", {x: 452, y: 305, "text-anchor": "middle", class: "t-sm trainonly"}, mid, "lr khối cuối 5e-5, ×0,75 mỗi khối xuống");
el("text", {x: 452, y: 321, "text-anchor": "middle", class: "t-sm trainonly"}, mid, "gradient checkpointing");
const ema = el("g", {id: "g-ema", class: "trainonly"}, mid);
el("rect", {x: 392, y: 334, width: 120, height: 26, rx: 13, style: {fill: "var(--joint-soft)", stroke: "var(--joint)"}}, ema);
el("text", {x: 452, y: 351, "text-anchor": "middle", class: "t-sm", style: {fill: "var(--joint)"}}, ema, "AdamW + EMA 0,9995");

// token maps (PCA RGB)
function pcaBlock(id, box, cols, rows, rgb, color, l1, l2, ly) {
  const gg = g(id); const cw = box[2] / cols, ch = box[3] / rows;
  rgb.forEach((c, i) => el("rect", {x: box[0] + (i % cols) * cw, y: box[1] + Math.floor(i / cols) * ch, width: cw + .3, height: ch + .3,
    style: {fill: `rgb(${c[0]},${c[1]},${c[2]})`}}, gg));
  el("rect", {x: box[0], y: box[1], width: box[2], height: box[3], style: {fill: "none", stroke: color, strokeWidth: 1.5}}, gg);
  el("text", {x: box[0] + box[2] / 2, y: ly, "text-anchor": "middle", class: "t-lbl", style: {fill: color}}, gg, l1);
  el("text", {x: box[0] + box[2] / 2, y: ly + 15, "text-anchor": "middle", class: "t-sm"}, gg, l2);
  return gg;
}
const vh = Math.round(D.pca_var_h.reduce((a, b) => a + b) * 100), vc = Math.round(D.pca_var_c.reduce((a, b) => a + b) * 100);
pcaBlock("g-ph", P.ph, 7, 14, D.pca_h, "var(--hplus)", "Token ra", `PCA→RGB · ${vh}% phương sai`, 26);
pcaBlock("g-pc", P.pc, 8, 16, D.pca_c, "var(--clip)", "Token ra", `PCA→RGB · ${vc}% phương sai`, 588);
document.querySelector("#g-ph .t-sm").setAttribute("y", 40);

// projections into the joint space
const gProj = g("g-proj");
el("rect", {x: 692, y: 236, width: 140, height: 144, rx: 10, style: {fill: "var(--paper)", stroke: "var(--joint)", strokeDasharray: "4 3"}}, gProj);
el("text", {x: 762, y: 256, "text-anchor": "middle", class: "t-lbl", style: {fontSize: "12.5px"}}, gProj, "Chiếu chung");
for (const [y, a, b, c] of [[276, "H+", "1280 → 1024", "var(--hplus)"], [324, "CLIP", "1024 → 1024", "var(--clip)"]]) {
  el("rect", {x: 700, y, width: 124, height: 24, rx: 5, style: {fill: c, opacity: .14}}, gProj);
  el("rect", {x: 700, y, width: 4, height: 24, rx: 2, style: {fill: c}}, gProj);
  el("text", {x: 710, y: y + 16, class: "t-sm", style: {fill: c, fontWeight: 600}}, gProj, a);
  el("text", {x: 818, y: y + 16, "text-anchor": "end", class: "t-sm", style: {fill: "var(--ink)"}}, gProj, b);
}
el("text", {x: 762, y: 314, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px"}}, gProj, "1 lớp tuyến tính / bên");
el("text", {x: 762, y: 366, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px"}}, gProj, "+ nhãn nguồn H+ / CLIP");

// heads: the query-head architecture drawn inside each box
function blk(gg, x, y, w, h, lines, stroke, fill = "var(--paper)") {
  el("rect", {x, y, width: w, height: h, rx: 7, style: {fill, stroke, strokeWidth: 1.2}}, gg);
  lines.forEach((t, i) => el("text", {x: x + w / 2, y: y + h / 2 + (i - (lines.length - 1) / 2) * 13 + 4, "text-anchor": "middle", class: "t-sm",
    style: {fill: "var(--ink)", fontSize: "10.5px"}}, gg, t));
}
function iarr(gg, d) { el("path", {d, style: {fill: "none", stroke: "var(--muted)", strokeWidth: 1.3}, "marker-end": "url(#arr)"}, gg); }
function headBox(id, box, color, soft, title, sub, kind) {
  const gg = g(id), [x, y, w, h] = box, c = y + 88;
  el("rect", {x, y, width: w, height: h, rx: 12, style: {fill: soft, stroke: color, strokeWidth: 1.4}}, gg);
  el("text", {x: x + 12, y: y + 22, class: "t-h"}, gg, title);
  el("text", {x: x + 12, y: y + 38, class: "t-sm", style: {fontSize: "10.5px"}}, gg, sub);
  // inputs: tokens become keys / values, the 40 learned queries ask the questions
  blk(gg, x + 8, kind === "j" ? c - 30 : c - 11, 58, kind === "j" ? 76 : 22, kind === "j" ? ["228", "token", "K, V"] : ["token K,V"], color);
  el("rect", {x: x + 8, y: y + 126, width: 58, height: 26, rx: 7, style: {fill: "var(--paper)", stroke: color}}, gg);
  for (let i = 0; i < 40; i++) el("rect", {x: x + 12 + i * 1.25, y: y + 130, width: .8, height: 8, style: {fill: color}}, gg);
  el("text", {x: x + 37, y: y + 149, "text-anchor": "middle", class: "t-sm", style: {fontSize: "9.5px", fill: "var(--ink)"}}, gg, "40 query");
  blk(gg, x + 82, y + 62, 92, 92, ["Cross-", "attention", "8 đầu", "softmax", "trên token"], color);
  iarr(gg, `M${x + 66},${c} L${x + 80},${c}`); iarr(gg, `M${x + 66},${y + 139} L${x + 80},${y + 139}`);
  blk(gg, x + 188, c - 22, 56, 44, ["+ dư", "MLP", "LN"], color);
  iarr(gg, `M${x + 174},${c} L${x + 186},${c}`);
  el("rect", {x: x + 256, y: c - 26, width: 22, height: 52, rx: 4, style: {fill: "var(--paper)", stroke: color}}, gg);
  for (let i = 0; i < 40; i++) el("rect", {x: x + 259, y: c - 23 + i * 1.15, width: 16, height: .7, style: {fill: color}}, gg);
  el("text", {x: x + 267, y: c + 38, "text-anchor": "middle", class: "t-sm", style: {fontSize: "9.5px"}}, gg, "40 vector");
  iarr(gg, `M${x + 244},${c} L${x + 254},${c}`);
  blk(gg, x + 288, c - 18, 48, 36, ["· w_a", "+ b_a"], color);
  iarr(gg, `M${x + 278},${c} L${x + 286},${c}`);
  return gg;
}
const HH = headBox("g-hh", P.hH, "var(--hplus)", "var(--hplus-soft)", "Head H+", "đọc 99 token × 1280 (CLS + 98 mảnh)", "h");
const HJ = headBox("g-hj", P.hJ, "var(--joint)", "var(--joint-soft)", "Query chung", "đọc 228 token × 1024 (99 H+ + 129 CLIP)", "j");
const HC = headBox("g-hc", P.hC, "var(--clip)", "var(--clip-soft)", "Head CLIP + chữ", "đọc 129 token × 1024 (CLS + 128 mảnh)", "c");
// query chung: which backbone each of the 8 attention heads reads, for the selected attribute
const routeCells = [];
for (let k = 0; k < 8; k++) routeCells.push(el("rect", {x: 952 + k * 11, y: 382, width: 9, height: 9, rx: 2, style: {fill: "var(--line)"}}, HJ));
el("text", {x: 1044, y: 390, class: "t-sm", style: {fontSize: "10px"}}, HJ, "nguồn của 8 đầu");
// head CLIP: the language path joins the output
const textPill = el("g", {}, HC);
el("rect", {x: 1050, y: 542, width: 148, height: 40, rx: 7, style: {fill: "var(--paper)", stroke: "var(--clip)", strokeDasharray: "3 2"}}, textPill);
el("text", {x: 1058, y: 557, class: "t-sm", style: {fontSize: "10px"}}, textPill, "+ phần chữ: CLS khớp");
el("text", {x: 1058, y: 572, class: "t-sm", style: {fontSize: "10px"}}, textPill, "câu mô tả, cộng =");
const textVal = el("text", {x: 1192, y: 572, "text-anchor": "end", class: "t-val", style: {fontSize: "11.5px", fill: "var(--clip)"}}, textPill, "");
el("path", {d: "M1174,542 L1174,528", style: {fill: "none", stroke: "var(--clip)", strokeWidth: 1.3}, "marker-end": "url(#arr)"}, HC);
const attrLbl = [];
const mH = [], mC = [], mJh = [], mJc = [];

// output stage (drawn by drawOutputs): 3 columns of 40 logits -> mean per attribute -> sigmoid -> threshold -> labels
const gVals = g("g-vals"), gLoss = g("g-loss", "trainonly"), gMean = g("g-mean", "infonly"), gOut = g("g-out", "infonly");
const COLS = {h: [1222, 40, "var(--hplus)", "z_H"], j: [1222, 232, "var(--joint)", "z_J"], c: [1222, 428, "var(--clip)", "z_C"]};
const ROW = 3.9;                     // height of one attribute row in a head's logit column (40 rows = 156)
function cellFill(v, col) { const a = Math.min(Math.abs(v) / 7, 1); return v > 0 ? [col, (.12 + .88 * a).toFixed(2)] : ["var(--muted)", (.06 + .5 * a).toFixed(2)]; }
function drawOutputs() {
  const ai = state.attr;
  for (const gg of [gVals, gLoss, gMean, gOut]) gg.innerHTML = "";
  // three logit vectors
  for (const [k, [x, y, col, nm]] of Object.entries(COLS)) {
    el("text", {x: x + 9, y: y - 6, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px"}}, gVals, nm);
    D.logits[k].forEach((v, a) => { const [f, o] = cellFill(v, col);
      el("rect", {x, y: y + a * ROW, width: 18, height: ROW - .5, style: {fill: f, opacity: o}}, gVals); });
    el("rect", {x: x - 1.5, y: y + ai * ROW - 1.2, width: 21, height: ROW + 1.9, style: {fill: "none", stroke: "var(--ink)", strokeWidth: 1.4}}, gVals);
    el("text", {x: x + 9, y: y + 40 * ROW + 12, "text-anchor": "middle", class: "t-sm", style: {fontSize: "9.5px"}}, gVals, "40 logit");
    el("text", {x: x + 24, y: y + ai * ROW + 4, class: "t-val infonly", style: {fontSize: "10.5px"}}, gVals, fmt(D.logits[k][ai], 1));
  }
  // inference: element-wise mean, sigmoid, threshold
  const mx = 1292, my = 160, R = 7.5;
  el("text", {x: mx + 10, y: my - 22, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px", fill: "var(--ink)"}}, gMean, "trung bình");
  el("text", {x: mx + 10, y: my - 9, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px"}}, gMean, "từng hàng");
  D.logit_final.forEach((v, a) => { const [f, o] = cellFill(v, "var(--ink)");
    el("rect", {x: mx, y: my + a * R, width: 20, height: R - .6, style: {fill: f, opacity: o}}, gMean); });
  el("rect", {x: mx - 1.5, y: my + ai * R - 1, width: 23, height: R + 1.4, style: {fill: "none", stroke: "var(--joint)", strokeWidth: 1.6}}, gMean);
  el("text", {x: mx + 10, y: my + 40 * R + 14, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px"}}, gMean, "z̄ (40)");
  const px = 1340, pw = 96;
  el("text", {x: px + pw / 2, y: my - 22, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px", fill: "var(--ink)"}}, gMean, "σ → xác suất");
  [[0, "0", "start"], [.5, "0,5", "middle"], [1, "1", "end"]].forEach(([f, t, an]) =>
    el("text", {x: px + f * pw, y: my - 7, "text-anchor": an, class: "t-sm", style: {fontSize: "9.5px"}}, gMean, t));
  D.prob.forEach((p, a) => {
    el("rect", {x: px, y: my + a * R + 1, width: pw, height: R - 2, style: {fill: "var(--soft)"}}, gMean);
    el("rect", {x: px, y: my + a * R + 1, width: Math.max(p * pw, .8), height: R - 2, style: {fill: p > .5 ? "var(--joint)" : "var(--muted)", opacity: p > .5 ? .95 : .55}}, gMean);
  });
  el("line", {x1: px + pw / 2, y1: my - 3, x2: px + pw / 2, y2: my + 40 * R + 3, style: {stroke: "var(--grad)", strokeWidth: 1.3, strokeDasharray: "3 2"}}, gMean);
  el("rect", {x: px - 1.5, y: my + ai * R - .5, width: pw + 3, height: R + 1, style: {fill: "none", stroke: "var(--joint)", strokeWidth: 1.4}}, gMean);
  el("text", {x: px + pw / 2, y: my + 40 * R + 14, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px", fill: "var(--grad)"}}, gMean, "ngưỡng 0,5");
  el("text", {x: mx - 4, y: my + 40 * R + 34, class: "t-val", style: {fontSize: "11px"}}, gMean,
    `${vn(A[ai])}: z̄ = ${fmt(D.logit_final[ai], 2)} → p = ${fmt(D.prob[ai], 3)}`);
  // labels above the threshold
  const pos = A.map((x, i) => i).filter(i => D.prob[i] > .5).sort((a, b) => D.prob[b] - D.prob[a]);
  el("text", {x: 1460, y: 176, class: "t-h"}, gOut, "Nhãn: có");
  el("text", {x: 1460, y: 192, class: "t-sm", style: {fontSize: "10px"}}, gOut, `${pos.length}/40 có p > 0,5`);
  pos.forEach((i, k) => el("text", {x: 1460, y: 214 + k * 19, class: "t-sm", style: {fill: i === ai ? "var(--joint)" : "var(--ink)", fontSize: "11px", fontWeight: i === ai ? 700 : 400}}, gOut, `${vn(A[i])}`));
  el("text", {x: 1460, y: 214 + pos.length * 19 + 10, class: "t-sm", style: {fontSize: "10px"}}, gOut, `${40 - pos.length} thuộc tính khác:`);
  el("text", {x: 1460, y: 214 + pos.length * 19 + 24, class: "t-sm", style: {fontSize: "10px"}}, gOut, "không");
  // training: each head's 40 logits against the 40 labels -> one loss per head -> sum
  [["h", 118, "L_H", "var(--hplus)"], ["j", 310, "L_J", "var(--joint)"], ["c", 502, "L_C", "var(--clip)"]].forEach(([k, y, n, c]) => {
    el("path", {d: `M1241,${y} L1262,${y}`, style: {fill: "none", stroke: c, strokeWidth: 1.4}, "marker-end": "url(#arr)"}, gLoss);
    el("rect", {x: 1264, y: y - 36, width: 96, height: 72, rx: 8, style: {fill: "var(--grad-soft)", stroke: "var(--grad)"}}, gLoss);
    D.gt.forEach((v, a) => el("rect", {x: 1272, y: y - 30 + a * 1.5, width: 12, height: 1.2,
      style: {fill: D.mask[a] ? (v ? "var(--ok)" : "var(--line)") : "var(--frozen)"}}, gLoss));
    el("text", {x: 1278, y: y + 34 - 2, "text-anchor": "middle", class: "t-sm", style: {fontSize: "8.5px"}}, gLoss, "");
    el("text", {x: 1292, y: y - 14, class: "t-h", style: {fill: "var(--grad)", fontSize: "15px"}}, gLoss, n);
    el("text", {x: 1292, y: y + 2, class: "t-sm", style: {fontSize: "9.5px"}}, gLoss, "so 40 logit");
    el("text", {x: 1292, y: y + 14, class: "t-sm", style: {fontSize: "9.5px"}}, gLoss, "với 40 nhãn y");
    el("text", {x: 1292, y: y + 27, class: "t-sm", style: {fontSize: "9.5px", fill: "var(--ink)"}}, gLoss, "w·BCE(z+s, ỹ)");
    el("path", {d: `M1360,${y} C1400,${y} 1404,${310 + (y - 310) * .12} 1424,${310 + (y - 310) * .12}`, style: {fill: "none", stroke: "var(--grad)", strokeWidth: 1.3}, "marker-end": "url(#arr)"}, gLoss);
  });
  el("text", {x: 1264, y: 66, class: "t-sm", style: {fontSize: "9.5px"}}, gLoss, "cột xanh lá: nhãn y = 1");
  el("circle", {cx: 1448, cy: 310, r: 22, style: {fill: "var(--ink)"}}, gLoss);
  el("text", {x: 1448, y: 316, "text-anchor": "middle", class: "t-h", style: {fill: "var(--paper)"}}, gLoss, "Σ");
  el("text", {x: 1448, y: 350, "text-anchor": "middle", class: "t-sm"}, gLoss, "L = L_H + L_J + L_C");
  el("text", {x: 1448, y: 365, "text-anchor": "middle", class: "t-sm", style: {fontSize: "10px"}}, gLoss, "→ lan truyền ngược");
}

// gradient paths (train)
const gGrad = g("g-grad", "trainonly");
const GRAD = ["M1440,289 C1420,10 1240,20 1210,60", "M1426,318 C1380,420 1300,420 1210,392", "M1442,331 C1420,612 1240,612 1210,580",
  "M862,70 C780,0 640,6 560,46", "M862,240 C780,226 640,220 562,214", "M862,380 C780,392 640,398 562,404", "M862,560 C780,618 640,612 560,572"];
const gradEls = GRAD.map(d => el("path", {d, class: "gradpath"}, gGrad));

/* ---------- data-driven parts ---------- */
function setAttr(ai) {
  state.attr = ai;
  const shares = D.j_head_hshare.map(r => r[ai]);
  shares.forEach((s, k) => routeCells[k].style.fill = s > .5 ? "var(--hplus)" : "var(--clip)");
  textVal.textContent = fmt(D.logits.c_text[ai], 2);
  drawOutputs();
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", +c.dataset.i === ai));
  drawMaps(ai); drawBars(ai, shares); drawQH(); drawLoss(ai); if (typeof QD !== "undefined") drawQueries();
}
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
const inImg = new Image(); inImg.src = D.input;
function heat(cv, vals, cols, rows, color) {
  const ctx = cv.getContext("2d"); const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (inImg.complete) ctx.drawImage(inImg, 0, 0, W, H);
  ctx.fillStyle = cssVar("--bg"); ctx.globalAlpha = .35; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1;
  const m = Math.max(...vals, 1e-9);
  const cw = W / cols, ch = H / rows;
  ctx.fillStyle = color;
  vals.forEach((v, i) => { ctx.globalAlpha = Math.pow(v / m, .75) * .82; ctx.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, cw, ch); });
  ctx.globalAlpha = 1;
}
function drawMaps(ai) {
  heat(document.getElementById("cv-h"), D.att_h[ai].slice(1), 7, 14, cssVar("--hplus"));
  heat(document.getElementById("cv-c"), D.att_c[ai].slice(1), 8, 16, cssVar("--clip"));
  const aj = D.att_j[ai];
  heat(document.getElementById("cv-jh"), aj.slice(1, 99), 7, 14, cssVar("--hplus"));
  heat(document.getElementById("cv-jc"), aj.slice(100), 8, 16, cssVar("--clip"));
  const sh = aj.slice(0, 99).reduce((a, b) => a + b), sc = aj.slice(99).reduce((a, b) => a + b);
  document.getElementById("jh-share").textContent = `${Math.round(sh * 100)}% attention`;
  document.getElementById("jc-share").textContent = `${Math.round(sc * 100)}% attention`;
}
function drawBars(ai, shares) {
  const L = 8;
  const rows = [["Head H+ · z_H", D.logits.h[ai], "var(--hplus)"], ["Head CLIP · z_C", D.logits.c[ai], "var(--clip)"],
    ["  trong đó phần chữ", D.logits.c_text[ai], "var(--clip)"], ["Query chung · z_J", D.logits.j[ai], "var(--joint)"],
    ["Trung bình 3 head", D.logit_final[ai], "var(--ink)"]];
  const box = document.getElementById("bars"); box.innerHTML = "";
  for (const [n, v, c] of rows) {
    const w = Math.min(Math.abs(v), L) / L * 50;
    const left = v >= 0 ? 50 : 50 - w;
    const r = document.createElement("div"); r.className = "bar";
    r.innerHTML = `<span>${n}</span><div class="track"><div class="fill" style="left:${left}%;width:${w}%;background:${c};opacity:${n.startsWith("  ") ? .55 : 1}"></div><div class="zero"></div></div><span class="val">${fmt(v, 2)}</span>`;
    box.appendChild(r);
  }
  const p = D.prob[ai], gt = D.gt[ai];
  const r = document.createElement("div"); r.className = "bar";
  r.innerHTML = `<span><b>Quyết định</b></span><span>p = ${fmt(p, 3)} ${p > .5 ? "&gt;" : "&lt;"} 0,5 nên báo <b>${p > .5 ? "có" : "không"}</b>. Nhãn thật là <b>${gt ? "có" : "không"}</b>.</span><span class="val" style="color:var(${(p > .5) === (gt === 1) ? "--ok" : "--grad"})">${(p > .5) === (gt === 1) ? "đúng" : "sai"}</span>`;
  box.appendChild(r);
}
function buildChips() {
  const order = ["Accessory-Hat", "UpperBody-Color-White", "LowerBody-Color-Black", "Accessory-Backpack", "Hair-Length-Short", "Age-Adult",
    "LowerBody-Type-Trousers&Shorts", "Gender-Female", "UpperBody-Color-Grey", "LowerBody-Color-Blue", "Accessory-Bag", "Accessory-Glasses-Normal"];
  const box = document.getElementById("chips");
  order.forEach(a => {
    const i = A.indexOf(a); const b = document.createElement("button");
    b.className = "chip" + (D.gt[i] ? " pos" : ""); b.dataset.i = i; b.textContent = vn(a) + (D.gt[i] ? " ●" : "");
    b.addEventListener("click", () => setAttr(i)); box.appendChild(b);
  });
}
function buildAttrs() {
  const box = document.getElementById("attrs");
  A.forEach((a, i) => {
    const p = D.prob[i], yes = p > .5, ok = yes === (D.gt[i] === 1);
    const r = document.createElement("div"); r.className = "arow" + (yes ? " yes" : "");
    r.innerHTML = `<span>${vn(a)}</span><div class="t"><i style="width:${(p * 100).toFixed(1)}%"></i><b></b></div><span class="val">${fmt(p, 3)}</span><span class="${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"}</span>`;
    box.appendChild(r);
  });
}

/* ---------- steps and animation ---------- */
const ALL = ["g-raw", "g-input", "g-aug", "g-gh", "g-gc", "g-sh", "g-sc", "g-mid", "g-ph", "g-pc", "g-proj", "g-hh", "g-hj", "g-hc", "g-vals", "g-loss", "g-mean", "g-out", "g-grad"];
const S_INF = [
  {lit: ["g-raw", "g-input"], fl: [["raw_in", "--ink"]],
    cap: `Ảnh gốc PETA 11178 có kích thước ${D.raw_size[0]}×${D.raw_size[1]} pixel. Gói nộp đổi ảnh về 224×112 và chia giá trị pixel cho 255 để đưa về khoảng [0, 1]. Sau đó mỗi backbone tự chuẩn hoá theo thống kê lúc pretrain của nó (H+ theo ImageNet, CLIP theo OpenAI), nên hai backbone dùng chung một tensor đầu vào.`},
  {lit: ["g-input", "g-gh", "g-gc"], fl: [["in_gh", "--hplus"], ["in_gc", "--clip"]],
    cap: "H+ chia ảnh thành các mảnh 16×16 pixel, được 14×7 = 98 mảnh. Cộng thêm một token CLS và bốn token register, H+ có 103 token, mỗi token 1280 chiều. CLIP chia ảnh thành các mảnh 14×14 pixel, được 16×8 = 128 mảnh, cộng token CLS thành 129 token, mỗi token 1024 chiều."},
  {lit: ["g-gh", "g-gc", "g-sh", "g-sc", "g-mid"], fl: [["gh_sh", "--hplus"], ["gc_sc", "--clip"]], scan: true,
    cap: "Hai backbone chạy song song. ViT-H+ của DINOv3 có 32 khối và 840 triệu tham số. ViT-L/14 của CLIP có 24 khối và 303 triệu tham số. Mỗi khối gồm self-attention và MLP, nên sau khối cuối, mỗi token mang thông tin của cả ảnh chứ không chỉ của mảnh ban đầu."},
  {lit: ["g-sh", "g-sc", "g-ph", "g-pc"], fl: [["sh_ph", "--hplus"], ["sc_pc", "--clip"]],
    cap: `Hai lưới bên phải tô mỗi token đầu ra theo ba thành phần chính (PCA) của nó, đổi sang màu RGB. Hai mảnh cùng màu có đặc trưng gần nhau, nên người, áo, quần và nền tách thành các vùng riêng. Ba thành phần này giữ ${vh}% phương sai của token H+ và ${vc}% phương sai của token CLIP.`},
  {lit: ["g-ph", "g-pc", "g-proj", "g-hh", "g-hj", "g-hc"], fl: [["ph_hh", "--hplus"], ["ph_pj", "--hplus"], ["pj_j", "--hplus"], ["pc_pj", "--clip"], ["qj_j", "--clip"], ["pc_hc", "--clip"]],
    cap: "Ba head đọc token cùng lúc. Head H+ và head CLIP giống hai thành viên của v30. Mỗi head có 40 query, mỗi query ứng với một thuộc tính và đọc token bằng cross-attention 8 đầu. Head CLIP cộng thêm phần chữ, là độ khớp giữa token CLS và câu mô tả thuộc tính. Query chung cần token của hai backbone có cùng số chiều, nên khối Chiếu chung đưa token H+ (1280 chiều) và token CLIP (1024 chiều) về 1024 chiều bằng một lớp tuyến tính cho mỗi bên, rồi cộng một vector nhãn nguồn để query phân biệt token của từng backbone. Sau đó query chung đọc cả 228 token."},
  {lit: ["g-hh", "g-hj", "g-hc", "g-vals", "g-mean", "g-out"], fl: [["hh_v", "--hplus"], ["hj_v", "--joint"], ["hc_v", "--clip"], ["v1_m", "--hplus"], ["v2_m", "--joint"], ["v3_m", "--clip"], ["m_o", "--ink"], ["o_l", "--joint"]],
    cap: "Mỗi head trả ra một vector 40 logit, mỗi hàng ứng với một thuộc tính. Ô càng đậm thì logit càng dương. Ba vector được lấy trung bình theo từng hàng thành z̄, và sigmoid đổi mỗi giá trị z̄ thành xác suất từ 0 đến 1. Thuộc tính có xác suất lớn hơn 0,5 được báo là có. Ngưỡng 0,5 dùng chung cho mọi thuộc tính, không có offset hay ngưỡng riêng. Với ảnh này, cả 7 thuộc tính có thật đều được báo đúng và không thuộc tính nào bị báo nhầm."}];
const S_TRAIN = [
  {lit: ["g-raw", "g-input", "g-aug"], fl: [["raw_in", "--grad"]],
    cap: "Lúc train, mỗi ảnh được tăng cường ngẫu nhiên bằng các phép giảm độ phân giải (xác suất 30%), lật ngang, đệm 10 pixel rồi cắt lại, đổi độ sáng, độ tương phản và độ bão hoà, làm mờ và xoá một vùng. Các phép đổi màu giữ nguyên hue, vì màu là nhãn cần dự đoán. Mỗi batch có 64 ảnh lấy từ train và val của PA100k, Market và PETA (131.076 ảnh). Mỗi dataset được lấy mẫu tỉ lệ với N^0,5."},
  {lit: ["g-input", "g-gh", "g-gc"], fl: [["in_gh", "--hplus"], ["in_gc", "--clip"]], cap: S_INF[1].cap},
  {lit: ["g-gh", "g-gc", "g-sh", "g-sc", "g-mid"], fl: [["gh_sh", "--hplus"], ["gc_sc", "--clip"]], scan: true,
    cap: "Patch embedding và 8 khối đầu của mỗi backbone được đóng băng (các khối gạch chéo, 312 triệu tham số). 24 khối còn lại của H+ và 16 khối còn lại của CLIP được fine-tune. Learning rate ở khối cuối là 5e-5 và giảm 0,75 lần qua mỗi khối xuống dưới. Gradient checkpointing giúp cả hai backbone vừa một GPU 32 GB."},
  {lit: ["g-sh", "g-sc", "g-ph", "g-pc"], fl: [["sh_ph", "--hplus"], ["sc_pc", "--clip"]], cap: S_INF[3].cap},
  {lit: ["g-ph", "g-pc", "g-proj", "g-hh", "g-hj", "g-hc"], fl: S_INF[4].fl, cap: S_INF[4].cap},
  {lit: ["g-hh", "g-hj", "g-hc", "g-vals", "g-loss"], fl: [["hh_v", "--hplus"], ["hj_v", "--joint"], ["hc_v", "--clip"]],
    cap: "Mỗi head có một loss riêng, và ba loss được cộng lại. Mỗi loss là BCE trên logit đã cộng độ dịch theo prior s = 0,32 · log(p/(1 − p)), có trọng số DeepMAR, label smoothing 0,05 và bỏ qua các ô mà dataset không gán nhãn. Phần Loss lúc huấn luyện ở dưới tính lại từng bước bằng số thật của ảnh này."},
  {lit: ["g-grad", "g-hh", "g-hj", "g-hc", "g-sh", "g-sc", "g-loss"], grad: true,
    cap: "Gradient của ba loss lan ngược về các head và hai backbone. Query chung đọc token của cả hai backbone, nên gradient của nó đi vào cả H+ và CLIP. Nhờ vậy hai backbone học để bổ sung cho nhau, thay vì học riêng như hai thành viên của v30. Các khối đóng băng không nhận gradient."},
  {lit: ["g-sh", "g-sc", "g-mid", "g-hh", "g-hj", "g-hc"], ema: true,
    cap: "Bộ tối ưu là AdamW với weight decay 0,05, và learning rate của head gấp 10 lần backbone. Learning rate tăng dần trong nửa epoch đầu rồi giảm theo cosine. Mô hình được train 3 epoch ở bf16. Bản được lưu và đóng gói là trọng số EMA với hệ số 0,9995. Gói nặng 2,36 GB và chạy 3,04 ms/ảnh trên RTX 5090."}];
const steps = () => state.mode === "train" ? S_TRAIN : S_INF;
let timers = [], raf = 0;
function clearAnim() { timers.forEach(clearTimeout); timers = []; cancelAnimationFrame(raf); document.querySelectorAll(".dot").forEach(d => d.remove());
  SH.scan.style.opacity = 0; SC.scan.style.opacity = 0; SH.ctr.style.opacity = 0; SC.ctr.style.opacity = 0;
  gradEls.forEach(p => p.classList.remove("run")); }
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
function dots(list) {
  if (reduce) return;
  const items = [];
  for (const [k, c] of list) { const p = flowEl[k]; const L = p.getTotalLength();
    for (let j = 0; j < 4; j++) items.push({p, L, c, d: j * 0.14, e: el("circle", {r: 3.4, class: "dot", style: {fill: `var(${c})`}})}); }
  const t0 = performance.now(), dur = 1100;
  const tick = now => {
    let alive = false;
    for (const it of items) {
      const t = (now - t0) / dur - it.d;
      if (t < 0) { it.e.style.opacity = 0; alive = true; continue; }
      if (t > 1) { it.e.style.opacity = 0; continue; }
      alive = true; const pt = it.p.getPointAtLength(it.L * t);
      it.e.setAttribute("cx", pt.x); it.e.setAttribute("cy", pt.y); it.e.style.opacity = 1;
    }
    if (alive) raf = requestAnimationFrame(tick); else items.forEach(it => it.e.remove());
  };
  raf = requestAnimationFrame(tick);
}
function scan(S, nFrozen) {
  if (reduce) return;
  S.scan.style.opacity = 1; S.ctr.style.opacity = 1;
  for (let i = 0; i < S.n; i++) timers.push(setTimeout(() => {
    S.scan.setAttribute("x", S.x0 + i * (S.sw + S.gap) - 2);
    S.ctr.textContent = `khối ${i + 1}/${S.n}`;
  }, i * (1500 / S.n)));
}
function show(k) {
  clearAnim();
  const st = steps(); state.step = (k + st.length) % st.length; const s = st[state.step];
  for (const id of ALL) { const e = document.getElementById(id); if (e) e.setAttribute("class", (e.getAttribute("class") || "").replace(/\b(dim|lit)\b/g, "").trim() + (s.lit.includes(id) ? " lit" : " dim")); }
  for (const [k2, p] of Object.entries(flowEl)) p.style.stroke = s.fl && s.fl.some(f => f[0] === k2) ? "var(--muted)" : "var(--line)";
  if (s.fl) dots(s.fl);
  if (s.scan) { scan(SH); scan(SC); }
  if (s.grad) { gradEls.forEach(p => p.classList.add("run"));
    [SH, SC].forEach(S => S.slabs.forEach(r => { if (r.dataset.frozen === "0") { r.style.transition = "opacity .4s"; r.style.opacity = 1; } })); }
  document.getElementById("capn").textContent = `${state.step + 1}/${st.length}`;
  document.getElementById("capt").textContent = s.cap;
  document.querySelectorAll(".step").forEach((b, i) => b.classList.toggle("on", i === state.step));
}
function buildSteps() {
  const names = state.mode === "train" ? ["tăng cường", "cắt mảnh", "backbone", "token ra", "3 head", "3 loss", "lan truyền ngược", "cập nhật"]
    : ["ảnh vào", "cắt mảnh", "backbone", "token ra", "3 head", "gộp & quyết định"];
  const box = document.getElementById("steps"); box.innerHTML = "";
  names.forEach((n, i) => { const b = document.createElement("button"); b.className = "step"; b.textContent = `${i + 1} ${n}`; b.addEventListener("click", () => { stop(); show(i); }); box.appendChild(b); });
}
let playT = 0;
function play() { state.playing = true; document.getElementById("play").textContent = "⏸ Dừng";
  const go = () => { show(state.step + 1); playT = setTimeout(go, 2900); }; show(state.step); playT = setTimeout(go, 2900); }
function stop() { state.playing = false; clearTimeout(playT); document.getElementById("play").textContent = "▶ Chạy"; }
function setMode(m) {
  state.mode = m; document.getElementById("app").classList.toggle("mode-train", m === "train");
  svg.classList.toggle("mode-train", m === "train");
  document.getElementById("m-inf").setAttribute("aria-pressed", m === "inf"); document.getElementById("m-train").setAttribute("aria-pressed", m === "train");
  buildSteps(); drawOutputs(); show(0);
}
document.getElementById("m-inf").addEventListener("click", () => { stop(); setMode("inf"); });
document.getElementById("m-train").addEventListener("click", () => { stop(); setMode("train"); });
document.getElementById("play").addEventListener("click", () => state.playing ? stop() : play());
document.getElementById("next").addEventListener("click", () => { stop(); show(state.step + 1); });
document.getElementById("prev").addEventListener("click", () => { stop(); show(state.step - 1); });


/* ---------- inside a query head ---------- */
let qhKind = "h";
const qsvg = document.getElementById("qh");
const sig = x => 1 / (1 + Math.exp(-x));
function qel(tag, at = {}, text) { return el(tag, at, qsvg, text); }
function box(x, y, w, h, fill, stroke, dash) { return qel("rect", {x, y, width: w, height: h, rx: 10, style: {fill, stroke, strokeWidth: 1.4, strokeDasharray: dash || ""}}); }
function txt(x, y, t, cls = "t-sm", st = {}, anchor = "start") { return qel("text", {x, y, class: cls, "text-anchor": anchor, style: st}, t); }
function arrow(d, color = "var(--muted)", w = 1.6) { return qel("path", {d, style: {fill: "none", stroke: color, strokeWidth: w}, "marker-end": "url(#arr2)"}); }
function gridImg(x, y, w, h, cols, rows, color, vals) {
  qel("image", {href: D.input, x, y, width: w, height: h, preserveAspectRatio: "none"});
  const cw = w / cols, ch = h / rows;
  if (vals) { const m = Math.max(...vals, 1e-9);
    vals.forEach((v, i) => qel("rect", {x: x + (i % cols) * cw, y: y + Math.floor(i / cols) * ch, width: cw + .2, height: ch + .2, style: {fill: color, opacity: (Math.pow(v / m, .75) * .85).toFixed(3)}})); }
  else { for (let i = 0; i <= cols; i++) qel("line", {x1: x + i * cw, y1: y, x2: x + i * cw, y2: y + h, style: {stroke: color, strokeWidth: .6}});
    for (let j = 0; j <= rows; j++) qel("line", {x1: x, y1: y + j * ch, x2: x + w, y2: y + j * ch, style: {stroke: color, strokeWidth: .6}}); }
  qel("rect", {x, y, width: w, height: h, style: {fill: "none", stroke: "var(--line)"}});
}
function strip(x, y, w, h, n, sel, color, vertical) {
  for (let i = 0; i < n; i++) {
    const on = i === sel;
    const at = vertical ? {x, y: y + i * h / n, width: w, height: h / n - .6} : {x: x + i * w / n, y, width: w / n - .6, height: h};
    qel("rect", Object.assign(at, {style: {fill: on ? color : "var(--soft)", stroke: on ? "var(--ink)" : "none", strokeWidth: on ? 1 : 0}}));
  }
}
function drawQH() {
  const ai = state.attr, K = qhKind; qsvg.innerHTML = "";
  const dfs = qel("defs"); const m2 = el("marker", {id: "arr2", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"}, dfs);
  el("path", {d: "M0,1 L9,5 L0,9 z", style: {fill: "var(--muted)"}}, m2);
  const color = {h: "var(--hplus)", c: "var(--clip)", j: "var(--joint)"}[K], soft = {h: "var(--hplus-soft)", c: "var(--clip-soft)", j: "var(--joint-soft)"}[K];
  const Dm = D.head_dims[K], N = {h: 99, c: 129, j: 228}[K], dh = Dm / 8;
  const name = vn(A[ai]);
  // 1. input tokens
  box(14, 120, 178, 300, "var(--paper)", "var(--line)");
  txt(24, 142, "1 · Token vào", "t-lbl");
  if (K === "h") { gridImg(36, 156, 66, 132, 7, 14, "var(--hplus)"); txt(110, 176, "CLS + 98 mảnh"); txt(110, 192, "(bỏ 4 register)"); }
  if (K === "c") { gridImg(36, 156, 66, 132, 8, 16, "var(--clip)"); txt(110, 176, "CLS + 128"); txt(110, 192, "mảnh"); }
  if (K === "j") { gridImg(24, 156, 60, 120, 7, 14, "var(--hplus)"); gridImg(90, 156, 60, 120, 8, 16, "var(--clip)");
    txt(24, 290, "H+ 99 + CLIP 129, đã qua", "t-sm", {fontSize: "10px"}); txt(24, 303, "Chiếu chung + nhãn nguồn", "t-sm", {fontSize: "10px"}); }
  txt(24, 330, `${N} token`, "t-val"); txt(24, 346, `× ${Dm} chiều`, "t-sm");
  txt(24, 372, "LayerNorm rồi chiếu thành", "t-sm", {fontSize: "10px"}); txt(24, 386, "khoá K và giá trị V cho", "t-sm", {fontSize: "10px"}); txt(24, 400, "mỗi đầu attention", "t-sm", {fontSize: "10px"});
  // 2. learned queries
  box(214, 14, 316, 82, "var(--paper)", "var(--line)");
  txt(226, 36, "2 · 40 query học được", "t-lbl");
  txt(226, 52, `40 × ${Dm} · giống nhau cho mọi ảnh, mỗi query hỏi 1 thuộc tính`, "t-sm", {fontSize: "10px"});
  strip(226, 62, 292, 18, 40, ai, color, false);
  txt(226 + (ai + .5) * 292 / 40, 92, "▲ " + name, "t-sm", {fill: "var(--ink)", fontSize: "10px"}, ai > 30 ? "end" : "middle");
  // 3. cross-attention with 8 heads
  box(214, 120, 640, 330, soft, color);
  txt(226, 142, `3 · Cross-attention, 8 đầu song song (mỗi đầu ${dh} chiều)`, "t-lbl");
  txt(226, 158, `a = softmax(q·k / √${dh}) trên ${N} token, đầu ra là Σ a·v. Mỗi bản đồ là trọng số của query “${name}” ở một đầu.`, "t-sm", {fontSize: "10px"});
  const heads = {h: D.att_h_heads, c: D.att_c_heads, j: D.att_j_heads}[K][ai];
  heads.forEach((w, k) => {
    const cx = 226 + (k % 4) * 156, cy = 172 + Math.floor(k / 4) * 138;
    qel("rect", {x: cx, y: cy, width: 148, height: 130, rx: 8, style: {fill: "var(--paper)", stroke: "var(--line)"}});
    let src = K === "c" ? "c" : "h", vals = w.slice(1), cols = 7, rows = 14;
    if (K === "c") { cols = 8; rows = 16; }
    if (K === "j") { const sh = w.slice(0, 99).reduce((a, b) => a + b); src = sh > .5 ? "h" : "c";
      vals = src === "h" ? w.slice(1, 99) : w.slice(100); cols = src === "h" ? 7 : 8; rows = src === "h" ? 14 : 16; }
    const scol = src === "h" ? "var(--hplus)" : "var(--clip)";
    gridImg(cx + 8, cy + 8, 50, 100, cols, rows, scol, vals);
    txt(cx + 66, cy + 24, `đầu ${k + 1}`, "t-lbl", {fontSize: "12px"});
    if (K === "j") txt(cx + 66, cy + 40, `đọc ${src === "h" ? "H+" : "CLIP"}`, "t-sm", {fill: scol, fontWeight: 600});
    const mx = Math.max(...vals);
    txt(cx + 66, cy + (K === "j" ? 58 : 42), `ô mạnh nhất`, "t-sm", {fontSize: "10px"});
    txt(cx + 66, cy + (K === "j" ? 72 : 56), `${Math.round(mx * 100)}% trọng số`, "t-sm", {fontSize: "10px", fill: "var(--ink)"});
    const cls = w[K === "j" && src === "c" ? 99 : 0];
    txt(cx + 66, cy + (K === "j" ? 88 : 72), `CLS: ${Math.round(cls * 100)}%`, "t-sm", {fontSize: "10px"});
  });
  // 4. merge, residual, MLP
  arrow("M854,286 L884,286");
  box(884, 214, 150, 146, "var(--paper)", "var(--line)");
  txt(894, 236, "4 · Gộp và tinh chỉnh", "t-lbl", {fontSize: "12.5px"});
  ["ghép 8 đầu → Linear", "+ query ban đầu", "LayerNorm → MLP", `${Dm} → ${Dm * 2} → ${Dm} (GELU)`, "+ cộng dư", "LayerNorm"].forEach((t, i) =>
    txt(894, 256 + i * 16, t, "t-sm", {fontSize: "10.5px", fill: i % 2 ? "var(--muted)" : "var(--ink)"}));
  arrow("M372,96 C372,108 960,100 960,212", "var(--muted)", 1.2);
  txt(660, 108, "đường cộng dư (query ban đầu)", "t-sm", {fontSize: "10px"});
  // 5. per-attribute vectors
  arrow("M1034,286 L1062,286");
  box(1062, 164, 120, 246, "var(--paper)", "var(--line)");
  txt(1072, 186, "5 · 40 vector", "t-lbl", {fontSize: "12.5px"});
  txt(1072, 201, `q_a, mỗi vector ${Dm} chiều`, "t-sm", {fontSize: "9.5px"});
  strip(1100, 212, 42, 186, 40, ai, color, true);
  // 6. classifier
  arrow("M1182,286 L1210,286");
  box(1210, 150, 278, K === "c" ? 290 : 200, "var(--paper)", color);
  txt(1222, 172, "6 · Phân loại riêng từng thuộc tính", "t-lbl", {fontSize: "12.5px"});
  txt(1222, 190, "mỗi thuộc tính a có vector w_a và số b_a riêng", "t-sm", {fontSize: "10px"});
  const z = D.logits[K][ai], zt = D.logits.c_text[ai];
  if (K !== "c") {
    txt(1222, 220, `z = q_a · w_a + b_a`, "t-val");
    txt(1222, 252, `z(${name}) = ${fmt(z, 2)}`, "t-h", {fill: color});
    txt(1222, 274, `→ logit của head này, 1 trong 3 số`, "t-sm", {fontSize: "10px"});
    txt(1222, 288, `được lấy trung bình ở bước cuối`, "t-sm", {fontSize: "10px"});
  } else {
    txt(1222, 216, `phần query: q_a · w_a + b_a = ${fmt(z - zt, 2)}`, "t-sm", {fill: "var(--ink)"});
    txt(1222, 240, "phần chữ: CLS → chiếu P (1024 → 768)", "t-sm", {fontSize: "10.5px"});
    txt(1222, 255, "→ chuẩn hoá → ×100 · (t_a + 0,1·δ_a) + c_a", "t-sm", {fontSize: "10.5px"});
    txt(1222, 270, "t_a: vector chữ của câu mô tả (cố định)", "t-sm", {fontSize: "10px"});
    txt(1222, 284, "δ_a, c_a: học được", "t-sm", {fontSize: "10px"});
    txt(1222, 304, `phần chữ = ${fmt(zt, 2)}`, "t-sm", {fill: "var(--ink)"});
    txt(1222, 336, `z(${name}) = ${fmt(z, 2)}`, "t-h", {fill: color});
    txt(1222, 358, "= phần query + phần chữ", "t-sm", {fontSize: "10px"});
  }
  // explanation list
  const L = document.getElementById("qhlist");
  const inp = K === "h" ? "99 token của H+, gồm token CLS và 98 mảnh 16×16 (bốn token register bị bỏ). Mỗi token có 1280 chiều"
    : K === "c" ? "129 token của CLIP, gồm token CLS và 128 mảnh 14×14. Mỗi token có 1024 chiều"
    : "228 token, gồm 99 token của H+ và 129 token của CLIP. Khối Chiếu chung đã đưa cả hai về 1024 chiều và cộng nhãn nguồn, nên query biết mỗi token đến từ backbone nào";
  const jn = K === "j" ? (() => { const n = D.att_j_heads[ai].filter(w => w.slice(0, 99).reduce((a, b) => a + b) > .5).length;
    return ` Ở query chung, mỗi đầu chỉ đọc token của một backbone. Với “${name}”, ${n} đầu đọc H+ (bản đồ xanh) và ${8 - n} đầu đọc CLIP (bản đồ cam). Số đầu đọc mỗi backbone thay đổi theo thuộc tính.`; })() : "";
  L.innerHTML = [
    `Đầu vào là ${inp}.`,
    `Mỗi head có 40 query. Mỗi query là một vector học được trong lúc train và ứng với một thuộc tính. Ảnh nào cũng dùng cùng 40 query này.`,
    `Trong bước cross-attention, mô hình đo độ khớp giữa mỗi query và từng token, đưa các độ khớp qua softmax để được trọng số có tổng bằng 1, rồi lấy trung bình có trọng số của các token. Tám đầu attention làm việc này song song. Mỗi đầu có phép chiếu riêng, nên các đầu có thể đọc những vùng khác nhau.${jn}`,
    `Kết quả của 8 đầu được ghép lại, cộng với query ban đầu, rồi đi qua một MLP nhỏ và LayerNorm. Đầu ra là 40 vector ${Dm} chiều, mỗi vector mô tả một thuộc tính trong ảnh này.`,
    `Mỗi thuộc tính có một vector trọng số và một hằng số riêng. Logit bằng tích vô hướng giữa vector của thuộc tính và vector trọng số, cộng với hằng số.${K === "c" ? " Head CLIP cộng thêm phần chữ, là độ khớp giữa token CLS (sau khi chiếu sang không gian ảnh và chữ của CLIP) với câu mô tả thuộc tính." : ""} Mỗi head trả về 40 logit, một logit cho mỗi thuộc tính.`
  ].map(t => `<li>${t}</li>`).join("");
}
document.querySelectorAll("#qhsec .seg button").forEach(b => b.addEventListener("click", () => {
  qhKind = b.dataset.k; document.querySelectorAll("#qhsec .seg button").forEach(x => x.setAttribute("aria-pressed", x === b)); drawQH(); drawQueries(); }));

/* ---------- loss and output ---------- */
function bceLogits(z, y) { return Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z))); }
function headLoss(k) {   // this image only: sum over labelled attributes of w * BCE(z + s, smoothed y) / number labelled
  let num = 0, den = 0;
  for (let a = 0; a < 40; a++) {
    if (!D.mask[a]) continue;
    const p = D.prior[a], y = D.gt[a], s = .32 * Math.log(p / (1 - p)), yt = y * .95 + .025, w = y ? Math.exp(1 - p) : Math.exp(p);
    num += w * bceLogits(D.logits[k][a] + s, yt); den += 1;
  }
  return num / den;
}
const tn = (x, k = 3) => (x < 0 ? "-" : "") + Math.abs(x).toFixed(k).replace(".", "{,}");   // number for TeX, decimal comma
function typeset(...ids) {
  const els = ids.map(i => document.getElementById(i));
  if (window.MathJax && MathJax.typesetPromise) { MathJax.typesetClear(els); MathJax.typesetPromise(els).catch(() => {}); }
  else setTimeout(() => typeset(...ids), 300);
}
function drv(rows) {
  return rows.map(([t, m], i) => `<li><span class="n">${i + 1}</span><div><p class="tx">${t}</p>${m ? `<div class="math">\\[${m}\\]</div>` : ""}</div></li>`).join("");
}
function drawLoss(ai) {
  const p = D.prior[ai], y = D.gt[ai], s = .32 * Math.log(p / (1 - p)), yt = y * .95 + .025, w = y ? Math.exp(1 - p) : Math.exp(p), m = D.mask[ai];
  const name = vn(A[ai]);
  document.getElementById("loss-sub").innerHTML = `Các con số dưới đây tính trên ảnh này cho thuộc tính <b>“${name}”</b>. Có thể đổi thuộc tính bằng các nút ở phần trên. Prior \\(p\\) là tỉ lệ ảnh có thuộc tính này trong ${D.n_train.toLocaleString("vi-VN")} ảnh train, ở đây \\(p = ${tn(p, 4)}\\). Nhãn thật của ảnh là <b>${y ? "có" : "không"}</b> (\\(y = ${y}\\))${m ? "" : ". PETA không gán nhãn thuộc tính này, nên ô này bị loại khỏi loss"}.`;
  document.getElementById("loss-steps").innerHTML = drv([
    ["Mỗi head cho một logit \\(z\\), ghi ở cột đầu của bảng bên dưới. Lúc train, logit được cộng thêm một độ dịch \\(s\\) tính từ prior (logit adjustment, \\(\\tau = 0{,}32\\)). Thuộc tính càng hiếm thì \\(s\\) càng âm, nên để loss nhỏ, head phải đẩy \\(z\\) lên cao hơn. Lúc suy luận không cộng \\(s\\), nên logit thô của thuộc tính hiếm vượt 0 sớm hơn và mô hình báo thuộc tính hiếm dễ hơn.",
     `s = \\tau \\log\\frac{p}{1-p} = 0{,}32 \\cdot \\log\\frac{${tn(p, 4)}}{${tn(1 - p, 4)}} = ${tn(s)}`],
    ["Nhãn được làm mượt với \\(\\varepsilon = 0{,}05\\).", `\\tilde{y} = y\\,(1-\\varepsilon) + \\frac{\\varepsilon}{2} = ${y} \\cdot 0{,}95 + 0{,}025 = ${tn(yt)}`],
    ["Loss BCE được tính trên logit đã dịch, với \\(\\sigma(x) = 1/(1+e^{-x})\\).", `\\ell = -\\Big[\\,\\tilde{y}\\,\\log \\sigma(z+s) + (1-\\tilde{y})\\,\\log\\big(1-\\sigma(z+s)\\big)\\Big]`],
    ["Trọng số DeepMAR tăng phần đóng góp của ảnh có thuộc tính hiếm.", `w = \\begin{cases} e^{\\,1-p} & \\text{nếu } y = 1 \\\\ e^{\\,p} & \\text{nếu } y = 0 \\end{cases} \;=\; e^{${tn(y ? 1 - p : p, 4)}} = ${tn(w)}`],
    ["Loss của một head là trung bình có trọng số trên mọi ô (ảnh \\(i\\), thuộc tính \\(a\\)) của batch. Ô mà dataset của ảnh không gán nhãn có \\(m_{ia} = 0\\) và bị bỏ qua.", `L_{\\text{head}} = \\frac{\\sum_{i,a} m_{ia}\\, w_{ia}\\, \\ell_{ia}}{\\sum_{i,a} m_{ia}}`],
    ["Loss tổng là tổng của ba loss, được lan truyền ngược một lần cho cả ba head và hai backbone.", `L = L_{\\text{H+}} + L_{\\text{CLIP}} + L_{\\text{chung}}`]]);
  const rows = [["Head H+", "h", "--hplus"], ["Head CLIP", "c", "--clip"], ["Query chung", "j", "--joint"]].map(([n, k, c]) => {
    const z = D.logits[k][ai], zs = z + s, l = bceLogits(zs, yt);
    return `<tr><td style="color:var(${c})">${n}</td><td class="val">${fmt(z, 2)}</td><td class="val">${fmt(zs, 2)}</td><td class="val">${fmt(sig(zs), 3)}</td><td class="val">${fmt(l, 4)}</td><td class="val">${m ? fmt(w * l, 4) : "bỏ qua"}</td></tr>`;
  }).join("");
  document.getElementById("loss-table").innerHTML = `<thead><tr><th>Head</th><th>\\(z\\)</th><th>\\(z+s\\)</th><th>\\(\\sigma(z+s)\\)</th><th>\\(\\ell\\)</th><th>\\(w\\,\\ell\\)</th></tr></thead><tbody>${rows}</tbody>`;
  const LH = headLoss("h"), LC = headLoss("c"), LJ = headLoss("j"), n = D.mask.reduce((a, b) => a + b);
  document.getElementById("loss-total").innerHTML = `Với ${n} thuộc tính có nhãn của ảnh này, loss của từng head và loss tổng bằng<div class="math">\\[L = \\underbrace{${tn(LH, 4)}}_{L_{\\text{H+}}} + \\underbrace{${tn(LC, 4)}}_{L_{\\text{CLIP}}} + \\underbrace{${tn(LJ, 4)}}_{L_{\\text{chung}}} = ${tn(LH + LC + LJ, 4)}\\]</div><span class="sub" style="font-size:13px">Loss nhỏ vì bản final đã được train trên chính ảnh này. Khi train thật, mẫu số là số ô có nhãn của cả batch 64 ảnh.</span>`;
  const zH = D.logits.h[ai], zC = D.logits.c[ai], zJ = D.logits.j[ai], zb = (zH + zC + zJ) / 3, pr = sig(zb), dec = pr > .5;
  document.getElementById("inf-steps").innerHTML = drv([
    ["Ba head cho ba logit thô, không cộng \\(s\\).", `z_{\\text{H+}} = ${tn(zH, 2)}, \\quad z_{\\text{CLIP}} = ${tn(zC, 2)}, \\quad z_{\\text{chung}} = ${tn(zJ, 2)}`],
    ["Ba logit được lấy trung bình.", `\\bar{z} = \\frac{z_{\\text{H+}} + z_{\\text{CLIP}} + z_{\\text{chung}}}{3} = \\frac{${tn(zH, 2)} + ${tn(zC, 2)} + ${tn(zJ, 2)}}{3} = ${tn(zb, 2)}`],
    ["Sigmoid đổi giá trị trung bình thành xác suất.", `\\hat{p} = \\sigma(\\bar{z}) = \\frac{1}{1 + e^{-\\bar{z}}} = ${tn(pr, 4)}`],
    [`Xác suất được so với ngưỡng 0,5 dùng chung cho mọi thuộc tính, tương đương điều kiện \\(\\bar z > 0\\). Nhãn thật là <b>${y ? "có" : "không"}</b>, nên quyết định <b style="color:var(${dec === (y === 1) ? "--ok" : "--grad"})">${dec === (y === 1) ? "đúng" : "sai"}</b>.`,
     `\\hat{p} = ${tn(pr, 4)} ${dec ? ">" : "<"} 0{,}5 \;\\Rightarrow\; \\text{${dec ? "có" : "không"} “${name}”}`],
    ["Gói nộp làm phép tính này cho cả 40 thuộc tính trong một lượt chạy và trả về 40 xác suất cho mỗi ảnh. Điểm được chấm ở ngưỡng 0,5, không có offset hay ngưỡng riêng cho từng thuộc tính.", ""]]);
  typeset("loss-sub", "loss-steps", "loss-table", "loss-total", "inf-steps");
}

/* ---------- the 40 learned queries ---------- */
function divColor(v) {      // v in [-1, 1] -> diverging fill from the theme (negative: clip, positive: hplus)
  const a = Math.min(Math.abs(v), 1);
  return [v >= 0 ? cssVar("--hplus") : cssVar("--clip"), Math.pow(a, .7)];
}
function canvasCtx(cv, w, h) {
  const r = window.devicePixelRatio || 1; cv.width = w * r; cv.height = h * r; cv.style.width = w + "px";
  const ctx = cv.getContext("2d"); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, w, h); return ctx;
}
function drawQueries() {
  const Q = QD[qhKind], Dm = Q.dim, name = {h: "head H+", c: "head CLIP", j: "query chung"}[qhKind];
  const ink = cssVar("--ink"), muted = cssVar("--muted"), bg = cssVar("--paper"), sel = state.attr;
  document.getElementById("qintro").innerHTML = `Query của ${name} là một ma trận tham số \\(Q \\in \\mathbb{R}^{40 \\times ${Dm}}\\), mỗi hàng \\(q_a\\) ứng với một thuộc tính. Ma trận này được khởi tạo bằng số ngẫu nhiên nhỏ và được học cùng mô hình qua gradient của loss. Sau khi train, ma trận cố định, nên ảnh nào cũng dùng đúng 40 hàng này. Phần thay đổi theo từng ảnh là thông tin query đọc được từ token.`;
  document.getElementById("qeq").innerHTML = `\\[\\begin{aligned}
q_a &\\sim \\mathcal{N}(0,\\,0{,}02^2)^{${Dm}} \\ \\text{lúc khởi tạo, rồi được học} \\\\
\\alpha^{h}_{a,n} &= \\operatorname{softmax}_{n}\\!\\left(\\frac{\\big(\\mathrm{LN}(q_a)\\,W^{h}_{Q}\\big)\\cdot\\big(\\mathrm{LN}(t_n)\\,W^{h}_{K}\\big)}{\\sqrt{${Dm / 8}}}\\right), \\quad h = 1..8 \\\\
u_a &= q_a + W_O\\Big[\\,{\\textstyle\\sum_n} \\alpha^{1}_{a,n}\\,\\mathrm{LN}(t_n) W^{1}_{V};\\ \\dots;\\ {\\textstyle\\sum_n} \\alpha^{8}_{a,n}\\,\\mathrm{LN}(t_n) W^{8}_{V}\\Big] \\\\
o_a &= \\mathrm{LN}\\big(u_a + \\mathrm{MLP}(\\mathrm{LN}(u_a))\\big), \\qquad z_a = o_a \\cdot w_a + b_a
\\end{aligned}\\]`;
  const n0 = Q.q_norm.reduce((a, b) => a + b) / 40;
  const off = []; for (let i = 0; i < 40; i++) for (let j = i + 1; j < 40; j++) off.push([Q.cos[i][j], i, j]);
  off.sort((a, b) => b[0] - a[0]);
  const mean = off.reduce((a, b) => a + b[0], 0) / off.length;
  const selRow = Q.cos[sel].map((v, j) => [v, j]).filter(x => x[1] !== sel).sort((a, b) => b[0] - a[0]).slice(0, 3);
  document.getElementById("qfacts").innerHTML = [
    `Trong công thức, \\(t_n\\) là token thứ \\(n\\). Các ma trận \\(W^{h}_{Q}, W^{h}_{K}, W^{h}_{V}\\) chiếu query và token về ${Dm / 8} chiều cho đầu \\(h\\). \\(\\alpha^{h}_{a,n}\\) là trọng số mà query \\(a\\) dành cho token \\(n\\), và tổng của nó theo \\(n\\) bằng 1. \\(w_a\\) và \\(b_a\\) là bộ phân loại riêng của thuộc tính \\(a\\).`,
    `Lúc khởi tạo, 40 hàng là số ngẫu nhiên và chưa mang thông tin gì. Độ dài mỗi hàng khoảng \\(0{,}02\\sqrt{${Dm}} = ${tn(Q.init_norm, 2)}\\). Sau khi train, độ dài trung bình tăng lên \\(${tn(n0, 2)}\\), cho thấy mỗi query đã học một hướng riêng.`,
    `Cos giữa hai query khác nhau trung bình chỉ \\(${tn(mean, 3)}\\), nên các query gần như vuông góc và mỗi thuộc tính dùng một hướng riêng. Ba cặp giống nhau nhất là ${off.slice(0, 3).map(([v, i, j]) => `${vn(A[i])} và ${vn(A[j])} (${fmt(v, 2)})`).join(", ")}.`,
    `Query của “${vn(A[sel])}” gần nhất với query của ${selRow.map(([v, j]) => `${vn(A[j])} (${fmt(v, 2)})`).join(", ")}.`,
    `Mỗi query tự chọn vùng ảnh của nó, ví dụ query mũ đọc vùng đỉnh đầu và query quần đọc nửa dưới. Vì vậy mỗi thuộc tính có một vector \\(o_a\\) riêng trước khi phân loại, thay vì cả 40 thuộc tính dùng chung một vector đặc trưng của toàn ảnh.`
  ].map(t => `<li>${t}</li>`).join("");
  // query matrix: 40 rows x first 96 dims (scaled by the largest |value| of the whole matrix)
  const LW = 112, cw = 4, rh = 9, W = LW + 96 * cw + 56, H = 40 * rh + 22;
  let ctx = canvasCtx(document.getElementById("qmat"), W, H);
  ctx.font = "10px 'IBM Plex Sans', sans-serif"; ctx.textBaseline = "middle";
  Q.q_first.forEach((row, a) => {
    ctx.fillStyle = a === sel ? cssVar("--joint") : muted; ctx.textAlign = "right"; ctx.fillText(vn(A[a]), LW - 6, 16 + a * rh + rh / 2);
    row.forEach((v, k) => { const [c, al] = divColor(v * 2.2); ctx.globalAlpha = al; ctx.fillStyle = c; ctx.fillRect(LW + k * cw, 16 + a * rh, cw - .3, rh - .8); });
    ctx.globalAlpha = 1;
    const nl = Q.q_norm[a] / Math.max(...Q.q_norm) * 44;
    ctx.fillStyle = muted; ctx.fillRect(LW + 96 * cw + 8, 16 + a * rh + 2, nl, rh - 4);
  });
  ctx.strokeStyle = ink; ctx.lineWidth = 1.2; ctx.strokeRect(LW - .5, 16 + sel * rh - .5, 96 * cw, rh);
  ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.font = "10px 'IBM Plex Mono', monospace";
  ctx.fillText(`96 / ${Dm} chiều đầu`, LW, 8); ctx.fillText("độ dài", LW + 96 * cw + 8, 8);
  document.getElementById("qmat-cap").textContent = `Ma trận query đã học của ${name}. Mỗi hàng là một query và mỗi cột là một chiều. Màu xanh là giá trị dương, màu cam là giá trị âm. Thanh bên phải là độ dài của từng query.`;
  // cosine similarity between the 40 queries
  const c2 = 8.4, W2 = LW + 40 * c2 + 6, H2 = 40 * c2 + 22;
  ctx = canvasCtx(document.getElementById("qcos"), W2, H2);
  ctx.font = "10px 'IBM Plex Sans', sans-serif"; ctx.textBaseline = "middle";
  Q.cos.forEach((row, i) => {
    ctx.fillStyle = i === sel ? cssVar("--joint") : muted; ctx.textAlign = "right"; ctx.fillText(vn(A[i]), LW - 6, 16 + i * c2 + c2 / 2);
    row.forEach((v, j) => { if (i === j) { ctx.globalAlpha = 1; ctx.fillStyle = ink; } else { const [c, al] = divColor(v / .6); ctx.globalAlpha = al; ctx.fillStyle = c; }
      ctx.fillRect(LW + j * c2, 16 + i * c2, c2 - .4, c2 - .4); });
    ctx.globalAlpha = 1;
  });
  ctx.strokeStyle = cssVar("--joint"); ctx.lineWidth = 1.4;
  ctx.strokeRect(LW - .5, 16 + sel * c2 - .5, 40 * c2, c2); ctx.strokeRect(LW + sel * c2 - .5, 15.5, c2, 40 * c2);
  ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.font = "10px 'IBM Plex Mono', monospace"; ctx.fillText("cos(q_a, q_b), cột theo cùng thứ tự", LW, 8);
  document.getElementById("qcos-cap").textContent = `Cosine giữa từng cặp query của ${name}. Đường chéo đậm có giá trị 1. Ô xanh là hai query cùng hướng, ô cam là hai query ngược hướng. Phần lớn ô gần 0, nên các query gần như độc lập với nhau. Hàng và cột màu tím ứng với thuộc tính đang chọn.`;
  typeset("qintro", "qeq", "qfacts");
}

/* ---------- start ---------- */
buildChips(); buildAttrs(); setMode("inf"); setAttr(state.attr);
inImg.onload = () => drawMaps(state.attr);
const rerender = () => { drawMaps(state.attr); drawQueries(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
new MutationObserver(rerender).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
if (!reduce) play();
