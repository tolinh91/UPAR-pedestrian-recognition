const {JSDOM} = require("jsdom");
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8").replace(/<script src="https:[^"]+"><\/script>/g, "");
const dom = new JSDOM("<!doctype html><html><head></head><body>" + html + "</body></html>", {runScripts: "dangerously", pretendToBeVisual: true,
  beforeParse(w) {
    w.SVGElement.prototype.getTotalLength = () => 100; w.SVGElement.prototype.getPointAtLength = () => ({x: 0, y: 0});
    w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {get: () => () => {}, set: () => true});
    w.matchMedia = () => ({matches: true, addEventListener() {}});
  }});
const errs = []; dom.window.addEventListener("error", e => errs.push(e.message));
setTimeout(() => {
  const d = dom.window.document;
  const click = sel => d.querySelector(sel).dispatchEvent(new dom.window.MouseEvent("click", {bubbles: true}));
  try {
    click("#m-train"); for (let i = 0; i < 8; i++) click("#next");
    click("#m-inf"); for (let i = 0; i < 6; i++) click("#next");
    d.querySelectorAll(".chip").forEach(c => c.dispatchEvent(new dom.window.MouseEvent("click", {bubbles: true})));
    d.querySelectorAll("#qhsec .seg button").forEach(c => c.dispatchEvent(new dom.window.MouseEvent("click", {bubbles: true})));
  } catch (e) { errs.push(String(e)); }
  console.log("errors:", errs.length ? errs : "none");
  console.log("svg nodes", d.querySelectorAll("#dg *").length, "qh nodes", d.querySelectorAll("#qh *").length,
    "attr rows", d.querySelectorAll(".arow").length, "chips", d.querySelectorAll(".chip").length);
  console.log("loss-total:", d.getElementById("loss-total").textContent.slice(0, 140));
  console.log("caption:", d.getElementById("capt").textContent.slice(0, 80));
  process.exit(0);
}, 500);
