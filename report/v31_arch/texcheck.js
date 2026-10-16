const {JSDOM} = require("jsdom"); const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8").replace(/<script src="https:[^"]+"><\/script>/g, "");
const dom = new JSDOM("<!doctype html><html><body>" + html + "</body></html>", {runScripts: "dangerously", pretendToBeVisual: true,
  beforeParse(w) { w.SVGElement.prototype.getTotalLength = () => 100; w.SVGElement.prototype.getPointAtLength = () => ({x: 0, y: 0});
    w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {get: () => () => {}, set: () => true}); w.matchMedia = () => ({matches: true, addEventListener() {}}); }});
setTimeout(() => { const d = dom.window.document;
  const t = ["loss-sub", "loss-steps", "inf-steps", "qintro", "qfacts"].map(i => d.getElementById(i).textContent).join(" ");
  const bad = t.match(/[\t\v]|(?<!\\)\((z|p|s|y)\)/g);
  console.log("tab/vtab or broken inline math:", bad ? bad.slice(0, 5) : "none");
  console.log("inline sample:", (t.match(/\\\([^)]{0,30}/g) || []).slice(0, 4));
  process.exit(0); }, 400);
