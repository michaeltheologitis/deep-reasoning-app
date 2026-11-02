// The tree's renderers on drives a crafted page may hold. Node has no DOM, so the document is faked at its boundary.
import assert from "node:assert/strict";
import test from "node:test";
import { known, opened, renderAgent, renderStep } from "../src/deep_reasoning/web/tree.js";

class FakeElement {
  constructor(tag) {
    Object.assign(this, { tag, className: "", attributes: {}, children: [] });
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  addEventListener() {}
  append(...kids) {
    this.children.push(...kids);
  }
  get textContent() {
    return this.children.map((kid) => (typeof kid === "object" ? kid.textContent : String(kid))).join("");
  }
}
globalThis.document = { createElement: (tag) => new FakeElement(tag) };

const drive = (fields) => ({ node: 2, drive: 1, task: "Count.", namespace: "root", status: "done", result: "1", seconds: 0.5, cost: 0.001, unpriced: false, steps: [], ...fields });
const ON_THE_PROTOTYPE = ["constructor", "__proto__", "toString", "hasOwnProperty"];
const ROOMS = ["A", "B", "C", "D"].map((room) => drive({ task: `Count room ${room}.` }));
const RAISED = "Traceback (most recent call last):\n  File \"<string>\", line 2\nValueError: invalid literal for int() with base 10: 'five'";
// A delegating cell, opened onto its sub-agents' rows; and its parts, by class, in the order drawn.
const delegation = (fields) => ({ kind: "cell", code: "seats = run_all(...)\nprint(sum(map(int, seats)))", agents: ROOMS, ...fields });
const openDrawn = (cell) => {
  opened.add(cell);
  return renderStep(cell, false);
};
const parts = (el) => el.children.map((kid) => kid.className.trim().split(" ")[0]);
// The meta's words after the code toggle that leads it, which CSS sets off with a separator.
const metaOf = (el) => {
  const [toggle, ...rest] = el.children[0].children.find((kid) => kid.className === "meta").children;
  assert.equal(toggle.className, "show-code");
  return rest.map((kid) => (typeof kid === "object" ? kid.textContent : String(kid))).join("");
};

test("a key on Object.prototype is no table's entry", () => {
  const table = { failed: ["failed", "err"] };
  for (const key of ON_THE_PROTOTYPE) assert.deepEqual(known(table, key), []);
  assert.deepEqual(known(table, "failed"), ["failed", "err"]);
});

for (const status of ON_THE_PROTOTYPE) {
  test(`a drive whose status is ${status}, as a crafted page may hold, draws as one of no known status`, () => {
    const text = renderAgent(drive({ status })).textContent;
    assert.match(text, /Count\./);
    assert.doesNotMatch(text, /function|Object/);
  });
}

test("a copied drive left running, which has no start, shows no time, alone or in its cell", () => {
  const running = drive({ status: "running", seconds: null, cost: null });
  const cell = { kind: "cell", code: "subagent('Count.')", output: null, status: "running", agents: [running] };
  for (const el of [renderAgent(running), renderStep(cell, false)]) assert.doesNotMatch(el.textContent, /NaN/);
});

test("a delegating cell that raised after its sub-agents answered says so, and shows its error under its own line, above their rows", () => {
  const el = openDrawn(delegation({ output: RAISED, status: "failed" }));
  assert.match(metaOf(el), /^cell failed · 4 done/);
  assert.deepEqual(parts(el), ["node", "out", "rows"]);
  assert.match(el.children[1].className, /\berr\b/);
  assert.match(el.children[1].textContent, /ValueError: invalid literal/);
});

test("a delegating cell that is done shows its sub-agents' rows under its line, and no failure", () => {
  const el = openDrawn(delegation({ output: "72", status: "done" }));
  assert.doesNotMatch(metaOf(el), /failed/);
  assert.deepEqual(parts(el), ["node", "rows"]);
});
