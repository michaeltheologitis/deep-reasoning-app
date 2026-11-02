// The tree's renderers on drives a crafted page may hold. Node has no DOM, so the document is faked at its boundary.
import assert from "node:assert/strict";
import test from "node:test";
import { known, renderAgent, renderStep } from "../src/deep_reasoning/web/tree.js";

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
