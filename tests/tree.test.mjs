// The tree's renderers on drives a crafted page may hold, and answers as they show. Node has no DOM, so the document is faked
// at its boundary; marked is the vendored copy the page loads.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { clamped, known, markdownHtml, opened, renderAgent, renderAnswer, renderStep } from "../src/deep_reasoning/web/tree.js";

globalThis.marked = createRequire(import.meta.url)("../src/deep_reasoning/web/vendor/marked.min.js");

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
const GROUP = [
  "  + Exception Group Traceback (most recent call last):",
  "  |   File \"<string>\", line 1, in <module>",
  "  | ExceptionGroup: x (2 sub-exceptions)",
  "  +-+---------------- 1 ----------------",
  "    | ValueError: 1",
  "    +---------------- 2 ----------------",
  "    | KeyError: 2",
  "    +------------------------------------",
].join("\n");
// What a failed cell printed, and the line it shows closed.
const ERRORS = [
  ["a traceback", RAISED, "ValueError: invalid literal for int() with base 10: 'five'"],
  ["a traceback, then the answer it gave before raising", `${RAISED}\nFinalAnswer: 'x'`, "ValueError: invalid literal for int() with base 10: 'five'"],
  ["a message over lines", "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nValueError: 2 problems:\ntools: bad", "ValueError: 2 problems:"],
  ["an exception group", GROUP, "ExceptionGroup: x (2 sub-exceptions)"],
  ["no traceback, then an answer", "worker gone\nFinalAnswer: 'x'", "worker gone"],
];
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

for (const [name, output, shown] of ERRORS) {
  test(`a failed cell that printed ${name} shows "${shown}" closed`, () => {
    const el = renderStep({ kind: "cell", code: "seats = count()", output, status: "failed", agents: [] }, false);
    assert.deepEqual(parts(el), ["line", "out"]);
    assert.equal(el.children[1].textContent, `⎿${shown}`);
  });
}

test("a cell closed failed as its agent ended, with no output, draws no empty error line, alone or delegating", () => {
  const closed = { code: "seats = count()", output: "", status: "failed" };
  assert.deepEqual(parts(renderStep({ kind: "cell", ...closed, agents: [] }, false)), ["line"]);
  const el = openDrawn(delegation(closed));
  assert.match(metaOf(el), /^cell failed · 4 done/);
  assert.deepEqual(parts(el), ["node", "rows"]);
});

// Answers seen mangled when every answer went through Markdown, and others that hold no Markdown.
const PLAIN_ANSWERS = [
  "List<String>",
  "The constructor is __init__.",
  "f(*args, **kwargs); 2*3*4",
  "Edit json/__init__.py",
  "<!doctype html><html><body><h1>Hi</h1><img src=\"http://example.com/x.png\"></body></html>",
  "x**2 + y**2, regex \\d+\\.\\d+, C:\\Users\\_temp and &lt;",
  "two\nlines\n\nthen a paragraph",
];
const MARKDOWN_ANSWERS = [
  ["a list", "- first\n- second", /<ul>\n<li>first<\/li>\n<li>second<\/li>\n<\/ul>/],
  ["a table", "| room | seats |\n|---|---:|\n| A | 12 |", /<table>[\s\S]*<th>room<\/th>[\s\S]*<td align="right">12<\/td>/],
  ["a code block", "```python\nprint(1 < 2)\n```", /<pre><code class="language-python">print\(1 &lt; 2\)\n<\/code><\/pre>/],
  ["bold text", "The answer is **42**.", /<p>The answer is <strong>42<\/strong>\.<\/p>/],
];
const anchors = (html) => html.match(/<a [^>]*>/g) ?? [];

for (const text of PLAIN_ANSWERS) {
  test(`an answer with no Markdown shows exactly as written: ${JSON.stringify(text)}`, () => {
    assert.equal(markdownHtml(text), null);
    const el = renderAnswer(text);
    assert.equal(el.textContent, text);
    assert.match(el.className, /\bplain\b/);
  });
}

for (const [name, text, html] of MARKDOWN_ANSWERS) {
  test(`an answer with ${name} shows as Markdown`, () => {
    assert.match(markdownHtml(text), html);
  });
}

test("in a Markdown answer, code-like text and raw HTML show as written", () => {
  const html = markdownHtml("- __init__, json/__init__.py and 2*3*4\n- List<String>\n\n<div onclick=\"x()\">hi</div>");
  assert.match(html, /__init__, json\/__init__\.py and 2\*3\*4/);
  assert.match(html, /List&#60;String&#62;/);
  assert.match(html, /&#60;div onclick=&#34;x\(\)&#34;&#62;hi&#60;\/div&#62;/);
  assert.doesNotMatch(html, /<(strong|em|div)\b/);
});

test("an image in an answer is a link to it, never fetched", () => {
  const html = markdownHtml("A ![chart](http://127.0.0.1:9/leak?q=secret) and ![](http://127.0.0.1:9/plain.png).");
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /<a [^>]*href="http:\/\/127\.0\.0\.1:9\/leak\?q=secret">chart<\/a>/);
  assert.match(html, /<a [^>]*href="http:\/\/127\.0\.0\.1:9\/plain\.png">http:\/\/127\.0\.0\.1:9\/plain\.png<\/a>/);
});

test("every link in an answer opens in a new tab, with no opener or referrer", () => {
  const links = anchors(markdownHtml("[a](https://a.example), https://b.example, <https://c.example> and ![d](https://d.example/d.png)"));
  assert.equal(links.length, 4);
  for (const link of links) assert.match(link, /^<a target="_blank" rel="noopener noreferrer" href=/);
});

test("a long text shows its first lines with a toggle to show it all, and all of it once shown whole", () => {
  const turn = {};
  const whole = new WeakSet();
  const draw = (clamp) => ({ className: clamp });
  const text = "line\n".repeat(30);
  const [shown, toggle] = clamped(turn, whole, "prompt", text, 12, draw);
  assert.equal(shown.className, "clamped");
  assert.equal(toggle.textContent, "show all · 31 lines");
  whole.add(turn);
  const [all, back] = clamped(turn, whole, "prompt", text, 12, draw);
  assert.equal(all.className, "");
  assert.equal(back.textContent, "show less");
  assert.equal(clamped({}, whole, "prompt", "short", 12, draw).className, "");
});

test("one very long line counts as the lines it wraps to", () => {
  const [, toggle] = clamped({}, new WeakSet(), "answer", "x".repeat(100000), 200, () => ({}));
  assert.equal(toggle.textContent, "show all · 1,000 lines");
});
