// The tree's renderers on drives a crafted page may hold, and answers as they show. Node has no DOM, so the document is faked
// at its boundary; marked is the vendored copy the page loads.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { CLAMPS, COST_NA, SANITIZE, UNFINISHED, WAITING, clamped, costView, element, formatCost, formatTotal, known, markdownHtml, opened, plainFailure, renderAgent, renderAnswer, renderStep, stepLabel } from "../src/deep_reasoning/web/tree.js";

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
  [
    "an exception from a module, which Python names with it",
    "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\ndeep_reasoner.v2.namespaces.NamespaceSpawnError: root may not spawn into librarian",
    "NamespaceSpawnError: root may not spawn into librarian",
  ],
  [
    "a provider's error",
    "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nopenai.NotFoundError: Error code: 404 - {'error': {'message': 'The model `fake-l` does not exist'}}",
    "The provider has no model fake-l (404) · NotFoundError: Error code: 404 - {'error': {'message': 'The model `fake-l` does not exist'}}",
  ],
];
// A delegating cell, opened onto its sub-agents' rows; and its parts, by class, in the order drawn.
const delegation = (fields) => ({ kind: "cell", code: "seats = run_all(...)\nprint(sum(map(int, seats)))", agents: ROOMS, ...fields });
const openDrawn = (cell) => {
  opened.add(cell);
  return renderStep(cell, false);
};
const parts = (el) => el.children.map((kid) => kid.className.trim().split(" ")[0]);
// Each element under el, at any depth, that passes test.
const findAll = (el, test) => el.children.filter((kid) => typeof kid === "object").flatMap((kid) => [...(test(kid) ? [kid] : []), ...findAll(kid, test)]);
// Every text in an element, a screen reader's included.
const textOf = (el) => (typeof el === "object" ? el.children.map(textOf).join("") : String(el));
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
  assert.match(metaOf(el), /^4 done, then the cell failed · /);
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
  assert.match(metaOf(el), /^4 done, then the cell failed · /);
  assert.deepEqual(parts(el), ["node", "rows"]);
});

test("an opened failed cell shows its traceback without the engine's frame that ran it, plain or an exception group's", () => {
  const events = readFileSync(new URL("runs/raising.jsonl", import.meta.url), "utf8").trim().split("\n").map(JSON.parse);
  const failed = events.filter((event) => event.kind === "cell.end" && event.status === "failed");
  assert.ok(failed.length >= 2 && failed.every((event) => event.output.includes("exec(code")));
  for (const { code, output } of failed) {
    const cell = { kind: "cell", code, output, status: "failed", agents: [] };
    const shown = openDrawn(cell).children.at(-1).textContent;
    assert.doesNotMatch(shown, /exec\(code|deep_reasoner/);
    assert.match(shown, /File "<string>", line \d+, in <module>\n/);
    const lines = output.split("\n");
    assert.equal(shown, `⎿${lines.toSpliced(lines.findIndex((line) => line.includes("backends.py")), 3).join("\n")}`);
  }
});

test("a cell's line says how it ended in words as well as colour, for a screen reader, and one stopped or cut off is hollow", () => {
  for (const [status, glyph] of [["failed", "●"], ["stopped", "○"], ["cut off", "○"], ["running", "●"]]) {
    const line = renderStep({ kind: "cell", code: "print(1/0)", output: null, status, agents: [] }, false).children[0];
    assert.match(textOf(line), new RegExp(`${glyph} ${status}Python\\(print\\(1/0\\)\\)`));
  }
  const done = renderStep({ kind: "cell", code: "print(1)", output: "1", status: "done", agents: [] }, false).children[0];
  assert.equal(textOf(done), "●Python(print(1))");
});

test("a sub-agent a failed run cut off says so on its row, and its cell's glyph takes one class", () => {
  assert.match(renderAgent(drive({ status: "cut off", result: null })).textContent, /cut off · 0\.5s/);
  const cell = renderStep({ kind: "cell", code: "print(1)", output: null, status: "cut off", agents: [] }, false);
  assert.equal(cell.children[0].children[1].className, "glyph cut-off");
});

test("a sub-agent on another model than the run's shows its model as a tag, whole in its title", () => {
  const model = "a-model-with-a-very-long-name-v2-instruct-128k";
  const tags = renderAgent(drive({ model })).children[0].children.filter((kid) => kid.className === "tag");
  assert.deepEqual(tags.map((tag) => [tag.textContent, tag.attributes.title]), [[model, model]]);
});

test("an open sub-agent waiting on its first call says so, and one that has steps doesn't", () => {
  const agent = drive({ status: "running", result: null, seconds: null, started: null, steps: [] });
  opened.add(agent);
  assert.match(renderAgent(agent).textContent, new RegExp(WAITING));
  const busy = { ...agent, steps: [{ kind: "think", text: "Hmm." }] };
  opened.add(busy);
  assert.doesNotMatch(renderAgent(busy).textContent, new RegExp(WAITING));
});

test("a run whose only calls never finished costs N/A, saying why, not $0", () => {
  const shown = costView({ cost: null, unpriced: false, unfinished: true });
  assert.deepEqual([shown.textContent, shown.attributes.title], [COST_NA, UNFINISHED]);
  assert.equal(costView({ cost: 0, unpriced: false, unfinished: false }), "$0");
});

// Costs as every row and the total show them: under a dollar to a hundredth of a cent, from a dollar to the cent.
const COSTS = [
  [0.00004, "<$0.0001"],
  [0.0001, "$0.0001"],
  [0.009996, "$0.0100"],
  [0.01, "$0.0100"],
  [0.0123, "$0.0123"],
  [0.56, "$0.5600"],
  [0.99996, "$1.00"],
  [1.23456, "$1.23"],
  [12345.678, "$12,345.68"],
];

test("every cost follows one rule: one cent always one way, a tiny one never $0.0000, thousands separated", () => {
  for (const [usd, shown] of COSTS) assert.equal(formatCost(usd), shown, usd);
});

test("a chat's total is never finer than its turns' costs, so the shown costs add up to it", () => {
  assert.equal(formatTotal([0.0123, 0.0049]), "$0.0172");
  assert.deepEqual([0.0123, 0.0049].map(formatCost), ["$0.0123", "$0.0049"]);
  assert.equal(formatTotal(Array(23).fill(0.0011)), "$0.0253");
  assert.equal(formatTotal([0.25, 0.0011, null]), "$0.2511");
  assert.equal(formatTotal([0.6, 0.6]), "$1.20");
  assert.equal(formatTotal([0.00004]), "<$0.0001");
  assert.equal(formatTotal([0, null]), null);
});

// A provider's errors as the OpenAI client words them, and what each shows first.
const PROVIDER_ERRORS = [
  ["NotFoundError: Error code: 404 - {'error': {'message': 'The model `gpt-6-lun` does not exist or you do not have access to it.'}}", "The provider has no model gpt-6-lun (404)"],
  ["NotFoundError: Error code: 404 - {'error': 'model \"llama9\" not found'}", "The provider has no model llama9 (404)"],
  ["NotFoundError: Error code: 404 - {'error': 'no such model'}", "The provider has no model by that name (404)"],
  ["NotFoundError: Error code: 404 - Not Found", "Nothing answers at the provider's address; is its URL right? (404)"],
  ["APIConnectionError: Connection error.", "Couldn't reach the provider"],
  ["APITimeoutError: Request timed out.", "The provider didn't answer in time"],
  ["AuthenticationError: Error code: 401 - {'error': 'bad key'}", "The provider refused the key; check it in settings (401)"],
  ["PermissionDeniedError: Error code: 403 - {}", "The provider denied this key access (403)"],
  ["RateLimitError: Error code: 429 - {}", "The provider's rate limit or quota ran out (429)"],
  ["InternalServerError: Error code: 503 - {}", "The provider failed (503)"],
];

for (const [raw, plain] of PROVIDER_ERRORS) {
  test(`a provider's error leads with plain words: ${raw.split(" -")[0]}, ${plain}`, () => {
    assert.equal(plainFailure(raw), plain);
  });
}

test("a 404 names the model asked for, when the caller knows it", () => {
  assert.equal(plainFailure("NotFoundError: Error code: 404 - {'error': 'no such model'}", "fake-l"), "The provider has no model fake-l (404)");
});

test("any other failure, a provider's own reason included, has no plain words of its own", () => {
  const reasons = [
    "BadRequestError: Error code: 400 - {'error': {'message': \"This model's maximum context length is 128000 tokens\"}}",
    "APIStatusError: Error code: 413 - {}",
    "UnprocessableEntityError: Error code: 422 - {}",
    "ValueError: Error code: 404",
    "RecursionBudgetExceeded: depth 7",
    "worker gone",
  ];
  for (const raw of reasons) assert.equal(plainFailure(raw), null);
});

test("a sub-agent that failed on a provider's error leads with plain words, a short reason after", () => {
  const raw = "NotFoundError: Error code: 404 - {'error': {'message': 'The model `fake-small` does not exist'}}";
  const [, under] = renderAgent(drive({ status: "failed", result: raw })).children;
  assert.equal(under.textContent, `⎿The provider has no model fake-small (404) · ${raw}`);
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
  "2**10** is big",
  "about ~5-~10 minutes",
  "café**x** and cafe**x**",
  "这是重要的结论。",
];
const MARKDOWN_ANSWERS = [
  ["a list", "- first\n- second", /<ul>\n<li>first<\/li>\n<li>second<\/li>\n<\/ul>/],
  ["a table", "| room | seats |\n|---|---:|\n| A | 12 |", /<table>[\s\S]*<th>room<\/th>[\s\S]*<td align="right">12<\/td>/],
  ["a code block", "```python\nprint(1 < 2)\n```", /<pre><code class="language-python">print\(1 &lt; 2\)\n<\/code><\/pre>/],
  ["bold text", "The answer is **42**.", /<p>The answer is <strong>42<\/strong>\.<\/p>/],
  ["a double-tilde strike", "~~struck~~ out", /<p><del>struck<\/del> out<\/p>/],
  // written with no spaces between words, so ** sits against a letter on each side
  ["bold in Chinese", "这是**重要**的结论。", /<p>这是<strong>重要<\/strong>的结论。<\/p>/],
  ["bold in Chinese before full-width punctuation", "北京是中国的**首都**，也是最大的城市之一。", /<p>北京是中国的<strong>首都<\/strong>，/],
  ["bold in Japanese", "これは**とても重要**なポイントです。", /<p>これは<strong>とても重要<\/strong>なポイントです。<\/p>/],
  ["bold in katakana", "答えは**データベース**です。", /<strong>データベース<\/strong>です/],
  ["bold in Korean", "이것은**중요한**결론입니다.", /<p>이것은<strong>중요한<\/strong>결론입니다\.<\/p>/],
  ["bold in Thai", "นี่คือ**คำตอบ**ที่ถูกต้อง", /<p>นี่คือ<strong>คำตอบ<\/strong>ที่ถูกต้อง<\/p>/],
  ["emphasis in Chinese", "这是*斜体*的字。", /<p>这是<em>斜体<\/em>的字。<\/p>/],
];
const anchors = (html) => html.match(/<a [^>]*>/g) ?? [];
// DOMPurify's own rule for an attribute, on the page's settings; that rule reads no DOM, so its window is a stub
const purify = createRequire(import.meta.url)("../src/deep_reasoning/web/vendor/purify.min.js")({ document: { nodeType: 9, createElement: () => ({}) }, Element: class {} });
purify.setConfig(SANITIZE);
// Each tag and attribute in marked's HTML that the sanitiser would drop.
const dropped = (html) =>
  [...html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*)>/g)].flatMap(([, tag, attributes]) => [
    ...(SANITIZE.ALLOWED_TAGS.includes(tag) ? [] : [tag]),
    ...[...attributes.matchAll(/([a-z-]+)="([^"]*)"/g)].filter(([, name, value]) => !purify.isValidAttribute(tag, name, value)).map(([attribute]) => `${tag} ${attribute}`),
  ]);

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

test("an answer's links, aligned columns, numbered list and task list pass the sanitiser whole", () => {
  const html = markdownHtml("See [docs](https://a.example/docs), <mailto:me@a.example> and [below](#part).\n\n| n | sq |\n|:---:|---:|\n| 3 | 9 |\n\n3. third\n4. fourth\n\n- [x] done\n- [ ] todo");
  assert.equal(anchors(html).length, 3);
  for (const link of anchors(html)) assert.match(link, /^<a target="_blank" rel="noopener noreferrer" href=/);
  assert.match(html, /<td align="center">3<\/td>\n<td align="right">9<\/td>/);
  assert.match(html, /<ol start="3">/);
  assert.match(html, /<input checked="" disabled="" type="checkbox">/);
  assert.deepEqual(dropped(html), []);
});

test("a link to a file, a sandbox: or a javascript: address reaches the sanitiser as its text and address", () => {
  const html = markdownHtml("# Saved\n\n[the CSV](squares.csv), [chart](sandbox:/mnt/data/chart.svg) and [run](javascript:alert(1)).");
  assert.equal(anchors(html).length, 0);
  assert.match(html, /<p>the CSV \(squares\.csv\), chart \(sandbox:\/mnt\/data\/chart\.svg\) and run \(javascript:alert\(1\)\)\.<\/p>/);
  assert.deepEqual(dropped(html), []);
});

test("a long prompt shows its first lines with a toggle to show it all, and all of it once shown whole", () => {
  const turn = {};
  const text = "line\n".repeat(30);
  const draw = () => element("div", { class: "user" }, text);
  const [shown, toggle] = clamped(turn, "prompt", text, draw);
  assert.equal(shown.className, "user clamped");
  assert.equal(shown.attributes.style, `--clamp-lines: ${CLAMPS.prompt.shown}`);
  assert.equal(toggle.textContent, "show all · 31 lines");
  assert.equal(toggle.attributes["aria-expanded"], "false");
  CLAMPS.prompt.whole.add(turn);
  const [all, back] = clamped(turn, "prompt", text, draw);
  assert.equal(all.className, "user");
  assert.equal(back.textContent, "show less");
  assert.equal(clamped({}, "prompt", "short", draw).className, "user");
});

test("one very long line counts as the lines it wraps to", () => {
  const [, toggle] = clamped({}, "answer", "x".repeat(100000), () => element("div"));
  assert.equal(toggle.textContent, "show all · 1,000 lines");
});

for (const tag of ["code", "pre", "kbd", "script"]) {
  test(`raw HTML after a <${tag}> tag in a Markdown answer still shows as text`, () => {
    const html = markdownHtml(`# Answer\n\nSee <${tag}><img/src=//evil.example/x.png> <p/style="background:url(//evil.example/y)">`);
    assert.doesNotMatch(html, /<(img|p\/)/);
  });
}

test("in a Markdown answer, a run of asterisks no emphasis takes, one tilde, or a letter beyond ASCII stay as written", () => {
  const html = markdownHtml("# Notes\n\n2**10** is big, about ~5-~10 minutes, café**x** and cafe**x**");
  assert.match(html, /2\*\*10\*\* is big, about ~5-~10 minutes, café\*\*x\*\* and cafe\*\*x\*\*/);
});

test("an image inside a link is the link's text, so no link holds another", () => {
  const html = markdownHtml("[![badge](https://img.example/b.png)](https://target.example)");
  assert.equal(anchors(html).length, 1);
  assert.match(html, /href="https:\/\/target\.example">badge<\/a>/);
});

test("a link or image to an address DOMPurify would drop shows as its text and the address, not as a link", () => {
  const html = markdownHtml("See [the chart](sandbox:/mnt/data/chart.svg), ![plot](sandbox:/x.png), ![](sandbox:/y.png) and [run](javascript:alert(1)).");
  assert.equal(anchors(html).length, 0);
  assert.match(html, /the chart \(sandbox:\/mnt\/data\/chart\.svg\), plot \(sandbox:\/x\.png\), sandbox:\/y\.png and run \(javascript:alert\(1\)\)\./);
});

test("a link to an address DOMPurify keeps stays a link, its address shown only by the link", () => {
  const html = markdownHtml("[home](https://a.example) and [mail](mailto:me@a.example) and [below](#part)");
  assert.equal(anchors(html).length, 3);
  assert.doesNotMatch(html, /\(https:|\(mailto:|\(#/);
});

test("a link to a file in the chat's folder, which the app doesn't serve, shows as its text and the address", () => {
  const html = markdownHtml("Saved: [Download the CSV](first_20_squares.csv), [cubes.csv](cubes.csv) and [chart](./out/chart.png) [page](/pages/x).");
  assert.equal(anchors(html).length, 0);
  assert.match(html, /Download the CSV \(first_20_squares\.csv\), cubes\.csv and chart \(\.\/out\/chart\.png\) page \(\/pages\/x\)\./);
});

test("the sanitiser keeps an answer's link to the web, mail or a place in it only", () => {
  for (const href of ["https://a.example", "http://a.example/x?y", "mailto:me@a.example", "#part"]) {
    assert.ok(purify.isValidAttribute("a", "href", href), href);
  }
  for (const href of ["javascript:alert(1)", "JaVaScRiPt:x", "data:text/html,x", "vbscript:x", "file:///etc/passwd", "ftp://a.example", "/pages/x", "notes.html", "./a.csv", "//a.example"]) {
    assert.ok(!purify.isValidAttribute("a", "href", href), href);
  }
});

test("a cell that raised or was stopped before its FinalAnswer gave one reads as a Python cell, not the final answer", () => {
  const cell = { kind: "cell", code: "FinalAnswer('it's)", output: "SyntaxError: unterminated string literal", status: "failed", agents: [] };
  assert.equal(stepLabel(cell), "Python(FinalAnswer('it's))");
  assert.equal(stepLabel({ ...cell, output: null, status: "stopped" }), "Python(FinalAnswer('it's))");
  assert.equal(stepLabel({ ...cell, output: "FinalAnswer: 'x'", status: "done" }), "Final answer");
});

test("a worked example left out as recorded before it said where it is stored names the agent's namespace, or none", () => {
  const head = (step) => renderStep({ kind: "left_out", name: "tally", why: "ZeroDivisionError", ...step }, false).children[0].textContent;
  assert.equal(head({ namespace: "stats" }), '◆Left out worked example "tally" from stats');
  assert.equal(head({ namespace: "stats", from: null }), '◆Left out worked example "tally" from stats');
  assert.equal(head({}), '◆Left out worked example "tally"');
});

test("a worked example left out says which, where it is stored and why, each field as text and why quoted as code", () => {
  const why = "UndefinedError: 'dict object' has no attribute 'the Library file is damaged: run reset'";
  const step = { kind: "left_out", name: "<b>tally</b>", namespace: "stats", from: { odd: 1 }, why };
  const el = renderStep(step, false);
  assert.deepEqual(parts(el), ["line", "out"]);
  assert.equal(el.children[0].textContent, '◆Left out worked example "<b>tally</b>" from {"odd":1}');
  assert.equal(el.children[1].textContent, `⎿couldn't be rendered: ${why}`);
  const [quoted] = findAll(el, (kid) => kid.tag === "code");
  assert.equal(quoted.textContent, why);
  assert.equal(stepLabel(step), 'Left out worked example "<b>tally</b>"');
  const long = renderStep({ ...step, why: "x".repeat(10_000) }, false);
  assert.equal(findAll(long, (kid) => kid.tag === "code")[0].textContent, `${"x".repeat(299)}…`);
});
