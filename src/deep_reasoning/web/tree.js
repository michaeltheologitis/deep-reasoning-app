// Draws a run's tree as the chat shows it: its steps, and its sub-agents as rows that click open and shut, each redrawn alone.
// The chat (app.js) and a page's tab (page.js) both draw with it.
import { CUT_OFF, asText, changeText, clip } from "./fold.js";

const FIRST_ROWS = 20;
const MORE_ROWS = 100;
const SECONDS_PER_MINUTE = 60;
const MS_PER_SECOND = 1000;
const FROM_NEXT = "from the next message";
const LEFT_OUT = "Left out";
// as the worker cuts why it left a worked example out; a shared file's record may hold more
const WHY_CHARS = 300;
const UNKNOWN_COST = "N/A";
// no-break spaces keep a cost whole where its line wraps
const COST_NA = `cost\u00a0${UNKNOWN_COST}`;
const PRICE_UNKNOWN = "this model's price isn't known";
const UNFINISHED = "a model call never finished, so what it cost isn't known";
const WAITING = "waiting for the model…";
// before the root agent starts: its run is building the tools its namespaces grant
const STARTING = "starting…";
const FINAL_ANSWER = "FinalAnswer: ";
// marked's tokens that leave an answer plain: shown as written, raw HTML included
const PLAIN_TOKENS = new Set(["paragraph", "space", "text", "escape", "br", "html"]);
// not the scripts written without spaces, where ** sits against letters
const WORD_CHAR = /(?![\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Hangul}\p{scx=Thai}\p{scx=Lao}\p{scx=Khmer}\p{scx=Myanmar}\uFF00-\uFFEF])[\p{L}\p{N}_]/u;
const NEW_TAB_LINK = '<a target="_blank" rel="noopener noreferrer" ';
const ANCHOR_TAG = /<\/?a\b[^>]*>/g;
// what the HTML marked makes may hold, and no more: its tags and attributes, and links to the web, mail or a place in the
// answer; the app serves nothing at a relative address, such as a file's name in the chat's folder
const SANITIZE = {
  ALLOWED_TAGS: ["a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "input", "li", "ol", "p", "pre", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul"],
  ALLOWED_ATTR: ["align", "checked", "class", "disabled", "href", "rel", "start", "target", "title", "type"],
  ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|#)/i,
};
const THEME_KEY = "theme";
const LINE_CHARS = 100;
// a prompt or answer past about `past` lines shows its first `shown` until shown whole
const CLAMPS = { prompt: { past: 12, shown: 8, whole: new WeakSet() }, answer: { past: 200, shown: 40, whole: new WeakSet() } };
const TRACEBACK = /^(?:Traceback| {2}\+ Exception Group Traceback) \(most recent call last\):$/;
// an exception group's lines hang off its frame, "  | "; the exception's own line is the first after the head not indented within it
const GROUP_FRAME = /^ {2}\| /;
const EXCEPTION = /^(?: {2}\| )?[^\s|+]/;
// the frame of the engine's REPL atop a cell's traceback, plain or an exception group's, with its source and carets
const EXEC_FRAME =
  /^((?:Traceback| {2}\+ Exception Group Traceback) \(most recent call last\):\n)((?: {2}\| )?) {2}File "[^"\n]*deep_reasoner[/\\][^"\n]*", line \d+, in \S+\n\2 {4}\S.*\n(?:\2 {4}[ ~^]*\^[ ~^]*\n)?/gm;
// Python names an exception from outside builtins with its module, deep_reasoner.v2.namespaces.X, which a cut line would show alone
const EXCEPTION_MODULE = /^(?:[A-Za-z_]\w*\.)+(?=[A-Za-z_]\w*(?::|$))/;
// as the OpenAI client words a provider's errors: "NotFoundError: Error code: 404 - {...}"
const PROVIDER_STATUS = /^(?:openai\.)?(?:APIStatus|Authentication|PermissionDenied|NotFound|RateLimit|InternalServer)Error: Error code: (\d{3})\b/;
const UNREACHABLE = /^(?:openai\.)?APIConnectionError: /;
const TIMED_OUT = /^(?:openai\.)?APITimeoutError: /;
// the model a provider's 404 names: "The model `x` does not exist", 'model "x" not found'
const NAMED_MODEL = /\bmodel\s*[`'"]([^`'"\n]+)[`'"]/i;
const NOT_FOUND = 404;
const SERVER_ERROR = 500;
const CUT_SEPARATOR = " · ";
const NO_KEY = "The provider refused the key; check it in settings";
const NO_ACCESS = "The provider denied this key access";
const RATE_LIMITED = "The provider's rate limit or quota ran out";
const PROVIDER_FAILED = "The provider failed";
const NO_ADDRESS = "Nothing answers at the provider's address; is its URL right?";
const UNREACHED = "Couldn't reach the provider";
const TOO_SLOW = "The provider didn't answer in time";
// others, such as 400 for a context too long, are best said in the provider's own words
const STATUS_WORDS = { 401: NO_KEY, 403: NO_ACCESS, 429: RATE_LIMITED };
// ︎ keeps ✳ a text glyph where the system would draw it as an emoji
const SPIN = ["·", "✢", "✳︎", "✶", "✻", "✽", "✻", "✶", "✳︎", "✢"];
// a cell's status as words for a screen reader, beside its glyph's colour
const SPOKEN_STATUS = new Set(["failed", "stopped", CUT_OFF, "running"]);
const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";
const STILL_FRAME = 2;
const STATUS = {
  done: ["done", ""],
  running: ["running", ""],
  exhausted: ["out of turns", "warn"],
  failed: ["failed", "err"],
  stopped: ["stopped", ""],
  [CUT_OFF]: [CUT_OFF, ""],
};

const opened = new WeakSet();
const showingCode = new WeakSet();
const shown = new WeakMap();
const views = new WeakMap();
const keys = new WeakMap();
const counter = new Intl.NumberFormat();
let nextId = 1;
let frame = 0;
// made on first use, from the vendored marked a page loads before its modules
let markdown = null;
// Whether a Library change shows that it applies from the next message: the chat says, for its newest turn's; a page's never do.
let appliesNext = () => false;
// made on the first tick
let reducedMotion = null;

function element(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (name.startsWith("on")) el.addEventListener(name.slice(2), value);
    else if (name === "class") el.className = value;
    else el.setAttribute(name, value);
  }
  el.append(...kids.flat(Infinity).filter((kid) => kid != null && kid !== false));
  return el;
}

// A table's entry for key, or none: a crafted key such as "constructor" mustn't reach Object.prototype.
const known = (table, key) => (Object.hasOwn(table, key) ? table[key] : []);
const join = (items, separator) => items.flatMap((item, i) => (i ? [separator, item] : [item]));
const since = (start) => Date.now() / MS_PER_SECOND - start;
// a cell that raised, or was stopped or cut off, gave no answer, whatever its code starts with
const NO_ANSWER = ["failed", "stopped", CUT_OFF];
const isFinal = (cell) => !NO_ANSWER.includes(cell.status) && cell.code.startsWith("FinalAnswer(");
const formatCount = (count) => counter.format(count);
const plural = (count, noun) => `${formatCount(count)} ${noun}${count === 1 ? "" : "s"}`;
const keyOf = (node) => {
  if (!keys.has(node)) keys.set(node, String(nextId++));
  return keys.get(node);
};

// One rule for every cost: under a dollar to a hundredth of a cent, from a dollar to the cent, so a total is never finer than
// the costs it adds up, which add up to it as near as rounding allows; one less than the least shown as "<$0.0001".
const LEAST_COST = 0.0001;
const SMALL_DIGITS = 4;
const LARGE_DIGITS = 2;
const money = (digits) => new Intl.NumberFormat("en", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const SMALL_COST = money(SMALL_DIGITS);
const LARGE_COST = money(LARGE_DIGITS);

function formatCost(usd) {
  if (usd == null) return null;
  if (usd === 0) return "$0";
  if (usd > 0 && usd < LEAST_COST) return `<$${SMALL_COST.format(LEAST_COST)}`;
  // judged as rounded, so $0.99996 reads $1.00, and a cent always $0.0100
  const small = Math.abs(Number(usd.toFixed(SMALL_DIGITS))) < 1;
  return `$${(small ? SMALL_COST : LARGE_COST).format(usd)}`;
}

// What priced turns cost together, by the same rule; null for none.
function formatTotal(costs) {
  const priced = costs.filter((cost) => cost > 0);
  if (!priced.length) return null;
  return formatCost(priced.reduce((sum, cost) => sum + cost, 0));
}

// An unpriced call's cost is unknown, not free; a holder with no calls yet shows none.
function costView({ cost, unpriced, unfinished }) {
  if (unpriced || unfinished) return element("span", { title: unfinished ? UNFINISHED : PRICE_UNKNOWN }, COST_NA);
  return formatCost(cost);
}

function formatSeconds(seconds) {
  if (seconds == null) return null;
  const tenths = seconds.toFixed(1);
  if (Number(tenths) < SECONDS_PER_MINUTE) return `${tenths}s`;
  const whole = Math.round(seconds);
  return `${Math.floor(whole / SECONDS_PER_MINUTE)}m ${whole % SECONDS_PER_MINUTE}s`;
}

const spinner = () => element("span", { class: "spin", "aria-hidden": "true" }, SPIN[frame]);
const clock = (start) => element("span", { "data-since": start }, formatSeconds(since(start)));
const caret = (open) => element("span", { class: "caret", "aria-hidden": "true" }, open ? "▾" : "▸");
const decor = (text, props = {}) => element("span", { "aria-hidden": "true", ...props }, text);

// A dotted namespace's name in two parts, for a line too narrow for it: its parents are cut first, so its own name shows.
function nameParts(name) {
  const own = name.lastIndexOf(".") + 1;
  return [own > 0 && element("span", { class: "ns-parents" }, name.slice(0, own)), element("span", { class: "ns-own" }, name.slice(own))];
}

// A toggle redraws only its own node, so a click stays quick with thousands of rows shown.
function remember(node, el, again) {
  views.set(node, { el, again });
  return el;
}

function refocus(...candidates) {
  const target = candidates.map((key) => key && document.querySelector(`[data-key="${key}"]`)).find(Boolean);
  target?.focus({ preventScroll: true });
}

function redraw(node, fallbackKey) {
  const { el, again } = views.get(node);
  const focused = document.activeElement?.dataset?.key;
  el.replaceWith(again());
  refocus(focused, fallbackKey);
}

function toggle(set, node) {
  if (set.has(node)) set.delete(node);
  else set.add(node);
  redraw(node);
}

function toggler(node, props, kids, set = opened, key = keyOf(node)) {
  const expanded = String(set.has(node));
  return element("button", { type: "button", "aria-expanded": expanded, "data-key": key, onclick: () => toggle(set, node), ...props }, kids);
}

function describe(step) {
  if (step.kind === "think") return { text: step.text };
  if (step.kind === "change") return { text: `Library: ${changeText(step)}` };
  if (step.kind === "page") return { text: `page: ${asText(step.title)}` };
  if (step.kind === "left_out") return { text: `${LEFT_OUT} worked example "${asText(step.name)}"` };
  const count = step.agents.length;
  if (count) return { text: count === 1 ? "Delegated to a sub-agent" : `Delegated to ${plural(count, "sub-agent")}` };
  if (isFinal(step)) return { text: "Final answer" };
  const [first, ...rest] = step.code.split("\n");
  return { tool: "Python", text: first + (rest.length ? " …" : "") };
}

function stepLabel(step, chars = Infinity) {
  const { tool, text } = describe(step);
  return tool ? `${tool}(${clip(text, chars)})` : clip(text, chars);
}

function headline(step) {
  const { tool, text } = describe(step);
  return element("span", { class: "head" }, tool ? [element("b", {}, tool), `(${text})`] : text);
}

function finalValue(output) {
  const value = output.replace(/^FinalAnswer: /, "");
  return /^(['"]).*\1$/s.test(value) ? value.slice(1, -1) : value;
}

function out(text, { tone = "", full = false, more = null } = {}) {
  return element(
    "div",
    { class: `out ${tone} ${full ? "full" : ""}` },
    decor("⎿"),
    element("span", { class: "text" }, text, more && element("span", { class: "more" }, more)),
  );
}

// What a provider's error means, or null for any other failure. model is the one asked for, when known.
function plainFailure(text, model = null) {
  if (TIMED_OUT.test(text)) return TOO_SLOW;
  if (UNREACHABLE.test(text)) return UNREACHED;
  const status = Number(text.match(PROVIDER_STATUS)?.[1]);
  if (status === NOT_FOUND) return `${missing(text, model)} (${status})`;
  const words = STATUS_WORDS[status] ?? (status >= SERVER_ERROR ? PROVIDER_FAILED : null);
  return words && `${words} (${status})`;
}

// A 404 that speaks of a model is the model's; any other, the address's.
function missing(text, model) {
  if (!/\bmodel\b/i.test(text)) return NO_ADDRESS;
  return `The provider has no model ${model ?? text.match(NAMED_MODEL)?.[1] ?? "by that name"}`;
}

// A failure led by what a provider's error means, then its own words: on a line of their own, or after on one cut short.
function failureView(text, { model = null, after = "\n" } = {}) {
  const plain = plainFailure(text, model);
  return plain ? [plain, after, element("span", { class: "raw" }, text)] : text;
}

// A failed cell's error: its last traceback's exception line, plain or an exception group's, its name first; else its last
// line, before the FinalAnswer line the engine adds after a traceback.
function errorLine(lines) {
  const head = lines.findLastIndex((line) => TRACEBACK.test(line));
  const exception = head < 0 ? null : lines.slice(head + 1).find((line) => EXCEPTION.test(line));
  if (exception) return failureView(exception.replace(GROUP_FRAME, "").replace(EXCEPTION_MODULE, ""), { after: CUT_SEPARATOR });
  return lines.length > 1 && lines.at(-1).startsWith(FINAL_ANSWER) ? lines.at(-2) : lines.at(-1);
}

function renderOutput(cell, full, root) {
  const tone = cell.status === "failed" ? "err" : "";
  if (cell.status === "stopped" || cell.status === CUT_OFF) return null;
  if (cell.output == null) return cell.agents.length ? null : out("Running…");
  // a cell closed as its agent ended has no output, and an empty red line would read as one
  if (cell.status === "failed" && !cell.output.trim()) return null;
  if (full) return out(cell.output.replace(EXEC_FRAME, "$1"), { tone, full: true });
  const lines = cell.output.trimEnd().split("\n");
  if (cell.status === "failed") return out(errorLine(lines), { tone });
  // each sub-agent's row opens onto its answer, and the answer below shows the root's
  if (cell.agents.length || (root && isFinal(cell))) return null;
  if (isFinal(cell)) return out(finalValue(cell.output));
  const more = lines.length > 1 ? ` … +${lines.length - 1} lines` : null;
  return out(lines[0] || "(no output)", { more });
}

// "cut off" as one class, cut-off
const glyph = (cell) => element("span", { class: `glyph ${asText(cell.status).replaceAll(" ", "-")}` }, decor("●"), SPOKEN_STATUS.has(cell.status) && element("span", { class: "sr-only" }, ` ${cell.status}`));

function renderThink(step) {
  const open = opened.has(step);
  return toggler(step, { class: `think ${open ? "open" : ""}` }, [
    element("span"),
    decor("✻"),
    element("span", { class: "body" }, step.text),
  ]);
}

function renderCell(cell, root) {
  if (cell.agents.length) return renderDelegation(cell);
  const open = opened.has(cell);
  return element(
    "div",
    { class: "cell" },
    toggler(cell, { class: "line" }, [element("span"), glyph(cell), headline(cell)]),
    open && element("pre", { class: "code" }, cell.code),
    renderOutput(cell, open, root),
  );
}

function renderDelegation(cell) {
  const open = opened.has(cell);
  const code = showingCode.has(cell);
  const output = renderOutput(cell, code, false);
  const rows = open && renderRows(cell);
  return element(
    "div",
    { class: "cell" },
    element(
      "div",
      { class: "node" },
      toggler(cell, { class: "line" }, [caret(open), glyph(cell), headline(cell)]),
      element(
        "span",
        { class: "meta" },
        toggler(cell, { class: "show-code", "aria-label": "code and output" }, "code", showingCode, `${keyOf(cell)}:code`),
        delegationMeta(cell),
      ),
    ),
    code && element("pre", { class: "code" }, cell.code),
    // the cell's own error goes under its line, above the rows, so it never reads as the last sub-agent's result
    cell.status === "failed" ? [output, rows] : [rows, output],
  );
}

function delegationMeta(cell) {
  const { agents } = cell;
  const counts = Object.entries(STATUS).flatMap(([status, [word, tone]]) => {
    const count = agents.filter((agent) => agent.status === status).length;
    return count > 0 ? [element("span", { class: `word ${tone}` }, `${formatCount(count)} ${word}`)] : [];
  });
  // a copied drive has no start, so it shows no clock
  const starts = agents.filter((agent) => agent.status === "running" && agent.started != null).map((agent) => agent.started);
  const longest = starts.length
    ? clock(Math.min(...starts))
    : formatSeconds(Math.max(...agents.map((agent) => agent.seconds ?? 0)));
  const costs = agents.map((agent) => agent.cost).filter((cost) => cost != null);
  const unpriced = agents.some((agent) => agent.unpriced);
  const total = costView({ cost: costs.length > 0 ? costs.reduce((sum, cost) => sum + cost, 0) : null, unpriced });
  // the cell's own failure after its sub-agents', so it doesn't read as theirs
  const ends = join(counts, " · ");
  if (cell.status === "failed") ends.push(element("span", { class: "word err" }, counts.length ? ", then the cell failed" : "cell failed"));
  return join([ends.length > 0 && ends, longest, total].filter(Boolean), " · ");
}

function renderRows(cell) {
  const count = Math.min(shown.get(cell) ?? FIRST_ROWS, cell.agents.length);
  const hidden = cell.agents.length - count;
  return element("div", { class: "rows" }, cell.agents.slice(0, count).map(renderAgent), hidden > 0 && renderMore(cell, count, hidden));
}

function renderMore(cell, count, hidden) {
  const key = keyOf(cell);
  const button = (label, rows, suffix) =>
    element("button", { type: "button", "data-key": `${key}:${suffix}`, onclick: () => showMore(cell, rows) }, label);
  return element(
    "div",
    { class: "more-rows" },
    `… ${formatCount(hidden)} more`,
    hidden > MORE_ROWS && [" · ", button(`show ${MORE_ROWS} more`, count + MORE_ROWS, "more")],
    " · ",
    button("show all", Infinity, "all"),
  );
}

function showMore(cell, count) {
  const first = cell.agents[shown.get(cell) ?? FIRST_ROWS];
  shown.set(cell, count);
  redraw(cell, keyOf(first));
}

// A change opens onto its YAML before (when it replaced one) and after; node is the step, or its entry under the answer, so each opens on its own.
function renderChange(change, node = change, by = null) {
  const yaml = (label, value) => [element("div", { class: "yaml-label" }, label), element("pre", { class: "code yaml", tabindex: "0", "aria-label": label }, asText(value))];
  return element(
    "div",
    { class: "cell change" },
    toggler(node, { class: "line" }, [
      element("span"),
      decor("◆", { class: "glyph" }),
      element("span", { class: "head" }, element("b", {}, "Library"), " ", changeText(change), element("span", { class: "note" }, [by, appliesNext(change) && FROM_NEXT].filter(Boolean).map((part) => ` · ${part}`))),
    ]),
    opened.has(node) && [change.before != null && yaml("before", change.before), yaml("after", change.after)],
  );
}

// A page's link opens its tab; its id and title come from the run's record, so each shows as text.
function pageLink(step, props, ...kids) {
  return element("a", { href: `/pages/${encodeURIComponent(asText(step.page))}`, target: "_blank", rel: "noopener noreferrer", ...props }, kids);
}

function renderPage(step) {
  const head = element("span", { class: "head" }, element("b", {}, "page"), " ", asText(step.title), decor(" ↗"));
  return element("div", { class: "cell page-step" }, pageLink(step, { class: "line" }, element("span"), decor("◆", { class: "glyph" }), head));
}

// A worked example the agent wasn't shown, as it couldn't be rendered, and why: each field from the run's record shows as text.
// why is the error's own text, which a shared file's template can word: quoted as code, never as the app's words
function renderLeftOut(step) {
  const head = element("span", { class: "head" }, element("b", {}, LEFT_OUT), ` worked example "${asText(step.name)}" from ${asText(step.from)}`);
  const why = ["couldn't be rendered: ", element("code", {}, clip(asText(step.why), WHY_CHARS))];
  return element("div", { class: "cell left-out" }, element("div", { class: "line" }, element("span"), decor("◆", { class: "glyph" }), head), out(why, { tone: "warn", full: true }));
}

function stepView(step, root) {
  if (step.kind === "think") return renderThink(step);
  if (step.kind === "change") return renderChange(step);
  if (step.kind === "page") return renderPage(step);
  if (step.kind === "left_out") return renderLeftOut(step);
  return renderCell(step, root);
}

function renderStep(step, root) {
  return remember(step, stepView(step, root), () => renderStep(step, root));
}

function renderUnder(agent) {
  const [, tone] = known(STATUS, agent.status);
  if (agent.status === "running") {
    const last = agent.steps.at(-1);
    return last ? out(stepLabel(last)) : null;
  }
  if (!tone || !agent.result) return null;
  return out(failureView(agent.result.split("\n")[0], { model: agent.model, after: CUT_SEPARATOR }), { tone });
}

const waiting = (word = WAITING) => element("div", { class: "waiting" }, word);

function renderAgent(agent) {
  const open = opened.has(agent);
  const running = agent.status === "running";
  const [word, tone] = known(STATUS, agent.status);
  const tags = [agent.namespace !== "root" && agent.namespace, typeof agent.model === "string" && agent.model, agent.drive > 1 && `message ${agent.drive}`];
  const meta = [
    agent.status !== "done" && element("span", { class: `word ${tone}` }, word),
    running ? agent.started != null && clock(agent.started) : formatSeconds(agent.seconds),
    costView(agent),
  ].filter(Boolean);
  const el = element(
    "div",
    { class: "agent" },
    toggler(agent, { class: "row" }, [
      caret(open),
      element("span", { class: "task" }, agent.task),
      tags.filter(Boolean).map((tag) => element("span", { class: "tag", title: tag }, tag)),
      element("span", { class: "meta" }, running && [spinner(), " "], join(meta, " · ")),
    ]),
    open
      ? element("div", { class: "children" }, agent.steps.map((step) => renderStep(step, false)), running ? agent.steps.length === 0 && waiting() : renderUnder(agent))
      : renderUnder(agent),
  );
  return remember(agent, el, () => renderAgent(agent));
}

const escapeHtml = (text) => text.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
// as DOMPurify checks the address marked hands it: encoding leaves a scheme's letters and colon as they are
const followable = (href) => SANITIZE.ALLOWED_URI_REGEXP.test(href);
const isPlain = (tokens) => tokens.every((token) => PLAIN_TOKENS.has(token.type) && isPlain(token.tokens ?? []));

// Markdown as answers show it, so code (__init__, 2*3*4, ~/a~b) and raw HTML show as written, and nothing loads.
function newMarkdown({ Marked, Renderer, Tokenizer }) {
  return new Marked({
    breaks: true,
    tokenizer: {
      // after an inline <code>, <pre>, <kbd> or <script>, marked would leave the text unescaped
      tag(src) {
        const token = Tokenizer.prototype.tag.call(this, src);
        this.lexer.state.inRawBlock = false;
        return token;
      },
      // undefined, as false would fall back to marked's own: one ~ never strikes
      del(src) {
        const token = Tokenizer.prototype.del.call(this, src);
        return token?.raw.startsWith("~~") ? token : undefined;
      },
      // _ never emphasises, nor * touching a letter or digit outside it; the whole run stays text, so marked doesn't retry within it
      emStrong(src, masked, before = "") {
        const token = Tokenizer.prototype.emStrong.call(this, src, masked, before);
        if (!token) return token;
        const after = src[token.raw.length] ?? "";
        if (!(token.raw.startsWith("_") || WORD_CHAR.test(before) || WORD_CHAR.test(after))) return token;
        const run = src.match(/^[*_]+/)[0];
        return { type: "text", raw: run, text: run };
      },
    },
    renderer: {
      html: (html, block) => (block ? `<pre><code>${escapeHtml(html)}</code></pre>` : escapeHtml(html)),
      image(href, title, alt) {
        return this.link(href, title, alt || escapeHtml(href));
      },
      // an image in a link is already a link, so its text alone goes in; an address DOMPurify would drop, such as sandbox:
      // or a file's name, shows as text after it, never as a link that does nothing
      link(href, title, text) {
        const inner = text.replace(ANCHOR_TAG, "");
        if (!followable(href)) return inner === escapeHtml(href) ? inner : `${inner} (${escapeHtml(href)})`;
        return Renderer.prototype.link.call(this, href, title, inner).replace(/^<a /, NEW_TAB_LINK);
      },
    },
  });
}

// An answer's HTML, or null when marked finds no Markdown in it, so it shows exactly as written.
function markdownHtml(text) {
  markdown ??= newMarkdown(globalThis.marked);
  const tokens = markdown.lexer(text);
  return isPlain(tokens) ? null : markdown.parser(tokens);
}

// Answers and a page's text are model output: shown as Markdown when they hold some, else as plain text.
// A block that scrolls takes focus, so a keyboard scrolls it too.
function renderAnswer(text) {
  const html = markdownHtml(text);
  if (html == null) return element("div", { class: "answer plain" }, text);
  const el = element("div", { class: "answer" });
  el.innerHTML = globalThis.DOMPurify.sanitize(html, SANITIZE);
  for (const list of el.querySelectorAll("ol")) {
    const last = Number(list.getAttribute("start") ?? 1) + list.children.length - 1;
    list.style.setProperty("--marker-chars", String(last).length);
  }
  for (const block of el.querySelectorAll("pre, table")) block.setAttribute("tabindex", "0");
  return el;
}

// A node's long prompt or answer (kind), as draw makes it, shows its first lines until shown whole.
function clamped(node, kind, text, draw) {
  const { past, shown, whole } = CLAMPS[kind];
  const lines = text.split("\n").reduce((sum, line) => sum + Math.max(1, Math.ceil(line.length / LINE_CHARS)), 0);
  const el = draw();
  if (lines <= past) return el;
  const all = whole.has(node);
  if (!all) {
    el.className += " clamped";
    el.setAttribute("style", `--clamp-lines: ${shown}`);
  }
  const label = all ? "show less" : `show all · ${plural(lines, "line")}`;
  return [el, toggler(node, { class: "show-all" }, label, whole, `${keyOf(node)}:${kind}`)];
}

const whenChangesApply = (test) => {
  appliesNext = test;
};

// Moves each spinner on a frame, unless motion is reduced, and each running clock to now.
function tick() {
  reducedMotion ??= matchMedia(REDUCED_MOTION);
  frame = reducedMotion.matches ? STILL_FRAME : (frame + 1) % SPIN.length;
  for (const el of document.querySelectorAll(".spin")) el.textContent = SPIN[frame];
  for (const el of document.querySelectorAll("[data-since]")) el.textContent = formatSeconds(since(Number(el.dataset.since)));
}

// The theme the chat's toggle stored, if any; else the system's.
function storedTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch {
    // storage can be blocked: follow the system's
    return null;
  }
}

// The theme picked in this tab or another, kept in localStorage; none, or one cleared, follows the system's. changed runs
// after each change another tab makes.
function followStoredTheme(changed = () => {}) {
  const use = () => {
    const theme = storedTheme();
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  };
  use();
  addEventListener("storage", (e) => {
    if (e.key !== THEME_KEY && e.key !== null) return;
    use();
    changed();
  });
}

export {
  CLAMPS,
  THEME_KEY,
  COST_NA,
  MS_PER_SECOND,
  PRICE_UNKNOWN,
  SANITIZE,
  STARTING,
  UNFINISHED,
  UNKNOWN_COST,
  WAITING,
  caret,
  clamped,
  clock,
  costView,
  decor,
  element,
  failureView,
  formatCost,
  formatSeconds,
  formatTotal,
  join,
  known,
  markdownHtml,
  nameParts,
  opened,
  out,
  pageLink,
  plainFailure,
  plural,
  redraw,
  refocus,
  remember,
  renderAgent,
  renderAnswer,
  renderChange,
  renderStep,
  spinner,
  stepLabel,
  tick,
  toggler,
  followStoredTheme,
  views,
  waiting,
  whenChangesApply,
};
