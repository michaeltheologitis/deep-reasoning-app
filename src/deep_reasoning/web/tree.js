// Draws a run's tree as the chat shows it: its steps, and its sub-agents as rows that click open and shut, each redrawn alone.
// The chat (app.js) and a page's tab (page.js) both draw with it.
import { asText, changeText, clip } from "./fold.js";

const FIRST_ROWS = 20;
const MORE_ROWS = 100;
const SECONDS_PER_MINUTE = 60;
const MS_PER_SECOND = 1000;
const FROM_NEXT = "from the next message";
const UNKNOWN_COST = "N/A";
// no-break spaces keep a cost whole where its line wraps
const COST_NA = `cost\u00a0${UNKNOWN_COST}`;
const PRICE_UNKNOWN = "this model's price isn't known";
// ︎ keeps ✳ a text glyph where the system would draw it as an emoji
const SPIN = ["·", "✢", "✳︎", "✶", "✻", "✽", "✻", "✶", "✳︎", "✢"];
const STATUS = {
  done: ["done", ""],
  running: ["running", ""],
  exhausted: ["out of turns", "warn"],
  failed: ["failed", "err"],
  stopped: ["stopped", ""],
};

const opened = new WeakSet();
const showingCode = new WeakSet();
const shown = new WeakMap();
const views = new WeakMap();
const keys = new WeakMap();
const counter = new Intl.NumberFormat();
let nextId = 1;
let frame = 0;
// Whether a Library change shows that it applies from the next message: the chat says, for its newest turn's; a page's never do.
let appliesNext = () => false;

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
const isFinal = (cell) => cell.code.startsWith("FinalAnswer(");
const formatCount = (count) => counter.format(count);
const plural = (count, noun) => `${formatCount(count)} ${noun}${count === 1 ? "" : "s"}`;
const keyOf = (node) => {
  if (!keys.has(node)) keys.set(node, String(nextId++));
  return keys.get(node);
};

function formatCost(usd) {
  if (usd == null) return null;
  if (usd === 0) return "$0";
  return `$${usd.toFixed(usd >= 0.01 ? 2 : 4)}`;
}

// An unpriced call's cost is unknown, not free; a holder with no calls yet shows none.
const costView = ({ cost, unpriced }) => (unpriced ? element("span", { title: PRICE_UNKNOWN }, COST_NA) : formatCost(cost));

function formatSeconds(seconds) {
  if (seconds == null) return null;
  const tenths = seconds.toFixed(1);
  if (Number(tenths) < SECONDS_PER_MINUTE) return `${tenths}s`;
  const whole = Math.round(seconds);
  return `${Math.floor(whole / SECONDS_PER_MINUTE)}m ${whole % SECONDS_PER_MINUTE}s`;
}

const spinner = () => element("span", { class: "spin" }, SPIN[frame]);
const clock = (start) => element("span", { "data-since": start }, formatSeconds(since(start)));
const caret = (open) => element("span", { class: "caret" }, open ? "▾" : "▸");

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
    element("span", {}, "⎿"),
    element("span", { class: "text" }, text, more && element("span", { class: "more" }, more)),
  );
}

function renderOutput(cell, full, root) {
  const tone = cell.status === "failed" ? "err" : "";
  if (cell.status === "stopped") return null;
  if (cell.output == null) return cell.agents.length ? null : out("Running…");
  if (full) return out(cell.output, { tone, full: true });
  const lines = cell.output.trimEnd().split("\n");
  if (cell.status === "failed") return out(lines.at(-1), { tone });
  // each sub-agent's row opens onto its answer, and the answer below shows the root's
  if (cell.agents.length || (root && isFinal(cell))) return null;
  if (isFinal(cell)) return out(finalValue(cell.output));
  const more = lines.length > 1 ? ` … +${lines.length - 1} lines` : null;
  return out(lines[0] || "(no output)", { more });
}

const glyph = (cell) => element("span", { class: `glyph ${cell.status}` }, "●");

function renderThink(step) {
  const open = opened.has(step);
  return toggler(step, { class: `think ${open ? "open" : ""}` }, [
    element("span"),
    element("span", {}, "✻"),
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
  return element(
    "div",
    { class: "cell" },
    element(
      "div",
      { class: "node" },
      toggler(cell, { class: "line" }, [caret(open), glyph(cell), headline(cell)]),
      toggler(cell, { class: "show-code", "aria-label": "code and output" }, "code", showingCode, `${keyOf(cell)}:code`),
      element("span", { class: "meta" }, delegationMeta(cell)),
    ),
    code && element("pre", { class: "code" }, cell.code),
    open && renderRows(cell),
    renderOutput(cell, code, false),
  );
}

function delegationMeta(cell) {
  const { agents } = cell;
  const counts = Object.entries(STATUS).map(([status, [word, tone]]) => {
    const count = agents.filter((agent) => agent.status === status).length;
    return count > 0 && element("span", { class: `word ${tone}` }, `${formatCount(count)} ${word}`);
  });
  // a copied drive has no start, so it shows no clock
  const starts = agents.filter((agent) => agent.status === "running" && agent.started != null).map((agent) => agent.started);
  const longest = starts.length
    ? clock(Math.min(...starts))
    : formatSeconds(Math.max(...agents.map((agent) => agent.seconds ?? 0)));
  const costs = agents.map((agent) => agent.cost).filter((cost) => cost != null);
  const unpriced = agents.some((agent) => agent.unpriced);
  const total = costView({ cost: costs.length > 0 ? costs.reduce((sum, cost) => sum + cost, 0) : null, unpriced });
  return join([...counts, longest, total].filter(Boolean), " · ");
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
  const yaml = (label, value) => [element("div", { class: "yaml-label" }, label), element("pre", { class: "code yaml" }, asText(value))];
  return element(
    "div",
    { class: "cell change" },
    toggler(node, { class: "line" }, [
      element("span"),
      element("span", { class: "glyph" }, "◆"),
      element("span", { class: "head" }, element("b", {}, "Library"), " ", changeText(change), element("span", { class: "note" }, [by, appliesNext(change) && FROM_NEXT].filter(Boolean).map((part) => ` · ${part}`))),
    ]),
    opened.has(node) && [change.before != null && yaml("before", change.before), yaml("after", change.after)],
  );
}

// A page's link opens its tab; its id and title come from the run's record, so each shows as text.
function pageLink(step, props, ...kids) {
  return element("a", { href: `/pages/${encodeURIComponent(asText(step.page))}`, target: "_blank", rel: "noopener", ...props }, kids);
}

function renderPage(step) {
  const head = element("span", { class: "head" }, element("b", {}, "page"), " ", asText(step.title), " ↗");
  return element("div", { class: "cell page-step" }, pageLink(step, { class: "line" }, element("span"), element("span", { class: "glyph" }, "◆"), head));
}

function stepView(step, root) {
  if (step.kind === "think") return renderThink(step);
  if (step.kind === "change") return renderChange(step);
  if (step.kind === "page") return renderPage(step);
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
  return out(agent.result.split("\n")[0], { tone });
}

function renderAgent(agent) {
  const open = opened.has(agent);
  const running = agent.status === "running";
  const [word, tone] = known(STATUS, agent.status);
  const tags = [agent.namespace !== "root" && agent.namespace, agent.drive > 1 && `message ${agent.drive}`];
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
      tags.filter(Boolean).map((tag) => element("span", { class: "tag" }, tag)),
      element("span", { class: "meta" }, running && [spinner(), " "], join(meta, " · ")),
    ]),
    open
      ? element("div", { class: "children" }, agent.steps.map((step) => renderStep(step, false)), !running && renderUnder(agent))
      : renderUnder(agent),
  );
  return remember(agent, el, () => renderAgent(agent));
}

function renderAnswer(text) {
  const el = element("div", { class: "answer" });
  // answers are model output, and marked passes raw HTML through
  if (window.marked && window.DOMPurify) el.innerHTML = window.DOMPurify.sanitize(window.marked.parse(text));
  else Object.assign(el, { textContent: text, style: "white-space: pre-wrap" });
  return el;
}

const whenChangesApply = (test) => {
  appliesNext = test;
};

// Moves each spinner on a frame, and each running clock to now.
function tick() {
  frame = (frame + 1) % SPIN.length;
  for (const el of document.querySelectorAll(".spin")) el.textContent = SPIN[frame];
  for (const el of document.querySelectorAll("[data-since]")) el.textContent = formatSeconds(since(Number(el.dataset.since)));
}

// The theme the chat's toggle stored, if any; else the system's.
function useStoredTheme() {
  try {
    const theme = localStorage.getItem("theme");
    if (theme) document.documentElement.dataset.theme = theme;
  } catch {
    // no stored theme: follow the system's
  }
}

export {
  COST_NA,
  MS_PER_SECOND,
  PRICE_UNKNOWN,
  UNKNOWN_COST,
  caret,
  clock,
  costView,
  element,
  formatCost,
  formatSeconds,
  join,
  known,
  opened,
  out,
  pageLink,
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
  useStoredTheme,
  views,
  whenChangesApply,
};
