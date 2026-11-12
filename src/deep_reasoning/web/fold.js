// Folds a run's events into the turn the chat draws: the root's steps, each sub-agent as a row under the cell that started it, Library changes, pages,
// worked examples left out, and the totals;
// and words a Library change.

const FIELD_CHARS = 60;
const STOPPED = "stopped";
export const CUT_OFF = "cut off";

// a cut never keeps half of an emoji, or of another character past U+FFFF: alone, it shows as �
export const clip = (text, chars) => (text.length > chars ? `${text.slice(0, chars - 1).replace(/[\uD800-\uDBFF]$/, "")}…` : text);
// A change event is untrusted, as any cell can log one and a shared file can hold any: each field shows as text, whatever it holds.
export const asText = (value) => (typeof value === "string" ? value : (JSON.stringify(value) ?? "?"));
export const field = (value) => clip(asText(value), FIELD_CHARS);

export function changeText({ what, namespace, name, before }) {
  if (what === "namespace") return `${before == null ? "added" : "changed"} namespace ${field(namespace)}`;
  if (what === "tool") return `${before == null ? "added" : "replaced"} tool ${field(name)}`;
  return `${before == null ? "added" : "replaced"} ${field(what)} "${field(name)}" in ${field(namespace)}`;
}

export function newFold() {
  return {
    turn: { prompt: "", steps: [], changes: [], pages: [], answer: null, shown: null, outcome: null, detail: null, cost: null, unpriced: false, unfinished: false, seconds: null, started: null, model: null, provider: null, namespace: null, run: null },
    seq: 0,
    // node → its holder (the turn for the root, else its row) and its parent's node
    nodes: new Map(),
    cells: new Map(),
    // the newest step, and the tasks of the sub-agents above it
    latest: null,
  };
}

// Folds one event in; false when it was ignored, as one already folded.
export function foldEvent(fold, event) {
  if (event.seq <= fold.seq) return false;
  fold.seq = event.seq;
  FOLDS[event.kind]?.(fold, event);
  return true;
}

export function foldEvents(events) {
  const fold = newFold();
  for (const event of events) foldEvent(fold, event);
  return fold;
}

const FOLDS = {
  "turn.start": (fold, { prompt, t, model, provider, namespace, run }) => Object.assign(fold.turn, { prompt, started: t, model, provider, namespace, run }),
  "agent.start": startAgent,
  think: (fold, event) => addStep(fold, event.node, { kind: "think", text: event.text }),
  "cell.start": startCell,
  "cell.end": endCell,
  "library.change": addChange,
  "page.open": addPage,
  // a worked example the agent wasn't shown, as it couldn't be rendered, and the namespace it is stored in
  "example.left_out": (fold, { node, name, namespace, from, why }) => addStep(fold, node, { kind: "left_out", name, namespace, from, why }),
  usage: addUsage,
  "agent.end": endAgent,
  "turn.end": endTurn,
};

function startAgent(fold, event) {
  if (event.parent == null) {
    fold.nodes.set(event.node, { holder: fold.turn, parent: null });
    return;
  }
  const cell = fold.cells.get(event.cell);
  if (!cell) return;
  const { task, namespace, drive, t } = event;
  const row = { task, namespace, drive, status: "running", result: null, cost: null, unpriced: false, seconds: null, started: t, steps: [] };
  cell.agents.push(row);
  fold.nodes.set(event.node, { holder: row, parent: event.parent });
}

// The node's entry and each one above it, once each: a file from someone else may name a node its own ancestor.
function lineage(fold, node) {
  const entries = new Set();
  for (let entry = fold.nodes.get(node); entry && !entries.has(entry); entry = fold.nodes.get(entry.parent)) entries.add(entry);
  return entries;
}

function addStep(fold, node, step) {
  const holder = fold.nodes.get(node)?.holder;
  if (!holder) return false;
  holder.steps.push(step);
  const tasks = [...lineage(fold, node)].filter((entry) => entry.parent != null).map((entry) => entry.holder.task);
  fold.latest = { tasks: tasks.reverse(), step };
  return true;
}

function startCell(fold, event) {
  const cell = { kind: "cell", code: event.code, output: null, status: "running", agents: [] };
  fold.cells.set(event.cell, cell);
  addStep(fold, event.node, cell);
}

// A step of the agent that made it, and listed on the turn with the tasks above it, so the answer shows it however deep.
function addChange(fold, event) {
  const { what, namespace, name, before, after } = event;
  if (addStep(fold, event.node, { kind: "change", what, namespace, name, before, after })) fold.turn.changes.push(fold.latest);
}

// A page the page tool opened: a step of the agent that opened it, and listed on the turn for the line under the answer.
function addPage(fold, event) {
  const step = { kind: "page", page: event.page, title: event.title };
  if (addStep(fold, event.node, step)) fold.turn.pages.push(step);
}

function endCell(fold, event) {
  const cell = fold.cells.get(event.cell);
  if (cell) Object.assign(cell, { code: event.code, output: event.output, status: event.status });
}

// A cost that isn't a number, or is below zero or past any number, as a shared file may hold, is unknown.
const isCost = (cost) => Number.isFinite(cost) && cost >= 0;

// A call's cost counts for its agent, every agent above it and the turn; once one call is unpriced, so is each of their totals
// (unpriced, cost null), unlike a holder with no calls yet (cost null alone).
// A sub-agent's own call names the model it asked for on its row, when that isn't the run's.
function addUsage(fold, event) {
  for (const { holder } of lineage(fold, event.node)) {
    holder.unpriced ||= !isCost(event.cost);
    holder.cost = holder.unpriced ? null : (holder.cost ?? 0) + event.cost;
  }
  const row = fold.nodes.get(event.node)?.holder;
  if (event.own !== true || !row || row === fold.turn || typeof event.model !== "string" || event.model === fold.turn.model) return;
  row.model = event.model;
}

function endAgent(fold, event) {
  const row = fold.nodes.get(event.node)?.holder;
  if (!row || row === fold.turn) return;
  Object.assign(row, { status: event.status, result: event.result ?? event.detail, seconds: event.t - row.started });
}

// The run's own total counts only calls that finished: one whose only calls were cut off or failed totals 0 but cost what no one knows.
// A dict or list answer is shown as the worker wrote it to be read (shown), its repr kept as the answer.
function endTurn(fold, event) {
  const { outcome, answer, shown, detail, cost, seconds } = event;
  const unfinished = cost === 0 && event.unfinished_calls > 0;
  const priced = isCost(cost) && !unfinished;
  Object.assign(fold.turn, { outcome, answer, shown: typeof shown === "string" ? shown : null, detail, cost: priced ? cost : null, unpriced: !priced && !unfinished, unfinished, seconds });
  markLeftGoing(fold.turn, event.t, outcome === STOPPED ? STOPPED : CUT_OFF);
}

// Each node of a turn's tree with where it is, "turn" or its steps' and sub-agents' places, the same each time its events are folded.
export function* nodesOf(turn) {
  yield ["turn", turn];
  yield* nodesBelow(turn, "");
}

function* nodesBelow(holder, path) {
  for (const [i, step] of holder.steps.entries()) {
    yield [`${path}${i}`, step];
    for (const [j, agent] of (step.agents ?? []).entries()) {
      yield [`${path}${i}.${j}`, agent];
      yield* nodesBelow(agent, `${path}${i}.${j}.`);
    }
  }
}

// What the run left going when it ended ends as status: stopped with a run that was stopped, else cut off, as when the app stopped under it.
function markLeftGoing(holder, t, status) {
  for (const step of holder.steps) {
    if (step.kind !== "cell") continue;
    if (step.status === "running") step.status = status;
    for (const agent of step.agents) {
      if (agent.status === "running") Object.assign(agent, { status, seconds: t - agent.started });
      markLeftGoing(agent, t, status);
    }
  }
}
