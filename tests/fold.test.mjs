// The fold on events recorded from real worker runs against the fake model (scripts/record_runs.py).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Worker } from "node:worker_threads";
import { changeText, foldEvent, foldEvents, nodesOf } from "../src/deep_reasoning/web/fold.js";

const FOLD_URL = new URL("../src/deep_reasoning/web/fold.js", import.meta.url).href;
const FOLD_TIMEOUT_MS = 5000;

const recorded = (name) => readFileSync(new URL(`runs/${name}.jsonl`, import.meta.url), "utf8").trim().split("\n").map(JSON.parse);
const ofKind = (events, kind) => events.filter((event) => event.kind === kind);
const last = (events) => events.at(-1);
const sum = (numbers) => numbers.reduce((total, number) => total + number, 0);
const costOf = (events, node) => sum(ofKind(events, "usage").filter((event) => event.node === node).map((event) => event.cost));
const cells = (holder) => holder.steps.filter((step) => step.kind === "cell");
const changes = (holder) => holder.steps.filter((step) => step.kind === "change");
const pages = (holder) => holder.steps.filter((step) => step.kind === "page");
const rowsOf = (holder) => cells(holder).flatMap((cell) => cell.agents);

function allRows(holder) {
  return rowsOf(holder).flatMap((row) => [row, ...allRows(row)]);
}

function allCells(holder) {
  return [...cells(holder), ...rowsOf(holder).flatMap(allCells)];
}

// Folds in a worker, so a fold that never returns fails its test instead of hanging the run: node:test's timeout can't stop a loop.
function foldInWorker(events) {
  const source = `
    const { parentPort, workerData } = require("node:worker_threads");
    import(workerData.url).then(({ foldEvents }) => {
      const { turn, latest } = foldEvents(workerData.events);
      parentPort.postMessage({ turn, latest });
    });`;
  const worker = new Worker(source, { eval: true, workerData: { url: FOLD_URL, events } });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => worker.terminate().then(() => reject(new Error("the fold never returned"))), FOLD_TIMEOUT_MS);
    worker.once("message", (fold) => {
      clearTimeout(timer);
      worker.terminate().then(() => resolve(fold));
    });
    worker.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

// What a crafted file may hold and no run writes: sub-agents 2 and 3 each named as the other's parent.
const parentCycle = [
  { seq: 1, t: 0, kind: "turn.start", prompt: "Loop." },
  { seq: 2, t: 0, kind: "agent.start", node: 1, parent: null, cell: null },
  { seq: 3, t: 1, kind: "cell.start", node: 1, cell: 1, code: "subagent('A.')" },
  { seq: 4, t: 2, kind: "agent.start", node: 2, parent: 3, cell: 1, task: "A.", drive: 1 },
  { seq: 5, t: 2, kind: "agent.start", node: 3, parent: 2, cell: 1, task: "B.", drive: 1 },
];

test("a saved turn has the prompt, the answer and the run's own totals", () => {
  const events = recorded("fanout");
  const { turn } = foldEvents(events);
  const end = last(events);
  assert.equal(turn.prompt, "Summarize two courses.");
  assert.deepEqual([turn.outcome, turn.answer, turn.cost, turn.seconds], [end.outcome, end.answer, end.cost, end.seconds]);
  assert.equal(turn.started, events[0].t);
});

test("a turn keeps the model, provider and namespace its run started on, and the run's id", () => {
  const [start] = recorded("fanout");
  const { turn } = foldEvents([start]);
  assert.deepEqual([turn.model, turn.provider, turn.namespace, turn.run], [start.model, start.provider, start.namespace, start.run]);
  assert.deepEqual([turn.model, turn.namespace], ["gpt-6-luna", "root"]);
  assert.match(turn.run, /^\d{8}-\d{6}-\d+$/);
});

test("a sub-agent whose own calls are on another model than the run's names it on its row", () => {
  const events = recorded("models");
  const [row] = rowsOf(foldEvents(events).turn);
  assert.equal(row.model, "fake-small");
  assert.equal(foldEvents(events).turn.model, "gpt-6-luna");
  // a tool's call inside the agent's cell, on its own model, isn't the agent's
  const toolCalls = events.map((event) => (event.kind === "usage" && event.node !== 1 ? { ...event, own: false } : event));
  assert.equal(rowsOf(foldEvents(toolCalls).turn)[0].model, undefined);
  assert.ok(rowsOf(foldEvents(recorded("fanout")).turn).every((each) => each.model === undefined));
});

test("a fan-out puts every sub-agent as a row under the cell that started it", () => {
  const events = recorded("fanout");
  const { turn } = foldEvents(events);
  assert.deepEqual(
    turn.steps.map((step) => step.kind),
    ["think", "cell", "cell"],
  );
  const [fanOut] = cells(turn);
  assert.match(fanOut.code, /run_all/);
  assert.equal(fanOut.status, "done");
  assert.deepEqual(
    fanOut.agents.map((row) => [row.task, row.status, row.result, row.drive]),
    [
      ["Summarize C1.", "done", "C1: 3", 1],
      ["Summarize C2.", "done", "C2: 4", 1],
    ],
  );
  assert.equal(fanOut.agents[0].steps.length, 2);
});

test("a row's time runs from its agent's start to its end, and its cost is its calls'", () => {
  const events = recorded("fanout");
  const { turn } = foldEvents(events);
  const starts = ofKind(events, "agent.start").filter((event) => event.parent != null);
  const ends = new Map(ofKind(events, "agent.end").map((event) => [event.node, event]));
  for (const [row, start] of rowsOf(turn).map((row, i) => [row, starts[i]])) {
    assert.equal(row.seconds, ends.get(start.node).t - start.t);
    assert.equal(row.cost, costOf(events, start.node));
  }
});

test("a nested sub-agent's cost counts for every agent above it", () => {
  const events = recorded("nested");
  const { turn } = foldEvents(events);
  const [survey] = rowsOf(turn);
  const [read] = rowsOf(survey);
  assert.deepEqual([survey.task, read.task], ["Survey CS.", "Read CS101."]);
  const [, surveyNode, readNode] = ofKind(events, "agent.start").map((event) => event.node);
  assert.equal(read.cost, costOf(events, readNode));
  assert.ok(Math.abs(survey.cost - (costOf(events, surveyNode) + read.cost)) < 1e-12);
});

test("the newest step comes with the tasks of the sub-agents above it", () => {
  const events = recorded("nested");
  const upTo = events.findIndex((event) => event.kind === "cell.start" && event.code.startsWith("FinalAnswer('CS101"));
  const fold = foldEvents(events.slice(0, upTo + 1));
  assert.deepEqual(fold.latest.tasks, ["Survey CS.", "Read CS101."]);
  assert.equal(fold.latest.step.code, "FinalAnswer('CS101 is light')");
});

test("a second drive of one sub-agent is a row of its own", () => {
  const { turn } = foldEvents(recorded("drives"));
  const [cell] = cells(turn);
  assert.deepEqual(
    cell.agents.map((row) => [row.task, row.drive, row.result, row.steps.length]),
    [
      ["Note that the answer is 42.", 1, "42", 1],
      ["Say it again.", 2, "again 42", 1],
    ],
  );
});

test("a failed cell and a failed sub-agent keep their status, and the failure shows as the row's result", () => {
  const events = recorded("failing");
  const { turn } = foldEvents(events);
  const [divide, delegate] = cells(turn);
  assert.equal(divide.status, "failed");
  assert.match(divide.output, /ZeroDivisionError/);
  const [broken] = delegate.agents;
  assert.equal(broken.status, "failed");
  assert.equal(broken.result, ofKind(events, "agent.end")[0].detail);
  assert.equal(turn.outcome, "answered");
});

test("an exhausted sub-agent and turn keep their outcome", () => {
  const { turn } = foldEvents(recorded("exhausted"));
  assert.deepEqual(
    rowsOf(turn).map((row) => row.status),
    ["exhausted"],
  );
  assert.equal(turn.outcome, "exhausted");
  assert.match(turn.answer, /failed to produce a final answer/);
});

for (const name of ["stopped", "killed"]) {
  test(`a ${name} run leaves nothing running`, () => {
    const events = recorded(name);
    const { turn } = foldEvents(events);
    assert.equal(turn.outcome, "stopped");
    assert.equal(turn.answer, null);
    assert.ok(allRows(turn).length > 0);
    assert.ok(allRows(turn).every((row) => row.status === "stopped" && row.seconds > 0));
    assert.ok(allCells(turn).every((cell) => cell.status !== "running"));
  });
}

test("what a run left going reads cut off, not stopped, when it ended without a stop, as when the app stopped under it", () => {
  const events = recorded("killed").slice(0, -1);
  const lost = { seq: last(events).seq + 1, t: last(events).t, kind: "turn.end", outcome: "failed", answer: null, detail: "The run's end wasn't saved." };
  const { turn } = foldEvents([...events, lost]);
  assert.equal(turn.outcome, "failed");
  assert.ok(allRows(turn).length > 0);
  assert.ok(allRows(turn).every((row) => row.status === "cut off"));
  assert.ok(allCells(turn).every((cell) => cell.status !== "running" && cell.status !== "stopped"));
});

test("a row the run's end stopped is timed to that end", () => {
  const events = recorded("killed");
  const { turn } = foldEvents(events);
  const [sleeper] = rowsOf(turn);
  const start = ofKind(events, "agent.start")[1];
  assert.equal(sleeper.seconds, last(events).t - start.t);
  assert.ok(allCells(turn).every((cell) => cell.status === "stopped"));
});

test("one unpriced call makes every total above it unknown, not zero", () => {
  const events = recorded("unpriced");
  const { turn } = foldEvents(events);
  assert.deepEqual([turn.cost, turn.unpriced], [null, true]);
  assert.ok(allRows(turn).every((row) => row.cost === null && row.unpriced));
  assert.equal(foldEvents(events.slice(0, -1)).turn.unpriced, true);
});

test("a run whose only call never finished, as its server failed it, costs what no one knows, not the 0 it totals", () => {
  const events = recorded("unanswered");
  assert.deepEqual([ofKind(events, "usage"), last(events).cost, last(events).unfinished_calls], [[], 0, 1]);
  const { turn } = foldEvents(events);
  assert.deepEqual([turn.cost, turn.unpriced, turn.unfinished], [null, false, true]);
});

// A run's end with what it cost and how many of its calls never finished, if it says.
const endedWith = (fields) => [
  ...parentCycle.slice(0, 2),
  { seq: 3, t: 2, kind: "turn.end", outcome: "failed", answer: null, detail: "ToolBuildError: x", seconds: 2, ...fields },
];

for (const [name, fields] of [
  ["that called no model, as one whose tools didn't build", { cost: 0, unfinished_calls: 0 }],
  ["recorded before its end said how many calls never finished", { cost: 0 }],
]) {
  test(`a run ${name} costs $0`, () => {
    const { turn } = foldEvents(endedWith(fields));
    assert.deepEqual([turn.cost, turn.unpriced, turn.unfinished], [0, false, false]);
  });
}

test("a run whose calls finished costs what they did, though a later one never finished", () => {
  const { turn } = foldEvents(endedWith({ cost: 0.25, unfinished_calls: 1 }));
  assert.deepEqual([turn.cost, turn.unfinished], [0.25, false]);
});

test("each node of a turn has the same path each time the run's events are folded, so a reload reopens what was open", () => {
  const events = recorded("nested");
  const paths = (turn) => [...nodesOf(turn)].map(([path, node]) => `${path} ${node.kind ?? node.task ?? "turn"}`);
  const whole = paths(foldEvents(events).turn);
  assert.deepEqual(whole.slice(0, 5), ["turn turn", "0 cell", "0.0 Survey CS.", "0.0.0 cell", "0.0.0.0 Read CS101."]);
  assert.deepEqual(paths(foldEvents(events).turn), whole);
  // a reload mid-run folds fewer events: each node it has keeps its path
  const sofar = paths(foldEvents(events.slice(0, Math.floor(events.length / 2))).turn);
  assert.ok(sofar.length > 3 && sofar.every((path) => whole.includes(path)));
});

test("a priced run's totals are known", () => {
  const { turn } = foldEvents(recorded("fanout"));
  assert.ok(turn.cost > 0);
  assert.deepEqual([turn.unpriced, turn.unfinished], [false, false]);
  assert.ok(allRows(turn).every((row) => row.cost > 0 && !row.unpriced));
});

test("a sub-agent with no calls yet has no cost, unlike one whose call was unpriced", () => {
  const events = recorded("unpriced");
  const firstCall = (start) => events.findIndex((event) => event.kind === "usage" && event.node === start.node);
  // the sub-agents run in parallel, so whichever calls first: just before the other's first call, it has started and not called
  const [early, late] = ofKind(events, "agent.start")
    .filter((event) => event.parent != null)
    .sort((a, b) => firstCall(a) - firstCall(b));
  const rows = rowsOf(foldEvents(events.slice(0, firstCall(late))).turn);
  const row = (start) => rows.find((candidate) => candidate.task === start.task);
  assert.deepEqual([row(early).cost, row(early).unpriced], [null, true]);
  assert.deepEqual([row(late).cost, row(late).unpriced], [null, false]);
});

for (const cost of ["0.5", "abc"]) {
  test(`a cost that isn't a number (${JSON.stringify(cost)}), as a crafted file may hold, reads as unknown`, () => {
    const events = [
      ...parentCycle.slice(0, 3),
      { seq: 4, t: 2, kind: "agent.start", node: 2, parent: 1, cell: 1, task: "A.", drive: 1 },
      { seq: 5, t: 3, kind: "usage", node: 2, cost },
      { seq: 6, t: 4, kind: "turn.end", outcome: "answered", answer: "done", detail: null, cost, seconds: 4 },
    ];
    for (const { turn } of [foldEvents(events.slice(0, -1)), foldEvents(events)]) {
      const [row] = rowsOf(turn);
      assert.deepEqual([turn.cost, turn.unpriced, row.cost, row.unpriced], [null, true, null, true]);
    }
  });
}

test("a running sub-agent is a running row, and it costs what its calls so far cost", () => {
  const events = recorded("fanout");
  const upTo = events.findIndex((event) => event.kind === "think" && event.text === "Easy.");
  const { turn } = foldEvents(events.slice(0, upTo + 1));
  const rows = rowsOf(turn);
  assert.deepEqual(
    rows.map((row) => row.status),
    ["running", "running"],
  );
  assert.equal(rows[0].started, ofKind(events, "agent.start")[1].t);
  assert.equal(rows[1].cost, costOf(events.slice(0, upTo + 1), 3));
  assert.equal(turn.outcome, null);
});

for (const name of ["nested", "librarian", "pages"]) {
  test(`folding ${name} live, after catching up on the events so far, gives the saved turn`, () => {
    const events = recorded(name);
    const fold = foldEvents(events.slice(0, 9));
    for (const event of events.slice(9)) foldEvent(fold, event);
    assert.deepEqual(fold.turn, foldEvents(events).turn);
  });
}

test("an event seen before, repeated or out of order, is ignored", () => {
  const events = recorded("fanout");
  const fold = foldEvents(events.slice(0, 10));
  assert.equal(foldEvent(fold, events[9]), false);
  assert.equal(foldEvent(fold, events[4]), false);
  const replayed = [...events.slice(0, 10), events[9], events[4], ...events.slice(5, 12), ...events.slice(10)];
  assert.deepEqual(foldEvents(replayed).turn, foldEvents(events).turn);
});

test("a call in a node named its own ancestor costs once for each agent in the cycle", async () => {
  const { turn } = await foldInWorker([...parentCycle, { seq: 6, t: 3, kind: "usage", node: 2, cost: 0.5 }]);
  assert.deepEqual(
    rowsOf(turn).map((row) => [row.task, row.cost]),
    [
      ["A.", 0.5],
      ["B.", 0.5],
    ],
  );
  assert.equal(turn.cost, null);
});

test("a step in a node named its own ancestor comes with each task above it once", async () => {
  const { turn, latest } = await foldInWorker([...parentCycle, { seq: 6, t: 3, kind: "think", node: 2, text: "Round." }]);
  assert.deepEqual(latest, { tasks: ["B.", "A."], step: { kind: "think", text: "Round." } });
  assert.deepEqual(rowsOf(turn)[0].steps, [latest.step]);
});

test("a Library change is a step of the agent that made it, after the cell it made it in", () => {
  const events = recorded("librarian");
  const { turn } = foldEvents(events);
  const [reworder] = rowsOf(turn);
  assert.deepEqual(
    turn.steps.map((step) => step.kind),
    ["cell", "change", "cell", "cell"],
  );
  assert.deepEqual(
    reworder.steps.map((step) => step.kind),
    ["cell", "change", "cell"],
  );
  const asFolded = ({ what, namespace, name, before, after }) => ({ kind: "change", what, namespace, name, before, after });
  assert.deepEqual([...changes(turn), ...changes(reworder)], ofKind(events, "library.change").map(asFolded));
  const [added] = changes(turn);
  const [replaced] = changes(reworder);
  assert.deepEqual([added.name, added.before, replaced.before], ["count the tickets", null, added.after]);
});

test("the turn lists every Library change, a sub-agent's with the tasks above it", () => {
  const { turn } = foldEvents(recorded("librarian"));
  const [reworder] = rowsOf(turn);
  assert.deepEqual(
    turn.changes.map(({ tasks }) => tasks),
    [[], ["Reword the example."]],
  );
  assert.equal(turn.changes[0].step, changes(turn)[0]);
  assert.equal(turn.changes[1].step, changes(reworder)[0]);
});

test("a change reads as what it did: a decomposition added or replaced in its namespace, a namespace added or changed, a tool added or replaced", () => {
  const worded = (name) => foldEvents(recorded(name)).turn.changes.map(({ step }) => changeText(step));
  assert.deepEqual(worded("librarian"), ['added decomposition "count the tickets" in librarian', 'replaced decomposition "count the tickets" in librarian']);
  assert.deepEqual(worded("namespaces"), ["added namespace research", "changed namespace research"]);
  assert.deepEqual(worded("tools"), ["added tool word_count", "added namespace counting", "replaced tool word_count"]);
});

test("a tool's change opens onto its block's YAML and its Python, before and after", () => {
  const [added, , replaced] = foldEvents(recorded("tools")).turn.changes.map(({ step }) => step);
  assert.equal(added.before, null);
  assert.match(added.after, /^factory: build\nfactory_from: word_count\.py\n# word_count\.py\nfrom deep_reasoner\.primitives import Func\n/);
  assert.equal(replaced.before, added.after);
  assert.match(replaced.after, /\nsep: ','\n[^]*text\.split\(params\["sep"\]\)/);
});

test("a change with odd fields, as a crafted file may hold, folds as it came, and one of no known agent is left out", () => {
  const odd = { what: 7, namespace: ["research"], name: { a: 1 }, before: 5 };
  const { turn } = foldEvents([
    { seq: 1, t: 0, kind: "turn.start", prompt: "Forge." },
    { seq: 2, t: 0, kind: "agent.start", node: 1, parent: null, cell: null },
    { seq: 3, t: 1, kind: "library.change", node: 1, cell: null, ...odd },
    { seq: 4, t: 1, kind: "library.change", node: 9, cell: 1, name: "nowhere" },
  ]);
  const step = { kind: "change", ...odd, after: undefined };
  assert.deepEqual(turn.steps, [step]);
  assert.deepEqual(turn.changes, [{ tasks: [], step }]);
});

test("a page opened is a step of the agent that opened it, after the cell it opened it in, and the turn lists each", () => {
  const events = recorded("pages");
  const { turn } = foldEvents(events);
  const [tabler] = rowsOf(turn);
  assert.deepEqual(
    turn.steps.map((step) => step.kind),
    ["cell", "page", "cell", "cell"],
  );
  assert.deepEqual(
    tabler.steps.map((step) => step.kind),
    ["cell", "page", "cell"],
  );
  const asFolded = ({ page, title }) => ({ kind: "page", page, title });
  assert.deepEqual([...pages(turn), ...pages(tabler)], ofKind(events, "page.open").map(asFolded));
  assert.deepEqual(
    turn.pages.map(({ title }) => title),
    ["How I counted", "Seats"],
  );
  assert.equal(turn.pages[0], pages(turn)[0]);
  assert.equal(turn.pages[1], pages(tabler)[0]);
});

test("a page of no known agent, as a crafted file may hold, is left out", () => {
  const { turn } = foldEvents([
    { seq: 1, t: 0, kind: "turn.start", prompt: "Forge." },
    { seq: 2, t: 0, kind: "agent.start", node: 1, parent: null, cell: null },
    { seq: 3, t: 1, kind: "page.open", node: 9, cell: 1, page: "nowhere", title: "Nowhere" },
  ]);
  assert.deepEqual([turn.steps, turn.pages], [[], []]);
});
