// The Deep Reasoning chat: chats and their runs from the app's API, each run's events folded into its tree, live as they stream.
import { asText, clip, field, foldEvent, foldEvents, newFold, nodesOf } from "./fold.js";
import { modelGroups } from "./models.js";
import {
  COST_NA,
  MS_PER_SECOND,
  PRICE_UNKNOWN,
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
  join,
  known,
  opened,
  out,
  pageLink,
  plural,
  redraw,
  refocus,
  remember,
  renderAnswer,
  renderChange,
  renderStep,
  spinner,
  stepLabel,
  tick,
  toggler,
  useStoredTheme,
  views,
  waiting,
  whenChangesApply,
} from "./tree.js";

const SPIN_MS = 120;
const STICK_PX = 80;
// past the widest the log is: CSS cuts each part to fit
const TRAIL_CHARS = 120;
const TRAIL_PARTS = 3;
const CONFLICT = 409;
const NOT_FOUND = 404;
const NEW_TITLE = "New chat";
// while the app isn't answering, how often the page asks again
const RETRY_MS = 2000;
// focus and visibilitychange often both fire on one return to the tab
const COME_BACK_MS = 500;
const RUN_FROM_OTHER_TAB = "this chat has a run going, started in another tab; it shows above now";
// the width at which the chat list is a drawer, as style.css has it
const PHONE = matchMedia("(max-width: 720px)");
const SCRATCH_NAME = "scratch";
const FOLDER_NAME_CHARS = 24;
const POPOVER_MARGIN_PX = 8;
const NOT_ANSWERING = "the app isn't answering; is deep-reasoning still running?";
const RECONNECTING = "Lost the connection to the run; reconnecting…";
const PANEL_CLOSE = "panel-close";
const PANEL_OPEN = "panel-open";
const LIBRARY_CHANGED = "library-changed";
const PLUS_NA = `\u00a0+\u00a0${UNKNOWN_COST}`;
const MENU_STEPS = { ArrowDown: 1, ArrowUp: -1 };
const OUTCOME = {
  exhausted: ["ran out of turns", "warn"],
  failed: ["failed", "err"],
  stopped: ["stopped by you", ""],
};
const STOP_LABEL = "esc to interrupt";
const STOPPING = "stopping…";
const OPEN_PREFIX = "open:";
// a touch keyboard has no shift+enter to speak of
const NEW_LINE_HINT = matchMedia("(pointer: coarse)").matches ? "" : " · shift+enter for a new line";

const byId = (id) => document.getElementById(id);
// sending: the chat whose message is on its way; problem shows in red, notice in plain words
const state = { chats: [], current: null, live: null, sending: null, problem: null, problemWhileDown: false, notice: null };
const ui = { renaming: null, deleting: null };
const drafts = new Map();
// this page's changes on their way: a list read meanwhile may predate them
const inflight = new Set();
let changes = 0;
let refreshing = null;
let notAnswering = false;
let retrying = 0;
let cameBack = -COME_BACK_MS;
const library = { namespaces: null, defaultNamespace: null };
const HOME_FOLDER = "~";
let popover = null;
// Each provider's models, as the server lists them with its key: read once per page load, and again after settings change.
let modelLists = null;
let drawing = 0;

function countAgents(holder, wanted = () => true) {
  let count = 0;
  for (const step of holder.steps) {
    for (const agent of step.agents ?? []) count += Number(wanted(agent)) + countAgents(agent, wanted);
  }
  return count;
}

// A change applies from the message after its turn's, so only the newest turn's says so.
whenChangesApply((change) => state.current?.turns?.at(-1)?.changes.some((entry) => entry.step === change));

// Every change the run made, however deep in its tree, so they show with the tree shut.
function renderListedChange(entry) {
  const by = entry.tasks.length > 0 && `by “${field(entry.tasks.at(-1))}”`;
  return remember(entry, renderChange(entry.step, entry, by), () => renderListedChange(entry));
}

// a server of your own may serve a model under a vendor's name
const servedBy = (model, provider) => (provider ? `${model} on ${provider}` : model);
const modelView = (turn) => turn.model && element("span", { title: servedBy(turn.model, turn.provider) }, turn.model);

function summary(turn, switched) {
  const [word, tone] = known(OUTCOME, turn.outcome);
  const seconds = formatSeconds(turn.seconds);
  const agents = countAgents(turn);
  const lead = word ? [element("span", { class: `word ${tone}` }, word), ` after ${seconds}`] : `Reasoned for ${seconds}`;
  return join([lead, agents > 0 && plural(agents, "sub-agent"), costView(turn), modelView(turn), switched].filter(Boolean), " · ");
}

function liveSummary(turn, live, switched) {
  const running = countAgents(turn, (agent) => agent.status === "running");
  const stop = live && (live.stopping ? element("span", { class: "working" }, STOPPING) : STOP_LABEL);
  const facts = [clock(turn.started), running > 0 && `${plural(running, "sub-agent")} running`, modelView(turn), switched, stop];
  return [element("span", { class: "working" }, "Reasoning… "), "(", join(facts.filter(Boolean), " · "), ")"];
}

// The tasks of the sub-agents above the run's newest step, then the step.
function renderTrail({ tasks, step }) {
  const trail = [...tasks.map((task) => clip(task, TRAIL_CHARS)), stepLabel(step, TRAIL_CHARS)];
  const parts = trail.length > TRAIL_PARTS ? [trail[0], "…", ...trail.slice(1 - TRAIL_PARTS)] : trail;
  const last = parts.length - 1;
  return element(
    "span",
    { class: "trail" },
    parts.map((part, i) => [i > 0 && decor(" › ", { class: "sep" }), element("span", { class: i === last ? "part step" : "part" }, part)]),
  );
}

function renderPrompt(turn) {
  return clamped(turn, "prompt", turn.prompt, () => element("div", { class: "user" }, element("span", { class: "prompt" }, "> "), turn.prompt));
}

// A run out of turns has no answer: the engine's words saying so show as its failure.
function renderTurnAnswer(turn) {
  if (turn.answer == null) return null;
  if (turn.outcome === "exhausted") return out(turn.answer, { tone: "warn", full: true });
  return clamped(turn, "answer", turn.answer, () => renderAnswer(turn.answer));
}

// A turn names its namespace when it differs from the turn before's, so a chat that switched reads clearly.
function renderTurn(turn, previous) {
  const live = state.live?.turn === turn ? state.live : null;
  // no turn.end yet: still going, though this page may not follow it (a run started in another tab)
  const going = turn.outcome == null;
  const open = opened.has(turn);
  const latest = live?.fold.latest;
  const switched = previous && turn.namespace !== previous.namespace && turn.namespace;
  // a slow model's first answer takes a while; another tab's run may have steps this one hasn't seen
  const steps = turn.steps.length ? turn.steps.map((step) => renderStep(step, true)) : live && waiting();
  const el = element(
    "section",
    { class: "turn" },
    renderPrompt(turn),
    toggler(turn, { class: "summary" }, [
      caret(open),
      going ? spinner() : decor("✻"),
      element("span", { class: "text" }, going ? liveSummary(turn, live, switched) : summary(turn, switched)),
    ]),
    live && !open && (latest ? out(renderTrail(latest)) : out(WAITING)),
    open && element("div", { class: "steps" }, steps),
    renderTurnAnswer(turn),
    turn.outcome === "failed" && turn.detail && out(failureView(turn.detail, { model: turn.model }), { tone: "err", full: true }),
    turn.changes.length > 0 && element("div", { class: "changes" }, turn.changes.map(renderListedChange)),
    turn.pages.length > 0 && element("div", { class: "pages" }, turn.pages.map((step) => pageLink(step, { class: "page-line" }, decor("↗ "), `page: ${asText(step.title)}`))),
  );
  return remember(turn, el, () => renderTurn(turn, previous));
}

function renderWelcome() {
  return element(
    "div",
    { class: "welcome" },
    element("div", { class: "hi" }, "✻ Deep Reasoning"),
    element("p", {}, "Ask anything. Each run shows how it got its answer as a tree you open level by level."),
  );
}

// Enter or Esc takes focus back to the chat's row; a click elsewhere leaves it there.
function renderRename(chat) {
  const input = element("input", {
    class: "rename",
    id: "rename",
    value: chat.title,
    "aria-label": "Chat name",
    onkeydown: (e) => {
      e.stopPropagation();
      if (e.key !== "Enter" && e.key !== "Escape") return;
      // else the key would go on to press the row it hands focus to
      e.preventDefault();
      finishRename(chat, e.key === "Enter" ? input.value : chat.title, true);
    },
    onblur: () => finishRename(chat, input.value, false),
  });
  requestAnimationFrame(() => {
    input.focus();
    input.select();
  });
  return element("li", { class: "chat current" }, input);
}

// Keep is focused, as in the Library panel's confirms.
function renderConfirmDelete(chat) {
  const question = chat.scratch_files ? "Delete this chat? Its scratch folder, and any files the agent wrote there, goes too." : "Delete this chat?";
  return element(
    "li",
    { class: "chat confirm", role: "group", "aria-labelledby": "chat-delete-question" },
    element("span", { id: "chat-delete-question" }, question),
    element("button", { type: "button", class: "yes", "data-key": `chat-yes:${chat.id}`, onclick: () => remove(chat) }, "Delete"),
    element("button", { type: "button", class: "no", "data-key": `chat-keep:${chat.id}`, onclick: () => keep(chat) }, "Keep"),
  );
}

function renderChat(chat) {
  if (ui.renaming === chat) return renderRename(chat);
  if (ui.deleting === chat) return renderConfirmDelete(chat);
  const current = chat === state.current;
  const live = chat.live != null;
  return element(
    "li",
    { class: `chat ${current ? "current" : ""}` },
    element(
      "button",
      { type: "button", class: "pick", "data-key": `chat:${chat.id}`, "aria-current": current && "page", onclick: () => select(chat), ondblclick: () => setUi({ renaming: chat }) },
      element("span", { class: "mark" }, live ? spinner() : current ? "●" : ""),
      element("span", { class: "name" }, chat.title),
    ),
    element("button", { type: "button", class: "icon", title: "Rename", "aria-label": `Rename “${chat.title}”`, onclick: () => setUi({ renaming: chat }) }, "✎"),
    element("button", { type: "button", class: "icon", title: "Delete", "aria-label": `Delete “${chat.title}”`, "data-key": `chat-delete:${chat.id}`, onclick: () => askDelete(chat) }, "✕"),
  );
}

function renderList() {
  const focused = document.activeElement?.dataset?.key;
  byId("chats").replaceChildren(...state.chats.map(renderChat));
  if (focused?.startsWith("chat")) refocus(focused);
}

const focusRow = (chat, kind = "chat") => chat && refocus(`${kind}:${chat.id}`);

// The scratch folder goes with the chat, so the question names it unless the server says it holds no files.
async function askDelete(chat) {
  chat.scratch_files = true;
  setUi({ deleting: chat });
  focusRow(chat, "chat-keep");
  let scratch;
  try {
    scratch = await request("GET", `api/chats/${chat.id}/scratch`);
  } catch (error) {
    lostWith(error, chat);
    return;
  }
  chat.scratch_files = scratch.files;
  if (ui.deleting === chat && !scratch.files) renderList();
}

function keep(chat) {
  setUi({ deleting: null });
  focusRow(chat, "chat-delete");
}

function revealCurrent() {
  byId("chats").querySelector("[aria-current]")?.scrollIntoView({ block: "nearest" });
}

const drawerOpen = () => PHONE.matches && byId("app").classList.contains("list-open");

// Only at phone width is the list a drawer.
function setDrawer(open) {
  const was = drawerOpen();
  byId("app").classList.toggle("list-open", open);
  byId("menu").setAttribute("aria-expanded", String(open));
  byId("sidebar").inert = PHONE.matches && !open;
  if (!open || was) return;
  revealCurrent();
  (byId("chats").querySelector("[aria-current]") ?? byId("new-chat")).focus();
}

function keepScroll(scroll, draw) {
  const box = byId("scroll");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < STICK_PX;
  draw();
  if (scroll === "bottom" || (scroll === "follow" && nearBottom)) box.scrollTop = box.scrollHeight;
}

function logOf(chat) {
  if (!chat.turns) return [];
  return chat.turns.length ? chat.turns.map((turn, i) => renderTurn(turn, chat.turns[i - 1])) : [renderWelcome()];
}

function renderLog(scroll) {
  const chat = state.current;
  byId("title").textContent = chat?.title ?? "";
  if (!chat) return byId("log").replaceChildren();
  const focused = document.activeElement?.dataset?.key;
  keepScroll(scroll, () => byId("log").replaceChildren(...logOf(chat)));
  refocus(focused);
}

const runElsewhere = (live) => `a run is going in “${live.chat.title}”: wait for it, or open it and press Esc to stop it`;

function usualHint(live, here) {
  if (here) return "";
  if (live) return runElsewhere(live);
  return `enter to send${NEW_LINE_HINT}`;
}

function renderComposer() {
  const live = state.live;
  const here = live?.chat === state.current;
  const hint = byId("hint");
  const text = state.problem ?? state.notice ?? usualHint(live, here);
  byId("stop").hidden = !here;
  byId("stop").textContent = live?.stopping ? STOPPING : STOP_LABEL;
  hint.classList.toggle("err", state.problem != null);
  // the hint is a live region: setting the same text again would announce it again
  if (hint.textContent !== text) hint.textContent = text;
  renderTotal(state.current?.turns ?? []);
  renderChoices(state.current);
}

// What the chat's turns cost, with N/A when any turn's is unknown.
function renderTotal(turns) {
  const sum = turns.reduce((total, turn) => total + (turn.cost ?? 0), 0);
  const known = sum > 0 && formatCost(sum);
  const unpriced = turns.some((turn) => turn.unpriced);
  const unknown = unpriced || turns.some((turn) => turn.unfinished);
  const total = byId("total");
  total.title = unknown ? (unpriced ? PRICE_UNKNOWN : UNFINISHED) : "";
  if (unknown) total.textContent = known ? `${known}${PLUS_NA} this chat` : `${COST_NA} this chat`;
  else total.textContent = known ? `${known} this chat` : "";
}

// Updated in place, as the composer redraws with each frame of a live run. The server says what the chat's next message runs on.
function renderChoices(chat) {
  const ready = library.namespaces && chat?.turns;
  byId("choices").hidden = !ready;
  if (!ready) return;
  byId("namespace-name").textContent = chat.namespace;
  byId("namespace").title = `${chat.namespace}: this chat's namespace, from its next message`;
  byId("model-name").textContent = chat.model?.name ?? "pick a model";
  byId("model-provider").textContent = chat.model ? ` · ${chat.model.provider}` : "";
  byId("model").title = `${chat.model ? servedBy(chat.model.name, chat.model.provider) : "No model"}: this chat's model, from its next message`;
  renderFolder(chat);
}

// A pick applies from the chat's next message; the server keeps it, so it survives a reload.
async function choose(chat, change) {
  let chosen;
  try {
    chosen = await request("PATCH", `api/chats/${chat.id}`, change);
  } catch (error) {
    if (!lostWith(error, chat)) report(`Not changed: ${error.message}`);
    return;
  }
  Object.assign(chat, { namespace: chosen.namespace, model: chosen.model });
  report(null);
}

// A chat with no folder of its own works in a scratch folder, which goes when the chat does.
function renderFolder(chat) {
  const name = chat.folder ? chat.folder.split("/").at(-1) || "/" : SCRATCH_NAME;
  const where = chat.folder ?? `scratch folder ${chat.scratch_folder}/, deleted with the chat`;
  const full = `This chat's working folder, from its next message: ${where}`;
  const button = byId("folder");
  byId("folder-name").textContent = clip(name, FOLDER_NAME_CHARS);
  button.title = full;
  button.setAttribute("aria-label", full);
}

function pickFolder() {
  const chat = state.current;
  folderPicker(byId("folder"), byId("above"), {
    label: "This chat's working folder",
    current: chat.folder,
    empty: "its own scratch folder, deleted with the chat",
    note: "Its Python runs there and reads and writes files there, from the next message. A folder you choose is never deleted.",
    pick: async (folder) => {
      try {
        ({ folder: chat.folder } = await request("PATCH", `api/chats/${chat.id}`, { folder }));
      } catch (error) {
        if (!lostWith(error, chat)) throw error;
        return;
      }
      if (chat === state.current) renderFolder(chat);
    },
  });
}

// A chat whose namespace was deleted still shows it, marked, so its next pick moves it on.
function pickNamespace() {
  const chat = state.current;
  const star = element("span", { class: "lib-star" }, "★ new chats start here");
  const options = library.namespaces.map((name) => ({ value: name, label: [name, name === library.defaultNamespace && star] }));
  if (!library.namespaces.includes(chat.namespace)) options.unshift({ value: chat.namespace, label: [chat.namespace, element("span", { class: "gone" }, "no longer exists")] });
  showMenu(byId("namespace"), byId("above"), {
    label: "Namespaces",
    groups: () => [{ options }],
    isCurrent: (name) => name === chat.namespace,
    pick: (name) => library.namespaces.includes(name) && choose(chat, { namespace: name }),
  });
}

function loadModels() {
  modelLists ??= request("GET", "api/models").catch((error) => {
    modelLists = null;
    throw error;
  });
  return modelLists;
}

// The menu opens at once, saying the lists are loading while a provider is slow to answer.
async function pickModel() {
  const button = byId("model");
  const host = byId("above");
  if (isOpenFrom(button)) return closePopover(true);
  const chat = state.current;
  const loading = notePopover(button, host, "Loading the model lists…");
  let lists;
  try {
    lists = await loadModels();
  } catch (error) {
    loading.textContent = `No model lists: ${error.message}`;
    return;
  }
  if (!popover?.el.contains(loading)) return;
  closePopover(false);
  if (!lists.length) return noProviderMenu(button, host);
  showMenu(button, host, {
    label: "Models",
    filter: "filter, or type a model name",
    groups: (query) => modelGroups(lists, query, chat.model),
    isCurrent: (model) => model.provider === chat.model?.provider && model.name === chat.model?.name,
    pick: (model) => choose(chat, { model }),
  });
}

// A popover with a line of text, and the button for what to do about it, if any.
function notePopover(button, host, text, action) {
  const note = element("p", { class: "menu-note" }, text);
  const el = element("div", { class: "menu-pop", role: "dialog", "aria-label": "Models", tabindex: "-1" }, note, action);
  openPopover(button, host, el, action ?? el);
  return note;
}

// With no key and no custom server there is no model to list, so the menu points to settings.
function noProviderMenu(button, host) {
  const add = element("button", { type: "button", class: "lib-action lib-primary", onclick: openSettings }, "Open settings");
  notePopover(button, host, "No models yet: add a provider's key, or a custom server (which needs none), in settings.", add);
}

function openSettings() {
  closePopover(false);
  byId("settings").dispatchEvent(new Event(PANEL_OPEN));
}

// After a change in settings the lists are read again, and each chat's model too: one that never picked follows what new chats start on.
async function settingsChanged() {
  modelLists = null;
  await refresh();
}

// After the Library panel read the Library, or it changed: its namespaces, and each chat's, as one that never picked follows the default.
async function libraryChanged(found) {
  useLibrary(found);
  renderComposer();
  await refresh();
}

// A popover opened by its button into host. Esc closes it and stops there, so it never also stops a run or closes a panel;
// focus leaving it (a click elsewhere, Tab) closes it too. Its buttons keep focus in it on mousedown, so a click toggles it.
function openPopover(button, host, el, focus) {
  closePopover(false);
  popover = { button, el };
  el.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    e.preventDefault();
    e.stopPropagation();
    closePopover(true);
  });
  // checked a frame later, as redrawing a popover's insides drops focus for a moment
  el.addEventListener("focusout", () => requestAnimationFrame(() => closeIfLeft(el)));
  host.append(el);
  // the composer's popovers open upward: one taller than the room above it scrolls inside, so its top stays on the screen
  if (host === byId("above")) el.style.maxHeight = `${el.getBoundingClientRect().bottom - POPOVER_MARGIN_PX}px`;
  button.setAttribute("aria-expanded", "true");
  focus.focus();
}

function closeIfLeft(el) {
  if (popover?.el === el && !el.contains(document.activeElement)) closePopover(false);
}

function closePopover(refocus) {
  const open = popover;
  if (!open) return;
  popover = null;
  open.el.remove();
  open.button.setAttribute("aria-expanded", "false");
  if (refocus) open.button.focus();
}

const isOpenFrom = (button) => popover?.button === button;

// A menu of grouped options, with a filter field when it has one: arrows move, Enter picks.
function showMenu(button, host, spec) {
  if (isOpenFrom(button)) return closePopover(true);
  const list = element("div", { class: "menu-list", id: "menu-list", role: "listbox", "aria-label": spec.label, tabindex: spec.filter ? null : "-1", onmousedown: (e) => e.preventDefault() });
  const field = spec.filter && element("input", { class: "menu-filter", role: "combobox", "aria-label": `Filter the ${spec.label.toLowerCase()}`, "aria-controls": list.id, "aria-expanded": "true", "aria-autocomplete": "list", placeholder: spec.filter, autocomplete: "off", spellcheck: "false" });
  const el = element("div", { class: "menu-pop" }, field, list);
  const open = { spec, list, field, options: [], active: 0 };
  field?.addEventListener("input", () => fillMenu(open));
  el.addEventListener("keydown", (e) => menuKey(open, e));
  fillMenu(open);
  openPopover(button, host, el, field || list);
  markActive(open);
}

function fillMenu(open) {
  const groups = open.spec.groups(open.field?.value ?? "");
  open.options = groups.flatMap((group) => group.options);
  const current = open.options.findIndex((option) => isCurrentOption(open, option));
  open.active = open.field?.value ? 0 : Math.max(current, 0);
  open.list.replaceChildren(...groups.map((group, i) => menuGroup(open, group, i)));
  open.list.scrollTop = 0;
  markActive(open);
}

const isCurrentOption = (open, option) => option.value != null && open.spec.isCurrent(option.value);

function menuGroup(open, group, number) {
  const head = group.name && element("div", { class: "menu-head", id: `menu-group-${number}`, role: "presentation" }, group.name);
  const note = group.note && element("div", { class: "menu-note", id: `menu-note-${number}`, role: "presentation" }, group.note);
  return element(
    "div",
    { role: "group", "aria-labelledby": head?.id, "aria-describedby": note?.id },
    head,
    note,
    group.options.map((option) => {
      const current = isCurrentOption(open, option);
      const id = `menu-option-${open.options.indexOf(option)}`;
      const mark = element("span", { class: "mark" }, current ? "✓" : "");
      return element("div", { class: "menu-option", id, role: "option", "aria-selected": String(current), onclick: () => chooseOption(open, option) }, mark, option.label);
    }),
  );
}

function markActive(open) {
  const id = open.options.length ? `menu-option-${open.active}` : null;
  const owner = open.field || open.list;
  for (const el of open.list.querySelectorAll(".active")) el.classList.remove("active");
  if (!id) return owner.removeAttribute("aria-activedescendant");
  owner.setAttribute("aria-activedescendant", id);
  const el = open.list.querySelector(`[id="${id}"]`);
  el.classList.add("active");
  el.scrollIntoView({ block: "nearest" });
}

function menuKey(open, e) {
  const step = MENU_STEPS[e.key];
  if (step && open.options.length) {
    e.preventDefault();
    open.active = (open.active + step + open.options.length) % open.options.length;
    markActive(open);
  } else if (e.key === "Enter" || (e.key === " " && !open.field)) {
    // also keeps Enter in the filter from sending the composer's message
    e.preventDefault();
    if (open.options.length) chooseOption(open, open.options[open.active]);
  }
}

// "use model name…" picks nothing: the filter field takes the name.
function chooseOption(open, option) {
  if (option.value == null) {
    open.field.placeholder = option.hint;
    open.field.focus();
    return;
  }
  closePopover(true);
  open.spec.pick(option.value);
}

// A folder typed or pasted, or found in the browser under the field, which follows what the field holds; empty means the scratch folder.
// The server lists the folders and checks the one picked: spec.pick throws its refusal, which the picker shows.
function folderPicker(button, host, spec) {
  if (isOpenFrom(button)) return closePopover(true);
  const field = element("input", { id: "folder-field", class: "menu-filter", value: spec.current ?? "", placeholder: `empty: ${spec.empty}`, autocomplete: "off", spellcheck: "false", "aria-describedby": "folder-problem folder-pop-note" });
  const problem = element("div", { id: "folder-problem", class: "folder-problem", "aria-live": "polite" });
  const browser = element("div", { class: "folder-browser", role: "group", "aria-label": "Folders", onmousedown: (e) => e.preventDefault() });
  const use = element("button", { type: "submit", class: "lib-action lib-primary" });
  const picker = { button, spec, field, problem, browser, use, asked: null };
  const el = element(
    "form",
    { class: "menu-pop folder-pop", method: "dialog", role: "dialog", "aria-label": spec.label, onsubmit: (e) => useFolder(e, picker) },
    element("label", { class: "menu-head", for: field.id }, spec.label),
    field,
    problem,
    element("p", { class: "menu-note", id: "folder-pop-note" }, spec.note),
    browser,
    element("div", { class: "lib-actions" }, use),
  );
  field.addEventListener("input", () => {
    problem.textContent = "";
    fillBrowser(picker);
  });
  fillBrowser(picker);
  openPopover(button, host, el, field);
}

const cleanFolder = (path) => path.trim().replace(/(.)\/+$/, "$1");

// Each keystroke asks again; an answer to an older question is dropped.
async function fillBrowser(picker) {
  const here = cleanFolder(picker.field.value) || HOME_FOLDER;
  picker.use.textContent = picker.field.value.trim() ? "Use this folder" : "Use a scratch folder";
  picker.asked = here;
  let listing;
  try {
    listing = await request("GET", `api/folders?path=${encodeURIComponent(here)}`);
  } catch (error) {
    if (picker.asked === here) picker.browser.replaceChildren(element("div", { class: "folder-here" }, here), element("p", { class: "menu-note" }, error.message));
    return;
  }
  if (picker.asked !== here) return;
  const row = (label, path) => element("button", { type: "button", class: "row", onclick: () => goToFolder(picker, path) }, label);
  const up = listing.parent == null ? [] : [row("..", listing.parent)];
  const rows = listing.folders.length ? listing.folders.map(({ name, path }) => row(`${name}/`, path)) : [element("p", { class: "menu-note" }, "no folders in here")];
  picker.browser.replaceChildren(element("div", { class: "folder-here" }, listing.path), ...up, ...rows);
}

async function goToFolder(picker, path) {
  picker.field.value = path;
  picker.problem.textContent = "";
  const fromKeyboard = picker.browser.contains(document.activeElement);
  await fillBrowser(picker);
  // a folder that can't be listed leaves no row, and focus mustn't fall out of the picker
  if (fromKeyboard) (picker.browser.querySelector(".row") ?? picker.field).focus();
}

async function useFolder(e, picker) {
  e.preventDefault();
  const path = cleanFolder(picker.field.value);
  try {
    await picker.spec.pick(path || null);
  } catch (error) {
    // a browser note saying the same gives way to the refusal, so it reads once
    const note = picker.browser.querySelector(".menu-note");
    if (note?.textContent === error.message) note.remove();
    picker.problem.textContent = error.message;
    picker.field.focus();
    return;
  }
  if (isOpenFrom(picker.button)) closePopover(true);
}

// The namespace the chat on screen runs in, for the Library panel; null with none on screen.
const chatNamespace = () => state.current?.namespace ?? null;

// One side panel at a time: opening one asks the others to close.
function closeOtherPanels(panel) {
  for (const other of document.querySelectorAll(".library:not([hidden])")) {
    if (other !== panel) other.dispatchEvent(new Event(PANEL_CLOSE));
  }
}

function render(scroll = "keep") {
  if (!ui.renaming) renderList();
  renderLog(scroll);
  renderComposer();
}

function setUi(change) {
  Object.assign(ui, { renaming: null, deleting: null }, change);
  renderList();
}

function report(problem) {
  Object.assign(state, { problem, problemWhileDown: problem != null && notAnswering, notice: null });
  renderComposer();
}

// A request that changes something is tracked until answered, so a list read meanwhile is read again after it.
async function request(method, url, body) {
  if (method === "GET") return answered(method, url, body);
  changes += 1;
  const sent = answered(method, url, body);
  inflight.add(sent);
  try {
    return await sent;
  } finally {
    inflight.delete(sent);
  }
}

async function answered(method, url, body) {
  const options = body === undefined ? { method } : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    answering(false);
    throw new Error(NOT_ANSWERING);
  }
  answering(true);
  const answer = await response.json().catch(() => ({}));
  if (response.ok) return answer;
  const problem = typeof answer.detail === "string" ? answer.detail : `${url} answered ${response.status}`;
  throw Object.assign(new Error(problem), { status: response.status });
}

// While the app isn't answering, the page asks again, and a hint shown meanwhile goes once it answers.
function answering(up) {
  if (notAnswering === !up) return;
  notAnswering = !up;
  if (notAnswering) {
    retrying = setInterval(refresh, RETRY_MS);
    return;
  }
  clearInterval(retrying);
  if (!state.problemWhileDown) return;
  Object.assign(state, { problem: null, problemWhileDown: false });
  renderComposer();
}

// What other tabs changed: the chats, and the open chat's turns when another tab ran in it.
function refresh(scroll = "follow") {
  refreshing ??= readAll(scroll).finally(() => {
    refreshing = null;
  });
  return refreshing;
}

// The log redraws only when its chat was read, so a selection in it survives a focus.
async function readAll(scroll) {
  try {
    await readChats();
  } catch (error) {
    report(error.message);
    return;
  }
  if (!ui.renaming) renderList();
  // a click, or a chat deleted elsewhere, may put another chat on screen while one is read
  for (let chat = state.current; chat && stale(chat); chat = state.current) {
    try {
      await readTurns(chat);
    } catch (error) {
      if (!lostWith(error, chat) && state.current === chat) return report(`Could not open this chat: ${error.message}`);
      continue;
    }
    if (state.current === chat) return render(scroll);
  }
  byId("title").textContent = state.current?.title ?? "";
  renderComposer();
}

// The list, the one used last first, read again while what this page sent may have changed it meanwhile.
async function readChats() {
  for (;;) {
    await Promise.allSettled(inflight);
    const asked = changes;
    const listed = await request("GET", "api/chats");
    if (asked === changes && !inflight.size) return useList(listed);
  }
}

// Each chat keeps its object, so its turns and draft stay; one no longer listed was deleted elsewhere.
function useList(listed) {
  const known = new Map(state.chats.map((chat) => [chat.id, chat]));
  const gone = state.chats.filter((chat) => !listed.some(({ id }) => id === chat.id));
  state.chats = listed.map((each) => adopt(known.get(each.id) ?? {}, each));
  for (const chat of gone) lose(chat);
}

// A hint is about what the chat runs on: once that changes, as when settings gave it a model, the hint goes.
function adopt(chat, listed) {
  const picks = (each) => JSON.stringify([each.namespace, each.model, each.folder]);
  if (chat === state.current && picks(chat) !== picks(listed)) state.problem = null;
  // a run this page follows is its own to end
  const { live, ...rest } = listed;
  return Object.assign(chat, rest, state.live?.chat === chat ? {} : { live });
}

// Read again when another tab ran something in it, or while its run goes on unfollowed, so its turn never stays half-folded.
const stale = (chat) => !chat.turns || (state.live?.chat !== chat && (chat.live != null || chat.last_run !== chat.read_run));

// A run still going in a chat is followed from where its turns end, unless the page follows another or the chat is gone.
async function readTurns(chat) {
  const { turns, ...rest } = await request("GET", `api/chats/${chat.id}`);
  const folds = turns.map((turn) => foldEvents(turn.events));
  for (const fold of folds) reopen(fold.turn);
  Object.assign(chat, rest, { turns: folds.map((fold) => fold.turn), read_run: rest.last_run });
  if (rest.live && !state.live && state.chats.includes(chat)) follow(chat, rest.live, folds.at(-1));
}

// Any 404 for a chat means it is gone: deleted in another tab.
function lostWith(error, chat) {
  if (error.status !== NOT_FOUND) return false;
  lose(chat);
  return true;
}

// A chat deleted elsewhere leaves the list with a note naming it; on screen, it gives way to the chat used last.
function lose(chat) {
  const index = state.chats.indexOf(chat);
  if (index >= 0) state.chats.splice(index, 1);
  drafts.delete(chat.id);
  if (state.live?.chat === chat) {
    state.live.source.close();
    state.live = null;
  }
  if (ui.renaming === chat || ui.deleting === chat) Object.assign(ui, { renaming: null, deleting: null });
  state.notice = `“${chat.title}” was deleted in another tab.`;
  if (state.current === chat) {
    state.current = null;
    if (state.chats.length) select(state.chats[0], { focus: null });
    else newChat({ focus: null });
  }
  render();
}

function keepDraft() {
  if (state.chats.includes(state.current)) drafts.set(state.current.id, byId("input").value);
}

function showDraft(chat) {
  byId("input").value = drafts.get(chat.id) ?? "";
  grow();
}

// focus: "input" (shutting the drawer), "row", or null when the page picked it, leaving focus and the notice.
async function select(chat, { focus = "input" } = {}) {
  keepDraft();
  state.current = chat;
  state.problem = null;
  if (focus) state.notice = null;
  ui.deleting = null;
  showDraft(chat);
  if (focus === "input") setDrawer(false);
  render("bottom");
  revealCurrent();
  if (focus === "input") byId("input").focus();
  if (focus === "row") focusRow(chat);
  await refresh("bottom");
}

// A chat with no messages and its first title is reused, so clicking "+ New chat" never piles up empty ones.
const isEmpty = (chat) => chat.last_run == null && chat.live == null && chat.title === NEW_TITLE && chat !== state.sending;

async function newChat(how) {
  const empty = state.chats.find(isEmpty);
  if (empty) return select(empty, how);
  let made;
  try {
    made = await request("POST", "api/chats");
  } catch (error) {
    report(`No new chat: ${error.message}`);
    return;
  }
  // a list read meanwhile may have it already
  const chat = state.chats.find(({ id }) => id === made.id) ?? Object.assign(made, { turns: [], read_run: null });
  if (!state.chats.includes(chat)) state.chats.unshift(chat);
  select(chat, how);
}

// The list drops the chat at once and focus goes to the next row; the server may take a few seconds, stopping a run that is going first.
async function remove(chat) {
  const index = state.chats.indexOf(chat);
  if (state.live?.chat === chat) {
    state.live.source.close();
    state.live = null;
  }
  state.chats.splice(index, 1);
  drafts.delete(chat.id);
  ui.deleting = null;
  state.notice = null;
  renderList();
  const next = state.chats[Math.min(index, state.chats.length - 1)];
  if (state.current !== chat) {
    renderComposer();
    if (next) focusRow(next);
    else byId("new-chat").focus();
  } else if (next) select(next, { focus: "row" });
  else newChat();
  try {
    await request("DELETE", `api/chats/${chat.id}`);
  } catch (error) {
    // gone already: deleted in another tab
    if (error.status === NOT_FOUND) return;
    // back in the list, read afresh when next opened
    state.chats.splice(Math.min(index, state.chats.length), 0, Object.assign(chat, { turns: null }));
    report(`Not deleted: ${error.message}`);
    render();
  }
}

async function finishRename(chat, typed, backToRow) {
  if (ui.renaming !== chat) return;
  ui.renaming = null;
  const before = chat.title;
  const title = typed.trim();
  if (title && title !== before) chat.title = title;
  render();
  if (backToRow) focusRow(chat);
  if (chat.title === before) return;
  let renamed;
  try {
    renamed = await request("PATCH", `api/chats/${chat.id}`, { title });
  } catch (error) {
    if (lostWith(error, chat)) return;
    chat.title = before;
    report(`Not renamed: ${error.message}`);
    render();
    return;
  }
  // the server cuts a long one
  if (renamed.title === chat.title) return;
  chat.title = renamed.title;
  render();
}

function follow(chat, run, fold) {
  const source = new EventSource(`api/runs/${run}/events?after=${fold.seq}`);
  const live = { chat, run, fold, turn: fold.turn, source, stopping: false };
  state.live = live;
  chat.live = run;
  source.onopen = () => {
    if (state.problem === RECONNECTING) report(null);
  };
  source.onmessage = (message) => receive(live, JSON.parse(message.data));
  source.onerror = () => lost(live);
}

function receive(live, event) {
  if (!foldEvent(live.fold, event)) return;
  if (event.kind === "library.change") byId("library").dispatchEvent(new Event(LIBRARY_CHANGED));
  // the server ended it as another tab deleted its chat
  if (event.kind === "turn.end" && event.outcome === "deleted") return lose(live.chat);
  if (event.kind === "turn.end") return finish(live);
  drawLive(live);
}

// Events come in bursts, as when catching up on a long run, so the live turn redraws at most once a frame.
function drawLive(live) {
  if (drawing) return;
  drawing = requestAnimationFrame(() => {
    drawing = 0;
    if (state.live !== live || live.chat !== state.current || !views.has(live.turn)) return;
    keepScroll("follow", () => redraw(live.turn));
    renderComposer();
  });
}

// The stream closes after turn.end; left open, EventSource would reconnect to it.
function finish(live) {
  live.source.close();
  live.chat.live = null;
  // a list read while the run went on would show it going again
  changes += 1;
  if (state.live === live) state.live = null;
  const [ended = "answered"] = known(OUTCOME, live.turn.outcome);
  byId("announce").textContent = `Run ${ended} after ${formatSeconds(live.turn.seconds)} in “${live.chat.title}”.`;
  render("follow");
}

// EventSource retries a dropped stream itself; it gives up only when refused (a run deleted meanwhile), so the chat is read afresh.
function lost(live) {
  if (live.source.readyState !== EventSource.CLOSED) return report(RECONNECTING);
  if (state.live === live) state.live = null;
  if (state.problem === RECONNECTING) state.problem = null;
  Object.assign(live.chat, { live: null, turns: null });
  if (live.chat === state.current) select(live.chat, { focus: null });
  else render();
}

// A run going here or in another chat keeps the message, and the hint says why.
async function send() {
  const input = byId("input");
  const prompt = input.value.trim();
  const chat = state.current;
  if (!prompt || state.sending || !chat?.turns) return;
  if (state.live) {
    report(`Not sent: ${state.live.chat === chat ? "this chat's run is going: wait for it, or press Esc to stop it" : runElsewhere(state.live)}.`);
    return;
  }
  state.sending = chat;
  let run, title;
  try {
    ({ run, title } = await request("POST", `api/chats/${chat.id}/runs`, { prompt }));
  } catch (error) {
    if (lostWith(error, chat)) return;
    if (error.status !== CONFLICT) return report(`Not sent: ${error.message}`);
    report(`Not sent: ${RUN_FROM_OTHER_TAB}.`);
    // read again, so the page follows the run another tab started
    refresh("bottom");
    return;
  } finally {
    state.sending = null;
  }
  if (state.current === chat && input.value.trim() === prompt) input.value = "";
  if (drafts.get(chat.id)?.trim() === prompt) drafts.delete(chat.id);
  grow();
  chat.title = title;
  // used last, so first in the list
  if (state.chats.includes(chat)) state.chats.unshift(...state.chats.splice(state.chats.indexOf(chat), 1));
  Object.assign(chat, { last_run: run, read_run: run });
  const fold = newFold();
  Object.assign(fold.turn, { prompt, started: Date.now() / MS_PER_SECOND });
  chat.turns.push(fold.turn);
  state.problem = null;
  follow(chat, run, fold);
  render("bottom");
}

// Says so at once: a cell blocked in C holds the stop up until the app ends its worker.
async function stop() {
  const live = state.live;
  if (!live || live.stopping) return;
  live.stopping = true;
  renderComposer();
  drawLive(live);
  try {
    await request("POST", `api/runs/${live.run}/stop`);
  } catch (error) {
    // a conflict: the run has just ended, and its turn.end is on its way
    if (error.status === CONFLICT) return;
    live.stopping = false;
    report(`Not stopped: ${error.message}`);
    drawLive(live);
  }
}

function grow() {
  const input = byId("input");
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

function isDark() {
  const theme = document.documentElement.dataset.theme;
  return theme ? theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

function labelTheme() {
  const next = isDark() ? "light" : "dark";
  byId("theme").textContent = next;
  byId("theme").setAttribute("aria-label", `Switch to ${next}`);
}

function toggleTheme() {
  const root = document.documentElement;
  root.dataset.theme = isDark() ? "light" : "dark";
  labelTheme();
  try {
    localStorage.setItem("theme", root.dataset.theme);
  } catch {
    // storage can be blocked; the choice then lasts this visit only
  }
}

function wire() {
  useStoredTheme();
  labelTheme();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", labelTheme);
  byId("theme").addEventListener("click", toggleTheme);
  byId("new-chat").addEventListener("click", () => newChat());
  byId("stop").addEventListener("click", stop);
  byId("namespace").addEventListener("click", pickNamespace);
  byId("model").addEventListener("click", pickModel);
  byId("folder").addEventListener("click", pickFolder);
  // a click on a popover's button leaves focus in the popover, so the click toggles it
  for (const id of ["namespace", "model", "folder"]) byId(id).addEventListener("mousedown", (e) => e.preventDefault());
  byId("menu").addEventListener("click", () => setDrawer(!drawerOpen()));
  // the event's path, as a click in the list may redraw the row it landed on
  byId("app").addEventListener("click", (e) => {
    const path = e.composedPath();
    if (drawerOpen() && !path.includes(byId("sidebar")) && !path.includes(byId("menu"))) setDrawer(false);
  });
  PHONE.addEventListener("change", () => setDrawer(false));
  setDrawer(false);
  window.addEventListener("focus", comeBack);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) comeBack();
  });
  byId("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  byId("input").addEventListener("input", grow);
  byId("input").addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    send();
  });
  // Esc keeps a chat it was asked to delete, else shuts the drawer, else stops this chat's run; the Library closes on an Esc left unused
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const runHere = state.live?.chat === state.current;
    if (!ui.deleting && !drawerOpen() && !runHere) return;
    e.preventDefault();
    if (ui.deleting) keep(ui.deleting);
    else if (drawerOpen()) {
      setDrawer(false);
      byId("menu").focus();
    } else stop();
  });
  setInterval(() => {
    if (state.live) tick();
  }, SPIN_MS);
  // beforeunload comes before a live run's stream is cut, which drops its chat's turns; some phones send only pagehide
  for (const unloading of ["beforeunload", "pagehide"]) addEventListener(unloading, keepOpen);
}

// The nodes open in each chat's turns, kept for this tab as it unloads, so a reload opens them again.
function keepOpen() {
  const turns = state.chats.flatMap((chat) => chat.turns ?? []).filter((turn) => turn.run);
  try {
    for (const turn of turns) {
      const paths = [...nodesOf(turn)].filter(([, node]) => opened.has(node)).map(([path]) => path);
      if (paths.length) sessionStorage.setItem(OPEN_PREFIX + turn.run, JSON.stringify(paths));
      else sessionStorage.removeItem(OPEN_PREFIX + turn.run);
    }
  } catch {
    // storage can be blocked; the tree then opens shut after a reload
  }
}

function reopen(turn) {
  let paths;
  try {
    paths = new Set(JSON.parse(sessionStorage.getItem(OPEN_PREFIX + turn.run) ?? "[]"));
  } catch {
    return;
  }
  for (const [path, node] of nodesOf(turn)) if (paths.has(path)) opened.add(node);
}

// Another tab, the librarian or an import may have changed the chats or the Library.
function comeBack() {
  const now = performance.now();
  if (now - cameBack < COME_BACK_MS) return;
  cameBack = now;
  if (byId("library").hidden) readNamespaces();
  else byId("library").dispatchEvent(new Event(LIBRARY_CHANGED));
  refresh();
}

async function readNamespaces() {
  try {
    useLibrary(await request("GET", "api/namespaces"));
  } catch {
    return;
  }
  renderComposer();
}

// The namespaces a chat can pick, and the default one new chats start in, as the Library panel reads them.
function useLibrary(found) {
  library.namespaces = found.namespaces.filter(({ usable }) => usable).map(({ name }) => name);
  library.defaultNamespace = found.default_namespace;
}

// A run going in a chat is what the page was showing before a reload, so that chat opens first; else the one used last.
async function start() {
  try {
    await readChats();
  } catch (error) {
    byId("log").textContent = `Could not load the chats (${error.message}). Start the app with: uv run deep-reasoning`;
    return;
  }
  const first = state.chats.find((chat) => chat.live) ?? state.chats[0];
  if (first) select(first);
  else newChat();
}

export { LIBRARY_CHANGED, PANEL_CLOSE, PANEL_OPEN, caret, chatNamespace, closeOtherPanels, element, folderPicker, libraryChanged, request, settingsChanged };

wire();
start();
// read ahead, so the model menu opens on its lists; a failure shows when the menu opens
loadModels().catch(() => {});
