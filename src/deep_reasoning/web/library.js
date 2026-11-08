// The Library panel: the decompositions, namespaces, tools and data every run draws on, read from the server's Library file; adding, editing and deleting save there.
import { LIBRARY_CHANGED, PANEL_CLOSE, TOUCH, caret, chatNamespace, closeOtherPanels, element, libraryChanged, request, showPanel } from "./app.js";
import { turnProblems } from "./turns.js";

const LIBRARY_URL = "api/library";
const ADD_URL = "api/decompositions";
const NAMESPACES_URL = "api/namespaces";
const TOOLS_URL = "api/tools";
const RUN_SETTINGS_URL = "api/run-settings";
// the server's answer to a save or delete opened on what has changed since
const CONFLICT = 409;
const TABS = ["Decompositions", "Namespaces", "Tools", "Data"];
const ROOT = "root";
const ROLES = ["user", "assistant", "system"];
const DECOMPOSITION_KEYS = ["name", "messages"];
const MESSAGE_KEYS = ["role", "content"];
const NAMESPACE_KEYS = ["repl", "reasoner", "tools", "vars", "spawn", "system_suffix"];
// a new namespace's name: plain words, joined by dots for a child
const NAMESPACE_NAME = /^[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*$/;
const INDENT = "  ";
// a touch keyboard has no Tab or Esc to speak of, so touch screens get no line about them
const EDITOR_KEYS = TOUCH ? null : "tab indents · esc leaves · ctrl/⌘+enter saves";
const CHECK_DELAY_MS = 300;
// What a namespace's save says while it runs, as the engine checks the whole Library, and a tool it grants afresh is built first.
const SAVING = "Saving…";
const BUILDING = "Saving. A custom tool this grants for the first time is built first, to check it, which can take up to a minute.";
// as the server's limit on a decomposition's YAML; a file of more bytes than 4 a character can't fit
const MAX_YAML_CHARS = 1_000_000;
const UTF8_MAX_BYTES = 4;
const counted = new Intl.NumberFormat("en");
// characters that don't show, as the server reads them: controls and formats, such as a zero-width space
const INVISIBLE = /[\p{Cc}\p{Cf}]/u;
// a code point as U+ writes it: hex, at least four digits
const CODE_POINT_DIGITS = 4;
const HEX = 16;
const codePoint = (char) => `U+${char.codePointAt(0).toString(HEX).toUpperCase().padStart(CODE_POINT_DIGITS, "0")}`;
// what heads a decomposition's turn problems in its form: they don't stop it saving, as a draft may have them, but agents are shown it as it is
const reach = (namespace) => (namespace === ROOT ? "every namespace" : `${namespace} and the namespaces under it`);
const headed = (action, namespace, problems) =>
  problems.length ? [`${action} saves it as it is, and from the next message agents in ${reach(namespace)} are shown it, marked ✗ until these are fixed:`, ...problems] : [];
// the shape of the generated decompositions, small enough to edit in place
const TEMPLATE = `name: count the open tickets
messages:
- role: user
  content: How many tickets are still open?
- role: assistant
  content: |
    <think>Count the tickets whose status is open, and print the count to check it.</think>
    <repl>
    open_tickets = sum(t["status"] == "open" for t in tickets)
    print(open_tickets)
    </repl>
- role: user
  content: |
    <observation>
    12
    </observation>
- role: assistant
  content: |
    <think>12 tickets are open. Answer from the count the code holds, not a number typed in.</think>
    <repl>
    FinalAnswer(f"{open_tickets} tickets are still open.")
    </repl>
`;
// a namespace's own fields, shown as the empty editor's placeholder; what it leaves out it inherits, as llm from root
const NAMESPACE_EXAMPLE = `vars:
  topic: machine learning
system_suffix: |
  Answer in plain words, and say where each fact comes from.
`;
// worded as the server words them; it checks again and has the last word
const NAMED_ELSEWHERE = {
  name: 'Remove "name": a namespace is named in the name field, not in its YAML.',
  decompositions: `Remove "decompositions": a namespace's decompositions are added in the Decompositions tab.`,
};
const SHAPES = {
  tools: 'tools should be a list of tool names, like "tools: [llm]".',
  spawn: 'spawn should be a list of namespace names, like "spawn: [research]", or [] for none.',
  vars: 'vars should be a mapping of names to values, like "vars: {year: 2026}".',
  system_suffix: 'system_suffix should be text; write "system_suffix: |" with the text indented below.',
  reasoner: 'reasoner should be a mapping, like "reasoner: {type: claude_code}".',
};

const panel = document.getElementById("library");
const toggleButton = document.getElementById("library-toggle");
const opened = new Set();
// each namespace's Edit YAML form and each decomposition's Edit form, by its row's key, built once, so a draft survives re-renders
const editForms = new Map();
let editFormCount = 0;
let library = null;
let problem = null;
let tab = TABS[0];
// the key of the row whose Delete is asking, and the refusal once there is one
let deleting = null;
// the confirmation showing, which whatever is done next in the panel clears
let confirmed = null;
// each read and change of the Library, numbered as sent: an answer is used only while none sent later has been
let sent = 0;
let applied = 0;

const dot = () => element("span", { class: "lib-dot", "aria-hidden": "true" }, "·");
// YAML and source keep their lines and scroll sideways, so each takes focus, for the arrows to scroll it, as a data table does
const sideways = (label, text) => element("pre", { class: "code lib-pre", role: "region", tabindex: "0", "aria-label": `${label}, scrolls sideways` }, text);
const namespaceNamed = (name) => library.namespaces.find((namespace) => namespace.name === name);
// one a chat may run in: a shared file's may hold one the engine can't read, listed with why (problems) to fix or delete
const usable = (name) => Boolean(namespaceNamed(name)?.usable);
const problemList = (problems) => problems.length > 0 && element("div", { class: "lib-problems lib-warning" }, problems.map((text) => element("p", {}, text)));
const decompositionKey = (namespace, name) => `d:${namespace}:${name}`;
const namespaceKey = (name) => `n:${name}`;
const toolKey = (name) => `t:${name}`;
const settingKey = (name) => `s:${name}`;
const parentOf = (name) => (name.includes(".") ? name.slice(0, name.lastIndexOf(".")) : ROOT);

const body = element("div", { class: "lib-body", id: "library-body", role: "tabpanel" });
const fileName = element("span", { class: "lib-filename" });
// while the Library can't be read again, the panel shows the copy it last read, saying so
const stale = element("p", { class: "lib-note lib-stale", role: "status" });
const tabButtons = TABS.map((name) =>
  element("button", { type: "button", role: "tab", class: "lib-tab", id: `library-tab-${name.toLowerCase()}`, "aria-controls": body.id, onclick: () => selectTab(name) }, name),
);
panel.append(
  element(
    "header",
    { class: "lib-head" },
    element("h2", { class: "lib-title" }, "Library", fileName),
    element("button", { type: "button", class: "lib-close", "aria-label": "Close the Library", onclick: () => setOpen(false) }, "×"),
  ),
  element("div", { class: "lib-tabs", role: "tablist", "aria-label": "Library", onkeydown: arrowTabs }, tabButtons.flatMap((button, i) => (i ? [dot(), button] : [button]))),
  stale,
  body,
);

// A YAML editor its toggle opens, checked as you type; an edit's form keeps what it opened on (was), and stored() reads what is there now.
// A quiet form says its problems only once something is typed in it or it is sent, as a blank one's would only nag; they still stop it saving.
function yamlForm({ id, label, action, text, example = null, quiet = false, problems, warnings = () => [], save, stored = null, gone = null, opening = null, before = [], beside = null, after = [], first = null, pending = null }) {
  const form = { problems, warnings, save, stored, gone, opening, first, pending, quiet, typed: false, was: text, saving: false, timer: null };
  form.toggle = element("button", { type: "button", class: "lib-action", "aria-expanded": "false", "aria-controls": id, "data-key": id, onclick: () => (form.el.hidden ? openForm(form) : closeForm(form)) }, label);
  form.editor = element(
    "textarea",
    {
      id: `${id}-yaml`,
      class: "lib-yaml",
      wrap: "off",
      spellcheck: "false",
      autocapitalize: "off",
      placeholder: example,
      rows: example?.trimEnd().split("\n").length,
      "aria-describedby": [EDITOR_KEYS && `${id}-keys`, `${id}-problems`].filter(Boolean).join(" "),
      oninput: () => edited(form),
      onkeydown: (e) => editorKey(form, e),
    },
    text,
  );
  form.problemList = element("div", { id: `${id}-problems`, class: "lib-problems", "aria-live": "polite" });
  form.newer = element("div", { class: "lib-newer" });
  form.status = element("p", { class: "lib-note", role: "status" });
  form.submit = element("button", { type: "submit", class: "lib-action lib-primary" }, action);
  form.el = element(
    "form",
    { id, class: "children lib-open lib-form", hidden: true, onsubmit: (e) => submit(form, e) },
    before,
    element("div", { class: "lib-form-line" }, element("label", { for: form.editor.id }, "YAML"), beside),
    form.editor,
    EDITOR_KEYS && element("div", { id: `${id}-keys`, class: "lib-keys" }, EDITOR_KEYS),
    after,
    form.problemList,
    form.newer,
    form.status,
    element("div", { class: "lib-actions" }, form.submit, element("button", { type: "button", class: "lib-action", onclick: () => closeForm(form) }, "Cancel")),
  );
  form.confirmation = element("p", { class: "lib-confirm", role: "status" });
  return form;
}

const addSection = (form) => element("section", { class: "lib-add" }, form.toggle, form.el, form.confirmation);

function openForm(form) {
  clearSaid();
  form.opening?.();
  form.el.hidden = false;
  form.typed = false;
  form.toggle.setAttribute("aria-expanded", "true");
  fitEditor(form.editor);
  check(form);
  (form.first ?? form.editor).focus();
}

// Cancel puts back what the form opened on.
function closeForm(form) {
  clearTimeout(form.timer);
  form.el.hidden = true;
  form.toggle.setAttribute("aria-expanded", "false");
  form.el.reset();
  showProblems(form, []);
  form.newer.replaceChildren();
  form.toggle.focus();
}

function edited(form) {
  form.typed = true;
  fitEditor(form.editor);
  clearTimeout(form.timer);
  form.timer = setTimeout(() => check(form), CHECK_DELAY_MS);
}

function fitEditor(editor) {
  if (!editor.isConnected) return;
  editor.style.height = "auto";
  editor.style.height = `${editor.scrollHeight + editor.offsetHeight - editor.clientHeight}px`;
}

// YAML wants Tab to indent, so Esc is the way out of the editor, to Save, or to the toggle while Save is disabled and can't take focus;
// a Delete asking is taken back first, as anywhere in the panel. Saving from the editor spares the trip.
function editorKey(form, e) {
  if (e.isComposing) return;
  if (e.key === "Escape") {
    e.stopPropagation();
    if (deleting && !deleting.refusal) {
      deleting = null;
      render();
    } else {
      // the check due after the last keystroke, run now, may disable Save
      check(form);
      (form.submit.disabled ? form.toggle : form.submit).focus();
    }
  } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    form.el.requestSubmit();
  } else if (e.key === "Tab" && !(e.shiftKey || e.altKey || e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    document.execCommand("insertText", false, INDENT);
  }
}

// Problems stop the form; warnings, read only once there are none, don't.
function check(form) {
  clearTimeout(form.timer);
  const found = form.problems(form.editor.value);
  if (form.quiet && !form.typed) showProblems(form, []);
  else showProblems(form, found, found.length ? [] : form.warnings(form.editor.value));
  form.submit.disabled = form.saving || found.length > 0;
  return found;
}

// unchanged problems are left alone, so the live region doesn't repeat them at every pause in typing
function showProblems(form, found, warned = []) {
  if ([...form.problemList.children].map((line) => line.textContent).join("\n") === [...found, ...warned].join("\n")) return;
  form.problemList.replaceChildren(...found.map((text) => element("p", {}, text)), ...warned.map((text) => element("p", { class: "lib-warning" }, text)));
}

// save() sends the YAML and says what was done, which row to show if any, and where to say it when not under the form (where()).
async function submit(form, e) {
  e.preventDefault();
  form.typed = true;
  if (form.saving || check(form).length) return;
  clearSaid();
  form.saving = form.submit.disabled = true;
  form.status.textContent = form.pending?.(form.editor.value) ?? "";
  let saved;
  try {
    saved = await form.save(form.editor.value, form);
  } catch (error) {
    // cleared first, so a refusal in the words already shown is said again
    showProblems(form, []);
    showProblems(form, error.message.split("\n"));
    if (error.status === CONFLICT) await showNewer(form);
  } finally {
    form.saving = form.submit.disabled = false;
    form.status.textContent = "";
  }
  if (!saved) return giveFocusBack(form);
  closeForm(form);
  render();
  if (saved.row) reveal(saved.row);
  say(saved.where?.() ?? form.confirmation, saved.said);
}

// Disabled while it saved, Save let focus fall to the page; a refusal gives it back.
function giveFocusBack(form) {
  if (document.activeElement === document.body && !form.el.hidden && form.el.isConnected) form.submit.focus();
}

// Changed meanwhile: the Library is read again, the draft kept, and what is stored now shown, which saving again replaces.
async function showNewer(form) {
  await load();
  const now = form.stored();
  // undefined once it is gone; null for one too long to show
  if (now === undefined) {
    editForms.delete(form.key);
    form.gone?.(form);
    return;
  }
  form.was = now;
  form.newer.replaceChildren(element("p", { class: "lib-note" }, "Stored now, which Save replaces with yours:"), sideways("What is stored now", now === null ? "(too long to show)" : now || "(blank: it sets nothing)"));
}

function say(where, text) {
  clearSaid();
  where.textContent = text;
  confirmed = where;
}

function clearSaid() {
  if (confirmed) confirmed.textContent = "";
  confirmed = null;
}

// it is drawn only if its tab is still the one showing
function reveal(key) {
  const row = body.querySelector(`[data-key="${CSS.escape(key)}"]`);
  row?.focus({ preventScroll: true });
  row?.parentElement.scrollIntoView({ block: "nearest" });
}

const fileInput = element("input", { type: "file", accept: ".yaml,.yml", onchange: readFile });
const picker = element("select", { id: "lib-namespace", onchange: () => check(decompositionForm) });
// only once the shape is right, so the YAML reads and its turns are as turnProblems takes them
const turnWarnings = (action, namespace) => (text) => headed(action, namespace(), turnProblems(readYaml(text).value.messages));
const decompositionForm = yamlForm({
  id: "lib-add-form",
  label: "+ Add decomposition",
  action: "Add",
  text: TEMPLATE,
  problems: (text) => decompositionProblems(text, namespaceNamed(picker.value)),
  warnings: turnWarnings("Add", () => picker.value),
  save: addDecomposition,
  // a new one goes where the chat on screen runs, else where new chats start: root would show a draft to every namespace
  opening: () => (picker.value = startingNamespace()),
  beside: element("label", { class: "lib-action lib-file-pick" }, "or choose a .yaml file", fileInput),
  after: element("div", { class: "lib-form-line" }, element("label", { for: picker.id }, "namespace"), picker),
});

// checked at once, not after a pause as the YAML is, so Add is ready as soon as the name is
const nameInput = element("input", {
  id: "lib-ns-name",
  class: "set-input",
  autocomplete: "off",
  spellcheck: "false",
  autocapitalize: "off",
  placeholder: "research",
  "aria-describedby": "lib-ns-form-problems",
  oninput: () => {
    namespaceForm.typed = true;
    check(namespaceForm);
  },
});
const namespaceForm = yamlForm({
  id: "lib-ns-form",
  label: "+ Add namespace",
  action: "Add",
  text: "",
  example: NAMESPACE_EXAMPLE,
  quiet: true,
  problems: (text) => [newNamespaceProblem(nameInput.value.trim()), ...namespaceProblems(text, nameInput.value.trim())].filter(Boolean),
  save: addNamespace,
  before: element("div", { class: "lib-form-line" }, element("label", { for: nameInput.id }, "name"), nameInput),
  first: nameInput,
  pending: building,
});

async function readFile() {
  const [file] = fileInput.files;
  fileInput.value = "";
  if (!file) return;
  if (file.size > MAX_YAML_CHARS * UTF8_MAX_BYTES) {
    showProblems(decompositionForm, [`${file.name} holds more than ${counted.format(MAX_YAML_CHARS)} characters, the most a decomposition's YAML may hold.`]);
    return;
  }
  decompositionForm.editor.value = await file.text();
  fitEditor(decompositionForm.editor);
  check(decompositionForm);
}

async function addDecomposition(text) {
  const namespace = picker.value;
  const { name, yaml, problems } = await request("POST", ADD_URL, { namespace, yaml: text });
  applied = ++sent;
  // looked up again: a read answered meanwhile may have replaced the Library, and may hold the new one already
  const { decompositions } = namespaceNamed(namespace);
  if (!decompositions.some((each) => each.name === name)) decompositions.push({ name, yaml, problems });
  const row = decompositionKey(namespace, name);
  opened.add(row);
  return { said: `Added "${name}" to ${namespace}${problems.length ? ", marked ✗" : ""}`, row };
}

// A namespace's change reaches every namespace under it, so the server answers with the whole Library.
async function changeLibrary(method, url, body) {
  const found = await request(method, url, body);
  applied = ++sent;
  useLibrary(found);
  libraryChanged(library);
}

const namespaceUrl = (name) => `${NAMESPACES_URL}/${encodeURIComponent(name)}`;

async function addNamespace(text) {
  const name = nameInput.value.trim();
  await changeLibrary("POST", NAMESPACES_URL, { name, yaml: text });
  opened.add(namespaceKey(name));
  return { said: `Added namespace "${name}"`, row: namespaceKey(name) };
}

const saveNamespace = (name) => async (text, form) => {
  await changeLibrary("PUT", namespaceUrl(name), { yaml: text, was: form.was });
  return { said: `Saved namespace "${name}"` };
};

// A row's edit form, built once; a closed one holds no draft, so it follows what is stored, and an open one keeps what it was opened on.
function editFormFor(key, stored, build) {
  if (!editForms.has(key)) editForms.set(key, Object.assign(yamlForm({ id: `lib-edit-${++editFormCount}`, text: stored, ...build() }), { key }));
  const form = editForms.get(key);
  // one too long to show (null) opens on a blank draft, which Save puts in its place
  if (form.el.hidden) {
    form.editor.defaultValue = stored ?? "";
    form.was = stored;
  }
  return form;
}

const editForm = (name) =>
  editFormFor(namespaceKey(name), namespaceNamed(name).yaml, () => ({
    label: "Edit YAML",
    action: "Save",
    problems: (text) => namespaceProblems(text, name),
    pending: building,
    save: saveNamespace(name),
    stored: () => namespaceNamed(name)?.yaml,
  }));

const storedDecomposition = (namespace, name) => namespaceNamed(namespace)?.decompositions.find((each) => each.name === name);

// Saved under the name its YAML gives, which may be a new one.
const saveDecomposition = (namespace, name) => async (text, form) => {
  await changeLibrary("PUT", ADD_URL, { namespace, name, was: form.was, yaml: text });
  const now = readYaml(text).value.name;
  const row = decompositionKey(namespace, now);
  if (now !== name) {
    editForms.delete(decompositionKey(namespace, name));
    opened.delete(decompositionKey(namespace, name));
  }
  opened.add(row);
  const marked = storedDecomposition(namespace, now)?.problems.length ? ", marked ✗" : "";
  // renamed, its row has a new form, drawn by then
  return { said: `Saved "${now}" in ${namespace}${marked}`, row, where: () => editForms.get(row)?.confirmation };
};

const decompositionEdit = (namespace, { name, yaml }) =>
  editFormFor(decompositionKey(namespace, name), yaml, () => ({
    label: "Edit",
    action: "Save",
    problems: (text) => decompositionProblems(text, namespaceNamed(namespace), name),
    warnings: turnWarnings("Save", () => namespace),
    save: saveDecomposition(namespace, name),
    stored: () => storedDecomposition(namespace, name)?.yaml,
    gone: (form) => moveToAdd(form.editor.value, namespace, name),
  }));

// Deleted or renamed meanwhile, its Edit draft moves to the add form, on its namespace, so Add saves it as a new one.
function moveToAdd(draft, namespace, name) {
  if (decompositionForm.el.hidden) openForm(decompositionForm);
  if (namespaceNamed(namespace)) picker.value = namespace;
  decompositionForm.editor.value = draft;
  check(decompositionForm);
  const note = `"${name}" was deleted or renamed in ${namespace} meanwhile, so your draft is here: Add saves it as a new one.`;
  decompositionForm.newer.replaceChildren(element("p", { class: "lib-note" }, note));
  decompositionForm.editor.focus();
  // measured once the panel has drawn it
  requestAnimationFrame(() => fitEditor(decompositionForm.editor));
}

// Each read: a closed edit form whose row went is dropped, and an open one hands its draft on; one saving is its save's to settle.
function pruneForms() {
  for (const [key, form] of editForms) {
    if (form.saving || form.stored() !== undefined) continue;
    editForms.delete(key);
    if (!form.el.hidden) form.gone?.(form);
  }
}

// The row whose Delete asks is keyed; a refusal given at once, from what the panel holds, skips the confirm.
function askDelete(key, refusal = null) {
  clearSaid();
  deleting = { key, refusal };
  render();
  if (!refusal) reveal(`keep:${key}`);
}

function keep(key) {
  deleting = null;
  render();
  reveal(`delete:${key}`);
}

// remove() deletes it and says what, and which row to show, in said; stored() says whether the Library, read again after a refusal, still holds it, else the refusal is said there too.
async function confirmDelete(key, { remove, said, stored }, button) {
  button.disabled = true;
  let done;
  try {
    done = await remove();
  } catch (error) {
    deleting = { key, refusal: error.message };
    // a refusal may come from a change made elsewhere, so the next confirm asks about what is there now
    await load();
    if (stored()) {
      render();
      reveal(`delete:${key}`);
      return;
    }
    // deleted meanwhile, so its row and any closed Edit form go
    opened.delete(key);
    editForms.delete(key);
    done = { text: error.message };
  }
  deleting = null;
  render();
  if (done.row) reveal(done.row);
  else {
    tabButtons[TABS.indexOf(tab)].focus();
    said.scrollIntoView({ block: "nearest" });
  }
  say(said, done.text);
}

// A Delete button beside the row's other actions, or the confirm in their place, as the chat list's; a refusal shows under them.
function deletable(key, actions, { question, refusal = () => null, remove, said, stored }) {
  if (deleting?.key === key && !deleting.refusal) {
    return element(
      "div",
      { class: "lib-actions confirm", role: "group", "aria-labelledby": "lib-delete-question" },
      element("span", { id: "lib-delete-question" }, question),
      element("button", { type: "button", class: "yes", onclick: (e) => confirmDelete(key, { remove, said, stored }, e.currentTarget) }, "Delete"),
      element("button", { type: "button", class: "no", "data-key": `keep:${key}`, onclick: () => keep(key) }, "Keep"),
    );
  }
  const ask = element("button", { type: "button", class: "lib-action", "data-key": `delete:${key}`, onclick: () => askDelete(key, refusal()) }, "Delete");
  return [element("div", { class: "lib-actions" }, actions, ask), deleting?.key === key && element("div", { class: "lib-problems", role: "alert" }, deleting.refusal)];
}

function deleteQuestion(namespace) {
  const count = namespace.decompositions.length;
  const what = count ? ` and its ${count} decomposition${count === 1 ? "" : "s"}` : "";
  const fallback = namespace.name === library.default_namespace ? ` New chats will start in ${ROOT} instead.` : "";
  const handed = namespace.handed_to;
  const handing = handed ? ` ${handed} worked example${handed === 1 ? " hands" : "s hand"} questions to it, and without it agents say they can't answer those.` : "";
  return `Delete namespace ${namespace.name}${what}?${fallback}${handing}`;
}

// It sends the count the confirm said, so the server deletes no more than that.
async function deleteNamespace(name) {
  const said = namespaceNamed(name).decompositions.length;
  await changeLibrary("DELETE", `${namespaceUrl(name)}?decompositions=${said}`);
  opened.delete(namespaceKey(name));
  editForms.delete(namespaceKey(name));
  return { text: `Deleted namespace "${name}"`, row: namespaceKey(parentOf(name)) };
}

// The server's refusal for the row, as one with namespaces under it or that another's spawn names, is said at once; anything else it may refuse is said after the confirm.
function namespaceActions(namespace) {
  const { name } = namespace;
  const form = editForm(name);
  const actions =
    name === ROOT
      ? element("div", { class: "lib-actions" }, form.toggle)
      : deletable(namespaceKey(name), form.toggle, {
          question: deleteQuestion(namespace),
          refusal: () => namespace.refusal,
          remove: () => deleteNamespace(name),
          said: namespaceForm.confirmation,
          stored: () => namespaceNamed(name) != null,
        });
  return [actions, form.el, form.confirmation];
}

// the YAML is checked as deep_reasoner's models would check it
const isMapping = (value) => Object.prototype.toString.call(value) === "[object Object]";
const isTextList = (value) => Array.isArray(value) && value.every((each) => typeof each === "string");
const quoted = (keys) => keys.map((key) => `"${key}"`).join(", ");
const FIELD_FITS = {
  tools: isTextList,
  spawn: (value) => value === null || isTextList(value),
  vars: isMapping,
  system_suffix: (value) => value === null || typeof value === "string",
  reasoner: (value) => value === null || isMapping(value),
};

function readYaml(text) {
  try {
    return { value: jsyaml.load(text) };
  } catch (error) {
    const where = error.mark ? `line ${error.mark.line + 1}: ` : "";
    return { problem: `The YAML doesn't parse (${where}${error.reason ?? error.message}). Check the indentation and quoting on that line.` };
  }
}

// characters as Python counts them, which a string's length overcounts past the BMP
const characters = (text) => (text.length > MAX_YAML_CHARS ? [...text].length : text.length);

// editing: the name of the one an Edit form holds, which its YAML may keep
function decompositionProblems(text, namespace, editing = null) {
  if (!text.trim()) return ["Write the decomposition's YAML above, or choose a .yaml file."];
  const length = characters(text);
  if (length > MAX_YAML_CHARS) return [`A decomposition's YAML may be at most ${counted.format(MAX_YAML_CHARS)} characters; this has ${counted.format(length)}.`];
  const { value: parsed, problem } = readYaml(text);
  if (problem) return [problem];
  if (!isMapping(parsed)) return ["The YAML should be a mapping: a name: line, then messages:, as in the template."];
  const extra = Object.keys(parsed).filter((key) => !DECOMPOSITION_KEYS.includes(key));
  return [
    extra.length > 0 && `Remove ${quoted(extra)}: only name and messages go in the YAML; pick the namespace below.`,
    nameProblem(parsed.name, namespace, editing),
    ...messageProblems(parsed.messages),
  ].filter(Boolean);
}

function nameProblem(name, namespace, editing) {
  if (name === undefined) return 'Add a name: line, like "name: count the open tickets".';
  if (typeof name !== "string" || !name.trim()) return 'The name should be text, like "name: count the open tickets".';
  if (name !== name.trim()) return `The name "${name}" starts or ends with a space; take it out, so it reads as it looks.`;
  const hidden = name.match(INVISIBLE);
  if (hidden) return `The name holds a character you can't see (${codePoint(hidden[0])}); take it out, so it reads as it looks.`;
  if (name !== editing && namespace?.decompositions.some((each) => each.name === name)) return `${namespace.name} already has "${name}"; rename it, pick another namespace, or Edit that one.`;
  return null;
}

function messageProblems(messages) {
  if (messages === undefined) return ["Add messages: a list of turns, each with a role and content, as in the template."];
  if (!Array.isArray(messages) || !messages.length) return ['messages should be a list of at least one turn, each starting "- role:".'];
  return messages.flatMap((message, i) => messageShapeProblems(message, `Message ${i + 1}`));
}

function messageShapeProblems(message, label) {
  if (!isMapping(message)) return [`${label} should be a mapping with role and content.`];
  const extra = Object.keys(message).filter((key) => !MESSAGE_KEYS.includes(key));
  const { role, content } = message;
  return [
    extra.length > 0 && `${label}: remove ${quoted(extra)}; a message holds only role and content.`,
    !ROLES.includes(role) && `${label}: role should be user, assistant or system${role === undefined ? "" : `, not ${JSON.stringify(role)}`}.`,
    content === undefined && `${label} needs content: the turn's text.`,
    content !== undefined && typeof content !== "string" && `${label}: content should be text; quote it, or write "content: |" with the text indented below.`,
  ].filter(Boolean);
}

function newNamespaceProblem(name) {
  if (!name) return 'Name the namespace, like "research"; "research.papers" goes under research.';
  if (!NAMESPACE_NAME.test(name)) return 'A namespace name is words of letters, digits, _ and -, each starting with a letter, joined by dots, like "research.papers".';
  if (name.toLowerCase().split(".").includes("root")) return `"${name}" reads as root, which every namespace goes under already: name it something else, like "research".`;
  if (namespaceNamed(name)) return `There is already a namespace "${name}"; open it and use Edit YAML.`;
  const parent = parentOf(name);
  if (!namespaceNamed(parent)) return `Add namespace "${parent}" first: "${name}" goes under it.`;
  return null;
}

// What a namespace's save says while it runs when its YAML grants a custom tool no namespace grants yet, which the server builds first.
function building(text) {
  const { value } = readYaml(text);
  const asked = isMapping(value) && isTextList(value.tools) ? value.tools : [];
  const afresh = (name) => library.tools.some((tool) => tool.name === name && tool.source != null && !tool.granted_in.length);
  return asked.some(afresh) ? BUILDING : SAVING;
}

// A namespace that sets nothing inherits everything, so blank YAML is fine.
function namespaceProblems(text, name) {
  const { value, problem } = readYaml(text);
  if (problem) return [problem];
  const fields = value ?? {};
  if (!isMapping(fields)) return ['The YAML should be a mapping of fields, like "tools: [llm]".'];
  const keys = Object.keys(fields);
  const unknown = keys.filter((key) => !NAMESPACE_KEYS.includes(key) && !Object.hasOwn(NAMED_ELSEWHERE, key));
  const misshapen = keys.filter((key) => Object.hasOwn(FIELD_FITS, key) && !FIELD_FITS[key](fields[key]));
  const tools = new Set(library.tools.map((tool) => tool.name));
  const namespaces = new Set([name, ...library.namespaces.map((namespace) => namespace.name)]);
  const fitting = (key) => (misshapen.includes(key) ? [] : (fields[key] ?? []));
  return [
    ...keys.filter((key) => Object.hasOwn(NAMED_ELSEWHERE, key)).map((key) => NAMED_ELSEWHERE[key]),
    unknown.length > 0 && `Remove ${quoted(unknown)}: a namespace's YAML holds only ${NAMESPACE_KEYS.slice(0, -1).join(", ")} and ${NAMESPACE_KEYS.at(-1)}.`,
    ...misshapen.map((key) => SHAPES[key]),
    ...fitting("tools").filter((tool) => !tools.has(tool)).map((tool) => `The Library has no tool "${tool}"; the Tools tab lists the ones it has.`),
    ...fitting("spawn").filter((spawned) => !namespaces.has(spawned)).map((spawned) => missingSpawn(spawned, name)),
  ].filter(Boolean);
}

// as the server words it; a namespace being added can't spawn into one to go under it, which needs it first
function missingSpawn(spawned, name) {
  if (!namespaceNamed(name) && spawned.startsWith(`${name}.`)) {
    return `spawn names "${spawned}", which goes under ${name} and isn't in the Library yet: add ${name} without it first, then ${spawned}, then put ${spawned} in the spawn of ${name}.`;
  }
  return `spawn names "${spawned}", which is not a namespace in the Library: add ${spawned} first, or take it out of spawn.`;
}

function disclosure(key, head, content) {
  const open = opened.has(key);
  return element(
    "div",
    { class: "lib-entry" },
    element("button", { type: "button", class: "row", "aria-expanded": String(open), "data-key": key, onclick: () => flip(key) }, caret(open), head),
    open && element("div", { class: "children lib-open" }, content()),
  );
}

// one span, so a long row cuts the detail before the name
function rowHead(name, detail, meta) {
  return [element("span", { class: "lib-name" }, name, " ", element("span", { class: "lib-detail" }, detail)), element("span", { class: "meta" }, meta)];
}

function fields(rows) {
  return element("dl", { class: "lib-fields" }, rows.map(([label, value]) => [element("dt", {}, label), element("dd", {}, value)]));
}

function sourced(items) {
  if (!items.length) return "—";
  return items.map(({ value, source, here }) => {
    const label = element("span", { class: `lib-source ${here ? "here" : ""}` }, source);
    if (value.includes("\n")) return element("div", { class: "lib-item lib-block" }, element("pre", { class: "code lib-doc" }, value), label);
    return element("div", { class: "lib-item" }, value, " ", label);
  });
}

const group = (title, rows) => element("section", { class: "lib-group" }, element("h3", { class: "lib-label" }, title), element("div", { class: "lib-rows" }, rows));

// ✓ or ✗: whether its turns make a worked example, as the server checked them
function checkMark(problems) {
  const count = problems.length;
  const label = count ? `${count} problem${count === 1 ? "" : "s"}` : "checks pass";
  return element("span", { class: `lib-check ${count ? "failed" : "passed"}`, role: "img", "aria-label": label, title: label }, count ? "✗" : "✓");
}

function decompositionRow(namespace, decomposition) {
  const { name, yaml, problems } = decomposition;
  const key = decompositionKey(namespace, name);
  const head = [element("span", { class: "lib-name" }, name), checkMark(problems)];
  // one too long to show is shown no YAML, and deleted as that (null)
  const remove = async () => {
    await changeLibrary("DELETE", ADD_URL, { namespace, name, was: yaml });
    opened.delete(key);
    editForms.delete(key);
    return { text: `Deleted "${name}" from ${namespace}`, row: decompositionForm.el.id };
  };
  const content = () => {
    const form = decompositionEdit(namespace, decomposition);
    const question = `Delete "${name}" from ${namespace}? Agents in ${reach(namespace)} stop seeing it from the next message.`;
    return [
      problemList(problems),
      yaml != null && sideways(`${name}'s YAML`, yaml),
      deletable(key, yaml == null ? [] : form.toggle, { question, remove, said: decompositionForm.confirmation, stored: () => storedDecomposition(namespace, name) != null }),
      form.el,
      form.confirmation,
    ];
  };
  return disclosure(key, head, content);
}

function renderDecompositions() {
  const groups = library.namespaces
    .filter((namespace) => namespace.decompositions.length)
    .map((namespace) => {
      const title = namespace.name === ROOT ? `${ROOT} · every namespace inherits these` : namespace.name;
      return group(title, namespace.decompositions.map((each) => decompositionRow(namespace.name, each)));
    });
  return [addSection(decompositionForm), ...groups];
}

const namespaceFields = (list) => fields(list.map(({ label, items }) => [label, sourced(items)]));

function chain(name) {
  const parts = name === ROOT ? [] : name.split(".");
  return [ROOT, ...parts.map((_, i) => parts.slice(0, i + 1).join("."))];
}

// a namespace's decompositions and its ancestors', as deep_reasoner resolves them: one it can't read isn't there for it
function decompositionsField(name) {
  const levels = new Map();
  for (const level of chain(name)) {
    for (const each of namespaceNamed(level)?.decompositions ?? []) {
      if (each.read) levels.set(each.name, level);
    }
  }
  const items = [...levels].map(([value, level]) => ({ value, source: level === name ? "attached here" : `inherited from ${level}`, here: level === name }));
  return { label: "Decompositions", items };
}

function renderNamespaces() {
  const tree = library.namespaces.map((namespace) => {
    const star = namespace.name === library.default_namespace && element("span", { class: "lib-star" }, "★ new chats start here");
    const head = [element("span", { class: "lib-name" }, namespace.name), star, namespace.problems.length > 0 && checkMark(namespace.problems)];
    const content = () => [problemList(namespace.problems), namespaceFields([...namespace.fields, decompositionsField(namespace.name)]), namespaceActions(namespace)];
    return element("div", { style: `--depth: ${namespace.depth}` }, disclosure(namespaceKey(namespace.name), head, content));
  });
  const run = disclosure("run-settings", element("span", { class: "lib-name" }, "Run settings"), runSettings);
  return [addSection(namespaceForm), element("div", { class: "lib-rows lib-tree" }, run, tree)];
}

// The engine reads only the settings it knows, so one it doesn't, as an import may bring, can be deleted; the rest stay as they are.
function runSettings() {
  const rows = library.run_settings.map(({ label, items, removable }) => {
    const remove = async () => {
      await changeLibrary("DELETE", `${RUN_SETTINGS_URL}/${encodeURIComponent(label)}`);
      return { text: `Deleted run setting "${label}"`, row: "run-settings" };
    };
    const stored = () => library.run_settings.some((setting) => setting.label === label);
    return [label, [sourced(items), removable && deletable(settingKey(label), [], { question: `Delete run setting ${label}?`, remove, said: namespaceForm.confirmation, stored })]];
  });
  return fields(rows);
}

const toolsSaid = element("p", { class: "lib-confirm", role: "status" });

// A tool any namespace grants is built by every run, so the server deletes it only once none does, and its refusal for the row is said at once.
function renderTool(tool) {
  const remove = async () => {
    await changeLibrary("DELETE", `${TOOLS_URL}/${encodeURIComponent(tool.name)}`);
    opened.delete(toolKey(tool.name));
    return { text: `Deleted tool "${tool.name}"` };
  };
  const stored = () => library.tools.some((each) => each.name === tool.name && each.stored);
  const granted = ["Granted in", tool.granted_in.join(", ") || "no namespace"];
  const made = [["Factory", tool.factory], ["File", tool.file], ["Defined in", tool.defined_in ?? "the deep_reasoner package"]];
  return [
    // one the engine can't read, as a shared file may hold, says why instead
    problemList(tool.problems),
    fields(tool.problems.length ? [granted] : [...made, granted]),
    tool.config && sideways(`${tool.name}'s settings`, tool.config),
    // What the agent is told it does, for a custom tool as far as its source says; it wraps, as one may run long.
    tool.description && element("pre", { class: "code lib-doc" }, tool.description),
    tool.source != null && [element("p", { class: "lib-note" }, `Its Python, ${tool.file}:`), sideways(`${tool.name}'s Python`, tool.source)],
    tool.stored && deletable(toolKey(tool.name), [], { question: `Delete tool ${tool.name}?`, refusal: () => tool.refusal, remove, said: toolsSaid, stored }),
  ];
}

function renderTools() {
  const rows = library.tools.map((tool) => {
    const granted = tool.granted_in.length ? `granted in ${tool.granted_in.join(", ")}` : "not granted";
    return disclosure(toolKey(tool.name), rowHead(tool.name, granted, tool.problems.length ? checkMark(tool.problems) : tool.badge), () => renderTool(tool));
  });
  return [element("div", { class: "lib-rows" }, toolsSaid, rows)];
}

function dataTable({ name, columns, rows }) {
  const cell = (tag, text) => element(tag, tag === "th" ? { scope: "col" } : {}, element("div", {}, text));
  return element(
    "div",
    { class: "lib-table", role: "region", tabindex: "0", "aria-label": `${name}, scrolls sideways` },
    element("table", {}, element("thead", {}, element("tr", {}, columns.map((column) => cell("th", column)))), element("tbody", {}, rows.map((row) => element("tr", {}, row.map((text) => cell("td", text)))))),
  );
}

function renderData() {
  if (!library.variables.length) return [element("p", { class: "lib-note" }, "No namespace's vars hold a table yet.")];
  const rows = library.variables.map((entry) => disclosure(`v:${entry.where}:${entry.name}`, rowHead(entry.name, entry.where, entry.summary), () => dataTable(entry)));
  return [group("Preloaded variables", rows)];
}

const RENDERERS = { Decompositions: renderDecompositions, Namespaces: renderNamespaces, Tools: renderTools, Data: renderData };

// A field being typed in when the agent's change redraws the panel is drawn again as itself, so it keeps focus and caret.
function render() {
  // one Tab stop for the tabs, the selected one; arrows move between them
  for (const button of tabButtons) {
    button.setAttribute("aria-selected", String(button.textContent === tab));
    button.tabIndex = button.textContent === tab ? 0 : -1;
  }
  body.setAttribute("aria-labelledby", `library-tab-${tab.toLowerCase()}`);
  const active = document.activeElement;
  const { selectionStart, selectionEnd } = active ?? {};
  const focused = active?.dataset?.key;
  if (problem || !library) body.replaceChildren(element("p", { class: "lib-note" }, problem ?? "Loading the Library…"));
  else body.replaceChildren(...RENDERERS[tab]());
  if (focused) body.querySelector(`[data-key="${CSS.escape(focused)}"]`)?.focus();
  else if (active !== document.activeElement && body.contains(active)) {
    active.focus({ preventScroll: true });
    if (selectionStart != null) active.setSelectionRange(selectionStart, selectionEnd);
  }
}

function flip(key) {
  deleting = null;
  clearSaid();
  if (opened.has(key)) opened.delete(key);
  else opened.add(key);
  render();
}

function selectTab(name) {
  deleting = null;
  clearSaid();
  tab = name;
  render();
  body.scrollTop = 0;
}

function arrowTabs(e) {
  const at = TABS.indexOf(tab);
  const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: TABS.length - 1 }[e.key];
  if (next === undefined) return;
  e.preventDefault();
  selectTab(TABS[(next + TABS.length) % TABS.length]);
  tabButtons[TABS.indexOf(tab)].focus();
}

// Opened, it reads the Library again, as an import or another tab may have changed it.
function setOpen(open) {
  deleting = null;
  clearSaid();
  showPanel(panel, toggleButton, open);
  if (!open) return;
  closeOtherPanels(panel);
  render();
  tabButtons[TABS.indexOf(tab)].focus();
  load();
}

// where the chat on screen runs, else where new chats start, else root
function startingNamespace() {
  return [chatNamespace(), library.default_namespace].find((name) => name && usable(name)) ?? ROOT;
}

// The decomposition form's picker keeps its pick while that namespace is there.
function useLibrary(found) {
  library = found;
  stale.textContent = "";
  fileName.textContent = ` · ${library.file}`;
  const picked = usable(picker.value) ? picker.value : startingNamespace();
  picker.replaceChildren(...library.namespaces.filter((namespace) => namespace.usable).map(({ name }) => element("option", { value: name }, name)));
  picker.value = picked;
  pruneForms();
}

// At start, and after each change the agent makes in a run; drafts live in their forms, so they survive.
async function load() {
  const asked = ++sent;
  let found;
  try {
    found = await request("GET", LIBRARY_URL);
  } catch (error) {
    if (!library) problem = `Could not load the Library (${error.message}).`;
    else if (asked > applied) stale.textContent = `Showing the Library as last read: ${error.message}`;
  }
  if (found && asked > applied) {
    applied = asked;
    stale.textContent = "";
    // as it was: nothing to redraw, so focusing the window costs nothing
    if (!problem && JSON.stringify(found) === JSON.stringify(library)) return;
    problem = null;
    useLibrary(found);
    libraryChanged(library);
  }
  if (!panel.hidden) render();
}

toggleButton.addEventListener("click", () => setOpen(panel.hidden));
panel.addEventListener(PANEL_CLOSE, () => setOpen(false));
panel.addEventListener(LIBRARY_CHANGED, load);
// Esc inside the panel takes back a Delete first, else closes it, and stops there, so it never also stops a run.
panel.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  e.stopPropagation();
  if (deleting && !deleting.refusal) keep(deleting.key);
  else setOpen(false);
});
// Back in this window with the panel open, app.js asks for the Library again (LIBRARY_CHANGED).
// Elsewhere, Esc closes the panel unless the chat used it, to stop its run or keep a chat; app.js's handler runs first.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !e.defaultPrevented && !panel.hidden) setOpen(false);
});
load();
