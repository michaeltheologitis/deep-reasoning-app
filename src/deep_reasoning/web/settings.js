// The settings panel: the providers, with each key's status, the custom servers, and the default folder. Keys go to the server, which keeps them in ~/.deep-reasoning/.env and never sends one back.
import { PANEL_CLOSE, PANEL_OPEN, SETTINGS_CHANGED, caret, closeOtherPanels, element, folderPicker, providersChanged, request, settingsChanged, showPanel } from "./app.js";

const ENV_FILE = "~/.deep-reasoning/.env";
// Ollama's, as an example: not the app's own port
const NEW_NAME = "ollama";
const NEW_URL = "http://localhost:11434/v1";
// not a name, so no server's name can be taken for it
const ADDING = Symbol("adding");
const NO_FOLDER = "each chat's own scratch folder";
const FOLDER_MISSING = "missing: new chats use a scratch folder";
const STATUS = { set: "key ✓", shell: "key from your shell" };
const NOT_FOUND = 404;
// what each status means for a key set in the shell too
const STATUS_TITLE = { set: `saved in ${ENV_FILE}, which wins over one set in your shell`, shell: "set in your shell; a key saved here would win over it" };

const panel = document.getElementById("settings");
const toggleButton = document.getElementById("settings-toggle");
// what the server last said: each provider, and where its key is, never the key
let providers = [];
let loadProblem = null;
// the folder new chats start in, as the server keeps it in the app file; null: each chat's own scratch folder
let defaultFolder = null;
// the name of the provider whose form shows, or ADDING for a new server's
let opened = null;
let cancelRemove = null;

const providerRows = element("div", { class: "lib-rows" });
const addToggle = element(
  "button",
  { type: "button", class: "lib-action", "aria-expanded": "false", "aria-controls": "server-new-form", "data-key": "add", onclick: () => flip(ADDING, opened === ADDING ? "add" : "name") },
  "+ Custom server",
);
const addArea = element("div", { class: "lib-add" }, addToggle);
const removedMeanwhile = element("p", { class: "lib-note", role: "status" });
const folderName = element("span", { id: "default-folder-name" }, NO_FOLDER);
const folderMissing = element("span", { id: "default-folder-missing", class: "set-missing" });
const folderHost = element("div");
const folderButton = pickerButton("default-folder", "dialog", folderName, pickDefaultFolder);
folderButton.setAttribute("aria-describedby", "default-folder-missing default-folder-note");

panel.append(
  element(
    "header",
    { class: "lib-head" },
    element("h2", { class: "lib-title" }, "Settings", element("span", { class: "lib-filename" }, ` · ${ENV_FILE}`)),
    element("button", { type: "button", class: "lib-close", "aria-label": "Close the settings", onclick: () => setOpen(false) }, "×"),
  ),
  element(
    "div",
    { class: "lib-body" },
    element("section", { class: "lib-group" }, element("h3", { class: "lib-label" }, "Providers"), providerRows, removedMeanwhile, addArea),
    element("p", { class: "lib-note" }, `Keys are kept in ${ENV_FILE}, readable only by you. Keys already set in your shell keep working; one saved here wins.`),
    pickerSection("default-folder", "Default folder", [folderButton, " ", folderMissing], folderHost, "New chats work in it; each chat can pick its own under the message box."),
  ),
);

// A value with ▾ that opens its picker below it; focus stays in the open picker on mousedown, so a click toggles it.
function pickerButton(id, popup, value, onclick) {
  const props = { type: "button", class: "choice set-model", "aria-haspopup": popup, "aria-expanded": "false", "aria-labelledby": `${id}-label ${value.id}`, "aria-describedby": `${id}-note` };
  return element("button", { ...props, onmousedown: (e) => e.preventDefault(), onclick }, value, element("span", { "aria-hidden": "true" }, " ▾"));
}

function pickerSection(id, title, button, host, note) {
  return element(
    "section",
    { class: "lib-group" },
    element("h3", { class: "lib-label", id: `${id}-label` }, title),
    element("div", {}, button),
    host,
    element("p", { class: "lib-note", id: `${id}-note` }, note),
  );
}

const vendors = () => providers.filter((provider) => !provider.custom);
const servers = () => providers.filter((provider) => provider.custom);

function render(focusKey) {
  const focused = focusKey ?? document.activeElement?.dataset?.key;
  cancelRemove = null;
  const rows = [...vendors().map(vendorRow), ...servers().map(serverRow)];
  providerRows.replaceChildren(...(loadProblem ? [element("p", { class: "lib-problems" }, loadProblem)] : rows));
  addToggle.setAttribute("aria-expanded", String(opened === ADDING));
  addArea.replaceChildren(addToggle, opened === ADDING ? serverForm(null) : "");
  if (focused) panel.querySelector(`[data-key="${focused}"]`)?.focus();
}

function flip(target, focusKey) {
  removedMeanwhile.textContent = "";
  opened = opened === target ? null : target;
  render(focusKey);
}

const keyStatus = (provider) => STATUS[provider.key] ?? (provider.custom ? "no key" : "add key");

function providerRow(provider, key, detail, form) {
  const open = opened === provider.name;
  return element(
    "div",
    { class: "lib-entry" },
    element(
      "button",
      { type: "button", class: "row", "aria-expanded": String(open), "data-key": key, onclick: () => flip(provider.name, key) },
      caret(open),
      element("span", { class: "lib-name" }, provider.name, detail && [" ", element("span", { class: "lib-detail" }, detail)]),
      element("span", { class: "meta", title: STATUS_TITLE[provider.key] }, keyStatus(provider)),
    ),
    open && form(provider),
  );
}

// A known vendor's address is the app's to know: its row takes only a key.
const vendorRow = (vendor, index) => providerRow(vendor, `v:${index}`, null, vendorForm);
const serverRow = (server, index) => providerRow(server, `s:${index}`, server.base_url, serverForm);
const rowKey = (name) => {
  const index = vendors().findIndex((vendor) => vendor.name === name);
  if (index >= 0) return `v:${index}`;
  const server = servers().findIndex((each) => each.name === name);
  return server >= 0 ? `s:${server}` : "add";
};

// Both forms use method="dialog", which never navigates, so a typed key can't end up in a URL even if this script failed.
function vendorForm(vendor) {
  const key = keyInput("vendor-key", "vendor-key-where");
  const problem = element("div", { class: "lib-problems", "aria-live": "polite" });
  const remove = element("button", { type: "button", class: "lib-action", disabled: vendor.key !== "set", onclick: () => confirmRemove(`Remove the ${vendor.name} key?`, () => forgetKey(vendor, problem), remove.parentElement) }, "Remove key");
  return element(
    "form",
    { class: "children lib-open lib-form", method: "dialog", novalidate: "", onsubmit: (e) => saveKey(e, vendor, key, problem) },
    element("div", { class: "set-fields" }, element("label", { for: key.id }, "API key"), key, element("span", { id: "vendor-key-where", class: "set-note" }, `${vendor.key_env} in `, element("span", { class: "set-path" }, ENV_FILE), vendor.key === "set" ? ", which wins over your shell's" : "")),
    problem,
    element("div", { class: "lib-actions" }, element("button", { type: "submit", class: "lib-action lib-primary" }, "Save"), remove),
  );
}

const keyInput = (id, note) => element("input", { id, class: "set-input", type: "password", autocomplete: "off", spellcheck: "false", "aria-describedby": note, "data-key": "key" });

// A server's name is fixed once added: chats name it as their model's provider, and its key is kept under it.
function serverForm(server) {
  const id = server ? "server" : "server-new";
  const input = (field, props) => element("input", { id: `${id}-${field}`, class: "set-input", autocomplete: "off", spellcheck: "false", ...props });
  const label = (field, text) => element("label", { for: `${id}-${field}` }, text);
  const name = server ? null : input("name", { placeholder: NEW_NAME, "data-key": "name" });
  const url = input("url", { type: "url", value: server?.base_url ?? "", placeholder: NEW_URL });
  const key = keyInput(`${id}-key`, `${id}-key-note`);
  const problem = element("div", { class: "lib-problems", "aria-live": "polite" });
  const keyNote = server?.key === "set" ? `set, as ${server.key_env}; a new one replaces it` : "optional: most local servers need none";
  return element(
    "form",
    { id: `${id}-form`, class: "children lib-open lib-form", method: "dialog", novalidate: "", onsubmit: (e) => saveServer(e, server, { name, url, key, problem }) },
    element(
      "div",
      { class: "set-fields" },
      name ? [label("name", "name"), name] : [],
      [label("url", "base URL"), url, label("key", "API key"), key],
      element("span", { id: `${id}-key-note`, class: "set-note" }, keyNote),
    ),
    problem,
    element(
      "div",
      { class: "lib-actions" },
      element("button", { type: "submit", class: "lib-action lib-primary" }, server ? "Save" : "Add"),
      server
        ? element("button", { type: "button", class: "lib-action", onclick: (e) => confirmRemove(`Remove ${server.name}?`, () => removeServer(server, problem), e.currentTarget.parentElement) }, "Remove")
        : element("button", { type: "button", class: "lib-action", onclick: () => flip(ADDING, "add") }, "Cancel"),
    ),
  );
}

// Each change goes to the server, which answers with every provider's status; the panel and the model menu follow that. A
// server another tab removed meanwhile is gone already (removing it again is no refusal): saving it reads the rows again.
async function change(method, url, body, problem, focusName) {
  try {
    providers = await request(method, url, body);
    removedMeanwhile.textContent = "";
  } catch (error) {
    if (error.status !== NOT_FOUND) {
      problem.textContent = error.message;
      return false;
    }
    await loadProviders();
    removedMeanwhile.textContent = `${focusName} was removed in another tab, so nothing was saved.`;
  }
  opened = null;
  render(rowKey(focusName));
  settingsChanged();
  return true;
}

// The field is emptied as the key is sent, so a key stays in the page no longer than that; a refused one comes back to be fixed.
async function sendKey(input, send) {
  const key = input.value.trim() || null;
  input.value = "";
  if (!(await send(key))) input.value = key ?? "";
}

function saveKey(e, vendor, input, problem) {
  e.preventDefault();
  if (!input.value.trim()) {
    problem.textContent = `Paste the ${vendor.name} key first.`;
    return;
  }
  problem.textContent = "";
  sendKey(input, (key) => change("PUT", providerUrl(vendor, "key"), { key }, problem, vendor.name));
}

const forgetKey = (vendor, problem) => change("DELETE", providerUrl(vendor, "key"), undefined, problem, vendor.name);
const providerUrl = (provider, part) => `api/providers/${encodeURIComponent(provider.name)}/${part}`;
const serverUrl = (server) => `api/servers/${encodeURIComponent(server.name)}`;

function saveServer(e, server, { name, url, key, problem }) {
  e.preventDefault();
  const base_url = url.value.trim();
  if (server) return sendKey(key, (typed) => change("PUT", serverUrl(server), { base_url, key: typed }, problem, server.name));
  const named = name.value.trim();
  sendKey(key, (typed) => change("POST", "api/servers", { name: named, base_url, key: typed }, problem, named));
}

const removeServer = (server, problem) => change("DELETE", serverUrl(server), undefined, problem, null);

// The confirm takes the buttons' place, as the chat list's does, leaving what was typed alone.
function confirmRemove(question, action, buttons) {
  const keep = element("button", { type: "button", class: "no" }, "Keep");
  const confirm = element("div", { class: "lib-actions confirm" }, question, element("button", { type: "button", class: "yes", onclick: action }, "Remove"), keep);
  cancelRemove = () => {
    cancelRemove = null;
    confirm.replaceWith(buttons);
    buttons.lastElementChild.focus();
  };
  keep.addEventListener("click", cancelRemove);
  buttons.replaceWith(confirm);
  keep.focus();
}

function pickDefaultFolder() {
  folderPicker(folderButton, folderHost, {
    label: "Default folder for new chats",
    current: defaultFolder,
    empty: "each chat gets its own scratch folder",
    note: "New chats start in it. A folder you choose is never deleted.",
    pick: async (folder) => showDefaultFolder(await request("PUT", "api/default-folder", { folder })),
  });
}

// One that isn't on this computer, as in a Library from another, is marked: new chats get their own scratch folder.
function showDefaultFolder({ folder, missing }) {
  defaultFolder = folder;
  folderName.textContent = folder ?? NO_FOLDER;
  folderMissing.textContent = missing ? FOLDER_MISSING : "";
}

async function loadDefaultFolder() {
  try {
    showDefaultFolder(await request("GET", "api/default-folder"));
  } catch {
    // the providers' row says the settings can't be read
  }
}

async function loadProviders() {
  try {
    providers = await request("GET", "api/providers");
    loadProblem = null;
  } catch (error) {
    loadProblem = `Could not read the settings: ${error.message}`;
  }
}

async function setOpen(open) {
  showPanel(panel, toggleButton, open);
  if (!open) return;
  closeOtherPanels(panel);
  render("v:0");
  await Promise.all([loadProviders(), loadDefaultFolder()]);
  if (!panel.hidden) render(panel.contains(document.activeElement) ? undefined : "v:0");
}

// read ahead, so a change another tab makes shows as one
const firstRead = loadProviders();

// Back in this window: what another tab changed shows, and the model menu reads its lists again once the providers changed;
// the chats are read anyway. A form open in the panel keeps what is typed in it; the rows follow when it closes.
async function readAgain() {
  await firstRead;
  const before = JSON.stringify(providers);
  await Promise.all([loadProviders(), loadDefaultFolder()]);
  if (JSON.stringify(providers) === before) return;
  if (!panel.hidden && opened === null) render();
  providersChanged();
}

toggleButton.addEventListener("click", () => setOpen(panel.hidden));
panel.addEventListener(PANEL_CLOSE, () => setOpen(false));
panel.addEventListener(PANEL_OPEN, () => setOpen(true));
panel.addEventListener(SETTINGS_CHANGED, readAgain);
// Esc inside the panel takes back a Remove first, else closes it, and stops there, so it never also stops a run.
panel.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  e.stopPropagation();
  if (cancelRemove) cancelRemove();
  else setOpen(false);
});
// Elsewhere, Esc closes the panel unless the chat used it, to stop its run or keep a chat; app.js's handler runs first.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !e.defaultPrevented && !panel.hidden) setOpen(false);
});
