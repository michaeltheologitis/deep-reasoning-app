// A page's tab: what an agent chose to show, read from the app's API and drawn with the chat's own renderers. All of it is untrusted text.
import { element, opened, renderAgent, renderAnswer, renderStep, useStoredTheme } from "./tree.js";

const byId = (id) => document.getElementById(id);

// A step or sub-agent copied from an earlier message, opened a level, as it was picked to be shown.
function renderRecord(block) {
  const from = [`from message ${block.turn}`, block.task].filter(Boolean).join(" · ");
  let shown;
  if (block.type === "step") {
    opened.add(block.step);
    shown = element("div", { class: "steps" }, renderStep(block.step, false));
  } else {
    for (const agent of block.agents) opened.add(agent);
    shown = element("div", { class: "steps" }, block.agents.map(renderAgent));
  }
  return element("section", { class: "ref" }, element("div", { class: "ref-from" }, from), block.note != null && renderAnswer(block.note), shown);
}

function renderTable({ columns, rows }) {
  const head = columns.length > 0 && element("thead", {}, element("tr", {}, columns.map((name) => element("th", {}, name))));
  const body = element("tbody", {}, rows.map((row) => element("tr", {}, row.map((value) => element("td", {}, value)))));
  return element("div", { class: "table-scroll" }, element("table", { class: "page-table" }, head, body));
}

function renderBlock(block) {
  if (block.type === "text") return renderAnswer(block.text);
  if (block.type === "table") return renderTable(block);
  return renderRecord(block);
}

async function start() {
  useStoredTheme();
  // the chat's theme toggle, in another tab
  addEventListener("storage", useStoredTheme);
  const id = location.pathname.split("/").at(-1);
  try {
    const response = await fetch(`/api/pages/${id}`);
    const page = await response.json();
    if (!response.ok) throw new Error(page.detail ?? `the app answered ${response.status}`);
    // drawn before any of it shows, so a block that fails to draw leaves the error, not a blank tab
    const blocks = page.blocks.map(renderBlock);
    document.title = `${page.title} · Deep Reasoning`;
    byId("title").textContent = page.title;
    byId("blocks").replaceChildren(...blocks);
  } catch (error) {
    byId("blocks").replaceChildren(element("p", { class: "word err" }, `This page can't be shown: ${error.message}`));
  }
}

start();
