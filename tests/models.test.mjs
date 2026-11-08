// What the composer's model and namespace menus list for what is typed.
import assert from "node:assert/strict";
import test from "node:test";
import { modelGroups, namespaceGroups } from "../src/deep_reasoning/web/models.js";

const LISTS = [
  { provider: "OpenAI", featured: ["gpt-6-astra", "gpt-6-luna"], models: ["gpt-4", "gpt-6-astra", "gpt-6-luna"], problem: null },
  { provider: "fake", featured: ["fake-large", "fake-small"], models: ["fake-large", "fake-small"], problem: null },
  { provider: "broken", featured: [], models: [], problem: "No list: it didn't answer." },
];
// Each group as "provider: label | label", in order.
const shown = (groups) => groups.map((group) => `${group.name}: ${group.options.map((option) => option.label).join(" | ")}`);

test("untyped, each provider lists its featured models, and the chat's own where it isn't one", () => {
  assert.deepEqual(shown(modelGroups(LISTS, "", { provider: "OpenAI", name: "gpt-4" })), [
    "OpenAI: gpt-6-astra | gpt-6-luna | gpt-4",
    "fake: fake-large | fake-small",
    "broken: use model name…",
  ]);
});

test("a name a later provider lists exactly comes first, so Enter picks it over another's use", () => {
  const groups = modelGroups(LISTS, "fake-large", null);
  assert.deepEqual(shown(groups), ["fake: fake-large", "OpenAI: use “fake-large”", "broken: use “fake-large”"]);
  assert.deepEqual(groups[0].options[0].value, { provider: "fake", name: "fake-large" });
});

test("within its provider, the exact name comes before the others that hold it", () => {
  const lists = [{ provider: "OpenAI", featured: [], models: ["gpt-6-luna-mini", "gpt-6-luna"], problem: null }];
  assert.deepEqual(shown(modelGroups(lists, "gpt-6-luna", null)), ["OpenAI: gpt-6-luna | gpt-6-luna-mini"]);
});

test("typed, a provider lists every model it knows that holds the text, in any case, then the text to use as it is", () => {
  assert.deepEqual(shown(modelGroups(LISTS, " GPT-6 ", null)).slice(0, 2), ["OpenAI: gpt-6-astra | gpt-6-luna | use “GPT-6”", "fake: use “GPT-6”"]);
});

test("with no name exact anywhere, the providers keep their order", () => {
  assert.deepEqual(modelGroups(LISTS, "luna", null).map((group) => group.name), ["OpenAI", "fake", "broken"]);
});

const NAMESPACES = ["root", "research", "research.papers", "historian"].map((name) => ({ value: name, label: name }));
const names = (groups) => groups.flatMap((group) => group.options.map((option) => option.value));

test("untyped, the namespace menu lists every namespace in the Library's order", () => {
  assert.deepEqual(names(namespaceGroups(NAMESPACES, "")), ["root", "research", "research.papers", "historian"]);
});

test("typed, the namespace menu keeps the names holding the text, in any case", () => {
  assert.deepEqual(names(namespaceGroups(NAMESPACES, " PAPER ")), ["research.papers"]);
  assert.deepEqual(names(namespaceGroups(NAMESPACES, "search")), ["research", "research.papers"]);
});

test("only spaces typed, the namespace menu keeps every namespace and says nothing", () => {
  const [group] = namespaceGroups(NAMESPACES, "  ");
  assert.deepEqual(names([group]), ["root", "research", "research.papers", "historian"]);
  assert.equal(group.note, null);
});

test("with no name holding the text, the namespace menu says so", () => {
  const [group] = namespaceGroups(NAMESPACES, "zzz");
  assert.deepEqual(group.options, []);
  assert.equal(group.note, "no namespace holds “zzz”");
});
