// What the composer's model and namespace menus list for what is typed.
import assert from "node:assert/strict";
import test from "node:test";
import { modelGroups, namespaceGroups } from "../src/deep_reasoning/web/models.js";

const LISTS = [
  { provider: "OpenAI", featured: ["gpt-6-astra", "gpt-6-luna"], models: ["gpt-4", "gpt-6-astra", "gpt-6-luna"], problem: null },
  { provider: "fake", featured: ["fake-large", "fake-small"], models: ["fake-large", "fake-small"], problem: null },
  { provider: "broken", featured: [], models: [], problem: "No list: it didn't answer." },
];
// Each group as "provider: label | label", in order; null names the group of names to use as typed.
const shown = (groups) => groups.map((group) => `${group.name}: ${group.options.map((option) => option.label).join(" | ")}`);

test("untyped, each provider lists its featured models, and the chat's own where it isn't one", () => {
  assert.deepEqual(shown(modelGroups(LISTS, "", { provider: "OpenAI", name: "gpt-4" })), [
    "OpenAI: gpt-6-astra | gpt-6-luna | gpt-4",
    "fake: fake-large | fake-small",
    "broken: use model name…",
  ]);
});

test("a name a later provider lists exactly comes first, then the text to use on each provider that doesn't list it", () => {
  const groups = modelGroups(LISTS, "fake-large", null);
  assert.deepEqual(shown(groups), ["fake: fake-large", "broken: ", "null: use “fake-large” on OpenAI | use “fake-large” on broken"]);
  assert.deepEqual(groups[0].options[0].value, { provider: "fake", name: "fake-large" });
});

test("within its provider, the exact name comes before the others that hold it", () => {
  const lists = [{ provider: "OpenAI", featured: [], models: ["gpt-6-luna-mini", "gpt-6-luna"], problem: null }];
  assert.deepEqual(shown(modelGroups(lists, "gpt-6-luna", null)), ["OpenAI: gpt-6-luna | gpt-6-luna-mini"]);
});

test("typed, a provider lists every model it knows that holds the text, in any case, and the text to use comes after them all", () => {
  assert.deepEqual(shown(modelGroups(LISTS, " GPT-6 ", null)), [
    "OpenAI: gpt-6-astra | gpt-6-luna",
    "broken: ",
    "null: use “GPT-6” on OpenAI | use “GPT-6” on fake | use “GPT-6” on broken",
  ]);
});

// Enter picks the first option: a model some provider lists, never a made-up name on another provider.
for (const [typed, picked] of [
  ["small", { provider: "fake", name: "fake-small" }],
  ["fake-l", { provider: "fake", name: "fake-large" }],
  ["FAKE-LARGE", { provider: "fake", name: "fake-large" }],
  ["Gpt-6-Luna", { provider: "OpenAI", name: "gpt-6-luna" }],
  ["luna", { provider: "OpenAI", name: "gpt-6-luna" }],
  ["nothing-lists-this", { provider: "OpenAI", name: "nothing-lists-this" }],
]) {
  test(`Enter on “${typed}” picks ${picked.name} on ${picked.provider}`, () => {
    assert.deepEqual(modelGroups(LISTS, typed, null).flatMap((group) => group.options)[0].value, picked);
  });
}

test("a provider whose list names the text in another case offers its listed model, not the text as typed", () => {
  const values = modelGroups(LISTS, "FAKE-LARGE", null).flatMap((group) => group.options.map((option) => option.value));
  assert.deepEqual(values[0], { provider: "fake", name: "fake-large" });
  assert.ok(!values.some((value) => value.provider === "fake" && value.name === "FAKE-LARGE"));
  assert.ok(values.some((value) => value.provider === "OpenAI" && value.name === "FAKE-LARGE"));
});

test("with no name exact anywhere, the providers that list a match keep their order", () => {
  const both = [LISTS[0], { provider: "other", featured: [], models: ["luna-2"], problem: null }];
  assert.deepEqual(modelGroups(both, "luna", null).map((group) => group.name), ["OpenAI", "other", null]);
});

test("a provider whose list failed keeps its note while a filter is typed", () => {
  const broken = modelGroups(LISTS, "luna", null).find((group) => group.name === "broken");
  assert.equal(broken.note, "No list: it didn't answer.");
  assert.deepEqual(broken.options, []);
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
