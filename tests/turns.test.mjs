// The page's turn check and YAML slip words on the cases test_library.py checks library.py's on, so the forms warn in the
// server's words; js-yaml is the vendored copy the page loads.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { slipWords, turnProblems } from "../src/deep_reasoning/web/turns.js";

// as the page's script tag sets it
globalThis.jsyaml = createRequire(import.meta.url)("../src/deep_reasoning/web/vendor/js-yaml.min.js");
const { jsyaml } = globalThis;
// a case's YAML: its text, then any line repeated to its length
const textOf = ({ text, repeat = "", times = 0, after = "" }) => text + repeat.repeat(times) + after;
// a pasted 32 KB line took the regex this replaced 4 s
const SLIP_MS = 1000;
const cases = (file) => JSON.parse(readFileSync(new URL(file, import.meta.url), "utf8"));
const CASES = cases("turn_cases.json");
const SLIP_CASES = cases("slip_cases.json");

for (const { case: name, messages, problems } of CASES) {
  test(`the page finds what the server finds in ${name}`, () => {
    assert.deepEqual(turnProblems(messages), problems);
  });
}

// the line js-yaml stops at, as library.js's readYaml reads it
function errorLine(text) {
  try {
    jsyaml.load(text);
  } catch (error) {
    return error.mark.line + 1;
  }
  assert.fail("the YAML parsed");
}

for (const slip of SLIP_CASES) {
  test(`the page words a YAML slip as the server does: ${slip.case}`, () => {
    const text = textOf(slip);
    const started = performance.now();
    assert.equal(slipWords(text, errorLine(text)), slip.words);
    assert.ok(performance.now() - started < SLIP_MS);
  });
}
