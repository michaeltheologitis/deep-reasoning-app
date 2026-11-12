// A decomposition's turns checked as a worked example, as library.py's turn_problems checks them, in its words; and the
// commonest slip behind a YAML error, as its slip_words says it.
const THINK = /<think>[\s\S]*?<\/think>/;
const REPL = "<repl>";
const REPL_END = "</repl>";
const OBSERVATION = /<observation>[\s\S]*?<\/observation>/;
const FINAL_ANSWER = /\bFinalAnswer[ \t]*\(/;
const COMMENT_LINE = /^[ \t]*#/;
// what Python's str.strip() strips, as the engine strips the code it runs: JS's trim() strips a different set
const PYTHON_SPACE = /[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/.source;
const PYTHON_STRIPPED = new RegExp(`^${PYTHON_SPACE}+|${PYTHON_SPACE}+$`, "g");
const BETWEEN = {
  user: "the assistant's turn goes between them",
  assistant: "the user turn with the <observation> goes between them",
};
const NO_TASK = "There is no task: add a user turn holding it.";

// the code the engine runs, as its code() takes it: after the last <repl>, up to the next </repl> or the end, stripped; blank for none
function runningCode(content) {
  const start = content.lastIndexOf(REPL);
  if (start === -1) return "";
  const rest = content.slice(start + REPL.length);
  const end = rest.indexOf(REPL_END);
  return (end === -1 ? rest : rest.slice(0, end)).replace(PYTHON_STRIPPED, "");
}

const answers = (content) => runningCode(content).split("\n").some((line) => !COMMENT_LINE.test(line) && FINAL_ANSWER.test(line));

function replyProblems(label, content) {
  const blocks = content.split(REPL).length - 1;
  return [
    !THINK.test(content) && `${label} has no <think>…</think>.`,
    blocks === 0 && `${label} has no <repl>…</repl> block.`,
    blocks > 1 && `${label} has ${blocks} <repl> blocks; the engine runs only the last, so keep one.`,
    blocks > 0 && !runningCode(content) && `${label} has an empty <repl> block; the engine treats it as no block.`,
  ].filter(Boolean);
}

// previous: the latest turn before this one but a system one, null for none
function problemsOfTurn({ number, role, content }, previous) {
  const label = `Turn ${number} (${role})`;
  if (role === "system") return previous ? [`${label} comes after the task; system turns go before it.`] : [];
  const problems = [
    !previous && role === "assistant" && `${label} should be the user's task; start with a user turn.`,
    previous?.role === role && `${label} follows another ${role} turn; ${BETWEEN[role]}.`,
  ];
  if (role === "assistant") return [...problems, ...replyProblems(label, content)].filter(Boolean);
  // what a cell printed follows it, unless it answered: then a new message may
  const owed = previous?.role === "assistant" && !answers(previous.content) && !OBSERVATION.test(content);
  problems.push(owed && `${label} should hold the <observation>…</observation> that turn ${previous.number}'s code printed.`);
  return problems.filter(Boolean);
}

// messages: a decomposition's turns, each with its role and content, as the shape check passed them
export function turnProblems(messages) {
  const problems = [];
  let previous = null;
  messages.forEach(({ role, content }, i) => {
    const turn = { number: i + 1, role, content };
    problems.push(...problemsOfTurn(turn, previous));
    if (role !== "system") previous = turn;
  });
  if (!previous) return [NO_TASK];
  const { number, role, content } = previous;
  if (role === "user") problems.push(`It ends on turn ${number} (user); end with the assistant's turn that calls FinalAnswer(...).`);
  else if (!answers(content)) problems.push(`Turn ${number} (assistant), the last, should call FinalAnswer(...) in its <repl>.`);
  return problems;
}

// how far above a YAML error a line less indented than the one before it is tried as its slip; the most lines of a slipped
// paragraph indented with its first; and enough of a line to find it by, as the editor numbers no lines (in characters as
// Python counts them)
const SLIP_LINES_BACK = 5;
const SLIP_PARAGRAPH_LINES = 20;
const SLIP_SHOWN_CHARS = 40;
const indentOf = (line) => line.match(/^ */)[0].length;
const isBlank = (line) => /^[ \t]*$/.test(line);
const stripped = (line) => line.replace(/^[ \t]+|[ \t]+$/g, "");
const isMapping = (value) => Object.prototype.toString.call(value) === "[object Object]";

// the parsed YAML, or undefined when it doesn't parse; js-yaml is the page's global, as the vendored copy sets it
function parsed(text) {
  try {
    return { value: globalThis.jsyaml.load(text) };
  } catch {
    return undefined;
  }
}

// whether a line, alone, is YAML text or a comment, never a key or an item
function readsAsText(line) {
  if (isBlank(line)) return false;
  const read = parsed(line);
  return read !== undefined && !(Array.isArray(read.value) || isMapping(read.value));
}

function lineBefore(lines, at) {
  for (let each = at - 1; each >= 0; each -= 1) if (!isBlank(lines[each])) return each;
  return null;
}

// the end of the paragraph of text line at begins: the lines after it, blank ones between, at its indent that read as text
function paragraphEnd(lines, at) {
  let end = at + 1;
  for (let each = at + 1; each < Math.min(at + SLIP_PARAGRAPH_LINES, lines.length); each += 1) {
    if (isBlank(lines[each])) continue;
    if (indentOf(lines[each]) !== indentOf(lines[at]) || !readsAsText(lines[each])) break;
    end = each + 1;
  }
  return end;
}

// the key, in document order, whose text holds line as a line of its own; null for none
function fieldHolding(data, line) {
  const items = Array.isArray(data) ? data.map((item) => [null, item]) : isMapping(data) ? Object.entries(data) : [];
  for (const [key, value] of items) {
    if (key !== null && typeof value === "string" && value.split("\n").map(stripped).includes(line)) return key;
    const found = fieldHolding(value, line);
    if (found !== null) return found;
  }
  return null;
}

function slipSaid(number, end, first, field, width) {
  const chars = [...first];
  const shown = chars.length > SLIP_SHOWN_CHARS ? `${chars.slice(0, SLIP_SHOWN_CHARS).join("")}…` : first;
  const text = field === null ? "the text" : `the ${field} text`;
  const spaces = `${width} space${width === 1 ? "" : "s"}`;
  if (end === number) return `Line ${number}, "${shown}", is less indented than ${text} above it, so it isn't part of that text: start it with the same ${spaces}.`;
  return `Lines ${number} to ${end}, from "${shown}", are less indented than ${text} above them, so they aren't part of that text: start each with the same ${spaces}.`;
}

// The commonest slip behind a YAML error at errorLine, in plain words, as library.py's slip_words finds it: each line at most
// SLIP_LINES_BACK above the error, less indented than the line before it and text alone, is indented like that line, with
// the lines of text right after it at its indent; the first that makes the YAML parse is the slip. null when none does.
export function slipWords(text, errorLine) {
  const lines = text.split("\n");
  const last = Math.min(errorLine, lines.length) - 1;
  for (let at = last; at >= Math.max(last - SLIP_LINES_BACK, 0); at -= 1) {
    const before = lineBefore(lines, at);
    if (before === null || indentOf(lines[at]) >= indentOf(lines[before]) || !readsAsText(lines[at])) continue;
    const end = paragraphEnd(lines, at);
    const width = indentOf(lines[before]);
    const moved = [...lines.slice(0, at), ...lines.slice(at, end).map((line) => " ".repeat(width) + line.replace(/^ +/, "")), ...lines.slice(end)];
    const read = parsed(moved.join("\n"));
    if (read === undefined) continue;
    const first = stripped(lines[at]);
    return slipSaid(at + 1, end, first, fieldHolding(read.value, first), width);
  }
  return null;
}
