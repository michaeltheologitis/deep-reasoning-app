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

// a block text's head, "key: |" or "- |", or > for folded text, with any indicators; and a line of its own, as may end one
const BLOCK_HEAD = /^ *(?:- +)*(?:(\S[^#]*?):[ \t]+)?[|>][-+1-9]*[ \t]*(?:#.*)?$/;
const OWN_LINE = /^ *(?:#|-(?: |$)|[^\s#].*?:(?: |$))/;
const indentOf = (line) => line.match(/^ */)[0].length;
const isBlank = (line) => /^[ \t]*$/.test(line);
// enough of a line to find it by, as the editor numbers no lines; in characters as Python counts them
const SLIP_SHOWN_CHARS = 40;

// A line at or before errorLine that ends a block text above it, being less indented than that text's lines, and is no key,
// item or comment of its own, in plain words; null for none.
export function slipWords(text, errorLine) {
  const lines = text.split("\n").slice(0, errorLine);
  let at = 0;
  while (at < lines.length) {
    const head = lines[at].match(BLOCK_HEAD);
    at += 1;
    if (!head) continue;
    while (at < lines.length && isBlank(lines[at])) at += 1;
    if (at === lines.length || indentOf(lines[at]) <= indentOf(head[0])) continue;
    const body = indentOf(lines[at]);
    while (at < lines.length && (isBlank(lines[at]) || indentOf(lines[at]) >= body)) at += 1;
    if (at < lines.length && !OWN_LINE.test(lines[at])) {
      const textOf = head[1] ? `the lines of ${head[1]}` : "the text's lines";
      const chars = [...lines[at].replace(/^[ \t]+|[ \t]+$/g, "")];
      const shown = chars.length > SLIP_SHOWN_CHARS ? `${chars.slice(0, SLIP_SHOWN_CHARS).join("")}…` : chars.join("");
      return `Line ${at + 1}, "${shown}", isn't indented like ${textOf} above it, so it ends that text: start it with ${body} space${body === 1 ? "" : "s"}, as they do.`;
    }
  }
  return null;
}
