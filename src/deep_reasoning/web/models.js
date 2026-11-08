// What the composer's menus list for what is typed: the models, from each provider's lists as the server reads them, and the namespaces.

// A name a menu's filter keeps: one holding the text typed, trimmed, in any case.
const holds = (name, typed) => name.toLowerCase().includes(typed.toLowerCase());
const same = (name, typed) => name.toLowerCase() === typed.toLowerCase();

// Untyped, each provider's featured models and the chat's own. Typed, the models any provider lists that hold the text, a
// provider whose list names it, in any case, first; then the text to use as it is, on each provider whose list doesn't name
// it, the chat's own provider first, so Enter picks a listed model before a name no list holds, and that on the chat's provider.
export function modelGroups(lists, query, current) {
  const typed = query.trim();
  const groups = lists.map(({ provider, featured, models, problem }) => {
    const own = current?.provider === provider ? [current.name] : [];
    const known = [...new Set([...featured, ...own, ...(typed ? models : [])])];
    const matches = known.filter((name) => holds(name, typed)).sort((a, b) => same(b, typed) - same(a, typed));
    const options = matches.map((name) => ({ value: { provider, name }, label: name }));
    if (!typed && !options.length) options.push({ value: null, label: "use model name…", hint: `type a model name for ${provider}` });
    return { name: provider, note: problem, options, exact: matches.some((name) => same(name, typed)) };
  });
  if (!typed) return groups;
  const shown = groups.filter((group) => group.options.length || group.note).sort((a, b) => b.exact - a.exact);
  const own = (group) => group.name === current?.provider;
  const uses = groups
    .filter((group) => !group.exact)
    .sort((a, b) => own(b) - own(a))
    .map(({ name }) => ({ value: { provider: name, name: typed }, label: `use “${typed}” on ${name}` }));
  return uses.length ? [...shown, { name: null, note: null, options: uses }] : shown;
}

// The namespace menu's options whose names hold what is typed, in the Library's order, or a note that none do.
export function namespaceGroups(options, query) {
  const typed = query.trim();
  const shown = options.filter((option) => holds(option.value, typed));
  return [{ options: shown, note: shown.length ? null : `no namespace holds “${typed}”` }];
}

// A namespace's name as the composer shows it. One holding more than printable ASCII, as only a shared file may, is written as
// Python's ascii() writes it, as the Library panel's reason does ('rоot'), so a look-alike never passes for another.
const PRINTABLE_ASCII = /^[\x20-\x7e]*$/;
const FIRST_PRINTABLE = 0x20;
const PAST_PRINTABLE = 0x7f;
const HEX = 16;
const NAMED_ESCAPES = { "\t": "\\t", "\n": "\\n", "\r": "\\r", "\\": "\\\\" };
// the widest code point each escape takes, its letter and its hex digits
const ESCAPES = [
  [0xff, "x", 2],
  [0xffff, "u", 4],
  [Infinity, "U", 8],
];

export function shownName(name) {
  if (PRINTABLE_ASCII.test(name)) return name;
  const quote = name.includes("'") && !name.includes('"') ? '"' : "'";
  return `${quote}${[...name].map((char) => escaped(char, quote)).join("")}${quote}`;
}

function escaped(char, quote) {
  if (char === quote) return `\\${char}`;
  if (Object.hasOwn(NAMED_ESCAPES, char)) return NAMED_ESCAPES[char];
  const code = char.codePointAt(0);
  if (code >= FIRST_PRINTABLE && code < PAST_PRINTABLE) return char;
  const [, letter, digits] = ESCAPES.find(([widest]) => code <= widest);
  return `\\${letter}${code.toString(HEX).padStart(digits, "0")}`;
}
