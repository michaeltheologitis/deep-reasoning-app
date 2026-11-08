// What the composer's menus list for what is typed: the models, from each provider's lists as the server reads them, and the namespaces.

// A name a menu's filter keeps: one holding the text typed, trimmed, in any case.
const holds = (name, typed) => name.toLowerCase().includes(typed.toLowerCase());
const same = (name, typed) => name.toLowerCase() === typed.toLowerCase();

// Untyped, each provider's featured models and the chat's own. Typed, the models any provider lists that hold the text, a
// provider whose list names it, in any case, first; then the text to use as it is, on each provider whose list doesn't name
// it, so Enter picks a listed model before a name no list holds.
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
  const uses = groups.filter((group) => !group.exact).map(({ name }) => ({ value: { provider: name, name: typed }, label: `use “${typed}” on ${name}` }));
  return uses.length ? [...shown, { name: null, note: null, options: uses }] : shown;
}

// The namespace menu's options whose names hold what is typed, in the Library's order, or a note that none do.
export function namespaceGroups(options, query) {
  const typed = query.trim();
  const shown = options.filter((option) => holds(option.value, typed));
  return [{ options: shown, note: shown.length ? null : `no namespace holds “${typed}”` }];
}
