// What the composer's menus list for what is typed: the models, from each provider's lists as the server reads them, and the namespaces.

// A name a menu's filter keeps: one holding the text typed, trimmed, in any case.
const holds = (name, typed) => name.toLowerCase().includes(typed.toLowerCase());

// Each provider's featured models and the chat's own, or those matching what is typed and the typed name; one that lists
// the typed name exactly comes first, with it first, so Enter picks it over another's "use".
export function modelGroups(lists, query, current) {
  const typed = query.trim();
  const groups = lists.map(({ provider, featured, models, problem }) => {
    const own = current?.provider === provider ? [current.name] : [];
    const known = [...new Set([...featured, ...own, ...(typed ? models : [])])];
    const exact = typed !== "" && known.includes(typed);
    const matches = known.filter((name) => holds(name, typed)).sort((a, b) => (b === typed) - (a === typed));
    const options = matches.map((name) => ({ value: { provider, name }, label: name }));
    if (typed && !exact) options.push({ value: { provider, name: typed }, label: `use “${typed}”` });
    if (!options.length) options.push({ value: null, label: "use model name…", hint: `type a model name for ${provider}` });
    return { name: provider, note: problem, options, exact };
  });
  return groups.sort((a, b) => b.exact - a.exact);
}

// The namespace menu's options whose names hold what is typed, in the Library's order, or a note that none do.
export function namespaceGroups(options, query) {
  const typed = query.trim();
  const shown = options.filter((option) => holds(option.value, typed));
  return [{ options: shown, note: shown.length ? null : `no namespace holds “${typed}”` }];
}
