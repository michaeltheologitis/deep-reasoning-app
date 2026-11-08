// What the model menu lists for what is typed, from each provider's lists as the server reads them.

// Each provider's featured models and the chat's own, or those matching what is typed and the typed name; one that lists
// the typed name exactly comes first, with it first, so Enter picks it over another's "use".
export function modelGroups(lists, query, current) {
  const typed = query.trim();
  const groups = lists.map(({ provider, featured, models, problem }) => {
    const own = current?.provider === provider ? [current.name] : [];
    const known = [...new Set([...featured, ...own, ...(typed ? models : [])])];
    const exact = typed !== "" && known.includes(typed);
    const matches = known.filter((name) => name.toLowerCase().includes(typed.toLowerCase())).sort((a, b) => (b === typed) - (a === typed));
    const options = matches.map((name) => ({ value: { provider, name }, label: name }));
    if (typed && !exact) options.push({ value: { provider, name: typed }, label: `use “${typed}”` });
    if (!options.length) options.push({ value: null, label: "use model name…", hint: `type a model name for ${provider}` });
    return { name: provider, note: problem, options, exact };
  });
  return groups.sort((a, b) => b.exact - a.exact);
}
