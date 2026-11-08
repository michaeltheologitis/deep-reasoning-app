"""`deep-reasoning import`: read a deep_reasoner config into the Library, refusing in
plain words what would break the app, and say what it changed and what it left out."""

import difflib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import yaml
from conflit.config import load_namespaces
from deep_reasoner.config import load_cli_config
from deep_reasoner.namespaces import ROOT
from deep_reasoner.v2.cli import V2Config, build_namespace_registry
from pydantic import ValidationError

from deep_reasoning.library import (
    APP_TOOLS,
    BUILDERS,
    CLAUDE_CODE,
    CONTENT,
    ENTRY_NAMESPACE,
    HISTORIAN,
    LIBRARIAN,
    MAX_YAML_CHARS,
    NAMESPACE_NAME,
    NOT_RUN_SETTINGS,
    UPSERT_DECOMPOSITION,
    UPSERT_NAMESPACE,
    UPSERT_SETTING,
    UPSERT_TOOL,
    LibraryError,
    accepted,
    check_decomposition_name,
    check_name,
    check_size,
    check_tool_block,
    check_written,
    count,
    decomposition_problems,
    dump,
    granted,
    literals,
    namespace_problem,
    parsed_python,
    problem,
    read_library,
    shown,
    taken,
    template_names,
    template_problems,
    texts_in,
    tool_file,
    writing,
)
from deep_reasoning.tool_check import refuse_unbuilt_keyless

# The app's own namespaces: a config may name neither, nor a namespace under either.
APP_NAMESPACES = (HISTORIAN, LIBRARIAN)
# Said after the tools that don't build.
NOT_CARRIED = (
    "Each tool is built from its own Python file alone: if it reads a file or module from"
    " beside the config, those aren't carried into the Library, so put what it reads in"
    " its settings or in its file; if it needs a package, install it in the app's"
    " environment."
)
NOT_A_MAPPING = 'it should be a mapping of settings, like "max_iter: 20".'
# Past this, a config's YAML is refused before the engine reads it, measured with each
# alias written out in full, as values plus their text's characters: the engine expands
# every alias, so a few lines of them can stand for millions of values.
MAX_EXPANDED = 4 * MAX_YAML_CHARS
# Run settings no run survives below.
AT_LEAST_ONE = ("max_iter", "max_depth")
# How much of a value an import's report shows.
BRIEF_CHARS = 60
UNSET = object()
BILLS = (
    "Root's reasoner is claude_code, which runs each of them on Claude Code, billed to"
    " your subscription."
)


def file_problem(error):
    """An OSError reading a config's file, in words, without its errno."""
    if isinstance(error, FileNotFoundError):
        return f"There is no file {error.filename}."
    if isinstance(error, IsADirectoryError):
        return f"{error.filename} is a folder: name the config's YAML file in it."
    return f"Could not read {error.filename}: {error.strerror}."


def yaml_problem(error, path):
    """A YAML error in a config's file, in words: where, and what."""
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return f"{path} doesn't parse as YAML: {error}."
    reason = getattr(error, "problem", None) or "it isn't YAML"
    return (
        f"{mark.name} doesn't parse as YAML at line {mark.line + 1}, column"
        f" {mark.column + 1}: {reason}. Check the indentation and quoting there."
    )


def tool_row(name, block, config_dir):
    """A tool's block as the Library stores it, and for a tool of your own its Python
    source (None for a built-in's). A tool may take a built-in's name only to give that
    built-in its settings."""
    if name in APP_TOOLS:
        raise LibraryError(
            f'Tool "{name}" has the name of one of the app\'s own tools; rename yours.'
        )
    factory = block.get("factory", name)
    factory_from = block.get("factory_from")
    the_builtin = factory_from is None and factory == name and name in BUILDERS
    if taken(name) and not the_builtin:
        raise LibraryError(
            f'Tool "{name}" has a name a built-in, the app or the REPL uses (such as llm,'
            " history or subagent), or a Python keyword, so it would take that one's"
            " place in every namespace. Rename it in the config; a block may use a"
            " built-in's name only to give that built-in its settings."
        )
    check_tool_block(name, block)
    if factory_from is None:
        if factory == "llm":
            # It runs on the chat's model, as the chat does; the config's model or preset may not exist there.
            block = {key: value for key, value in block.items() if key != "model"}
        return block, None
    file = config_dir / factory_from
    try:
        source = file.read_text(encoding="utf-8")
    except OSError as error:
        raise LibraryError(
            f'Tool "{name}" names factory_from {factory_from}. {file_problem(error)}'
        ) from error
    return {**block, "factory_from": tool_file(name)}, source


def config_problem(error):
    """One of V2Config's validation errors, in the panel's words where it has them."""
    match error["loc"]:
        case ("namespaces", str(name), "decompositions", int(index), _, *_):
            inner = problem(error | {"loc": error["loc"][4:]})
            return f"Namespace {name}, decomposition {index + 1}: {inner}"
        case ("namespaces", str(name), _, *_):
            inner = namespace_problem(error | {"loc": error["loc"][2:]})
            return f"Namespace {name}: {inner}"
    return f"{'.'.join(map(str, error['loc']))}: {error['msg']}."


def read_config(path):
    """The deep_reasoner config at *path*, as the engine loads it, or a LibraryError in
    plain words; a file none of whose keys the engine reads, as a data file, is refused."""
    not_one = f"{path} is not a deep_reasoner config"
    try:
        check_expanded(path)
        config = load_cli_config(path, schema=V2Config)
    except ValidationError as error:
        raise LibraryError("\n".join(map(config_problem, error.errors()))) from error
    except OSError as error:
        raise LibraryError(file_problem(error)) from error
    except yaml.YAMLError as error:
        raise LibraryError(yaml_problem(error, path)) from error
    except ValueError as error:
        # conflit's YamlRootError, a ValueError, says "YAML root", after the file it
        # means, which may be one the config composes.
        file, mapping, _ = str(error).partition(": YAML root")
        if mapping:
            raise LibraryError(
                f"{file} is not a deep_reasoner config: {NOT_A_MAPPING}"
            ) from error
        raise LibraryError(f"{not_one}: {error}") from error
    except RecursionError as error:
        raise LibraryError(f"{path} is nested too deeply to read.") from error
    except TypeError as error:
        # What conflit raises of a _compose or _conflit that isn't as it reads them.
        raise LibraryError(
            f"{path} is not a config the engine can read: {error}."
        ) from error
    unknown = list(config.model_extra or {})
    if unknown and not config.model_fields_set - {"config_path", *unknown}:
        raise LibraryError(
            f"{not_one}: none of its keys ({', '.join(unknown)}) is a setting the engine"
            " reads."
        )
    return config


def unknown_key(key):
    """A key a config holds that the engine doesn't read, as an import leaves it out."""
    close = difflib.get_close_matches(key, V2Config.model_fields, n=1)
    meant = f" (did you mean {close[0]}?)" if close else ""
    return f"Left out: {key}, which is not a setting the engine reads{meant}."


def check_not_the_apps(names):
    """Raise if any of *names*, the namespaces a config names, is the app's own or under it."""
    taken_ones = dict.fromkeys(
        name for name in names if name and name.split(".")[0] in APP_NAMESPACES
    )
    if taken_ones:
        raise LibraryError(
            f"The config names {', '.join(taken_ones)}, but historian and librarian are"
            " the app's own namespaces (historian answers questions about a chat's"
            " earlier messages, and librarian edits the Library), so a config may"
            " neither replace them nor add a namespace under them. Rename them in the"
            " config, and wherever it names them (spawn, entry_namespace, subagent calls)."
        )


def holds_a_key(place):
    return LibraryError(
        f"{place} holds a key, and the Library file is shared, so keys never go in it:"
        " take it out of the config."
    )


def source_texts(name, source):
    """A tool's Python *source* and its literals, parsed here once; none for a built-in's."""
    if source is None:
        return []
    return [source, *literals(parsed_python(source, f"Tool {name}'s source"))]


def check_imports_no_key(settings, namespaces, tools, secrets):
    """Raise if what an import would store holds any of *secrets*, naming where; each of
    *tools* as tool_row gives it, whose source is checked to parse too."""
    places = [
        (f"Setting {key}", [key, *texts_in(value)]) for key, value in settings.items()
    ]
    places += [
        (f"Namespace {each.name}", list(texts_in(each.model_dump())))
        for each in namespaces
    ]
    places += [
        (f"Tool {name}", [name, *texts_in(block), *source_texts(name, source)])
        for name, (block, source) in tools.items()
    ]
    for place, texts in places:
        if any(secret in text for secret in secrets for text in texts):
            raise holds_a_key(place)


@contextmanager
def about(where):
    """Say *where* a refusal in the block is, before its own words."""
    try:
        yield
    except LibraryError as error:
        raise LibraryError(f"{where}: {error}") from error


def expanded_size(data, cap):
    """How big *data* is with each alias written out in full, as values plus their
    text's characters, as the engine expands them; the walk stops past *cap*."""
    size, stack = 0, [data]
    while stack and size <= cap:
        value = stack.pop()
        size += 1
        if isinstance(value, str | bytes):
            size += len(value)
        elif isinstance(value, dict):
            stack += [*value, *value.values()]
        elif isinstance(value, list):
            stack += value
    return size


def check_expanded(path):
    """Raise if the config at *path*, with the files conflit composes into it and those
    in its namespaces_dir, found as conflit and the engine find them, passes MAX_EXPANDED
    with each alias written out, before the engine writes them out."""
    composed = load_namespaces(Path(path))
    files = [payload for _, payload in composed]
    folders = [each.get("namespaces_dir") for at, each in composed if at == "."]
    folder = next((each for each in reversed(folders) if isinstance(each, str)), None)
    if folder:
        found = (Path(path).resolve().parent / folder).resolve()
        files += [
            yaml.safe_load(file.read_text(encoding="utf-8"))
            for file in sorted(found.glob("**/*.yaml"))
        ]
    size = 0
    for data in files:
        size += expanded_size(data, MAX_EXPANDED - size)
        if size > MAX_EXPANDED:
            raise LibraryError(
                f"{path} is too big to import: with each alias written out, as the engine"
                f" reads it, it holds more than {MAX_EXPANDED:,} values and characters."
            )


def check_settings(settings):
    """Raise unless each run setting fits in the Library and leaves a run something to
    do."""
    for key, value in settings.items():
        with about(f"Setting {key}"):
            check_size(dump(value), "Its YAML")
    for key in AT_LEAST_ONE:
        value = settings.get(key)
        if value is not None and value < 1:
            raise LibraryError(
                f"{key} is {value}, but a run needs at least 1 to take a step: make it 1"
                " or more, or take it out."
            )


def check_tools_fit(tools):
    """Raise unless each tool's block and source, of *tools* as tool_row gives them,
    fits."""
    for name, (block, source) in tools.items():
        with about(f"Tool {name}"):
            check_size(dump(block), "Its YAML")
            check_size(source or "", "Its source")


def check_decompositions(namespaces):
    """Raise unless each of *namespaces*' decompositions is one the panel would save:
    well named, fitting, and the only one of its name there."""
    pairs = [
        (namespace.name, each)
        for namespace in namespaces
        for each in namespace.decompositions
    ]
    held = Counter((name, each.name) for name, each in pairs)
    twice = [key for key, times in held.items() if times > 1]
    if twice:
        name, decomposition = twice[0]
        raise LibraryError(
            f"Namespace {name} holds more than one decomposition named"
            f' "{decomposition}": give each its own name.'
        )
    for name, each in pairs:
        with about(f'Namespace {name}, decomposition "{each.name}"'):
            check_decomposition_name(each.name)
            check_size(dump(each.model_dump()), "Its YAML")


def namespace_named(name):
    """Namespace *name*, as a refusal names it: quoted when it isn't a plain name, with
    each character that doesn't show escaped."""
    if NAMESPACE_NAME.fullmatch(name):
        return f"Namespace {name}"
    shown = "".join(each if each.isprintable() else ascii(each)[1:-1] for each in name)
    return f'Namespace "{shown}"'


def check_new_names(before, changes):
    """Raise unless each namespace new to the Library *before* the import is named as
    the panel names one."""
    for name in changes["namespaces"].keys() - before["namespaces"].keys():
        with about(namespace_named(name)):
            check_name(name)


def check_namespaces(connection, before, changes):
    """Raise unless each namespace the import writes fits, and is one the panel saves, as
    the Library stands with the import written (check_written())."""
    for name, own in changes["namespaces"].items():
        with about(namespace_named(name)):
            check_size(dump(own), "Its YAML")
            check_written(connection, name, name not in before["namespaces"])


def check_renders(changes, decompositions, names):
    """Raise unless each decomposition the import writes renders, its templates reading
    only *names*, as the panel refuses one that doesn't; its other problems only mark it
    ✗. The config's *decompositions* are by (namespace, name)."""
    problems = [
        f'Namespace {namespace}, decomposition "{name}": {problem}'
        for namespace, name in changes["decompositions"]
        for problem in template_problems(decompositions[namespace, name], names)
    ]
    if problems:
        raise LibraryError("\n".join(problems))


def check_builds(tools, environment, files):
    """Raise unless each of *tools*, (name, block, source), builds as a run builds it, in
    a process of its own with *environment*: each that doesn't, in words, naming the
    config's own file for it where *files* has one."""
    if tools:
        asked = {name: (block, source) for name, block, source in tools}
        refuse_unbuilt_keyless(asked, environment, note=NOT_CARRIED, files=files)


def library_state(connection):
    """The Library as an import compares with it: the run settings; each namespace's own
    fields; each decomposition as the engine reads it, by (namespace, name), None for one
    left out; and each tool's block and source. A row the engine can't read is left out
    with a warning, as the Library's readers leave it out."""
    read = read_library(connection)
    return {
        "settings": read["settings"],
        "namespaces": read["namespaces"],
        "decompositions": {
            (row["namespace"], row["name"]): (
                row["decomposition"].model_dump() if row["decomposition"] else None
            )
            for row in read["decompositions"]
        },
        "tools": read["tools"],
    }


def imported_state(settings, namespaces, tools, before):
    """What a config holds, in library_state's shape, over the Library *before* it: root's
    fields over root's own, and none for a root that sets nothing."""
    fields = {}
    for namespace in namespaces:
        own = namespace.model_dump(
            exclude_unset=True, exclude={"name", "decompositions"}
        )
        if namespace.name == ROOT and own:
            own = before["namespaces"].get(ROOT, {}) | own
        if namespace.name != ROOT or own:
            fields[namespace.name] = own
    decompositions = {
        (namespace.name, each.name): each.model_dump()
        for namespace in namespaces
        for each in namespace.decompositions
    }
    return {
        "settings": settings,
        "namespaces": fields,
        "decompositions": decompositions,
        "tools": tools,
    }


def canonical(value):
    """*value* as text that is the same for equal values, a NaN's included, which no
    NaN equals; a tool's (block, source) as a list."""
    plain = list(value) if isinstance(value, tuple) else value
    return yaml.safe_dump(plain, sort_keys=True, allow_unicode=True)


def same(one, other):
    return canonical(one) == canonical(other)


def changes_of(before, imported):
    """What *imported* holds and the Library *before* it doesn't, as it is, by kind
    (settings, namespaces, decompositions, tools) and then by key; a run setting the
    Library doesn't hold is the engine's default."""
    held = before | {"settings": V2Config().model_dump() | before["settings"]}
    return {
        kind: {
            key: value
            for key, value in imported[kind].items()
            if key not in held[kind] or not same(held[kind][key], value)
        }
        for kind in imported
    }


def store_changes(connection, changes):
    """Write *changes*, leaving the rest of the Library, comments and all, as it is."""
    rows = {
        UPSERT_SETTING: [
            (key, dump(value)) for key, value in changes["settings"].items()
        ],
        UPSERT_NAMESPACE: [
            (name, dump(own)) for name, own in changes["namespaces"].items()
        ],
        UPSERT_DECOMPOSITION: [
            (namespace, name, dump(each))
            for (namespace, name), each in changes["decompositions"].items()
        ],
        UPSERT_TOOL: [
            (name, dump(block), source)
            for name, (block, source) in changes["tools"].items()
        ],
    }
    for statement, values in rows.items():
        connection.executemany(statement, values)


def to_build(before, imported, changes):
    """The tools of your own to build: each a namespace grants after the import, whether
    the config or the Library holds it, but one the Library held as it is, granted
    already."""
    granted_before = set(granted(before["namespaces"]))
    granted_after = set(granted(before["namespaces"] | imported["namespaces"]))
    tools = before["tools"] | imported["tools"]
    return [
        (name, block, source)
        for name, (block, source) in tools.items()
        if source is not None
        and name in granted_after
        and (name in changes["tools"] or name not in granted_before)
    ]


def import_config(connection, path, secrets=(), environment=None):
    """Read the deep_reasoner config at *path* into the Library, all of it or none, and
    say what it changed and left out, as {"changes": lines, "left_out": lines}. *secrets*
    may not be in it; the tools it newly grants are built with *environment* (none if not
    given)."""
    config = read_config(path)
    registry = build_namespace_registry(config)
    unread = NOT_RUN_SETTINGS | set(config.model_extra or {})
    settings = config.model_dump(exclude_unset=True, exclude=unread)
    check_settings(settings)
    check_not_the_apps([*registry.names, settings.get(ENTRY_NAMESPACE)])
    tools = {
        name: tool_row(name, block, config.config_dir)
        for name, block in config.tools.items()
    }
    check_tools_fit(tools)
    # A root that sets nothing, like the bare one the engine adds, must not blank the Library's.
    namespaces = [
        namespace
        for namespace in map(registry.config, registry.names)
        if namespace.name != ROOT
        or namespace.decompositions
        or namespace.model_fields_set - {"name", "decompositions"}
    ]
    check_decompositions(namespaces)
    check_imports_no_key(settings, namespaces, tools, secrets)
    decompositions = {
        (namespace.name, each.name): each
        for namespace in namespaces
        for each in namespace.decompositions
    }
    # The command holds the app's lock on the file (owning), so no other writer waits on
    # the builds this holds the write lock through; a config the engine refuses runs no
    # code, as the builds come after its check.
    with writing(connection):
        before = library_state(connection)
        imported = imported_state(settings, namespaces, tools, before)
        changes = changes_of(before, imported)
        check_new_names(before, changes)
        store_changes(connection, changes)
        stored, _ = accepted(connection)
        check_namespaces(connection, before, changes)
        names = template_names(stored)
        check_renders(changes, decompositions, names)
        built = to_build(before, imported, changes)
        files = {
            name: block.get("factory_from") for name, block in config.tools.items()
        }
        check_builds(built, environment or {}, files)
    return {
        "changes": import_report(before, changes, built, decompositions, names),
        "left_out": left_out(config),
    }


def brief(value):
    """*value* as an import's report shows it, on one line, cut past BRIEF_CHARS."""
    if value is UNSET:
        return "(unset)"
    if isinstance(value, str):
        lines = value.strip().splitlines()
        text = f'"{lines[0] if lines else ""}"'
    else:
        text = shown(value)
    return text if len(text) <= BRIEF_CHARS else f"{text[: BRIEF_CHARS - 1]}…"


def differing(before, after):
    """The keys whose values differ between mappings *before* and *after*, a key only one
    holds included."""
    keys = dict.fromkeys([*before, *after])
    return [
        key
        for key in keys
        if key not in before or key not in after or not same(before[key], after[key])
    ]


def field_lines(before, after):
    """Each field that differs between fields *before* and *after*, as "key old → new"."""
    return [
        f"{key} {brief(before.get(key, UNSET))} → {brief(after.get(key, UNSET))}"
        for key in differing(before, after)
    ]


def setting_lines(before, changes):
    defaults = V2Config().model_dump()
    held = before["settings"]
    lines = []
    for key, value in changes["settings"].items():
        default = "" if key in held else " (the engine's default)"
        old = held.get(key, defaults.get(key))
        lines.append(f"  {key}: {brief(old)}{default} → {brief(value)}")
    return lines


def namespace_lines(before, changes):
    """What an import did to each namespace, a line each."""
    new = changes["decompositions"]
    names = dict.fromkeys([*changes["namespaces"], *(name for name, _ in new)])
    lines = []
    for name in names:
        held = [key in before["decompositions"] for key in new if key[0] == name]
        added, replaced = held.count(False), held.count(True)
        if name not in before["namespaces"]:
            with_them = f", with {count(added, 'decomposition')}" if added else ""
            lines.append(f"  {name}: added{with_them}")
            continue
        own = changes["namespaces"].get(name, before["namespaces"][name])
        parts = field_lines(before["namespaces"][name], own)
        parts += ["the rest of root kept"] if parts and name == ROOT else []
        parts += [f"{count(added, 'decomposition')} added"] if added else []
        parts += [f"{count(replaced, 'decomposition')} replaced"] if replaced else []
        lines += [f"  {name}: {'; '.join(parts)}"] if parts else []
    return lines


def tool_lines(before, changes, built):
    names = {name for name, _, _ in built}
    lines = []
    for name, (_, source) in changes["tools"].items():
        done = "replaced" if name in before["tools"] else "added"
        if source is not None and name not in names:
            done += "; not built: no namespace grants it"
        lines.append(f"  {name}: {done}")
    return lines


def root_lines(before, changes):
    """What of an import's change to root reaches every namespace, the historian and the
    librarian too, in words."""
    root = changes["namespaces"].get(ROOT, {})
    fields = differing(before["namespaces"].get(ROOT, {}), root) if root else []
    if not fields:
        return []
    named = " and ".join(fields)
    lines = [
        (
            f"Every namespace that sets none of its own gets root's {named}, historian and"
            " librarian too."
        )
    ]
    if "reasoner" in fields and (root.get("reasoner") or {}).get("type") == CLAUDE_CODE:
        lines.append(BILLS)
    return lines


def import_report(before, changes, built, decompositions, names):
    """What an import changed, in lines of plain words, from the Library *before* it and
    its *changes*, having *built* those tools; the config's *decompositions* by
    (namespace, name), whose templates may read *names*."""
    sections = {
        "Run settings, for every chat:": setting_lines(before, changes),
        "Namespaces:": namespace_lines(before, changes),
        "Tools:": tool_lines(before, changes, built),
        "The panel will mark these decompositions ✗, stored all the same:": (
            crossed_lines(changes, decompositions, names)
        ),
    }
    lines = [
        line for title, rows in sections.items() if rows for line in [title, *rows]
    ]
    lines += root_lines(before, changes)
    after = before["namespaces"] | changes["namespaces"]
    ungranted = set(granted(before["namespaces"])) - set(granted(after))
    if ungranted:
        lines.append(
            f"No namespace grants {', '.join(sorted(ungranted))} now: grant it again in"
            " the panel if chats need it."
        )
    return lines


def crossed_lines(changes, decompositions, names):
    """Each decomposition an import changes that the panel will mark ✗, with why: the
    config's *decompositions*, by (namespace, name), whose templates may read *names*."""
    lines = []
    for namespace, name in changes["decompositions"]:
        each = decompositions[namespace, name]
        problems = decomposition_problems(each, names)
        if problems:
            lines.append(f"  {namespace} / {name}: {' '.join(problems)}")
    return lines


def left_out(config):
    """What of *config* an import leaves out, in lines of plain words."""
    lines = [unknown_key(key) for key in config.model_extra or {}]
    openings = [each.name for each in config.decompositions]
    if openings:
        lines.append(
            f"Left out: the top-level decompositions {', '.join(openings)}, which open a"
            " run of `dr --main-decomposition`; a chat has none."
        )
    unread = sorted((config.model_fields_set & NOT_RUN_SETTINGS) - CONTENT)
    if unread:
        lines.append(
            f"Left out, as the app sets them for each chat and run: {', '.join(unread)}."
        )
    return lines
