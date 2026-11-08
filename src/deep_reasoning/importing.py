"""`deep-reasoning import`: read a deep_reasoner config into the Library, refusing in
plain words what would break the app, and say what it changed and what it left out."""

import difflib
import tempfile
from functools import partial
from pathlib import Path

import structlog
import yaml
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
    NOT_RUN_SETTINGS,
    UPSERT_DECOMPOSITION,
    UPSERT_NAMESPACE,
    UPSERT_SETTING,
    UPSERT_TOOL,
    LibraryError,
    accepted,
    check_backbones,
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
    suffix_problem,
    taken,
    template_names,
    texts_in,
    tool_file,
    writing,
)
from deep_reasoning.tool_check import refuse_unbuilt

log = structlog.get_logger()

# The app's own namespaces: a config may name neither, nor a namespace under either.
APP_NAMESPACES = (HISTORIAN, LIBRARIAN)
# The run config an import's tool check builds with: a client that reaches nothing, so a
# factory that calls the model while it is built fails rather than spends; and, as an
# import leaves out the config's models presets and llm_kwargs, no run has them either.
NOWHERE = {
    "client": {
        "base_url": "http://127.0.0.1:9/v1",
        "api_key_env": "DEEP_REASONING_NO_KEY",
        "max_retries": 0,
    }
}
# Said after the tools that don't build.
NOT_CARRIED = (
    "Each tool is built from its own Python file alone: if it reads a file or module from"
    " beside the config, those aren't carried into the Library, so put what it reads in"
    " its settings or in its file; if it needs a package, install it in the app's"
    " environment."
)
NOT_A_MAPPING = 'it should be a mapping of settings, like "max_iter: 20".'
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
    if not isinstance(factory, str) or not isinstance(factory_from, str | None):
        raise LibraryError(f'Tool "{name}": factory and factory_from should be text.')
    the_builtin = factory_from is None and factory == name and name in BUILDERS
    if taken(name) and not the_builtin:
        raise LibraryError(
            f'Tool "{name}" has a name a built-in, the app or the REPL uses (such as llm,'
            " history or subagent), or a Python keyword, so it would take that one's"
            " place in every namespace. Rename it in the config; a block may use a"
            " built-in's name only to give that built-in its settings."
        )
    if factory_from is None:
        if factory not in BUILDERS:
            raise LibraryError(
                f'Tool "{name}" names factory "{factory}", which is not built in; '
                "give its block a factory_from: file."
            )
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
        config = load_cli_config(path, schema=V2Config)
    except ValidationError as error:
        raise LibraryError("\n".join(map(config_problem, error.errors()))) from error
    except OSError as error:
        raise LibraryError(file_problem(error)) from error
    except yaml.YAMLError as error:
        raise LibraryError(yaml_problem(error, path)) from error
    except ValueError as error:
        # conflit's YamlRootError, a ValueError, is the one that says "YAML root".
        detail = NOT_A_MAPPING if "YAML root" in str(error) else str(error)
        raise LibraryError(f"{not_one}: {detail}") from error
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


def check_suffixes(connection, namespaces):
    """Raise unless the engine renders each of *namespaces*' system_suffix, as the panel
    refuses one."""
    problems = [
        each for each in map(partial(suffix_problem, connection), namespaces) if each
    ]
    if problems:
        raise LibraryError("\n".join(problems))


def check_builds(tools, environment):
    """Raise unless each of *tools*, (name, block, source), builds as a run builds it, in
    a process of its own with *environment*: each that doesn't, in words."""
    if not tools:
        return
    asked = {name: (block, source) for name, block, source in tools}
    log.info("Building tools to check them", tools=list(asked))
    with tempfile.TemporaryDirectory() as work:
        refuse_unbuilt(asked, NOWHERE, environment, Path(work), note=NOT_CARRIED)


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
    check_not_the_apps([*registry.names, settings.get(ENTRY_NAMESPACE)])
    tools = {
        name: tool_row(name, block, config.config_dir)
        for name, block in config.tools.items()
    }
    # A root that sets nothing, like the bare one the engine adds, must not blank the Library's.
    namespaces = [
        namespace
        for namespace in map(registry.config, registry.names)
        if namespace.name != ROOT
        or namespace.decompositions
        or namespace.model_fields_set - {"name", "decompositions"}
    ]
    check_imports_no_key(settings, namespaces, tools, secrets)
    # The command holds the app's lock on the file (owning), so no other writer waits on
    # the builds this holds the write lock through; a config the engine refuses runs no
    # code, as the builds come after its check.
    with writing(connection):
        before = library_state(connection)
        imported = imported_state(settings, namespaces, tools, before)
        changes = changes_of(before, imported)
        store_changes(connection, changes)
        stored, registry = accepted(connection)
        check_backbones(registry, registry.names)
        check_suffixes(connection, namespaces)
        built = to_build(before, imported, changes)
        check_builds(built, environment or {})
    decompositions = {
        (namespace.name, each.name): each
        for namespace in namespaces
        for each in namespace.decompositions
    }
    return {
        "changes": import_report(
            before, changes, built, decompositions, template_names(stored)
        ),
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
