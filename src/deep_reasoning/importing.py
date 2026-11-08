"""`deep-reasoning import`: read a deep_reasoner config into the Library, refusing in
plain words what would break the app, and say what it changed and what it left out."""

import ast
import difflib
import os
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml
from conflit.config import _parse_compose_entry, load_namespaces, read_yaml_strict
from deep_reasoner.config import load_cli_config
from deep_reasoner.namespaces import CONFIG_SOURCE, ROOT
from deep_reasoner.v2.cli import V2Config, build_namespace_registry
from pydantic import ValidationError

from deep_reasoning.app_tools import claude_code_tools
from deep_reasoning.library import (
    APP_TOOLS,
    CLAUDE_CODE,
    CONTENT,
    ENTRY_NAMESPACE,
    HISTORIAN,
    LIBRARIAN,
    LOCAL_REPL,
    MAX_YAML_CHARS,
    NAMESPACE_NAME,
    NOT_RUN_SETTINGS,
    ROOTS_EXAMPLES,
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
    check_tool_name,
    check_written,
    claim_refusal,
    config_of,
    decomposition_problems,
    dump,
    expanded_size,
    false_claims,
    granted,
    granted_in,
    handed_to,
    in_chain_order,
    literals,
    namespace_problem,
    parsed,
    parsed_python,
    problem,
    read_library,
    run_config_and_registry,
    running_code,
    shown,
    template_names,
    template_problems,
    texts_in,
    tool_file,
    unreachable,
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
COMPOSE = "_compose"
KEEP_VARIABLES = (
    "Those the Library holds read a template variable the config's"
    " prompt_template_variables takes away, so no run would show them: keep the variable"
    " in the config, or replace those worked examples in it."
)
OUTSIDE = (
    "an import reads only files in the config's folder and below: copy the file in, or"
    " pass --within with a folder holding both."
)
# Past this, a config's YAML is refused before the engine reads it, measured with each
# alias written out in full, as values plus their text's characters: the engine expands
# every alias, so a few lines of them can stand for millions of values.
MAX_EXPANDED = 4 * MAX_YAML_CHARS
# Run settings no run survives below.
AT_LEAST_ONE = ("max_iter", "max_depth")
# How much of a value an import's report shows, and of a text before where it changed.
BRIEF_CHARS = 60
LEAD_CHARS = 20
# How many keys of a mapping's change a report shows.
SHOWN_KEYS = 3
# Root's fields that reach every namespace beside its own, and those only a namespace that
# sets none along its chain gets, with what one gets when root sets none.
ADDED_UP = ("tools", "vars", "system_suffix")
NEAREST = {
    "spawn": "may spawn into any namespace",
    "repl": "run code in the run settings' repl",
    "reasoner": "run on the run settings' backbone",
}
# The questions root's worked examples hand each of the app's own namespaces.
HANDED = {HISTORIAN: "earlier messages", LIBRARIAN: "the Library"}
# Root's worked examples in a new Library, by name: where each hands its question, and
# the example as the engine reads it, to tell one still as seeded.
SEEDED = {
    example.name: (handed_to(example), example.model_dump())
    for example in map(parsed, ROOTS_EXAMPLES)
}
UNSET = object()


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


def tool_row(name, block, config_dir, folder):
    """A tool's block as the Library stores it, and for a tool of your own its Python
    source (None for a built-in's), read only from *folder* (reading_folder()). A tool
    may take a built-in's name only to give that built-in its settings."""
    if name in APP_TOOLS:
        raise LibraryError(
            f'Tool "{name}" has the name of one of the app\'s own tools; rename yours.'
        )
    factory_from = block.get("factory_from")
    check_tool_name(name, block)
    check_tool_block(name, block)
    if runs_on_the_chats_model(name, block):
        # The config's model or preset may not exist where the chat runs.
        block = {key: value for key, value in block.items() if key != "model"}
    if factory_from is None:
        return block, None
    named = f'Tool "{name}" names factory_from'
    file = inside(config_dir / factory_from, factory_from, folder, named)
    try:
        source = file.read_text(encoding="utf-8")
    except OSError as error:
        raise LibraryError(
            f'Tool "{name}" names factory_from {factory_from}. {file_problem(error)}'
        ) from error
    return {**block, "factory_from": tool_file(name)}, source


def runs_on_the_chats_model(name, block):
    """Whether tool *name*, as its config *block* makes it, is an llm tool, which runs on
    each chat's model, as the chat does."""
    return block.get("factory_from") is None and block.get("factory", name) == "llm"


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


def read_config(path, folder):
    """The deep_reasoner config at *path*, as the engine loads it, reading only from
    *folder* (reading_folder()), or a LibraryError in plain words; a file none of whose
    keys the engine reads, as a data file, is refused."""
    not_one = f"{path} is not a deep_reasoner config"
    try:
        check_composes_within(Path(path).resolve(), folder)
        composed = [payload for _, payload in load_namespaces(Path(path))]
        size = check_expanded(path, composed)
        config = load_cli_config(path, schema=V2Config)
        check_expanded(path, namespace_files(config, folder), size)
    except LibraryError:
        raise
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
    meant = closest(key, V2Config.model_fields)
    return f"Left out: {key}, which is not a setting the engine reads{meant}."


def closest(word, choices):
    """' (did you mean X?)' for the one of *choices* closest to *word*; blank for none."""
    close = difflib.get_close_matches(word, choices, n=1)
    return f" (did you mean {close[0]}?)" if close else ""


def check_entry(entry, names):
    """Raise unless *entry*, the entry_namespace a config sets (None for none), is one
    of *names*, the namespaces the Library will have, as new chats start there."""
    if entry is None or entry in names:
        return
    raise LibraryError(
        f'entry_namespace names "{entry}", which is not a namespace in the config or the'
        f" Library{closest(entry, names)}: add it, or name one that is."
    )


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


def reading_folder(path, within=None):
    """The folder an import of the config at *path* reads in, as its real path: the
    config's own, or *within*, a folder holding it; a LibraryError for one that isn't."""
    config = Path(path).resolve()
    if within is None:
        return config.parent
    folder = Path(within).resolve()
    if not folder.is_dir() or not config.is_relative_to(folder):
        raise LibraryError(
            f"--within {within} should be a folder holding the config, {config}."
        )
    return folder


def inside(path, given, folder, named):
    """The real path of *path*, which the config gives as *given*, *named* saying where;
    a LibraryError, before anything reads it, when it is outside *folder*."""
    found = path.resolve()
    if not found.is_relative_to(folder):
        which = "," if str(found) == str(given) else f", which is {found},"
        raise LibraryError(f"{named} {given}{which} outside {folder}; {OUTSIDE}")
    return found


def check_composes_within(path, folder, stack=()):
    """Raise unless each file the config at *path*, a real path, composes, and each they
    compose, as conflit finds them, is in *folder*, each checked before it is read; what
    conflit refuses, such as a cycle, is left to it."""
    composed = read_yaml_strict(path).get(COMPOSE)
    for entry in composed if isinstance(composed, list) else []:
        given, _ = _parse_compose_entry(entry, COMPOSE)
        found = inside(path.parent / given, given, folder, f"{path} composes")
        if found not in (*stack, path):
            check_composes_within(found, folder, (*stack, path))


def namespace_files(config, folder):
    """The data of each file in the *config*'s namespaces_dir, found as the engine finds
    them: the folder, and each file through a link, refused outside *folder* before it
    is read."""
    if config.namespaces_dir is None:
        return []
    given = config.namespaces_dir
    found = inside(config.config_dir / given, given, folder, "Its namespaces_dir is")
    held = "Its namespaces_dir holds"
    return [
        yaml.safe_load(inside(file, file, folder, held).read_text(encoding="utf-8"))
        for file in sorted(found.glob("**/*.yaml"))
    ]


def check_expanded(path, files, size=0):
    """How big the config at *path* is, *size* so far, with *files*, the data of more of
    the files it reads, as expanded_size() measures them; raise past MAX_EXPANDED, before
    the engine writes the aliases out. Those conflit composes into it are measured first,
    then those in its namespaces_dir."""
    for data in files:
        size += expanded_size(data, MAX_EXPANDED - size)
        if size > MAX_EXPANDED:
            raise LibraryError(
                f"{path} is too big to import: with each alias written out, as the engine"
                f" reads it, it holds more than {MAX_EXPANDED:,} values and characters."
            )
    return size


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
    each character that isn't printable ASCII escaped, as a look-alike's would pass for
    another's."""
    if NAMESPACE_NAME.fullmatch(name):
        return f"Namespace {name}"
    return f'Namespace "{ascii(name)[1:-1]}"'


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


def check_claims(changes, registry):
    """Raise unless each decomposition the import writes makes no claim in the shape of a
    run's line that isn't so in its namespace, as *registry* resolves it."""
    for (namespace, name), each in changes["decompositions"].items():
        if namespace in registry.names and (
            claims := false_claims(registry, namespace, texts_in(each))
        ):
            refusal = claim_refusal("It", claims[0])
            raise LibraryError(
                f'Namespace {namespace}, decomposition "{name}": {refusal}'
            )


def check_renders(changes, decompositions, names, stored):
    """Raise unless each decomposition the import writes renders, its templates reading
    only *names*, as the panel refuses one that doesn't, and so does each of the
    Library's *stored* ones (read_decompositions()) that renders now and that the import
    keeps, as one that stops is left out of every run; their other problems only mark
    them ✗. The config's *decompositions* are by (namespace, name)."""
    problems = [
        f'Namespace {namespace}, decomposition "{name}": {problem}'
        for namespace, name in changes["decompositions"]
        for problem in template_problems(decompositions[namespace, name], names)
    ]
    kept = [
        f'Namespace {row["namespace"]}, decomposition "{row["name"]}", kept: {problem}'
        for row in stored
        if row["decomposition"] is not None
        and (row["namespace"], row["name"]) not in changes["decompositions"]
        for problem in template_problems(row["decomposition"], names)
    ]
    if kept:
        kept.append(KEEP_VARIABLES)
    if problems or kept:
        raise LibraryError("\n".join(problems + kept))


def check_builds(tools, environment, files):
    """Raise unless each of *tools*, (name, block, source), builds as a run builds it, in
    a process of its own with *environment*: each that doesn't, in words, naming the
    config's own file for it where *files* has one."""
    if tools:
        asked = {name: (block, source) for name, block, source in tools}
        refuse_unbuilt_keyless(asked, environment, note=NOT_CARRIED, files=files)


def library_state(read):
    """The Library, as read_library() *read* it, as an import compares with it: the run
    settings; each namespace's own fields; each decomposition as the engine reads it, by
    (namespace, name), None for one left out; and each tool's block and source. A row the
    engine can't read is left out with a warning, as the Library's readers leave it out."""
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


def import_config(connection, path, secrets=(), environment=None, within=None):
    """Read the deep_reasoner config at *path* into the Library, all of it or none, and
    say what it changed and left out, as {"changes": lines, "left_out": lines}. *secrets*
    may not be in it; the tools it newly grants are built with *environment* (none if not
    given). It reads files only in the config's folder, or *within*, a folder holding it."""
    folder = reading_folder(path, within)
    config = read_config(path, folder)
    registry = build_namespace_registry(config)
    unread = NOT_RUN_SETTINGS | set(config.model_extra or {})
    settings = config.model_dump(exclude_unset=True, exclude=unread)
    check_settings(settings)
    check_not_the_apps([*registry.names, settings.get(ENTRY_NAMESPACE)])
    tools = {
        name: tool_row(name, block, config.config_dir, folder)
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
        read = read_library(connection)
        before, earlier = library_state(read), as_the_engine_reads(read)
        imported = imported_state(settings, namespaces, tools, before)
        changes = changes_of(before, imported)
        namespace_names = [ROOT, *registry.names, *before["namespaces"]]
        check_entry(settings.get(ENTRY_NAMESPACE), namespace_names)
        check_new_names(before, changes)
        store_changes(connection, changes)
        later = accepted(connection)
        check_namespaces(connection, before, changes)
        check_claims(changes, later[1])
        names = template_names(later[0].prompt_template_variables)
        check_renders(changes, decompositions, names, read["decompositions"])
        built = to_build(before, imported, changes)
        files = {
            name: block.get("factory_from") for name, block in config.tools.items()
        }
        check_builds(built, environment or {}, files)
        # Inside the transaction, so a report that raises rolls the import back.
        said = import_report(
            before, changes, decompositions, names, earlier, later, environment or {}
        )
    return {"changes": said, "left_out": left_out(config)}


def as_the_engine_reads(read):
    """The Library, as read_library() *read* it, as the engine reads it: (config,
    registry); None when it can't, as for a shared file's it refuses."""
    try:
        return run_config_and_registry(config_of(read))
    except ValueError:
        return None


def visible(text):
    """*text* on one line as a terminal shows it: each line break as ⏎, and each other
    character that doesn't show (a control character, an escape) escaped."""
    return "".join(
        "⏎" if each == "\n" else each if each.isprintable() else ascii(each)[1:-1]
        for each in text
    )


def brief(value, start=0):
    """*value* as an import's report shows it, on one line, cut past BRIEF_CHARS: a text
    from *start*."""
    if value is UNSET:
        return "(unset)"
    if not isinstance(value, str):
        text = visible(shown(value))
        return text if len(text) <= BRIEF_CHARS else f"{text[: BRIEF_CHARS - 1]}…"
    part = visible(value[start : start + BRIEF_CHARS])
    cut_before = "…" if start else ""
    cut_after = "…" if len(value) > start + BRIEF_CHARS else ""
    return f'{cut_before}"{part}"{cut_after}'


def differing(before, after):
    """The keys whose values differ between mappings *before* and *after*, a key only one
    holds included."""
    keys = dict.fromkeys([*before, *after])
    return [
        key
        for key in keys
        if key not in before or key not in after or not same(before[key], after[key])
    ]


def changed(key, before, after):
    """How *key* changed from *before* to *after* (UNSET for none), as (key, old, new) as
    a report shows them: a mapping on both sides key by key, unless it is a block of
    another type (a repl or reasoner's), and a text on both sides from the word just
    before where the two first differ."""
    if (
        isinstance(before, dict)
        and isinstance(after, dict)
        and before.get("type") == after.get("type")
    ):
        return [
            line
            for each in differing(before, after)
            for line in changed(
                f"{key}.{visible(str(each))}",
                before.get(each, UNSET),
                after.get(each, UNSET),
            )
        ]
    start = 0
    if isinstance(before, str) and isinstance(after, str):
        same_up_to = len(os.path.commonprefix([before, after]))
        start = max(same_up_to - LEAD_CHARS, 0)
        space = re.search(r"\s", before[start:same_up_to]) if start else None
        start += space.end() if space else 0
    return [(key, brief(before, start), brief(after, start))]


def capped(field, lines):
    """The first SHOWN_KEYS of *lines*, the changes to *field*, and how many more."""
    if len(lines) <= SHOWN_KEYS + 1:
        return lines
    return [*lines[:SHOWN_KEYS], f"and {len(lines) - SHOWN_KEYS} more in {field}"]


def field_lines(before, after):
    """Each field that differs between fields *before* and *after*, as "key old → new"."""
    lines = []
    for field in differing(before, after):
        found = changed(field, before.get(field, UNSET), after.get(field, UNSET))
        lines += capped(field, [f"{key} {old} → {new}" for key, old, new in found])
    return lines


def and_list(names):
    """*names* in words: "a", "a and b", "a, b and c"."""
    return " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)


def setting_lines(before, changes):
    """Each run setting an import changed, but where new chats start, which the app's
    lines say."""
    defaults = V2Config().model_dump()
    held = before["settings"]
    lines = []
    for setting, value in changes["settings"].items():
        if setting == ENTRY_NAMESPACE:
            continue
        old = held.get(setting, defaults.get(setting, UNSET))
        default = "" if setting in held else " (the engine's default)"
        found = [
            f"{key}: {was}{default if key == setting else ''} → {now}"
            for key, was, now in changed(setting, old, value)
        ]
        lines += capped(setting, found)
    return [f"  {line}" for line in lines]


def quoted(names):
    return ", ".join(f'"{visible(name)}"' for name in names)


def namespace_lines(before, changes):
    """What an import did to each namespace, a line each, naming the decompositions it
    added and replaced."""
    new = changes["decompositions"]
    names = dict.fromkeys([*changes["namespaces"], *(name for name, _ in new)])
    lines = []
    for name in names:
        held = {
            each: (at, each) in before["decompositions"]
            for at, each in new
            if at == name
        }
        added = [each for each, there in held.items() if not there]
        replaced = [each for each, there in held.items() if there]
        if name not in before["namespaces"]:
            with_them = f", with decompositions {quoted(added)}" if added else ""
            lines.append(f"  {name}: added{with_them}")
            continue
        own = changes["namespaces"].get(name, before["namespaces"][name])
        parts = field_lines(before["namespaces"][name], own)
        parts += ["the rest of root kept"] if parts and name == ROOT else []
        parts += [f"decompositions added: {quoted(added)}"] if added else []
        which = "replaced" if added else "decompositions replaced"
        parts += [f"{which}: {quoted(replaced)}"] if replaced else []
        lines += [f"  {name}: {'; '.join(parts)}"] if parts else []
    return lines


def tool_lines(before, changes, registry):
    """What an import did to each tool, and which namespaces have it, as *registry*
    resolves them after the import."""
    lines = []
    for name, (block, source) in changes["tools"].items():
        done = "added"
        if name in before["tools"]:
            held, held_source = before["tools"][name]
            parts = field_lines(held, block)
            parts += ["its Python"] if source != held_source else []
            done = f"replaced {', '.join(parts)}" if parts else "replaced"
        lines.append(f"  {name}: {done}; {reach(name, source, registry)}")
    return lines


def reach(tool, source, registry):
    """Which namespaces of *registry* have *tool*, whose Python is *source* (None for a
    built-in's), in words."""
    granting = granted_in(registry, in_chain_order(registry), tool)
    if ROOT in granting:
        return "every namespace has it, as root grants it"
    if granting:
        return f"granted in {', '.join(granting)}"
    if source is None:
        return "no namespace grants it"
    return "not built: no namespace grants it"


def app_lines(before, changes, earlier, later, environment):
    """What an import changes in how the app's chats run, in words, a line each: only
    what applies. *earlier* and *later* are the Library as the engine reads it, (config,
    registry), before the import (None when it couldn't) and after; *environment* is
    what runs get of this one."""
    held = before["settings"].get(ENTRY_NAMESPACE, ROOT)
    entry = changes["settings"].get(ENTRY_NAMESPACE)
    lines = [f"New chats start in {entry}, not {held}."] if entry else []
    lines += root_lines(before, changes, later[1])
    lines += librarian_lines(earlier, later[1])
    lines += hand_off_lines(earlier, later[1], entry or held)
    lines += billing_lines(earlier, later)
    lines += key_lines(earlier, later[1], environment)
    lines += seeded_lines(before, changes)
    lines += stale_lines(before, changes, earlier, later[1])
    after = before["namespaces"] | changes["namespaces"]
    ungranted = sorted(set(granted(before["namespaces"])) - set(granted(after)))
    if ungranted:
        lines.append(
            f"No namespace grants {', '.join(ungranted)} now: grant it again in the panel"
            " if chats need it."
        )
    return lines


def root_lines(before, changes, registry):
    """What of an import's change to root reaches which namespaces, in words."""
    root = changes["namespaces"].get(ROOT)
    fields = differing(before["namespaces"].get(ROOT, {}), root) if root else []
    lines = []
    added_up = [field for field in fields if field in ADDED_UP]
    if added_up:
        lines.append(
            "Every namespace, historian and librarian included, gets root's"
            f" {and_list(added_up)} beside its own."
        )
    for field in (each for each in fields if each in NEAREST):
        lines.append(nearest_line(field, root.get(field), registry))
    return lines


def nearest_line(field, value, registry):
    """What root's *field*, now *value* (None for unset), means for the namespaces that
    set none along their chain, in words."""
    if value is None:
        unset = [
            name
            for name in in_chain_order(registry)
            if getattr(registry.resolve(name).sources, field) in (None, CONFIG_SOURCE)
        ]
        return f"Root sets no {field} now, so {and_list(unset)} {NEAREST[field]}."
    reached = [
        name
        for name in in_chain_order(registry)
        if name != ROOT and getattr(registry.resolve(name).sources, field) == ROOT
    ]
    if not reached:
        return f"Root's {field} reaches no other namespace: each sets its own."
    return f"Root's {field} reaches only the namespaces that set none: {and_list(reached)}."


def librarian_lines(earlier, registry):
    """That the librarian can no longer write tools, if the import moved it off the local
    REPL."""
    if LIBRARIAN not in registry.names:
        return []
    now = registry.resolve(LIBRARIAN).repl.type
    was = LOCAL_REPL
    if earlier is not None and LIBRARIAN in earlier[1].names:
        was = earlier[1].resolve(LIBRARIAN).repl.type
    if now == LOCAL_REPL or was != LOCAL_REPL:
        return []
    return [
        (
            f"The librarian now runs code in the {now} REPL, so it can't write tools: that"
            " needs the local REPL."
        )
    ]


def hand_off_lines(earlier, registry, entry):
    """Which namespaces' agents can't hand root's worked examples' questions to the
    app's own namespaces now, though they could before, as the engine's check_spawn
    decides; *entry*, where new chats start, first."""
    lost = defaultdict(list)
    # Stable, so the rest keep the registry's order.
    for name in sorted(in_chain_order(registry), key=lambda each: each != entry):
        if name.split(".")[0] in HANDED:
            continue
        targets = tuple(
            target
            for target in HANDED
            if unreachable(registry, name, target)
            and not could_not_hand(earlier, name, target)
        )
        where = f"{name} (where new chats start)" if name == entry else name
        lost[targets].append(where)
    return [
        f"Agents in {and_list(names)} can't hand questions about"
        f" {' or '.join(HANDED[each] for each in targets)} to the"
        f" {and_list(list(targets))} now: their spawn leaves"
        f" {'it' if len(targets) == 1 else 'them'} out."
        for targets, names in lost.items()
        if targets
    ]


def could_not_hand(earlier, name, target):
    """Whether namespace *name*'s agents couldn't hand a question to *target* before the
    import, as *earlier* reads the Library."""
    return (
        earlier is not None
        and name in earlier[1].names
        and unreachable(earlier[1], name, target) is not None
    )


def bills(engine, name):
    """Whether namespace *name* runs Claude Code, as its backbone or a tool, in *engine*,
    (config, registry)."""
    config, registry = engine
    if name not in registry.names:
        return False
    tools = registry.resolve(name).tools
    return registry.backbone(name) == CLAUDE_CODE or bool(
        claude_code_tools(config, tools)
    )


def billing_lines(earlier, later):
    """Which namespaces the import newly puts on Claude Code, as backbone or a tool."""
    newly = [
        name
        for name in in_chain_order(later[1])
        if bills(later, name) and not (earlier is not None and bills(earlier, name))
    ]
    if not newly:
        return []
    return [
        f"Agents in {and_list(newly)} now run Claude Code, billed to your subscription."
    ]


def key_lines(earlier, registry, environment):
    """Each key a namespace's REPL now reads that *environment* doesn't hold, and who
    needs it."""
    needing = defaultdict(list)
    for name in in_chain_order(registry):
        repl = registry.resolve(name).repl
        key = getattr(repl, "api_key_env", None)
        if key and not environment.get(key) and not same_repl(earlier, name, repl):
            needing[key].append(name)
    return [
        f"{and_list(names)} {'needs' if len(names) == 1 else 'need'} {key} to run code,"
        " and it isn't set here: set it in the shell you start the app from, or runs"
        " there fail."
        for key, names in needing.items()
    ]


def same_repl(earlier, name, repl):
    """Whether namespace *name* ran code in *repl* before the import, as *earlier*
    reads it."""
    return (
        earlier is not None
        and name in earlier[1].names
        and earlier[1].resolve(name).repl == repl
    )


def seeded_lines(before, changes):
    """Each of the app's own worked examples in root that the import replaced, while the
    Library held it as seeded."""
    lines = []
    for namespace, name in changes["decompositions"]:
        target, seed = SEEDED.get(name, (None, None))
        held = before["decompositions"].get((namespace, name))
        if namespace == ROOT and target and held is not None and same(held, seed):
            lines.append(
                f"Root's \"{name}\" replaces the app's own example of handing questions"
                f" about {HANDED[target]} to the {target}, so agents learn it from yours."
            )
    return lines


def stale_lines(before, changes, earlier, registry):
    """Each system_suffix or worked example the import kept that names a tool its
    namespace no longer has."""
    if earlier is None:
        return []
    lines = []
    for name in [each for each in in_chain_order(registry) if each in earlier[1].names]:
        lost = set(earlier[1].resolve(name).tools) - set(registry.resolve(name).tools)
        lines += kept_naming(before, changes, name, lost) if lost else []
    return lines


def kept_naming(before, changes, name, lost):
    """Namespace *name*'s own system_suffix and worked examples the import kept that name
    a tool of *lost*, in words."""
    label = "Root" if name == ROOT else name
    own = before["namespaces"].get(name, {})
    suffix = own.get("system_suffix")
    kept = same(suffix, changes["namespaces"].get(name, own).get("system_suffix"))
    words = re.findall(r"\w+", suffix) if kept and isinstance(suffix, str) else []
    named = ", ".join(sorted(lost & set(words)))
    lines = []
    if named:
        lines.append(f"{label}'s system_suffix names {named}, which it no longer has.")
    for (at, example), data in before["decompositions"].items():
        if at != name or data is None or (at, example) in changes["decompositions"]:
            continue
        called = ", ".join(sorted(lost & code_names(data)))
        if called:
            lines.append(
                f'{label}\'s worked example "{visible(example)}" calls {called}, which it'
                " no longer has."
            )
    return lines


def code_names(decomposition):
    """Every name the code of a worked example, as a mapping, reads."""
    names = set()
    for message in decomposition["messages"]:
        if message["role"] != "assistant":
            continue
        try:
            tree = parsed_python(running_code(message["content"]))
        except LibraryError:
            continue
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return names


def import_report(before, changes, decompositions, names, earlier, later, environment):
    """What an import changed, in lines of plain words, from the Library *before* it and
    its *changes*; the config's *decompositions* by (namespace, name), whose templates may
    read *names*; and what that changes in the app (app_lines(), which takes *earlier*,
    *later* and *environment*)."""
    sections = {
        "Run settings, for every chat:": setting_lines(before, changes),
        "Namespaces:": namespace_lines(before, changes),
        "Tools:": tool_lines(before, changes, later[1]),
        "The panel will mark these decompositions ✗, stored all the same:": (
            crossed_lines(changes, decompositions, names)
        ),
        "What this changes in the app:": [
            f"  {line}"
            for line in app_lines(before, changes, earlier, later, environment)
        ],
    }
    return [line for title, rows in sections.items() if rows for line in [title, *rows]]


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
    lines += [
        f"Left out: tool {name}'s model {block['model']}, as an llm tool runs on each"
        " chat's model."
        for name, block in config.tools.items()
        if "model" in block and runs_on_the_chats_model(name, block)
    ]
    unread = sorted((config.model_fields_set & NOT_RUN_SETTINGS) - CONTENT)
    if unread:
        lines.append(
            f"Left out, as the app sets them for each chat and run: {', '.join(unread)}."
        )
    return lines
