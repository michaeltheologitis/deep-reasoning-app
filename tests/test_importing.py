"""`deep-reasoning import`: what a config brings into the Library, what it refuses, and
what it says."""

import math
import re
import shutil
import tempfile
import time
from contextlib import closing
from datetime import date
from pathlib import Path

import pytest
from deep_reasoner.config import load_cli_config
from deep_reasoner.namespaces import ROOT
from deep_reasoner.v2.cli import (
    V2Config,
)
from structlog.testing import capture_logs

from deep_reasoning.importing import NOT_CARRIED, import_config
from deep_reasoning.library import (
    HISTORIAN,
    HISTORIANS_EXAMPLES,
    LIBRARIAN,
    LIBRARIAN_HAND_OFFS,
    MAX_YAML_CHARS,
    UPSERT_NAMESPACE,
    LibraryError,
    add_namespace,
    default_namespace,
    engine_config,
    hand_off,
    open_library,
    parsed,
    stored_config,
)

FAKE_KEY = "sk-test-not-a-real-key"


CHAT = "a-chat"


RUN = "20261027-120000-000000"


# advisors.yaml's namespaces, imported into a fresh Library beside its historian and librarian.
IMPORTED_NAMESPACES = {
    "root",
    "router",
    "course_advisor",
    "health_advisor",
    HISTORIAN,
    LIBRARIAN,
}


ROUTER_DECOMPOSITIONS = [
    "route a course question",
    "route a wellbeing question",
    "decline anything else",
]


TYPED = """system_prompt: |
  Solve it.
namespaces:
  root:
    vars: {by_year: {2024: 10}, due: 2026-10-25, cap: .inf}
    decompositions:
      - name: count
        messages:
          - {role: user, content: "How many?"}
          - role: assistant
            content: |
              <repl>
              FinalAnswer(3)
              </repl>
"""


# Far longer than the guard takes, far shorter than writing a bomb out.
BOMB_SECONDS = 5


RAISES_ON_BUILD = "def build(client, params):\n    raise RuntimeError('offline')\n"


CONFIGS = Path(__file__).parent / "configs"


# A fresh root's worked examples: its hand-offs to the librarian, then to the historian.
ROOTS = [
    *(parsed(text) for text in LIBRARIAN_HAND_OFFS),
    *(parsed(hand_off(text)) for text in HISTORIANS_EXAMPLES),
]


# What the app puts in for a chat on OpenAI's gpt-6-luna.
CHATS_MODEL = {
    "model": "gpt-6-luna",
    "client": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


ROUTER_SHOWN = ROUTER_DECOMPOSITIONS + [each.name for each in ROOTS]


def config_file(tmp_path, text):
    """A config holding *text*, as config.yaml in *tmp_path*."""
    written = tmp_path / "config.yaml"
    written.write_text(text)
    return written


def run_config(library, tmp_path, namespace=ROOT):
    """The config of a run on gpt-6-luna, as the app makes it."""
    return V2Config.model_validate(
        engine_config(library, namespace, tmp_path / "tools", CHAT, RUN) | CHATS_MODEL
    )


def import_into(library, config):
    """What importing *config* into *library* changed, as the lines it says."""
    with closing(open_library(library)) as connection:
        return import_config(connection, config)["changes"]


def dumped(library):
    with closing(open_library(library)) as connection:
        return list(connection.iterdump())


def change(library, act, *args):
    with closing(open_library(library)) as connection:
        return act(connection, *args)


@pytest.mark.parametrize(
    ("root", "tools", "variables", "said"),
    [
        ("{tools: []}", [], {}, "  root: tools [llm] → []; the rest of root kept"),
        (
            "{vars: {term: fall}}",
            ["llm"],
            {"term": "fall"},
            "  root: vars (unset) → {term: fall}; the rest of root kept",
        ),
        (
            "{decompositions: [{name: hi, messages: [{role: user, content: hi}]}]}",
            ["llm"],
            {},
            "  root: 1 decomposition added",
        ),
    ],
)
def test_an_imported_root_gives_the_fields_it_sets_and_root_keeps_the_rest(
    library, tmp_path, root, tools, variables, said
):
    written = config_file(tmp_path, f"namespaces:\n  root: {root}\n")
    assert said in import_into(library, written)
    stored = run_config(library, tmp_path).namespaces[ROOT]
    assert (stored.tools, stored.vars) == (tools, variables)


@pytest.mark.parametrize("root", ["{}", "{decompositions: []}"])
def test_an_imported_root_that_sets_nothing_leaves_the_librarys_as_it_is(
    library, tmp_path, root
):
    import_into(library, config_file(tmp_path, "namespaces:\n  root: {tools: []}\n"))
    nothing = f"namespaces:\n  root: {root}\n  helper: {{}}\n"
    import_into(library, config_file(tmp_path, nothing))
    assert run_config(library, tmp_path).namespaces[ROOT].tools == []


def test_import_brings_the_configs_namespaces_decompositions_and_prompt(
    imported, tmp_path
):
    config = run_config(imported, tmp_path)
    router = config.namespaces["router"]
    assert set(config.namespaces) == IMPORTED_NAMESPACES
    # Its spawn list leaves the historian out, so it is shown root's hand-offs saying so.
    assert [each.name for each in router.decompositions] == ROUTER_SHOWN
    assert router.spawn == ["course_advisor", "health_advisor"]
    assert config.namespaces["course_advisor"].vars["catalog"]["CS201"]["prereqs"] == [
        "CS101"
    ]
    assert config.system_prompt.startswith("You solve each request")
    assert default_namespace(imported) == "router"


def test_import_brings_no_model_or_client(library, tmp_path):
    written = config_file(
        tmp_path,
        "model: qwen/qwen3-30b\nclient: {base_url: 'https://api.novita.ai/v3/openai'}\n",
    )
    import_into(library, written)
    config = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)
    assert "model" not in config and "client" not in config
    assert run_config(library, tmp_path).llm_kwargs == {}


def test_an_imported_llm_tool_runs_on_the_chats_model(library, tmp_path):
    written = config_file(
        tmp_path,
        "models: {coder: {model: qwen/qwen3-coder-30b-a3b-instruct}}\n"
        "tools:\n"
        "  code_reviewer: {factory: llm, model: coder, system_prompt: Review it.}\n"
        "namespaces: {root: {tools: [llm, code_reviewer]}}\n",
    )
    import_into(library, written)
    config = run_config(library, tmp_path)
    assert config.tools["code_reviewer"] == {
        "factory": "llm",
        "system_prompt": "Review it.",
    }
    assert "models" not in engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)


def test_an_import_reads_back_as_the_engine_loads_the_config(library, tmp_path):
    written = config_file(tmp_path, TYPED)
    import_into(library, written)
    expected = load_cli_config(written, schema=V2Config)
    config = run_config(library, tmp_path)
    assert config.system_prompt == expected.system_prompt == "Solve it.\n"
    assert config.namespaces[ROOT].vars == expected.namespaces[ROOT].vars
    assert config.namespaces[ROOT].vars["by_year"][2024] == 10
    assert config.namespaces[ROOT].vars["due"] == date(2026, 10, 25)
    assert config.namespaces[ROOT].vars["cap"] == math.inf
    assert config.namespaces[ROOT].decompositions == [
        *ROOTS,
        *expected.namespaces[ROOT].decompositions,
    ]


def test_an_imported_var_holding_a_nel_keeps_it(library, tmp_path):
    written = config_file(tmp_path, 'namespaces:\n  root:\n    vars: {note: "a\\Nb"}\n')
    import_into(library, written)
    assert run_config(library, tmp_path).namespaces[ROOT].vars["note"] == "a\x85b"


def test_a_configs_run_level_openings_are_left_out(library, tmp_path):
    written = config_file(
        tmp_path,
        "decompositions:\n"
        "  - name: full_triage\n"
        "    messages: [{role: user, content: '{{ task }}'}]\n",
    )
    with closing(open_library(library)) as connection:
        said = import_config(connection, written)
    assert said == {
        "changes": [],
        "left_out": [
            (
                "Left out: the top-level decompositions full_triage, which open a run of"
                " `dr --main-decomposition`; a chat has none."
            )
        ],
    }
    assert run_config(library, tmp_path).namespaces[ROOT].decompositions == ROOTS


def test_import_replaces_same_named_items_and_keeps_the_rest(imported, tmp_path):
    newer = config_file(
        tmp_path,
        "max_iter: 3\n"
        "namespaces:\n"
        "  router:\n"
        "    spawn: [course_advisor]\n"
        "    decompositions:\n"
        "      - name: decline anything else\n"
        '        messages: [{role: user, content: "Only courses, please."}]\n',
    )
    assert import_into(imported, newer) == [
        "Run settings, for every chat:",
        "  max_iter: 20 → 3",
        "Namespaces:",
        (
            "  router: spawn [course_advisor, health_advisor] → [course_advisor];"
            " 1 decomposition replaced"
        ),
        "The panel will mark these decompositions ✗, stored all the same:",
        (
            "  router / decline anything else: It ends on turn 1 (user); end with the"
            " assistant's turn that calls FinalAnswer(...)."
        ),
    ]
    config = run_config(imported, tmp_path)
    router = config.namespaces["router"]
    assert router.spawn == ["course_advisor"]
    assert [each.name for each in router.decompositions] == ROUTER_SHOWN
    declining = router.decompositions[
        ROUTER_DECOMPOSITIONS.index("decline anything else")
    ]
    assert declining.messages[0].content == "Only courses, please."
    assert config.max_iter == 3
    assert config.namespaces[ROOT].tools == ["llm"]
    assert set(config.namespaces) == IMPORTED_NAMESPACES


def test_a_failed_import_leaves_the_library_unchanged(imported, tmp_path):
    broken = config_file(
        tmp_path, "max_iter: 1\nnamespaces:\n  router: {tools: [ghost]}\n  extra: {}\n"
    )
    before = dumped(imported)
    with pytest.raises(ValueError, match="grants unknown tool 'ghost'"):
        import_into(imported, broken)
    assert dumped(imported) == before


@pytest.mark.parametrize(
    ("block", "files", "problem"),
    [
        (
            "{factory: build, factory_from: ./broken.py}",
            {"broken.py": "def build(:\n"},
            "Tool mine's source is not valid Python",
        ),
        (
            "{factory: telepathy}",
            {},
            'names factory "telepathy", which is not built in',
        ),
        ("{factory: [a, b]}", {}, "factory and factory_from should be text"),
        (
            "{factory: build, factory_from: 5}",
            {},
            "factory and factory_from should be text",
        ),
    ],
)
def test_import_refuses_a_tool_the_engine_could_not_build(
    library, tmp_path, block, files, problem
):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    config = config_file(tmp_path, f"tools:\n  mine: {block}\n")
    with pytest.raises(LibraryError, match=problem):
        import_into(library, config)


@pytest.mark.parametrize("tool", ["history", "library", "page"])
def test_an_import_may_not_give_a_tool_an_app_tools_name(library, tmp_path, tool):
    config = config_file(tmp_path, f"tools:\n  {tool}: {{factory: llm}}\n")
    with pytest.raises(LibraryError, match="one of the app's own tools"):
        import_into(library, config)
    with closing(open_library(library)) as connection:
        assert connection.execute("SELECT count(*) FROM tools").fetchone() == (0,)


def test_import_says_each_setting_namespace_and_tool_it_changed(library, waitlist):
    said = import_into(library, waitlist)
    assert said[:2] == [
        "Run settings, for every chat:",
        (
            '  system_prompt: "You solve a request by writing Python in a REPL and observ… →'
            ' "You manage course waitlists by writing Python in a REPL an…'
        ),
    ]
    assert said[2:] == [
        "  max_iter: 30 (the engine's default) → 6",
        "Namespaces:",
        "  root: tools [llm] → [waitlist]; the rest of root kept",
        "Tools:",
        "  waitlist: added",
        (
            "Every namespace that sets none of its own gets root's tools, historian and"
            " librarian too."
        ),
        "No namespace grants llm now: grant it again in the panel if chats need it.",
    ]


def test_import_says_where_new_chats_start_and_which_namespaces_it_added(
    library, advisors
):
    said = import_into(library, advisors)
    assert ('  entry_namespace: "root" (the engine\'s default) → "router"') in said
    assert said[said.index("Namespaces:") + 1 :][:3] == [
        "  course_advisor: added, with 1 decomposition",
        "  health_advisor: added, with 1 decomposition",
        "  router: added, with 3 decompositions",
    ]


@pytest.mark.parametrize(
    "config",
    [
        CONFIGS / "catalog" / "namespaces.yaml",
        CONFIGS / "catalog" / "namespaces_dir.yaml",
        "entry_namespace: historian\n",
        "namespaces:\n  historian.notes: {}\n",
    ],
    ids=["namespaces", "namespaces_dir", "entry", "under historian"],
)
def test_import_refuses_a_config_naming_the_apps_own_namespaces(
    library, tmp_path, config
):
    if isinstance(config, str):
        written = config_file(tmp_path, config)
        config = written
    open_library(library).close()
    before = dumped(library)
    with pytest.raises(LibraryError, match="are the app's own namespaces"):
        import_into(library, config)
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        (
            CONFIGS / "catalog" / "credits.yaml",
            (
                "credits.yaml is not a deep_reasoner config: none of its keys (CS101, MATH200,"
                " CS201, CS310, CS330, CS450) is a setting the engine reads."
            ),
        ),
        (
            "just: text\n",
            (
                "config.yaml is not a deep_reasoner config: none of its keys (just) is a setting"
                " the engine reads."
            ),
        ),
    ],
    ids=["a data file", "text"],
)
def test_import_refuses_a_file_none_of_whose_keys_the_engine_reads(
    library, tmp_path, config, problem
):
    if isinstance(config, str):
        written = config_file(tmp_path, config)
        config = written
    open_library(library).close()
    before = dumped(library)
    with pytest.raises(LibraryError) as refused:
        import_into(library, config)
    assert str(refused.value).endswith(problem)
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        ("max_iter: lots\n", "max_iter: Input should be a valid integer"),
        (
            (
                "namespaces:\n  roles:\n    decompositions:\n"
                "      - {name: a, messages: [{role: tool, content: hi}]}\n"
            ),
            (
                "Namespace roles, decomposition 1: Message 1: role should be user, assistant"
                ' or system, not "tool".'
            ),
        ),
        (
            "namespaces:\n  typo: {toolz: [llm]}\n",
            'Namespace typo: Remove "toolz": a namespace\'s YAML holds only',
        ),
    ],
)
def test_an_import_the_engine_refuses_says_why_in_plain_words(
    library, tmp_path, config, problem
):
    written = config_file(tmp_path, config)
    with pytest.raises(LibraryError) as refused:
        import_into(library, written)
    assert problem in str(refused.value)
    assert "pydantic" not in str(refused.value)


def test_import_refuses_a_namespace_whose_suffix_no_run_renders(library, tmp_path):
    written = config_file(
        tmp_path,
        "namespaces:\n  research:\n    system_suffix: 'Answer as {{ task.__class__ }}.'\n",
    )
    before = dumped(library)
    refused = (
        "The system_suffix of research, line 1: the engine reads text holding {{ or {%"
        " as a Jinja template, and this one reaches into Python internals (__class__),"
        " which a run refuses."
    )
    with pytest.raises(LibraryError, match=re.escape(refused)):
        import_into(library, written)
    assert dumped(library) == before


def test_an_imported_suffix_may_read_the_configs_own_template_variables(
    library, tmp_path
):
    written = config_file(
        tmp_path,
        "prompt_template_variables: {dept: CS}\n"
        "namespaces:\n  research:\n    system_suffix: 'Answer for {{ dept }}.'\n",
    )
    import_into(library, written)
    namespace = run_config(library, tmp_path).namespaces["research"]
    assert namespace.system_suffix == "Answer for {{ dept }}."


NOT_A_MAPPING = 'is not a deep_reasoner config: it should be a mapping of settings, like "max_iter: 20".'


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("- a list\n", NOT_A_MAPPING),
        ("just text\n", NOT_A_MAPPING),
        ("", NOT_A_MAPPING),
        (
            "namespaces:\n  a: {tools: [llm}\n",
            (
                "doesn't parse as YAML at line 2, column 18: expected ',' or ']', but got"
                " '}'. Check the indentation and quoting there."
            ),
        ),
    ],
    ids=["a list", "text", "empty", "broken YAML"],
)
def test_an_import_of_a_file_that_is_not_a_config_says_so(
    library, tmp_path, text, problem
):
    written = config_file(tmp_path, text)
    with pytest.raises(LibraryError) as refused:
        import_into(library, written)
    assert str(refused.value) == f"{written} {problem}"


@pytest.mark.parametrize(
    ("path", "problem"),
    [
        ("missing.yaml", lambda config: f"There is no file {config}."),
        ("", lambda config: f"{config} is a folder"),
    ],
)
def test_an_import_of_a_file_that_is_not_there_says_so(
    library, tmp_path, path, problem
):
    config = tmp_path / path
    with pytest.raises(LibraryError, match=re.escape(problem(config))):
        import_into(library, config)


def test_import_names_the_decompositions_the_panel_will_mark_crossed(library, tmp_path):
    written = config_file(
        tmp_path,
        "namespaces:\n  helper:\n    decompositions:\n      - name: no think\n"
        "        messages: [{role: user, content: 'How many?'},"
        " {role: assistant, content: '<repl>FinalAnswer(3)</repl>'}]\n",
    )
    assert import_into(library, written)[-2:] == [
        "The panel will mark these decompositions ✗, stored all the same:",
        "  helper / no think: Turn 2 (assistant) has no <think>…</think>.",
    ]


TOOL_SOURCES = {
    "raises": (
        "def build(client, params):\n    raise RuntimeError('offline')\n",
        "build in mine.py raised RuntimeError: offline",
    ),
    "reads a data file": (
        "def build(client, params):\n    open(params['catalog_path'])\n",
        "build in mine.py raised FileNotFoundError",
    ),
    "imports a module beside it": (
        "from helpers import shout\n\ndef build(client, params): ...\n",
        "importing 'mine.py' raised ModuleNotFoundError: No module named 'helpers'",
    ),
}


@pytest.mark.parametrize("case", TOOL_SOURCES)
def test_import_refuses_a_tool_that_does_not_build_and_says_why(
    library, tmp_path, case
):
    source, problem = TOOL_SOURCES[case]
    (tmp_path / "mine.py").write_text(source)
    (tmp_path / "helpers.py").write_text("def shout(text): return text\n")
    shutil.copy(CONFIGS / "catalog" / "credits.yaml", tmp_path)
    config = config_file(
        tmp_path,
        "tools:\n"
        "  mine: {factory: build, factory_from: mine.py, catalog_path: credits.yaml}\n"
        "namespaces:\n  helper: {tools: [mine]}\n",
    )
    open_library(library).close()
    before = dumped(library)
    with pytest.raises(LibraryError) as refused:
        import_into(library, config)
    said = str(refused.value)
    assert said.startswith(f"Tool mine doesn't build: {problem}")
    assert said.endswith(NOT_CARRIED)
    assert tempfile.gettempdir() not in said
    assert dumped(library) == before


def grants_mine(tmp_path, source, settings="{}"):
    """A config with tool mine, built from *source* with *settings*, granted in helper."""
    (tmp_path / "mine.py").write_text(source)
    config = config_file(
        tmp_path,
        "tools:\n"
        f"  mine: {{factory: build, factory_from: mine.py, settings: {settings}}}\n"
        "namespaces:\n  helper: {tools: [mine]}\n",
    )
    return config


def test_a_tool_the_library_holds_is_built_when_an_import_first_grants_it(
    library, tmp_path
):
    (tmp_path / "scrape.py").write_text(RAISES_ON_BUILD)
    stored = config_file(
        tmp_path, "tools:\n  scrape: {factory: build, factory_from: scrape.py}\n"
    )
    import_into(library, stored)
    granting = config_file(tmp_path, "namespaces:\n  research: {tools: [scrape]}\n")
    before = dumped(library)
    with pytest.raises(LibraryError, match="Tool scrape doesn't build: build in"):
        import_into(library, granting)
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("config", "note"),
    [
        (
            "x-defaults: &defaults {spawn: []}\nnamespaces:\n  helper: *defaults\n",
            "Left out: x-defaults, which is not a setting the engine reads.",
        ),
        (
            "max_iters: 3\nmax_depth: 4\n",
            (
                "Left out: max_iters, which is not a setting the engine reads (did you mean"
                " max_iter?)."
            ),
        ),
    ],
    ids=["an anchor block", "a typo"],
)
def test_a_key_the_engine_doesnt_read_is_left_out_and_said_so(
    library, tmp_path, config, note
):
    written = config_file(tmp_path, config)
    with closing(open_library(library)) as connection:
        said = import_config(connection, written)
        settings = stored_config(connection)
    assert note in said["left_out"]
    assert not {"x-defaults", "max_iters"} & settings.keys()


def test_a_nan_the_library_holds_reads_as_unchanged(library, tmp_path):
    config = config_file(tmp_path, "namespaces:\n  helper: {vars: {limit: .nan}}\n")
    assert import_into(library, config) == ["Namespaces:", "  helper: added"]
    assert import_into(library, config) == []


def test_a_row_that_isnt_yaml_is_left_out_with_a_warning_and_an_import_can_mend_it(
    library, tmp_path
):
    with closing(open_library(library)) as connection:
        connection.execute(UPSERT_NAMESPACE, ("helper", "vars: [unclosed\n"))
    config = config_file(tmp_path, "namespaces:\n  helper: {vars: {a: 1}}\n")
    with capture_logs() as logs:
        assert import_into(library, config) == ["Namespaces:", "  helper: added"]
    left_out = {
        (entry["table"], entry["row"])
        for entry in logs
        if entry["event"] == "library.row_left_out"
    }
    assert left_out == {("namespaces", repr("helper"))}
    assert run_config(library, tmp_path).namespaces["helper"].vars == {"a": 1}


def test_a_tool_no_namespace_grants_is_not_built_and_the_import_says_so(
    library, tmp_path
):
    (tmp_path / "mine.py").write_text(RAISES_ON_BUILD)
    config = config_file(
        tmp_path, "tools:\n  mine: {factory: build, factory_from: mine.py}\n"
    )
    assert import_into(library, config) == [
        "Tools:",
        "  mine: added; not built: no namespace grants it",
    ]


def test_an_import_of_what_the_library_holds_changes_nothing_and_keeps_its_text(
    library, tmp_path
):
    change(library, add_namespace, "helper", "# mine\nvars: {a: 1}\n")
    config = config_file(
        tmp_path, "max_iter: 30\nnamespaces:\n  helper: {vars: {a: 1}}\n"
    )
    before = dumped(library)
    assert import_into(library, config) == []
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("root", "said"),
    [
        (
            "{system_suffix: Be brief., spawn: [helper]}",
            [
                (
                    "Every namespace that sets none of its own gets root's system_suffix"
                    " and spawn, historian and librarian too."
                )
            ],
        ),
        (
            "{reasoner: {type: claude_code}}",
            [
                (
                    "Every namespace that sets none of its own gets root's reasoner,"
                    " historian and librarian too."
                ),
                (
                    "Root's reasoner is claude_code, which runs each of them on Claude Code,"
                    " billed to your subscription."
                ),
            ],
        ),
    ],
    ids=["fields", "a reasoner that bills"],
)
def test_import_says_what_of_root_reaches_the_apps_own_namespaces_too(
    library, tmp_path, root, said
):
    config = config_file(tmp_path, f"namespaces:\n  root: {root}\n  helper: {{}}\n")
    assert import_into(library, config)[-len(said) :] == said


@pytest.mark.parametrize("name", ["llm", "safe_url", "subagent"])
def test_import_refuses_a_tool_of_your_own_named_as_a_built_in(library, tmp_path, name):
    (tmp_path / "mine.py").write_text("def build(client, params): ...\n")
    config = config_file(
        tmp_path, f"tools:\n  {name}: {{factory: build, factory_from: mine.py}}\n"
    )
    with pytest.raises(
        LibraryError, match=f'Tool "{name}" has a name a built-in, the app or the REPL'
    ):
        import_into(library, config)


def test_import_gives_a_built_in_its_settings_under_its_own_name(library, tmp_path):
    config = config_file(
        tmp_path,
        "tools:\n  safe_url: {call_in: [helper]}\n"
        "namespaces:\n  helper: {tools: [safe_url]}\n",
    )
    assert "  safe_url: added" in import_into(library, config)
    assert run_config(library, tmp_path).tools["safe_url"] == {"call_in": ["helper"]}


@pytest.mark.parametrize(
    "config",
    [
        f"tools:\n  keyed: {{factory: build, factory_from: mine.py, token: {FAKE_KEY}}}\n",
        'namespaces:\n  helper: {vars: {token: "\\x73' + FAKE_KEY[1:] + '"}}\n',
        f"system_prompt: Use {FAKE_KEY}.\n",
        # FAKE_KEY in base64.
        "tools:\n  keyed: {factory: llm, token: !!binary c2stdGVzdC1ub3QtYS1yZWFsLWtleQ==}\n",
    ],
    ids=[
        "a tool's settings",
        "a namespace's vars, escaped",
        "the system prompt",
        "a tool's settings as !!binary",
    ],
)
def test_import_refuses_a_providers_key(library, tmp_path, config):
    (tmp_path / "mine.py").write_text("def build(client, params): ...\n")
    written = config_file(tmp_path, config)
    open_library(library).close()
    before = dumped(library)
    with (
        closing(open_library(library)) as connection,
        pytest.raises(LibraryError, match="holds a key"),
    ):
        import_config(connection, written, secrets=[FAKE_KEY])
    assert dumped(library) == before


def test_an_import_with_a_reasoner_the_engine_doesnt_know_is_refused(library, tmp_path):
    config = config_file(
        tmp_path,
        "namespaces:\n  odd:\n    reasoner: {type: chatgpt}\n",
    )
    before = dumped(library)
    with pytest.raises(LibraryError, match="namespace 'odd': unknown reasoner type"):
        import_into(library, config)
    assert dumped(library) == before


# A worked example's last turn, as the panel saves one.
ANSWERS = (
    '{role: assistant, content: "<think>x</think>\\n<repl>\\nFinalAnswer(1)\\n</repl>"}'
)


def refused_unchanged(library, config, said):
    """Import *config*, which must be refused saying *said*, leaving *library* as it was."""
    open_library(library).close()
    before = dumped(library)
    with pytest.raises(LibraryError) as refused:
        import_into(library, config)
    assert said in str(refused.value)
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("asked", "problem"),
    [
        ("Do {{ nothere }}", 'this one names "nothere"; a template may name only task'),
        ("Do {{ task ", "this one doesn't parse (unexpected end of template"),
    ],
    ids=["a name the engine doesn't fill in", "one that doesn't parse"],
)
def test_import_refuses_a_worked_example_whose_template_fails(
    library, tmp_path, asked, problem
):
    config = config_file(
        tmp_path,
        "namespaces:\n  helper:\n    decompositions:\n      - name: a\n        messages:\n"
        f"          - {{role: user, content: '{asked}'}}\n          - {ANSWERS}\n",
    )
    said = (
        'Namespace helper, decomposition "a": Turn 1 (user), line 1: the engine reads'
        f" text holding {{{{ or {{% as a Jinja template, and {problem}"
    )
    refused_unchanged(library, config, said)


@pytest.mark.parametrize(
    ("variables", "problem"),
    [
        ("{max: 1}", 'Var "max" would hide Python\'s own max'),
        ("{class: 1}", 'Var "class" is a Python keyword'),
        ("{'my var': 1}", 'Var "my var" needs a Python name'),
        ("{_x: 1}", 'Var "_x" starts with _, which templates may not read'),
    ],
)
def test_import_refuses_a_var_the_panel_refuses(library, tmp_path, variables, problem):
    config = config_file(tmp_path, f"namespaces:\n  helper: {{vars: {variables}}}\n")
    refused_unchanged(library, config, f"Namespace helper: {problem}")


def test_import_refuses_a_var_a_tool_it_sees_takes_the_place_of(library, tmp_path):
    config = config_file(tmp_path, "namespaces:\n  helper: {vars: {llm: 1}}\n")
    said = 'In helper, var "llm" (set in helper) has the name of the tool llm (granted in root)'
    refused_unchanged(library, config, said)


SHAPE = (
    "A namespace name is words of letters, digits, _ and -, each starting with a letter"
)
READS_AS_ROOT = "reads as root, which every namespace goes under already"


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("''", 'Namespace "": Name the namespace'),
        ('"a\\nb"', f'Namespace "a\\nb": {SHAPE}'),
        ('"a\\u200bb"', f'Namespace "a\\u200bb": {SHAPE}'),
        ("a/b", f'Namespace "a/b": {SHAPE}'),
        ('"a\\U0001F600"', f'Namespace "a\U0001f600": {SHAPE}'),
        ("Root", f'Namespace Root: "Root" {READS_AS_ROOT}'),
        ("root.child", f'Namespace root.child: "root.child" {READS_AS_ROOT}'),
        ("-dash", f'Namespace "-dash": {SHAPE}'),
        ("1abc", f'Namespace "1abc": {SHAPE}'),
    ],
    ids=[
        "empty",
        "newline",
        "zero width space",
        "slash",
        "emoji",
        "Root",
        "under root",
        "dash first",
        "digit first",
    ],
)
def test_import_refuses_a_namespace_name_the_panel_refuses(
    library, tmp_path, name, said
):
    config = config_file(tmp_path, f"namespaces:\n  {name}: {{vars: {{year: 1}}}}\n")
    refused_unchanged(library, config, said)


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("' a '", 'decomposition " a ": The name " a " starts or ends with a space'),
        ('"a\\u200b"', "The name holds a character you can't see (U+200B)"),
    ],
)
def test_import_refuses_a_decomposition_name_that_doesnt_read_as_it_looks(
    library, tmp_path, name, said
):
    config = config_file(
        tmp_path,
        "namespaces:\n  helper:\n    decompositions:\n"
        f"      - {{name: {name}, messages: [{{role: user, content: hi}}]}}\n",
    )
    refused_unchanged(library, config, said)


def test_import_refuses_two_decompositions_of_one_name_in_a_namespace(
    library, tmp_path
):
    config = config_file(
        tmp_path,
        "namespaces:\n  helper:\n    decompositions:\n"
        "      - {name: same, messages: [{role: user, content: hi}]}\n"
        "      - {name: same, messages: [{role: user, content: ho}]}\n",
    )
    said = 'Namespace helper holds more than one decomposition named "same": give each its own name.'
    refused_unchanged(library, config, said)


# Past MAX_YAML_CHARS in bytes, though not in characters: é is two bytes as UTF-8.
WIDE = "é" * (MAX_YAML_CHARS // 2 + 1)
TOO_BIG = "may be at most 1,000,000 bytes as UTF-8"


@pytest.mark.parametrize(
    ("config", "where"),
    [
        (
            f"namespaces:\n  helper: {{system_suffix: '{WIDE}'}}\n",
            "Namespace helper: Its YAML",
        ),
        (
            (
                "namespaces:\n  helper:\n    decompositions:\n"
                f"      - {{name: a, messages: [{{role: user, content: '{WIDE}'}}]}}\n"
            ),
            'Namespace helper, decomposition "a": Its YAML',
        ),
        (
            f"prompt_template_variables: {{dept: '{WIDE}'}}\n",
            "Setting prompt_template_variables: Its YAML",
        ),
        (f"system_prompt: '{WIDE}'\n", "Setting system_prompt: Its YAML"),
        (
            f"tools:\n  mine: {{factory: build, factory_from: mine.py, note: '{WIDE}'}}\n",
            "Tool mine: Its YAML",
        ),
        (
            "tools:\n  mine: {factory: build, factory_from: wide.py}\n",
            "Tool mine: Its source",
        ),
    ],
    ids=[
        "a namespace",
        "a decomposition",
        "a setting",
        "the system prompt",
        "a tool's block",
        "a tool's source",
    ],
)
def test_import_refuses_what_passes_the_librarys_size_limit_in_bytes(
    library, tmp_path, config, where
):
    (tmp_path / "mine.py").write_text("def build(client, params): ...\n")
    (tmp_path / "wide.py").write_text(f"def build(client, params): ...\n# {WIDE}\n")
    refused_unchanged(library, config_file(tmp_path, config), f"{where} {TOO_BIG}")


# What conflit says of a _compose entry it can't read.
ENTRIES = "_compose entries must be a file path string or a {namespace: path} mapping"


# Nine aliases deep, each a list of ten of the one before: 10**9 strings, written out.
ALIAS_BOMB = (
    "a: &a [x, x, x, x, x, x, x, x, x, x]\n"
    + "".join(
        f"{name}: &{name} [{', '.join([f'*{before}'] * 10)}]\n"
        for before, name in zip("abcdefgh", "bcdefghi")
    )
    + "max_iter: 3\n"
)


@pytest.mark.parametrize(
    "composed", [False, True], ids=["itself", "in a file it composes"]
)
def test_import_refuses_an_alias_bomb_without_writing_it_out(
    library, tmp_path, composed
):
    (tmp_path / "bomb.yaml").write_text(ALIAS_BOMB)
    config = (
        config_file(tmp_path, "_compose: [bomb.yaml]\n")
        if composed
        else tmp_path / "bomb.yaml"
    )
    said = "is too big to import: with each alias written out, as the engine reads it"
    started = time.monotonic()
    refused_unchanged(library, config, said)
    assert time.monotonic() - started < BOMB_SECONDS


@pytest.mark.parametrize(
    "setting",
    ["max_iter: 0", "max_iter: -3", "max_depth: 0"],
)
def test_import_refuses_a_run_setting_no_run_takes_a_step_with(
    library, tmp_path, setting
):
    key, value = setting.split(": ")
    said = f"{key} is {value}, but a run needs at least 1 to take a step: make it 1 or more"
    refused_unchanged(library, config_file(tmp_path, f"{setting}\n"), said)


@pytest.mark.parametrize(
    ("composed", "problem"),
    [
        ("5", "_compose must be a list."),
        ("{a: b}", "_compose must be a list."),
        ("[5]", f"{ENTRIES}, got 'int'."),
        ("[null]", f"{ENTRIES}, got 'NoneType'."),
        ("[[a]]", f"{ENTRIES}, got 'list'."),
    ],
)
def test_import_words_a_compose_the_engine_cant_read(
    library, tmp_path, composed, problem
):
    config = config_file(tmp_path, f"_compose: {composed}\n")
    said = f"{config} is not a config the engine can read: {problem}"
    refused_unchanged(library, config, said)


def test_import_blames_a_file_it_composes_that_is_not_a_config(library, tmp_path):
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    config = config_file(tmp_path, "_compose: [list.yaml]\n")
    said = f"{tmp_path / 'list.yaml'} {NOT_A_MAPPING}"
    refused_unchanged(library, config, said)


def test_a_tool_that_doesnt_build_is_blamed_on_the_configs_own_file(library, tmp_path):
    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers" / "scraping.py").write_text(RAISES_ON_BUILD)
    config = config_file(
        tmp_path,
        "tools:\n  scrape: {factory: build, factory_from: helpers/scraping.py}\n"
        "namespaces:\n  helper: {tools: [scrape]}\n",
    )
    said = "Tool scrape doesn't build: build in helpers/scraping.py raised RuntimeError"
    refused_unchanged(library, config, said)


def test_import_refuses_a_spawn_naming_a_namespace_it_doesnt_bring(library, tmp_path):
    config = config_file(tmp_path, "namespaces:\n  helper: {spawn: [nowhere]}\n")
    said = 'Namespace helper: spawn names "nowhere", which is not a namespace in the Library'
    refused_unchanged(library, config, said)
