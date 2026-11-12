import json
import stat
from contextlib import closing
from dataclasses import replace

import pytest
from fake_model import FakeOpenAI, scripted
from structlog.testing import capture_logs

from deep_reasoning import settings as settings_module
from deep_reasoning.library import open_library
from deep_reasoning.settings import (
    ANTHROPIC_MODELS,
    DEFAULT_FOLDER,
    ENV_FILE,
    GEMINI_MODELS,
    LAST_MODEL,
    LIBRARIAN_MODEL,
    NO_KEY,
    NOT_A_KEY,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
    SERVERS,
    VENDORS,
    Provider,
    SettingsError,
    add_server,
    default_folder,
    drop_key,
    find_provider,
    is_chat_model,
    key_of,
    key_status,
    keyless_environment,
    list_models,
    provider_list,
    providers,
    read_settings,
    remember_pick,
    remove_server,
    run_key,
    save_default_folder,
    save_key,
    saved_keys,
    server_key_env,
    starting_model,
    update_server,
)

pytestmark = pytest.mark.usefixtures("no_vendor_keys")

OPENAI = VENDORS[0]
KEY = "sk-saved-in-settings"
SHELL_KEY = "sk-from-the-shell"
LOCAL_URL = "http://127.0.0.1:8000/v1"
MOVED_URL = "http://127.0.0.1:9000/v1"
OPENAI_LIVE_LIST = ("gpt-4", "gpt-6-luna", "gpt-image-2", "text-embedding-3-small")
THEIRS = """\
# my own settings
export EDITOR=vim
OPENAI_API_KEY='sk-older'
QUOTED="two words"
"""


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def plant(library, key, text):
    """Write a setting's JSON straight into the file, as one from someone else may hold it."""
    with closing(open_library(library)) as connection:
        connection.execute("INSERT INTO app_settings VALUES (?, ?)", (key, text))


def test_a_key_saved_in_a_new_env_file_is_readable_by_its_owner_alone(app_home):
    save_key(app_home, OPENAI, f"  {KEY}\n")
    assert mode(app_home / ENV_FILE) == 0o600
    assert (app_home / ENV_FILE).read_text() == f"OPENAI_API_KEY='{KEY}'\n"
    assert saved_keys(app_home) == {"OPENAI_API_KEY": KEY}


def test_saving_a_key_keeps_the_files_other_lines_and_narrows_who_can_read_it(
    app_home,
):
    app_home.mkdir(parents=True)
    env = app_home / ENV_FILE
    env.write_text(THEIRS)
    env.chmod(0o644)
    save_key(app_home, OPENAI, KEY)
    assert env.read_text() == THEIRS.replace("sk-older", KEY)
    assert mode(env) == 0o600
    assert list(app_home.iterdir()) == [env]


def test_a_saved_key_beats_the_shells(app_home, monkeypatch):
    monkeypatch.setenv(OPENAI.key_env, SHELL_KEY)
    assert key_status(OPENAI, saved_keys(app_home)) == "shell"
    save_key(app_home, OPENAI, KEY)
    saved = saved_keys(app_home)
    assert key_status(OPENAI, saved) == "set"
    assert key_of(OPENAI, saved) == KEY == run_key(app_home, OPENAI)


def test_a_shell_key_with_a_stray_line_ending_runs_without_it(app_home, monkeypatch):
    monkeypatch.setenv(OPENAI.key_env, f"{SHELL_KEY}\r\n")
    assert run_key(app_home, OPENAI) == SHELL_KEY


def test_saving_a_key_through_a_symlinked_env_file_leaves_its_target_alone(
    app_home, tmp_path
):
    app_home.mkdir(parents=True)
    target = tmp_path / "dotfiles.env"
    target.write_text(THEIRS)
    target.chmod(0o644)
    (app_home / ENV_FILE).symlink_to(target)
    save_key(app_home, OPENAI, KEY)
    env = app_home / ENV_FILE
    assert not env.is_symlink() and mode(env) == 0o600
    assert env.read_text() == THEIRS.replace("sk-older", KEY)
    assert target.read_text() == THEIRS and mode(target) == 0o644


def test_dropping_a_key_keeps_the_other_lines_and_lets_the_shells_show_again(
    app_home, monkeypatch
):
    app_home.mkdir(parents=True)
    (app_home / ENV_FILE).write_text(THEIRS)
    monkeypatch.setenv(OPENAI.key_env, SHELL_KEY)
    drop_key(app_home, OPENAI.key_env)
    assert (app_home / ENV_FILE).read_text() == THEIRS.replace(
        "OPENAI_API_KEY='sk-older'\n", ""
    )
    assert key_status(OPENAI, saved_keys(app_home)) == "shell"
    drop_key(app_home, OPENAI.key_env)


def test_the_provider_list_says_where_each_key_is_and_never_what_it_is(
    app_home, library, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", SHELL_KEY)
    save_key(app_home, OPENAI, KEY)
    listed = provider_list(app_home, library)
    assert {each["name"]: each["key"] for each in listed} == {
        "OpenAI": "set",
        "OpenRouter": "shell",
        "Anthropic": "none",
        "Google Gemini": "none",
    }
    assert KEY not in json.dumps(listed) and SHELL_KEY not in json.dumps(listed)


def test_anthropics_key_is_kept_apart_from_the_one_claude_code_bills():
    anthropic = next(each for each in VENDORS if each.name == "Anthropic")
    assert anthropic.key_env == "DEEP_REASONING_ANTHROPIC_KEY"
    assert "ANTHROPIC_API_KEY" not in {each.key_env for each in VENDORS}


@pytest.mark.parametrize("key", ["sk two words", "sk-one\nOTHER=line", "", "  "])
def test_a_key_that_is_not_one_plain_line_is_refused_and_nothing_is_written(
    app_home, key
):
    with pytest.raises(SettingsError, match="paste the OpenAI key as one line"):
        save_key(app_home, OPENAI, key)
    assert not (app_home / ENV_FILE).exists()


def test_a_custom_server_is_added_changed_and_removed_with_its_key(app_home, library):
    add_server(app_home, library, " LM Studio ", f"{LOCAL_URL} ", "sk-local")
    server = find_provider(library, "LM Studio")
    assert (server.base_url, server.key_env, server.custom) == (
        LOCAL_URL,
        "DEEP_REASONING_LM_STUDIO_KEY",
        True,
    )
    url_env = "DEEP_REASONING_LM_STUDIO_URL"
    assert saved_keys(app_home) == {url_env: LOCAL_URL, server.key_env: "sk-local"}
    update_server(app_home, library, server, MOVED_URL, None)
    moved = find_provider(library, "LM Studio")
    assert moved.base_url == MOVED_URL
    assert saved_keys(app_home) == {url_env: MOVED_URL, server.key_env: "sk-local"}
    assert run_key(app_home, moved) == "sk-local"
    update_server(app_home, library, moved, MOVED_URL, "sk-newer")
    assert saved_keys(app_home) == {url_env: MOVED_URL, server.key_env: "sk-newer"}
    remove_server(app_home, library, moved)
    assert find_provider(library, "LM Studio") is None
    assert saved_keys(app_home) == {}


def test_a_custom_server_with_no_key_runs_on_a_placeholder(app_home, library):
    add_server(app_home, library, "vllm", LOCAL_URL, None)
    assert run_key(app_home, find_provider(library, "vllm")) == NO_KEY
    assert saved_keys(app_home) == {"DEEP_REASONING_VLLM_URL": LOCAL_URL}


def test_a_tool_is_built_keyless_with_only_the_key_variables_every_run_has(
    app_home, library, monkeypatch
):
    monkeypatch.setenv(OPENAI.key_env, SHELL_KEY)
    openrouter, anthropic, gemini = VENDORS[1:]
    save_key(app_home, openrouter, KEY)
    add_server(app_home, library, "vllm", LOCAL_URL, None)
    built_in = keyless_environment(library, app_home)
    assert built_in[OPENAI.key_env] == NOT_A_KEY
    # A saved key, or a custom server's placeholder, reaches only its own provider's
    # chats: a tool reading one would stop the others.
    others = {openrouter.key_env, server_key_env("vllm"), anthropic.key_env}
    assert not built_in.keys() & (others | {gemini.key_env})


def test_a_shell_key_for_a_custom_server_is_used_only_once_saved_for_its_address(
    app_home, library, tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEP_REASONING_VLLM_KEY", SHELL_KEY)
    plant(library, SERVERS, json.dumps([{"name": "vllm", "base_url": LOCAL_URL}]))
    planted = find_provider(library, "vllm")
    unbound = (key_status(planted, saved_keys(app_home)), run_key(app_home, planted))
    add_server(app_home, tmp_path / "mine.sqlite", "vllm", LOCAL_URL, None)
    saved_here = (key_status(planted, saved_keys(app_home)), run_key(app_home, planted))
    assert unbound == ("none", NO_KEY)
    assert saved_here == ("shell", SHELL_KEY)


def test_a_server_added_under_a_name_bound_elsewhere_gets_its_key_only_when_given_one(
    app_home, library, tmp_path
):
    add_server(app_home, library, "vllm", LOCAL_URL, KEY)
    other = tmp_path / "other.sqlite"
    add_server(app_home, other, "vllm", MOVED_URL, None)
    blank = key_status(find_provider(other, "vllm"), saved_keys(app_home))
    mine = key_status(find_provider(library, "vllm"), saved_keys(app_home))
    assert (blank, mine) == ("none", "set")


def test_a_vendor_with_no_key_cannot_run(app_home):
    with pytest.raises(SettingsError, match="OpenAI has no key: add one in settings"):
        run_key(app_home, OPENAI)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("", "Name the server with letters"),
        ("vllm/../x", "Name the server with letters"),
        ("x" * 41, "Name the server with letters"),
        ("lm studio", r"There is a provider named lm studio already\.$"),
        ("openai", "named OpenAI already, and names differing only in case count as"),
        # its key would be kept under Anthropic's variable
        (
            "anthropic.",
            "Anthropic already, and anthropic. would keep its key under the",
        ),
        ("lm-studio", "same variable, DEEP_REASONING_LM_STUDIO_KEY: pick a name that"),
    ],
)
def test_a_server_name_that_is_bad_or_taken_is_refused(
    app_home, library, name, problem
):
    add_server(app_home, library, "lm studio", LOCAL_URL, None)
    with pytest.raises(SettingsError, match=problem):
        add_server(app_home, library, name, LOCAL_URL, None)
    assert [each.name for each in settings_module.servers(library)] == ["lm studio"]


@pytest.mark.parametrize(
    "url",
    [
        "localhost:8000/v1",
        "ftp://example.com/v1",
        "http:///v1",
        "http://example.com/v1?x=1",
        "http://example.com/v1#top",
        "http://exa mple.com/v1",
        "http://[::1/v1",
    ],
)
def test_a_base_url_that_is_not_a_plain_web_address_is_refused(app_home, library, url):
    with pytest.raises(SettingsError, match="should be an http:// or https:// address"):
        add_server(app_home, library, "vllm", url, None)
    assert read_settings(library) == {}


def test_a_base_url_carrying_a_password_is_refused(app_home, library):
    with pytest.raises(SettingsError, match="Put the key in the key field"):
        add_server(
            app_home, library, "vllm", "http://me:sk-secret@example.com/v1", None
        )
    assert read_settings(library) == {}


def test_server_key_variables_are_derived_from_their_names():
    assert server_key_env("vllm") == "DEEP_REASONING_VLLM_KEY"
    assert server_key_env("LM Studio 2.0") == "DEEP_REASONING_LM_STUDIO_2_0_KEY"


@pytest.mark.parametrize(
    ("key", "text"),
    [
        (SERVERS, "{not json"),
        (SERVERS, '"x"'),
        (LAST_MODEL, "1"),
        (LAST_MODEL, '{"provider": "vllm"}'),
        (DEFAULT_FOLDER, '["/home/me"]'),
        (DEFAULT_FOLDER, '"relative/folder"'),
        (DEFAULT_FOLDER, json.dumps("C:\\Users\\me")),
        (DEFAULT_FOLDER, json.dumps("/home/me/caf\udce9")),
        (LAST_MODEL, json.dumps({"provider": "vllm", "name": "qwen\ud800"})),
    ],
)
def test_a_setting_this_app_could_not_have_written_reads_as_unset(library, key, text):
    plant(library, key, text)
    with capture_logs() as logs:
        settings = read_settings(library)
    assert settings == {}
    assert [entry["event"] for entry in logs] == ["settings.left_out"]


@pytest.mark.parametrize(
    "planted",
    [
        pytest.param(1, id="not a mapping"),
        pytest.param({"name": "lone"}, id="no base URL"),
        pytest.param(
            {"name": "a/b", "base_url": LOCAL_URL}, id="a name add_server refuses"
        ),
        pytest.param(
            {"name": "lm ", "base_url": LOCAL_URL}, id="a space after its name"
        ),
        pytest.param({"name": "ftp", "base_url": "ftp://x.example/v1"}, id="a bad URL"),
        pytest.param({"name": "OpenAI", "base_url": LOCAL_URL}, id="a vendor's name"),
        pytest.param({"name": "anthropic", "base_url": LOCAL_URL}, id="a vendor's key"),
        pytest.param(
            {"name": "VLLM", "base_url": "http://x.example/v1"}, id="a server's name"
        ),
    ],
)
def test_a_server_entry_add_server_could_not_have_written_is_left_out(library, planted):
    plant(
        library, SERVERS, json.dumps([{"name": "vllm", "base_url": LOCAL_URL}, planted])
    )
    with capture_logs() as logs:
        kept = settings_module.servers(library)
    assert [(each.name, each.base_url) for each in kept] == [("vllm", LOCAL_URL)]
    assert [entry["event"] for entry in logs] == ["settings.server_left_out"]


def test_new_chats_start_on_gpt_6_luna_when_openai_has_a_key_and_nothing_was_picked(
    app_home, library, monkeypatch
):
    assert starting_model(app_home, library, providers(library)) is None
    monkeypatch.setenv(OPENAI.key_env, SHELL_KEY)
    luna = {"provider": "OpenAI", "name": "gpt-6-luna"}
    assert starting_model(app_home, library, providers(library)) == luna


def test_new_chats_start_on_the_model_picked_last_while_its_provider_can_answer(
    app_home, library, monkeypatch
):
    monkeypatch.setenv(OPENAI.key_env, SHELL_KEY)
    add_server(app_home, library, "vllm", LOCAL_URL, None)
    remember_pick(library, "vllm", "qwen3")
    assert starting_model(app_home, library, providers(library)) == {
        "provider": "vllm",
        "name": "qwen3",
    }
    remove_server(app_home, library, find_provider(library, "vllm"))
    luna = {"provider": "OpenAI", "name": "gpt-6-luna"}
    assert starting_model(app_home, library, providers(library)) == luna


def test_the_default_folder_is_kept_beside_the_other_settings(
    app_home, library, tmp_path
):
    assert default_folder(library) is None
    add_server(app_home, library, "vllm", LOCAL_URL, None)
    save_default_folder(library, str(tmp_path / "code"))
    assert default_folder(library) == str(tmp_path / "code")
    save_default_folder(library, None)
    assert default_folder(library) is None
    assert find_provider(library, "vllm") is not None


@pytest.mark.parametrize("vendor", VENDORS, ids=lambda vendor: vendor.name)
def test_each_vendor_features_the_model_new_chats_start_on(vendor):
    assert vendor.first_model is None or vendor.first_model in vendor.featured


def test_librarian_chats_start_on_a_model_their_vendor_features():
    vendor = next(each for each in VENDORS if each.name == LIBRARIAN_MODEL["provider"])
    assert LIBRARIAN_MODEL["name"] in vendor.featured


def test_openrouter_features_the_vendors_models_under_its_own_ids():
    vendors = OPENAI_MODELS + ANTHROPIC_MODELS + GEMINI_MODELS
    theirs = {name.split("/", 1)[1] for name in OPENROUTER_MODELS}
    assert {name.replace(".", "-") for name in vendors} == {
        name.replace(".", "-") for name in theirs
    }


@pytest.mark.anyio
async def test_a_vendors_listing_features_its_current_tiers_beside_its_whole_list():
    with FakeOpenAI(scripted({}), models=OPENAI_LIVE_LIST) as fake:
        listing = await list_models(replace(OPENAI, base_url=fake.base_url), KEY)
    assert listing["featured"] == list(OPENAI_MODELS)
    assert listing["models"] == ["gpt-4", "gpt-6-luna"]


@pytest.mark.parametrize(
    ("name", "chat"),
    [
        ("gpt-6-luna", True),
        ("gpt-4o-search-preview", True),
        ("o3-mini", True),
        ("claude-sonnet-5-5", True),
        ("models/gemini-3.8-flash", True),
        ("openai/gpt-6-luna", True),
        ("qwen3-32b", True),
        ("deliverable-7b", True),
        ("text-embedding-3-small", False),
        ("gemini-embedding-001", False),
        ("tts-1-hd", False),
        ("gpt-4o-mini-tts", False),
        ("gpt-4o-transcribe", False),
        ("whisper-1", False),
        ("gpt-realtime", False),
        ("gpt-4o-audio-preview", False),
        ("gpt-image-2", False),
        ("dall-e-3", False),
        ("imagen-4.0-generate-001", False),
        ("veo-3.0-generate-preview", False),
        ("omni-moderation-latest", False),
        ("gemini-2.0-flash-live-001", False),
        ("davinci-002", False),
        ("computer-use-preview", False),
    ],
)
def test_a_providers_list_keeps_its_chat_models_only(name, chat):
    assert is_chat_model(name) is chat


@pytest.mark.anyio
async def test_a_custom_server_features_only_the_chat_models_it_lists():
    with FakeOpenAI(scripted({}), models=("qwen3-32b", "bge-embedding-m3")) as fake:
        server = Provider("vllm", fake.base_url, server_key_env("vllm"), custom=True)
        listing = await list_models(server, None)
    assert listing["featured"] == listing["models"] == ["qwen3-32b"]


@pytest.mark.anyio
async def test_a_vendor_whose_list_fails_still_features_its_current_tiers():
    with FakeOpenAI(scripted({}), models=None) as fake:
        listing = await list_models(replace(OPENAI, base_url=fake.base_url), KEY)
    assert listing == {
        "provider": "OpenAI",
        "models": [],
        "problem": f"No list: {fake.base_url}/models answered 404 Not Found.",
        "featured": list(OPENAI_MODELS),
    }


@pytest.mark.anyio
async def test_a_custom_servers_listing_features_its_whole_list():
    with FakeOpenAI(scripted({}), models=("qwen3-32b", "qwen3-8b")) as fake:
        server = Provider("vllm", fake.base_url, server_key_env("vllm"), custom=True)
        listing = await list_models(server, None)
    assert listing["featured"] == listing["models"] == ["qwen3-32b", "qwen3-8b"]
