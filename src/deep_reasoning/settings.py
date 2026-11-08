"""Settings: the providers a chat's model runs on, their keys, the custom servers, and the
folder new chats work in.

Keys live in .env in the app's home, readable only by you: never in the app file, a chat,
a run's folder, an event, a log or a response. A key saved there wins over the same
variable in the shell. The custom servers, the model picked last and the default folder,
none of it secret, live in the app file's app_settings table, so a copy of the file
carries them. A custom server's key, saved or from the shell, goes only to the address
recorded beside it in .env when the server was saved here: a shared file's server may
borrow the name of one of yours.
"""

import asyncio
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import structlog
from dotenv import dotenv_values, set_key, unset_key

from deep_reasoning.library import open_library, writing
from deep_reasoning.runs import WITHHELD, secrets_of, worker_env

log = structlog.get_logger()

ENV_FILE = ".env"
UPSERT_APP_SETTING = (
    "INSERT INTO app_settings VALUES (?, ?)"
    " ON CONFLICT (key) DO UPDATE SET json = excluded.json"
)
SERVERS, LAST_MODEL, DEFAULT_FOLDER = "servers", "last_model", "default_folder"
MODEL_PARTS = ("provider", "name")
SERVER_FIELDS = ("name", "base_url")
# What this app writes under each setting; each server entry is checked as it is read.
SHAPES = {
    SERVERS: lambda value: isinstance(value, list),
    LAST_MODEL: lambda value: (
        isinstance(value, dict)
        and all(isinstance(value.get(part), str) for part in MODEL_PARTS)
    ),
    DEFAULT_FOLDER: lambda value: (
        value is None or (isinstance(value, str) and os.path.isabs(value))
    ),
}
OWNER_ONLY = 0o600
# One line of printable ASCII, no spaces: every key and base URL is, and nothing else
# can then break a line of .env.
PLAIN_TEXT = re.compile(r"[!-~]{1,4096}")
SERVER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}")
EXAMPLE_NAME = "vllm"
EXAMPLE_URL = "http://localhost:8000/v1"
WEB_SCHEMES = ("http", "https")
# A server without a key still needs one for the OpenAI client: the placeholder local
# servers document.
NO_KEY = "EMPTY"
LIST_TIMEOUT_S = 10.0
# What a provider's whole list holds beside its chat models: speech, images, video,
# embeddings, moderation, realtime and live audio, and completions-only models.
NOT_CHAT = re.compile(
    r"(?:^|[-_./:])(?:tts|transcribe|realtime|audio|speech|whisper|image|imagen|dall-e"
    r"|embed|embedding|embeddings|moderation|veo|sora|live|babbage|davinci|aqa"
    r"|computer-use)(?=$|[-_./:\d])",
    re.IGNORECASE,
)
SET, SHELL, NONE = "set", "shell", "none"


class SettingsError(ValueError):
    """Something settings refused, in words for the person who asked."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    custom: bool = False
    # what a new chat starts on when no chat has picked a model yet
    first_model: str | None = None
    # the models the menu shows before anything is typed, strongest first
    featured: tuple[str, ...] = ()


# Each vendor's current tiers, strongest first, from its own models page on 2026-10-27:
# refresh them when a vendor ships a new tier. OpenRouter serves the same under its own ids.
OPENAI_MODELS = ("gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna")
ANTHROPIC_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)
GEMINI_MODELS = ("gemini-4-argon", "gemini-3.8-flash", "gemini-3.5-flash-lite")
OPENROUTER_MODELS = (
    "openai/gpt-6-astra",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-luna",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-haiku-5.5",
    "google/gemini-4-argon",
    "google/gemini-3.8-flash",
    "google/gemini-3.5-flash-lite",
)

# The app knows these vendors' addresses, each from the vendor's own docs: they take only a key.
VENDORS = (
    Provider(
        "OpenAI",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        first_model="gpt-6-luna",
        featured=OPENAI_MODELS,
    ),
    Provider(
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        featured=OPENROUTER_MODELS,
    ),
    # never ANTHROPIC_API_KEY: with it set, Claude Code bills that key instead of the subscription
    Provider(
        "Anthropic",
        "https://api.anthropic.com/v1/",
        "DEEP_REASONING_ANTHROPIC_KEY",
        featured=ANTHROPIC_MODELS,
    ),
    Provider(
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        featured=GEMINI_MODELS,
    ),
)


# A custom server's key, and the address it is for, as .env or the shell holds them.
SERVER_KEY_VARIABLE = re.compile(r"DEEP_REASONING_[A-Z0-9_]+_KEY")
SERVER_URL_VARIABLE = re.compile(r"DEEP_REASONING_[A-Z0-9_]+_URL")
# Claude Code's: an API key it would bill in place of the subscription, and the
# subscription's own token.
CLAUDE_CODES_KEYS = (WITHHELD, "CLAUDE_CODE_OAUTH_TOKEN")
# What a provider's key variable holds where a tool is built to check it: no key.
NOT_A_KEY = "not-a-key"


def saved_keys(home: Path) -> dict[str, str]:
    """What the app's .env holds, by variable name: the keys, and the address each custom
    server's key is for."""
    path = home / ENV_FILE
    if not path.is_file():
        return {}
    values = dotenv_values(path, interpolate=False)
    return {name: value for name, value in values.items() if value}


def bound(provider: Provider, saved: dict[str, str]) -> bool:
    """Whether the provider's key is meant for its address: a vendor's always; a custom
    server's only when .env records its base URL as the address the key was saved for."""
    if not provider.custom:
        return True
    return saved.get(server_url_env(provider.name)) == provider.base_url


def key_status(provider: Provider, saved: dict[str, str]) -> str:
    if not bound(provider, saved):
        return NONE
    if saved.get(provider.key_env):
        return SET
    if os.environ.get(provider.key_env):
        return SHELL
    return NONE


def key_of(provider: Provider, saved: dict[str, str]) -> str | None:
    """The provider's key: the one saved in settings, else the shell's; none for a custom
    server at an address the key wasn't saved for."""
    if not bound(provider, saved):
        return None
    key = saved.get(provider.key_env) or os.environ.get(provider.key_env) or ""
    # A shell or hand-edited key may end in a stray newline or carriage return.
    return key.strip() or None


def checked_key(key: str, provider: Provider) -> str:
    key = key.strip()
    if not PLAIN_TEXT.fullmatch(key):
        raise SettingsError(
            f"That doesn't look like a key: paste the {provider.name} key as one line, "
            "without spaces."
        )
    return key


def optional_key(key: str | None, provider: Provider) -> str | None:
    """A key given with a custom server, checked; None when the field was left blank."""
    return checked_key(key, provider) if key and key.strip() else None


def write_key(home: Path, name: str, key: str) -> None:
    """Set *name* in .env, leaving its other lines as they are; the file is the user's alone.
    A key or a base URL: printable ASCII on one line, so it can't break a line."""
    path = home / ENV_FILE
    home.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not path.is_symlink():
        # The rewrite keeps the file's mode, so it is narrowed before the key goes in.
        path.chmod(OWNER_ONLY)
    # A new file, or one replacing a symlink, is made from a temporary one, which
    # starts readable by its owner only.
    set_key(path, name, key)


def drop_key(home: Path, name: str) -> None:
    path = home / ENV_FILE
    if path.is_file() and name in dotenv_values(path, interpolate=False):
        unset_key(path, name)


def bind_address(home: Path, server: Provider) -> None:
    """Record in .env that the custom server's key, saved or from the shell, is for its
    base URL, and for no other server of its name."""
    write_key(home, server_url_env(server.name), server.base_url)


def record_server(
    home: Path, server: Provider, key: str | None, claimable: bool
) -> None:
    """Record in .env the address the custom server's key is for, and the key if one was
    typed. With no key typed, only when the name's binding is *claimable*, so saving a
    shared file's server under one of your servers' names never hands it your key."""
    if key or claimable:
        bind_address(home, server)
    if key:
        write_key(home, server.key_env, key)


def save_key(home: Path, provider: Provider, key: str) -> None:
    key = checked_key(key, provider)
    if provider.custom:
        bind_address(home, provider)
    write_key(home, provider.key_env, key)


def forget_key(home: Path, provider: Provider) -> None:
    """Drop the provider's saved key, unless it was saved for another server's address."""
    if bound(provider, saved_keys(home)):
        drop_key(home, provider.key_env)


def well_shaped(key: str, value: Any) -> bool:
    if key not in SHAPES or SHAPES[key](value):
        return True
    log.warning("settings.left_out", key=key, why="not what this app writes there")
    return False


def settings_in(connection: sqlite3.Connection) -> dict[str, Any]:
    """The app's settings in the open file; one this app couldn't have written, as a file
    from someone else may hold, is left out with a warning, so it reads as unset."""
    settings = {}
    for key, text in connection.execute("SELECT key, json FROM app_settings"):
        try:
            value = json.loads(text)
            # An escape can make half a character, which no response can carry.
            json.dumps(value, ensure_ascii=False).encode()
        except UnicodeEncodeError:
            log.warning("settings.left_out", key=key, why="half a character")
        except ValueError:
            log.warning("settings.left_out", key=key, why="not JSON")
        else:
            settings[key] = value
    return {key: value for key, value in settings.items() if well_shaped(key, value)}


def read_settings(library: Path) -> dict[str, Any]:
    with closing(open_library(library)) as connection:
        return settings_in(connection)


def write_setting(library: Path, key: str, value: Any) -> None:
    """Set one of the app's settings: one statement, so it lands whole or not at all."""
    with closing(open_library(library)) as connection:
        save_setting(connection, key, value)


def save_setting(connection: sqlite3.Connection, key: str, value: Any) -> None:
    """Save *value* as setting *key*. A refusal, as a shared file's own constraints may
    make, is a SettingsError."""
    try:
        connection.execute(UPSERT_APP_SETTING, (key, json.dumps(value)))
    except sqlite3.IntegrityError as error:
        raise SettingsError(
            f"The Library file refused the setting: {error}."
        ) from error


@contextmanager
def changing_servers(library: Path) -> Iterator[list[Any]]:
    """The stored server entries, to change in place: written back as the block ends, in
    the transaction they were read in, so another process's change waits, never lost."""
    with closing(open_library(library)) as connection, writing(connection):
        entries = settings_in(connection).get(SERVERS, [])
        yield entries
        save_setting(connection, SERVERS, entries)


def env_stem(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def server_key_env(name: str) -> str:
    """The variable a custom server's key is kept under: DEEP_REASONING_<NAME>_KEY."""
    return f"DEEP_REASONING_{env_stem(name)}_KEY"


def server_url_env(name: str) -> str:
    """The variable holding the address a custom server's key is for:
    DEEP_REASONING_<NAME>_URL."""
    return f"DEEP_REASONING_{env_stem(name)}_URL"


def refuse_taken(name: str, others: list[Provider]) -> None:
    """Raise if another provider has the name, in any case, or the key variable it would
    get: a server under it would be handed that provider's key."""
    key_env = server_key_env(name)
    for other in others:
        taken = f"There is a provider named {other.name} already"
        if other.name == name:
            raise SettingsError(f"{taken}.")
        if other.name.lower() == name.lower():
            raise SettingsError(
                f"{taken}, and names differing only in case count as one."
            )
        if other.key_env == key_env:
            raise SettingsError(
                f"{taken}, and {name} would keep its key under the same variable,"
                f" {key_env}: pick a name that differs in its letters or digits."
            )


def stored_server(entry: Any, others: list[Provider]) -> Provider:
    """A server entry read from the app file, if add_server could have written it beside
    *others*; else SettingsError saying why."""
    if not isinstance(entry, dict) or not all(
        isinstance(entry.get(field), str) for field in SERVER_FIELDS
    ):
        raise SettingsError("A server should be a name and a base URL.")
    name, base_url = entry["name"], entry["base_url"]
    if checked_name(name) != name or checked_url(base_url) != base_url:
        raise SettingsError(
            "A server's name and base URL should have no spaces around them."
        )
    refuse_taken(name, others)
    return Provider(name, base_url, server_key_env(name), True)


def trusted_servers(entries: list[Any]) -> list[Provider]:
    """The server entries a server added here could be, as read from a file that may
    have come from someone else; the others are left out, so none gets another's key."""
    kept = []
    for position, entry in enumerate(entries):
        try:
            kept.append(stored_server(entry, [*VENDORS, *kept]))
        except SettingsError as error:
            log.warning("settings.server_left_out", position=position, why=str(error))
    return kept


def servers(library: Path) -> list[Provider]:
    return trusted_servers(read_settings(library).get(SERVERS, []))


def providers(library: Path) -> list[Provider]:
    return [*VENDORS, *servers(library)]


def key_names(library: Path) -> list[str]:
    """Each provider's key variable: what a run's worker may get, and withholds."""
    return [each.key_env for each in providers(library)]


def every_key(library: Path, home: Path) -> tuple[list[str], list[str]]:
    """Each variable that may hold a key, and each key: *library*'s providers', every
    custom server's, whichever Library lists it, set in the shell or saved in .env in
    *home*, and Claude Code's; and every key, saved there or set in the shell."""
    saved = saved_keys(home)
    servers = [
        name for name in [*os.environ, *saved] if SERVER_KEY_VARIABLE.fullmatch(name)
    ]
    names = list(dict.fromkeys([*key_names(library), *servers, *CLAUDE_CODES_KEYS]))
    # Every .env entry is a key, but the address each custom server's is for.
    keys = [name for name in saved if not SERVER_URL_VARIABLE.fullmatch(name)]
    return names, [*secrets_of(os.environ, names), *secrets_of(saved, keys)]


def keyless_environment(library: Path, home: Path) -> dict[str, str]:
    """What a run's worker gets of this environment, but any key: what a tool built to
    check it gets. Each of *library*'s providers' key variables holds NOT_A_KEY, so a
    tool that reads one while it is built still builds, and reads no key."""
    names, secrets = every_key(library, home)
    kept = {
        name: value
        for name, value in worker_env({}).items()
        if name not in names and value not in secrets
    }
    return kept | dict.fromkeys(key_names(library), NOT_A_KEY)


def find_provider(library: Path, name: str) -> Provider | None:
    return next((each for each in providers(library) if each.name == name), None)


def usable(library: Path, saved: dict[str, str]) -> list[Provider]:
    """The providers a chat can run on: vendors with a key, and custom servers, which may need none."""
    return [
        each
        for each in providers(library)
        if each.custom or key_status(each, saved) != NONE
    ]


def provider_list(home: Path, library: Path) -> list[dict[str, Any]]:
    """Each provider as the page shows it: never its key, only whether one is saved here,
    comes from the shell, or is missing."""
    saved = saved_keys(home)
    return [
        {
            "name": each.name,
            "base_url": each.base_url,
            "key_env": each.key_env,
            "custom": each.custom,
            "key": key_status(each, saved),
        }
        for each in providers(library)
    ]


def checked_url(base_url: str) -> str:
    base_url = base_url.strip()
    problem = (
        f"The base URL should be an http:// or https:// address with no ?query or "
        f"#fragment, like {EXAMPLE_URL}."
    )
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
    except ValueError as error:
        raise SettingsError(problem) from error
    if (
        not PLAIN_TEXT.fullmatch(base_url)
        or parts.scheme not in WEB_SCHEMES
        or not host
    ):
        raise SettingsError(problem)
    if parts.query or parts.fragment or base_url.endswith(("?", "#")):
        raise SettingsError(problem)
    if parts.username or parts.password:
        raise SettingsError("Put the key in the key field, not in the base URL.")
    return base_url


def checked_name(name: str) -> str:
    name = name.strip()
    if not SERVER_NAME.fullmatch(name):
        raise SettingsError(
            "Name the server with letters, digits, spaces, dots, dashes or underscores, "
            f"up to 40, like {EXAMPLE_NAME}."
        )
    return name


def named(entry: Any, name: str) -> bool:
    return isinstance(entry, dict) and entry.get("name") == name


def add_server(
    home: Path, library: Path, name: str, base_url: str, key: str | None
) -> None:
    """Add a custom server to the app file, and its key, if it has one, to .env in *home*;
    without a key, it takes the name's binding only if no other address holds it."""
    name, base_url = checked_name(name), checked_url(base_url)
    server = Provider(name, base_url, server_key_env(name), True)
    key = optional_key(key, server)
    unbound = server_url_env(name) not in saved_keys(home)
    with changing_servers(library) as entries:
        refuse_taken(name, [*VENDORS, *trusted_servers(entries)])
        entries.append({"name": name, "base_url": base_url})
    record_server(home, server, key, claimable=unbound)


def update_server(
    home: Path, library: Path, server: Provider, base_url: str, key: str | None
) -> None:
    """Change a custom server's base URL, and its key when a new one is given; without
    a key, the name's binding follows only a server whose own address it was."""
    base_url = checked_url(base_url)
    key = optional_key(key, server)
    its_own = bound(server, saved_keys(home))
    with changing_servers(library) as entries:
        entries[:] = [
            each | {"base_url": base_url} if named(each, server.name) else each
            for each in entries
        ]
    record_server(home, replace(server, base_url=base_url), key, claimable=its_own)


def remove_server(home: Path, library: Path, server: Provider) -> None:
    """Remove a custom server, and its key and address from .env when they are its own,
    not those of another file's server of its name."""
    keep_key = not bound(server, saved_keys(home))
    with changing_servers(library) as entries:
        entries[:] = [each for each in entries if not named(each, server.name)]
    if keep_key:
        return
    drop_key(home, server.key_env)
    drop_key(home, server_url_env(server.name))


def remember_pick(library: Path, provider: str, model: str) -> None:
    """The model picked last, in any chat: where new chats start."""
    write_setting(library, LAST_MODEL, {"provider": provider, "name": model})


def starting_model(home: Path, library: Path) -> dict[str, str] | None:
    """What a new chat runs on: the model picked last, while its provider can still answer;
    else the first model of the first vendor with a key that the app knows one for."""
    can_answer = usable(library, saved_keys(home))
    last = read_settings(library).get(LAST_MODEL)
    if last and any(each.name == last["provider"] for each in can_answer):
        return last
    first = next((each for each in can_answer if each.first_model), None)
    if first is None:
        return None
    return {"provider": first.name, "name": first.first_model}


def default_folder(library: Path) -> str | None:
    """The folder new chats work in; None: each in its own scratch folder."""
    return read_settings(library).get(DEFAULT_FOLDER)


def save_default_folder(library: Path, folder: str | None) -> None:
    write_setting(library, DEFAULT_FOLDER, folder)


def run_key(home: Path, provider: Provider) -> str:
    """The key a run on *provider* authenticates with, or why it can't run."""
    key = key_of(provider, saved_keys(home))
    if key:
        return key
    if provider.custom:
        return NO_KEY
    raise SettingsError(f"{provider.name} has no key: add one in settings.")


def is_chat_model(name: str) -> bool:
    return NOT_CHAT.search(name) is None


async def list_models(provider: Provider, key: str | None) -> dict[str, Any]:
    """The provider's chat models from its own GET {base_url}/models, or why there are
    none, and the ones the menu shows before anything is typed: a vendor's current tiers,
    even with no list, or a custom server's whole list."""
    url = f"{provider.base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    listing = {"provider": provider.name, "models": [], "problem": None}
    try:
        async with httpx.AsyncClient(timeout=LIST_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        names = {each["id"] for each in response.json()["data"]}
        listing["models"] = sorted(filter(is_chat_model, names))
    except httpx.HTTPStatusError as error:
        status = f"{error.response.status_code} {error.response.reason_phrase}".strip()
        listing["problem"] = f"No list: {url} answered {status}."
    except (httpx.LocalProtocolError, UnicodeEncodeError):
        # Raised for a key no header can carry; httpx's words would repeat the key.
        listing["problem"] = (
            f"No list: the {provider.name} key can't be sent; is it one line of "
            "plain text?"
        )
    except httpx.HTTPError as error:
        listing["problem"] = (
            f"No list: {url} didn't answer ({error or type(error).__name__})."
        )
    except (ValueError, KeyError, TypeError):
        listing["problem"] = f"No list: {url} didn't answer with a list of models."
    listing["featured"] = list(provider.featured or listing["models"])
    return listing


async def model_lists(home: Path, library: Path) -> list[dict[str, Any]]:
    """Each provider a chat can run on, with its models, fetched with its key here, so no
    key reaches the page."""
    saved = saved_keys(home)
    lists = [list_models(each, key_of(each, saved)) for each in usable(library, saved)]
    return list(await asyncio.gather(*lists))
