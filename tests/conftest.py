from contextlib import closing
from pathlib import Path

import pytest
from fake_model import SERVER_KEY, SERVER_NAME

from deep_reasoning.importing import import_config
from deep_reasoning.library import open_library
from deep_reasoning.settings import VENDORS, add_server, remember_pick

# Copies of deep_reasoner 0.3.0's docs/configs, in their layout, so _compose paths still resolve.
CONFIGS = Path(__file__).parent / "configs"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """A home directory of the test's own, so no test touches the real ~/.deep-reasoning."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def app_home(home):
    """The app's home in the test's home, where the command keeps it."""
    return home / ".deep-reasoning"


@pytest.fixture
def no_vendor_keys(monkeypatch):
    """No vendor's key from the shell, so nothing reaches a real vendor: a test sets the
    ones it means."""
    for vendor in VENDORS:
        monkeypatch.delenv(vendor.key_env, raising=False)


@pytest.fixture
def advisors():
    return CONFIGS / "catalog" / "advisors.yaml"


@pytest.fixture
def waitlist():
    return CONFIGS / "examples" / "waitlist" / "waitlist.yaml"


@pytest.fixture
def configs():
    """The engine's example configs, which those in examples/ compose from: what an
    import of one is given as the folder it reads within."""
    return CONFIGS


@pytest.fixture
def library(tmp_path):
    return tmp_path / "library.sqlite"


@pytest.fixture
def imported(library, advisors):
    """A Library holding advisors.yaml."""
    with closing(open_library(library)) as connection:
        import_config(connection, advisors)
    return library


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fake_client(fake):
    return {
        "base_url": fake.base_url,
        "api_key_env": "OPENAI_API_KEY",
        "max_retries": 0,
    }


@pytest.fixture
def fake_config(monkeypatch):
    """The engine config for a run whose model is a FakeOpenAI, as Runs.start takes it:
    call it with the fake."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    def config_for(fake, **settings):
        config = {
            "model": "gpt-6-luna",
            "client": fake_client(fake),
            "system_prompt": "You are a scripted test agent.",
            "max_iter": 6,
            "entry_namespace": "root",
        }
        return lambda folder: config | settings

    return config_for


@pytest.fixture
def library_on(library, home):
    """The Library, with a FakeOpenAI in the app's settings as a custom server, with a
    key, that new chats start on: call it with the fake."""

    def on(fake):
        add_server(home, library, SERVER_NAME, fake.base_url, SERVER_KEY)
        remember_pick(library, SERVER_NAME, "gpt-6-luna")
        return library

    return on
