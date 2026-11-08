"""The build check's own process: what it says of a tool that doesn't build, and that
nothing a tool leaves running holds it up or outlives it."""

import re
import time
from pathlib import Path

import pytest
import yaml

from deep_reasoning import app_tools
from deep_reasoning.app_tools import WRITTEN, check_builds
from deep_reasoning.library import LibraryError

BLOCK = {"factory": "build"}
BUILDS = """from deep_reasoner.primitives import Func


def build(client, params):
    return Func(len)
"""
RAISES = "def build(client, params):\n    raise RuntimeError('no index')\n"
LEAVES_A_CHILD = """import subprocess
from pathlib import Path

from deep_reasoner.primitives import Func


def build(client, params):
    child = subprocess.Popen(["sleep", "60"])
    Path({pid_file!r}).write_text(str(child.pid))
    return Func(len)
"""
LEAVES_A_THREAD = """import threading
import time

from deep_reasoner.primitives import Func


def build(client, params):
    threading.Thread(target=time.sleep, args=(60,)).start()
    return Func(len)
"""
SLOW = """import time


def build(client, params):
    time.sleep(60)
"""
ENDS_THE_CHECK = "import os\nimport sys\n\nsys.stderr.write('bye\\n')\nos._exit(3)\n"
ASKS_THE_CLIENT = """from deep_reasoner.primitives import Func


def build(client, params):
    return Func(lambda: str(client.base_url))
"""
WRONG_CLIENT = ASKS_THE_CLIENT.replace(
    "return Func", "assert 'not-the-runs' in str(client.base_url)\n    return Func"
)


def checked(tools, tools_dir):
    """Check *tools*, each its source by name, built by their build functions."""
    check_builds(
        {name: (BLOCK, source) for name, source in tools.items()}, tools_dir, WRITTEN
    )


def running(pid):
    """Whether process *pid* still runs a second after: not ended, nor a zombie."""
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            state = stat.read_text().rpartition(")")[2].split()[0]
        except FileNotFoundError:
            return False
        if state in "ZX":
            return False
        time.sleep(0.05)
    return True


def test_a_tool_that_builds_passes_and_leaves_no_files(tmp_path):
    checked({"counter": BUILDS}, tmp_path / "tools")
    assert list((tmp_path / "tools" / "checks").iterdir()) == []


def test_each_tool_that_doesnt_build_is_named_with_its_file_not_where_it_was_built(
    tmp_path,
):
    tools = {"counter": BUILDS, "broken": RAISES, "missing": "import nowhere_at_all\n"}
    with pytest.raises(LibraryError) as refused:
        checked(tools, tmp_path / "tools")
    assert str(refused.value) == (
        "Tool broken doesn't build, so it wasn't written: build in broken.py raised"
        " RuntimeError: no index\n"
        "Tool missing doesn't build, so it wasn't written: importing 'missing.py' raised"
        " ModuleNotFoundError: No module named 'nowhere_at_all'"
    )


@pytest.mark.parametrize(
    "source", [LEAVES_A_CHILD, LEAVES_A_THREAD], ids=["child", "thread"]
)
def test_what_a_tool_leaves_running_neither_holds_the_check_up_nor_outlives_it(
    tmp_path, monkeypatch, source
):
    monkeypatch.setattr(app_tools, "BUILD_TIMEOUT_S", 30)
    pid_file = tmp_path / "pid.txt"
    started = time.monotonic()
    checked({"lingers": source.format(pid_file=str(pid_file))}, tmp_path / "tools")
    assert time.monotonic() - started < 30
    if pid_file.exists():
        assert not running(int(pid_file.read_text()))


def test_a_tool_slower_than_the_limit_is_refused_in_words(tmp_path, monkeypatch):
    monkeypatch.setattr(app_tools, "BUILD_TIMEOUT_S", 8)
    refusal = "Tool slow didn't build within 8 seconds, so it wasn't written."
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        checked({"slow": SLOW}, tmp_path / "tools")


def test_a_tool_that_ends_the_check_is_refused_with_what_it_printed(tmp_path):
    refusal = "Tool ends doesn't build, so it wasn't written: bye"
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        checked({"ends": ENDS_THE_CHECK}, tmp_path / "tools")


def test_a_tool_is_built_with_the_runs_client_from_its_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("A_KEY_VARIABLE", "sk-not-used")
    client = {"base_url": "http://127.0.0.1:9/v1", "api_key_env": "A_KEY_VARIABLE"}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump({"config": {"client": client}}))
    checked({"asks": ASKS_THE_CLIENT}, tmp_path / "tools")
    with pytest.raises(LibraryError, match="raised AssertionError"):
        checked({"asks": WRONG_CLIENT}, tmp_path / "tools")
