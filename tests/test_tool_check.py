"""The build check, as the library tool and an import call it: a verdict per tool, built
as a run of the config given builds it, in the environment given; and nothing a tool does
(raising, exiting, hanging, leaving a thread or child) holds it up, outlives it, or gets
another tool blamed."""

import os
import time
from pathlib import Path

import pytest

from deep_reasoning import tool_check
from deep_reasoning.tool_check import check_tools

BLOCK = {"factory": "build"}
CONFIG = {
    "model": "gpt-6-luna",
    "client": {"base_url": "http://127.0.0.1:9/v1", "api_key_env": "A_TEST_MODEL_KEY"},
}
BUILDS = """from deep_reasoner.primitives import Func


def build(client, params):
    return Func(len)
"""
RAISES = "def build(client, params):\n    raise RuntimeError('no index')\n"
# As a tool reaching the run's model through a models preset does.
USES_A_PRESET = """from deep_reasoner.primitives import Func
from deep_reasoner.v2 import LLM


def build(client, params):
    llm = LLM("fast")
    if (llm.model, llm.client) != ("m-fast", client):
        raise RuntimeError(f"not as the run builds it: {llm.model}")
    if not str(client.base_url).startswith("http://127.0.0.1:9"):
        raise RuntimeError(f"not the run's client: {client.base_url}")
    return Func(llm)
"""
EXITS = "import sys\n\n\ndef build(client, params):\n    sys.exit(3)\n"
ENDS_THE_CHECK = "import os\nimport sys\n\nsys.stderr.write('bye\\n')\nos._exit(3)\n"
SLOW = "import time\n\n\ndef build(client, params):\n    time.sleep(60)\n"
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
READS_THE_ENVIRONMENT = """import os

from deep_reasoner.primitives import Func


def build(client, params):
    setting = os.environ["A_SETTING"]
    return Func(lambda: setting)
"""


def verdicts(tools, work, config=CONFIG, env=None):
    """The check's verdict on *tools*, each its source by name, built by their build."""
    tools = {name: (BLOCK, source) for name, source in tools.items()}
    return check_tools(tools, config, os.environ if env is None else env, work)


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


def test_each_tool_gets_a_verdict_naming_its_file_not_where_it_was_built(tmp_path):
    tools = {"counter": BUILDS, "broken": RAISES, "missing": "import nowhere_at_all\n"}
    work = tmp_path / "work"
    assert verdicts(tools, work) == {
        "counter": None,
        "broken": "build in broken.py raised RuntimeError: no index",
        "missing": "importing 'missing.py' raised ModuleNotFoundError: No module named"
        " 'nowhere_at_all'",
    }
    assert list(work.iterdir()) == []


def test_a_tool_is_built_as_the_run_builds_it_with_its_client_and_presets(tmp_path):
    with_presets = CONFIG | {"models": {"fast": {"model": "m-fast"}}}
    assert verdicts({"fast": USES_A_PRESET}, tmp_path, with_presets) == {"fast": None}
    [why] = verdicts({"fast": USES_A_PRESET}, tmp_path).values()
    assert "not as the run builds it: fast" in why


def test_a_tool_that_exits_is_refused_and_the_rest_still_checked(tmp_path):
    assert verdicts({"exits": EXITS, "counter": BUILDS}, tmp_path) == {
        "exits": "it called sys.exit(3)",
        "counter": None,
    }


def test_a_check_that_cant_start_says_why_for_every_tool(tmp_path):
    broken_config = CONFIG | {"max_iter": "many"}
    found = verdicts({"counter": BUILDS, "other": BUILDS}, tmp_path, broken_config)
    assert set(found) == {"counter", "other"}
    assert all(
        why.startswith("the check couldn't start: ValidationError")
        for why in found.values()
    )


def test_a_tool_that_ends_the_check_is_blamed_alone_and_the_rest_checked_again(
    tmp_path,
):
    tools = {"counter": BUILDS, "ends": ENDS_THE_CHECK, "broken": RAISES}
    assert verdicts(tools, tmp_path) == {
        "counter": None,
        "ends": "building it ended the check, printing: bye",
        "broken": "build in broken.py raised RuntimeError: no index",
    }


def test_a_tool_slower_than_the_limit_is_blamed_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_check, "BUILD_TIMEOUT_S", 8)
    assert verdicts({"slow": SLOW, "counter": BUILDS}, tmp_path) == {
        "slow": "it didn't build within 8 seconds",
        "counter": None,
    }


@pytest.mark.parametrize(
    "source", [LEAVES_A_CHILD, LEAVES_A_THREAD], ids=["child", "thread"]
)
def test_what_a_tool_leaves_running_neither_holds_the_check_up_nor_outlives_it(
    tmp_path, monkeypatch, source
):
    monkeypatch.setattr(tool_check, "BUILD_TIMEOUT_S", 30)
    pid_file = tmp_path / "pid.txt"
    started = time.monotonic()
    found = verdicts(
        {"lingers": source.format(pid_file=str(pid_file))}, tmp_path / "work"
    )
    assert (found, time.monotonic() - started < 30) == ({"lingers": None}, True)
    if pid_file.exists():
        assert not running(int(pid_file.read_text()))


STARTS_A_CHILD_AND_ENDS_THE_CHECK = """import os
import subprocess
from pathlib import Path


def build(client, params):
    child = subprocess.Popen(["sleep", "1234"])
    Path({pid_file!r}).write_text(str(child.pid))
    os._exit(3)
"""


def test_a_child_of_a_tool_that_ends_the_check_doesnt_outlive_it(tmp_path):
    pid_file = tmp_path / "pid.txt"
    source = STARTS_A_CHILD_AND_ENDS_THE_CHECK.format(pid_file=str(pid_file))
    found = verdicts({"ends": source}, tmp_path / "work")
    assert found["ends"].startswith("building it ended the check")
    assert not running(int(pid_file.read_text()))


def test_the_tools_are_built_in_the_environment_given(tmp_path):
    given = dict(os.environ) | {"A_SETTING": "given"}
    assert verdicts({"env": READS_THE_ENVIRONMENT}, tmp_path, env=given) == {
        "env": None
    }
    without = {key: value for key, value in given.items() if key != "A_SETTING"}
    [why] = verdicts({"env": READS_THE_ENVIRONMENT}, tmp_path, env=without).values()
    assert why == "build in env.py raised KeyError: 'A_SETTING'"


FORGES = """import os
from pathlib import Path


def build(client, params):
    for folder in (Path(__file__).parent, Path.cwd()):
        with open(folder / "notes", "a") as notes:
            notes.write('{"built": "forger", "verdict": null}\\n')
    os._exit(0)
"""


def test_a_verdict_a_tool_writes_beside_its_file_or_where_it_runs_counts_for_nothing(
    tmp_path,
):
    verdicts = check_tools({"forger": (BLOCK, FORGES)}, CONFIG, os.environ, tmp_path)
    assert verdicts["forger"].startswith("building it ended the check")
