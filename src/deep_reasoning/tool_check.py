"""Build tools as a run builds them, in a process of its own, as `python -P -m
deep_reasoning.tool_check` from an empty folder: so it sees no chat's files or modules,
as a run of another chat wouldn't.

It reads YAML on stdin: the tools, each its block by name (factory_from naming its file),
the run's spec.yaml, whose client it builds them with (None for none), and the file to
write the result in: why each tool that doesn't build doesn't, by name, as JSON. Then it
ends at once, so a thread or process a tool left running holds nothing up.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from deep_reasoner import ClientConfig, build_client
from deep_reasoner.tools import ToolBuildError
from deep_reasoner.v2.cli import make_tool

YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def failures(asked: dict[str, Any]) -> dict[str, str]:
    """Why each tool *asked* holds doesn't build, by name: none when all do."""
    client = None
    if asked["spec"]:
        spec = yaml.load(Path(asked["spec"]).read_text(), Loader=YAML_LOADER)
        client = build_client(ClientConfig.model_validate(spec["config"]["client"]))
    found = {}
    for name, block in asked["tools"].items():
        try:
            make_tool(name, block, client=client, config_path=None)
        except ToolBuildError as error:
            found[name] = error.detail
    return found


def main() -> None:
    asked = yaml.load(sys.stdin, Loader=YAML_LOADER)
    Path(asked["result"]).write_text(json.dumps(failures(asked)))
    os._exit(0)


if __name__ == "__main__":
    main()
