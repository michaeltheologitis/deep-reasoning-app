"""An OpenAI-compatible chat endpoint on 127.0.0.1 for scripted runs: real HTTP, so the
worker runs unmodified. A plan scripts it: for an agent whose task contains a key, the
reply at each turn of its drive. It lists its models too, as GET /v1/models."""

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

import structlog

log = structlog.get_logger()

Messages = list[dict[str, Any]]
MODELS = ("fake-large", "fake-small")
PRICE_PER_TOKEN = 1e-6
BEARER = "Bearer "
PLAIN = "text/plain"
# What the tests call the fake, and the key they give it, as a custom server in settings.
SERVER_NAME = "fake"
SERVER_KEY = "sk-fake-server-key"
# A decomposition for the library tool to write, and the same reworded, to replace it.
ASKED = "How many tickets are open?"
REWORDED_ASK = "How many open tickets are left?"
EXAMPLE = f"""name: count the tickets
messages:
- role: user
  content: {ASKED}
- role: assistant
  content: Count them with len(open_tickets).
"""
REWORDED = EXAMPLE.replace(ASKED, REWORDED_ASK)
# A namespace for the library tool to add, and the same with its suffix changed.
SUFFIX = "Cite a source for every claim."
CHANGED_SUFFIX = "Cite two sources for every claim."
RESEARCH = f"""system_suffix: |
  {SUFFIX}
vars:
  year: 2026
"""
RESEARCH_CHANGED = RESEARCH.replace(SUFFIX, CHANGED_SUFFIX)
# A tool for the library tool to write, and the same splitting on its settings' sep.
WORD_COUNT_DOC = "How many words text has."
WORD_COUNT = f'''from deep_reasoner.primitives import Func


def build(client, params):
    def word_count(text):
        """{WORD_COUNT_DOC}"""
        return len(text.split())

    return Func(word_count)
'''
SPLIT_COUNT = WORD_COUNT.replace("text.split()", 'text.split(params["sep"])')


@dataclass(frozen=True)
class FakeCall:
    started: float
    model: str
    messages: Messages
    key: str | None


class ProviderFailure(Exception):
    """A reply a plan gives as its provider's failure: the fake answers *status* with the
    plain *text* a gateway gives, as testers' 502s read "upstream request failed"."""

    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status, self.text = status, text


Plan = Mapping[str, list[str | ProviderFailure]]


def bearer_key(authorization: str | None) -> str | None:
    """The key an Authorization header carries, if it is a bearer one."""
    if authorization is None or not authorization.startswith(BEARER):
        return None
    return authorization.removeprefix(BEARER)


def word_count_usage(messages: Messages, reply: str) -> dict[str, Any]:
    """One token per whitespace-separated word, plus one: the same on every machine."""
    prompt = sum(len(str(m.get("content", "")).split()) for m in messages) + 1
    completion = len(reply.split()) + 1
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def own_cost_usage(messages: Messages, reply: str) -> dict[str, Any]:
    """word_count_usage with the server's own figure for what the call cost, as OpenRouter
    sends one: a server that is no vendor's prices its calls no other way."""
    usage = word_count_usage(messages, reply)
    return usage | {"cost": usage["total_tokens"] * PRICE_PER_TOKEN}


def repl(*lines: str, think: str = "") -> str:
    """One assistant turn: an optional <think>, then a <repl> block."""
    head = f"<think>{think}</think>\n" if think else ""
    return head + "<repl>\n" + "\n".join(lines) + "\n</repl>"


def folded_answer(literal: str) -> str:
    """An earlier answer as a follow-up's model is shown it: the cell that gave it, with
    the note that its work is folded away."""
    return (
        "<repl>\n# folded: the work behind this answer (its thoughts, cells and any "
        "sub-agents) isn't shown here, only the answer; asked about it, in any words or "
        f"language, do as the worked examples do\nFinalAnswer({literal})\n</repl>"
    )


def library_cell(method: str, *args: Any) -> str:
    """A turn whose cell calls the library tool's *method* with *args* and prints what it
    returns."""
    return repl(f"print(library.{method}({', '.join(map(repr, args))}))")


def write(namespace: str, text: str) -> str:
    """A turn whose cell writes the decomposition *text* into *namespace* with the
    library tool."""
    return library_cell("write_decomposition", namespace, text)


def task_and_turn(messages: Messages) -> tuple[str, int]:
    """The live task (the last user message that is not an observation) and how many
    assistant turns its drive has had."""
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        content = str(message.get("content", ""))
        if message.get("role") == "user" and not content.startswith("<observation>"):
            turns = sum(m.get("role") == "assistant" for m in messages[i + 1 :])
            return content, turns
    return "", 0


def scripted(plan: Plan) -> Callable[[Messages], str]:
    """A responder that follows the plan; its last reply repeats."""

    def respond(messages: Messages) -> str:
        task, turn = task_and_turn(messages)
        for key, replies in plan.items():
            if key in task:
                reply = replies[min(turn, len(replies) - 1)]
                if isinstance(reply, ProviderFailure):
                    raise reply
                return reply
        raise AssertionError(f"no plan for task {task!r}")

    return respond


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


class FakeOpenAI:
    """POST /v1/chat/completions and GET /v1/models on 127.0.0.1, as a context manager.

    `responder(messages)` gives each reply; one that raises answers HTTP 500 with its
    text. `.calls` records every request; `.base_url` is the endpoint. `models` are what
    it lists, or None to answer that listing with 404; `.listed` records the key each
    listing was asked with.
    """

    def __init__(
        self,
        responder: Callable[[Messages], str],
        *,
        usage: Callable[[Messages, str], dict[str, Any]] = word_count_usage,
        models: Sequence[str] | None = MODELS,
    ) -> None:
        self.calls: list[FakeCall] = []
        self.listed: list[str | None] = []
        self._responder = responder
        self._usage = usage
        self._models = models
        self._server: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def answer(
        self, request: dict[str, Any], key: str | None
    ) -> tuple[int, dict[str, Any] | str]:
        messages = request.get("messages", [])
        model = request.get("model", "")
        self.calls.append(FakeCall(time.time(), model, messages, key))
        try:
            reply = self._responder(messages)
        except ProviderFailure as failure:
            return failure.status, failure.text
        except Exception as error:
            log.exception("fake_model.answered_500")
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(error)}}
        return HTTPStatus.OK, {
            "id": f"chatcmpl-{len(self.calls)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": self._usage(messages, reply),
        }

    def listing(self, key: str | None) -> tuple[int, dict[str, Any]]:
        self.listed.append(key)
        if self._models is None:
            return HTTPStatus.NOT_FOUND, {"error": {"message": "no model list here"}}
        data = [{"id": name, "object": "model"} for name in self._models]
        return HTTPStatus.OK, {"object": "list", "data": data}

    def __enter__(self) -> Self:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
                self.reply(*fake.answer(request, self.key()))

            def do_GET(self) -> None:
                if not self.path.endswith("/models"):
                    self.reply(HTTPStatus.NOT_FOUND, {})
                    return
                self.reply(*fake.listing(self.key()))

            def key(self) -> str | None:
                return bearer_key(self.headers.get("Authorization"))

            def reply(self, status: int, body: dict[str, Any] | str) -> None:
                text = isinstance(body, str)
                data = body.encode() if text else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", PLAIN if text else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = _Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
