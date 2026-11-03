import json
from contextlib import closing

import pytest
from structlog.testing import capture_logs

from deep_reasoning import pages
from deep_reasoning.library import LibraryError, open_library, writing
from deep_reasoning.pages import (
    ID_TRIES,
    MAX_PAGE_CHARS,
    add_page,
    read_page,
    update_page,
)

CHAT = "a-chat"
RUN = "20261027-120000-000000"
CELL = {"kind": "cell", "code": "print(1)", "output": "1", "status": "done"}
DRIVE = {
    "node": 2,
    "drive": 1,
    "task": "Count.",
    "namespace": "root",
    "status": "done",
    "result": "1",
    "seconds": 0.5,
    "cost": None,
    "unpriced": True,
    "steps": [CELL | {"agents": []}],
}
# One block of each type, as the page tool saves them.
BLOCKS = [
    {"type": "text", "text": "It **counted**."},
    {
        "type": "step",
        "turn": 1,
        "node": None,
        "index": 0,
        "task": None,
        "note": "Here.",
        "step": CELL | {"agents": [DRIVE]},
    },
    {"type": "agent", "turn": 1, "node": 2, "note": None, "agents": [DRIVE]},
    {"type": "table", "columns": ["course"], "rows": [["CS101"], []]},
    {
        "type": "step",
        "turn": 1,
        "node": 2,
        "index": 1,
        "task": "Count.",
        "note": None,
        "step": {"kind": "change", "what": 7, "name": ["odd"], "before": None},
    },
]


def saved(library, blocks, title="Counted"):
    """A page holding *blocks*, saved as the page tool saves one; its id."""
    with closing(open_library(library)) as connection:
        return add_page(connection, CHAT, RUN, 1, title, json.dumps(blocks))


def read(library, page):
    with closing(open_library(library)) as connection:
        return read_page(connection, page)


def planted(library, title, blocks_text):
    """A page row holding *title* and *blocks_text* as they are, as a file from someone
    else might; its id."""
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(
            "INSERT INTO pages VALUES ('planted', ?, ?, 1, ?, ?, 0)",
            (CHAT, RUN, title, blocks_text),
        )
    return "planted"


def test_a_saved_page_reads_back_with_every_block(library):
    page = saved(library, BLOCKS)
    assert read(library, page) == {"id": page, "title": "Counted", "blocks": BLOCKS}


def test_each_page_gets_an_id_of_its_own(library):
    assert len({saved(library, BLOCKS) for _ in range(20)}) == 20


@pytest.mark.parametrize("page", ["no-such-page", "../../etc", "a b", "x" * 101, ""])
def test_a_page_not_there_or_with_no_pages_id_reads_as_none(library, page):
    saved(library, BLOCKS)
    assert read(library, page) is None


@pytest.mark.parametrize(
    ("title", "blocks_text"),
    [
        (b"Counted", json.dumps(BLOCKS)),
        ("x" * 201, json.dumps(BLOCKS)),
        ("Counted", "not json"),
        ("Counted", json.dumps({"blocks": BLOCKS})),
        ("Counted", json.dumps([BLOCKS[0]] * 201)),
        ("Counted", json.dumps([{"type": "text", "text": "x" * MAX_PAGE_CHARS}])),
        ("Counted", "[" * 100_000 + "]" * 100_000),
        ("Counted", '[{"type": "step", "turn": 9007199254740993}]'),
        ("Counted", '[{"type": "agent", "turn": NaN}]'),
        ("Counted", b"[]"),
    ],
    ids=[
        "a-title-not-text",
        "a-long-title",
        "blocks-not-json",
        "blocks-not-a-list",
        "too-many-blocks",
        "too-long",
        "nested-too-deep",
        "a-number-past-2-to-the-53",
        "nan",
        "bytes",
    ],
)
def test_a_shared_files_page_row_the_tool_could_not_have_saved_reads_as_none(
    library, title, blocks_text
):
    with capture_logs() as logs:
        found = read(library, planted(library, title, blocks_text))
    assert found is None
    assert "pages.left_out" in [entry["event"] for entry in logs]


def with_field(block, **fields):
    return block | fields


@pytest.mark.parametrize(
    "block",
    [
        {"type": "chart", "text": "x"},
        {"type": ["text"], "text": "x"},
        "a text",
        {"type": "text"},
        {"type": "text", "text": 5},
        {"type": "text", "text": "x" * 100_001},
        with_field(BLOCKS[1], turn=True),
        with_field(BLOCKS[1], turn=1.0),
        with_field(BLOCKS[1], node=[2]),
        with_field(BLOCKS[1], note="x" * 100_001),
        with_field(BLOCKS[1], step={"kind": "cell", "code": 5}),
        with_field(BLOCKS[1], step={"kind": "think"}),
        with_field(BLOCKS[1], step={"kind": "page", "page": "x", "title": None}),
        with_field(BLOCKS[1], step={"kind": {"a": 1}}),
        with_field(BLOCKS[1], step=CELL | {"agents": [DRIVE | {"steps": None}]}),
        with_field(BLOCKS[1], step=CELL | {"agents": ["a drive"]}),
        with_field(BLOCKS[2], agents=[DRIVE | {"cost": "0.5"}]),
        with_field(BLOCKS[2], agents=[DRIVE | {"unpriced": 0}]),
        with_field(BLOCKS[2], agents=[DRIVE | {"task": None}]),
        with_field(BLOCKS[2], agents=[DRIVE | {"steps": [{"kind": "think"}]}]),
        with_field(BLOCKS[2], agents=[DRIVE | {"status": "constructor"}]),
        with_field(BLOCKS[2], agents=[DRIVE | {"status": ["done"]}]),
        with_field(BLOCKS[1], step=CELL | {"status": "exhausted", "agents": []}),
        with_field(BLOCKS[3], columns=[1]),
        with_field(BLOCKS[3], rows=[[1]]),
        with_field(BLOCKS[3], rows=["CS101"]),
        with_field(BLOCKS[3], rows=[["x"] * 51]),
        with_field(BLOCKS[3], columns=["x"] * 51),
        with_field(BLOCKS[3], rows=[[]] * 1_001),
    ],
    ids=[
        "an-unknown-type",
        "a-type-not-text",
        "not-a-dict",
        "a-text-missing",
        "a-text-not-text",
        "a-long-text",
        "a-bool-turn",
        "a-float-turn",
        "a-node-not-a-node",
        "a-long-note",
        "a-cell-whose-code-is-not-text",
        "a-thought-with-no-text",
        "a-page-step-with-no-title",
        "a-kind-not-text",
        "a-sub-agent-with-no-steps",
        "a-sub-agent-not-a-dict",
        "a-cost-not-a-number",
        "unpriced-not-a-bool",
        "a-task-not-text",
        "a-sub-agents-thought-with-no-text",
        "a-status-no-drive-has",
        "a-status-not-text",
        "a-status-no-cell-has",
        "a-column-not-text",
        "a-value-not-text",
        "a-row-not-a-list",
        "too-wide",
        "too-many-columns",
        "too-many-rows",
    ],
)
def test_a_block_the_tool_could_not_have_saved_is_left_out_and_the_rest_kept(
    library, block
):
    blocks = [BLOCKS[0], block, BLOCKS[3]]
    page = planted(library, "Counted", json.dumps(blocks))
    with capture_logs() as logs:
        found = read(library, page)
    assert found["blocks"] == [BLOCKS[0], BLOCKS[3]]
    left_out = [entry for entry in logs if entry["event"] == "pages.block_left_out"]
    assert [entry["block"] for entry in left_out] == [1]


def test_fields_a_blocks_type_has_not_are_left_as_they_are(library):
    odd = {"type": "text", "text": "x", "note": 5, "agents": 5}
    assert read(library, saved(library, [odd]))["blocks"] == [odd]


def deep_agent(levels):
    """An agent block whose sub-agent nests *levels* deep, as a file from someone else
    may hold."""
    drive = DRIVE
    for _ in range(levels):
        drive = DRIVE | {"steps": [CELL | {"agents": [drive]}]}
    return {"type": "agent", "turn": 1, "node": 2, "note": None, "agents": [drive]}


def test_a_record_nested_past_pythons_recursion_limit_is_still_checked(library):
    blocks = [deep_agent(1_000)]
    assert read(library, saved(library, blocks))["blocks"] == blocks


def test_a_taken_id_is_tried_again_with_another(library, monkeypatch):
    taken = saved(library, BLOCKS)
    ids = iter([taken, taken, "a-fresh-id"])
    monkeypatch.setattr(pages, "generate_slug", lambda words: next(ids))
    assert saved(library, BLOCKS) == "a-fresh-id"


def test_a_page_whose_every_id_tried_is_taken_is_refused_in_words(library, monkeypatch):
    taken = saved(library, BLOCKS)
    monkeypatch.setattr(pages, "generate_slug", lambda words: taken)
    with pytest.raises(LibraryError, match=f"no free page id in {ID_TRIES} tries"):
        saved(library, BLOCKS)


def test_a_shared_files_pages_table_that_refuses_every_row_refuses_the_page_in_words(
    library,
):
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("DROP TABLE pages")
        connection.execute(
            "CREATE TABLE pages (id TEXT PRIMARY KEY, chat TEXT NOT NULL,"
            " run TEXT NOT NULL, node, title TEXT NOT NULL, blocks TEXT NOT NULL,"
            " created REAL NOT NULL, CHECK (0))"
        )
    with pytest.raises(LibraryError, match="refused the page: CHECK constraint failed"):
        saved(library, BLOCKS)


def saved_again(library, page, blocks, chat=CHAT, run=RUN):
    with closing(open_library(library)) as connection:
        update_page(connection, chat, run, page, json.dumps(blocks))


@pytest.mark.parametrize(
    ("chat", "run"), [("another-chat", RUN), (CHAT, "20261027-120000-000001")]
)
def test_a_page_is_saved_again_only_by_the_chat_and_run_that_made_it(
    library, chat, run
):
    page = saved(library, BLOCKS[:1])
    with pytest.raises(LibraryError, match="no longer in the Library file"):
        saved_again(library, page, BLOCKS, chat, run)
    assert read(library, page)["blocks"] == BLOCKS[:1]


def test_a_shared_files_pages_table_that_refuses_a_page_saved_again_says_so_in_words(
    library,
):
    page = saved(library, BLOCKS[:1])
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("ALTER TABLE pages RENAME TO kept")
        connection.execute(
            "CREATE TABLE pages (id TEXT PRIMARY KEY, chat TEXT NOT NULL,"
            " run TEXT NOT NULL, node, title TEXT NOT NULL, blocks TEXT NOT NULL,"
            " created REAL NOT NULL, CHECK (length(blocks) < 100))"
        )
        connection.execute("INSERT INTO pages SELECT * FROM kept")
    with pytest.raises(LibraryError, match="refused the page: CHECK constraint failed"):
        saved_again(library, page, BLOCKS)
    assert read(library, page)["blocks"] == BLOCKS[:1]
