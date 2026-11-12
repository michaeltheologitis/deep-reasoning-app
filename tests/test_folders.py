import ast
import errno
import os
import re

import pytest
from structlog.testing import capture_logs

from deep_reasoning.folders import (
    GONE,
    READ_ENTRIES,
    SHOWN_NAMES,
    FolderError,
    chosen_folder,
    folder_lines,
    list_folders,
    remove_unless_kept,
    scratch_folder,
    working_folder,
)

CHAT = "quiet-blue-heron"


@pytest.fixture
def code(home):
    """~/code with two folders, a hidden one, and a file."""
    folder = home / "code"
    for name in ("thesis", "Notes", ".git"):
        (folder / name).mkdir(parents=True)
    (folder / "todo.txt").write_text("write")
    return folder


def with_sentinel(folder):
    folder.mkdir(parents=True, exist_ok=True)
    sentinel = folder / "keep.txt"
    sentinel.write_text("mine")
    return sentinel


def test_a_listing_has_the_subfolders_hidden_last_and_no_files(code):
    listing = list_folders("~/code")
    assert listing["path"] == "~/code" and listing["parent"] == "~"
    assert listing["problem"] is None
    assert listing["folders"] == [
        {"name": "Notes", "path": "~/code/Notes"},
        {"name": "thesis", "path": "~/code/thesis"},
        {"name": ".git", "path": "~/code/.git"},
    ]


@pytest.mark.parametrize(
    "typed", ["~/code/", "code", " ~/code/thesis/.. ", "{home}/code"]
)
def test_a_typed_path_is_expanded_made_absolute_and_tidied(code, home, typed):
    assert list_folders(typed.format(home=home))["path"] == "~/code"
    assert chosen_folder(typed.format(home=home)) == str(code)


def test_the_root_has_no_parent_and_outside_home_paths_stay_whole():
    assert list_folders("/")["parent"] is None
    assert list_folders("/usr")["path"] == "/usr"
    assert list_folders("/usr")["parent"] == "/"


@pytest.mark.parametrize(
    ("typed", "problem"),
    [
        ("/no/such/folder", "There is no folder /no/such/folder."),
        ("~/code/todo.txt", "~/code/todo.txt is a file, not a folder."),
        ("~/a\x00b", "There is no folder ~/a\x00b."),
        ("~/" + "x" * 300, f"There is no folder ~/{'x' * 300}."),
    ],
)
def test_a_path_that_is_not_a_folder_is_refused_in_plain_words(code, typed, problem):
    with pytest.raises(FolderError) as choosing:
        chosen_folder(typed)
    assert str(choosing.value) == problem


@pytest.mark.parametrize(
    "typed", ["/no/such/folder", "~/cod", "~/a\x00b", "~/" + "x" * 300]
)
def test_a_path_with_nothing_there_lists_nothing_and_says_so_without_a_refusal(
    code, typed
):
    listing = list_folders(typed)
    with pytest.raises(FolderError) as choosing:
        chosen_folder(typed)
    assert (listing["folders"], listing["parent"]) == ([], None)
    assert listing["problem"] == str(choosing.value)


def test_a_file_is_refused_by_the_listing_too(code):
    with pytest.raises(
        FolderError, match=r"^~/code/todo\.txt is a file, not a folder\.$"
    ):
        list_folders("~/code/todo.txt")


def test_a_listing_leaves_out_what_the_page_could_not_show_or_open(code):
    (code / "loop").symlink_to(code / "loop")
    os.mkdir(os.fsencode(code / "caf") + b"\xe9")  # a name that isn't UTF-8
    names = [each["name"] for each in list_folders("~/code")["folders"]]
    assert names == ["Notes", "thesis", ".git"]


def test_a_folder_that_cannot_be_read_is_refused_in_plain_words(code, monkeypatch):
    def refused(path):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(os, "scandir", refused)
    with pytest.raises(FolderError, match=r"^Can't open ~/code: Permission denied\.$"):
        list_folders("~/code")


@pytest.mark.parametrize("typed", [None, "", "  "])
def test_a_blank_choice_is_the_scratch_folder(typed):
    assert chosen_folder(typed) is None


def test_a_chat_without_a_folder_works_in_its_scratch_folder_made_on_first_use(
    app_home,
):
    folder = working_folder(app_home, {"id": CHAT, "folder": None})
    assert folder == app_home / "work" / CHAT and folder.is_dir()


def test_a_chosen_folder_that_is_gone_refuses_the_run(app_home, code):
    gone = code / "thesis"
    gone.rmdir()
    with pytest.raises(FolderError, match=f"^{GONE}$"):
        working_folder(app_home, {"id": CHAT, "folder": str(gone)})


@pytest.mark.parametrize("chat", ["..", ".", "a/b", "../../etc", ""])
def test_only_a_plain_chat_id_names_a_scratch_folder(app_home, chat):
    with pytest.raises(ValueError):
        scratch_folder(app_home, chat)


def test_removing_a_chats_scratch_folder_takes_its_files_and_nothing_else(app_home):
    with_sentinel(app_home / "work" / CHAT / "deep")
    other = with_sentinel(app_home / "work" / "another-chat")
    remove_unless_kept(scratch_folder(app_home, CHAT), [])
    remove_unless_kept(scratch_folder(app_home, "never-ran"), [])
    assert not (app_home / "work" / CHAT).exists()
    assert other.read_text() == "mine"


@pytest.mark.parametrize("chosen", ["", "deep"])
def test_a_folder_that_is_or_holds_a_kept_one_is_kept(app_home, chosen):
    scratch = app_home / "work" / CHAT
    sentinel = with_sentinel(scratch / chosen)
    with capture_logs() as logs:
        remove_unless_kept(scratch, ["/elsewhere", str(scratch / chosen)])
    assert sentinel.read_text() == "mine"
    assert [each["event"] for each in logs] == ["folders.kept_as_chosen"]


def test_a_folder_that_is_a_symlink_is_never_followed(app_home, home):
    precious = with_sentinel(home / "precious")
    (app_home / "work").mkdir(parents=True)
    (app_home / "work" / CHAT).symlink_to(precious.parent)
    with capture_logs() as logs:
        remove_unless_kept(scratch_folder(app_home, CHAT), [])
    assert precious.read_text() == "mine"
    assert [each["event"] for each in logs] == ["folders.not_removed"]


def test_a_kept_folder_that_is_a_symlink_loop_does_not_stop_the_delete(app_home, home):
    loop = home / "loop"
    loop.symlink_to(loop)
    with_sentinel(app_home / "work" / CHAT)
    remove_unless_kept(scratch_folder(app_home, CHAT), [str(loop)])
    assert not (app_home / "work" / CHAT).exists()


def how_to_pick(button):
    return (
        "There is no upload: for files that aren't in this chat's folder, ask the user to"
        f" click {button!r} under the message box and pick the folder holding them."
    )


def quoted_names(where):
    """The names a folder line lists, read back as Python reads them."""
    listed = where.split("(a / marks a folder): ", 1)[1].removesuffix(".")
    return ast.literal_eval(f"[{listed}]")


def test_the_folder_lines_name_the_folder_what_it_holds_but_hidden_ones_and_its_button(
    code,
):
    where = (
        f"This chat's working folder is {str(code)!r}, the user's own, holding (a /"
        " marks a folder): 'Notes/', 'thesis/', 'todo.txt'."
    )
    assert folder_lines(code, scratch=False) == (where, how_to_pick("code"))


def test_the_folder_lines_say_a_scratch_folder_is_the_chats_own_and_empty(app_home):
    scratch = working_folder(app_home, {"id": CHAT, "folder": None})
    where = (
        f"This chat's working folder is {str(scratch)!r}, the chat's own scratch folder,"
        " not one of the user's: it is empty."
    )
    assert folder_lines(scratch, scratch=True) == (where, how_to_pick("scratch"))


def test_an_empty_folder_of_the_users_is_said_to_be_empty(code):
    where, _ = folder_lines(code / "thesis", scratch=False)
    assert where.endswith(", the user's own: it is empty.")


def test_a_folder_that_cannot_be_read_is_named_without_its_names(code, monkeypatch):
    def refused(path):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(os, "scandir", refused)
    where, _ = folder_lines(code, scratch=False)
    assert where == f"This chat's working folder is {str(code)!r}, the user's own."


@pytest.fixture
def notes(home):
    """~/notes, empty."""
    folder = home / "notes"
    folder.mkdir()
    return folder


def test_the_folder_line_lists_so_many_names_and_counts_the_rest(notes):
    for number in range(SHOWN_NAMES + 5):
        (notes / f"part{number:03}.txt").touch()
    where, _ = folder_lines(notes, scratch=False)
    assert quoted_names(where.replace(", and 5 more", "")) == [
        f"part{number:03}.txt" for number in range(SHOWN_NAMES)
    ]


def test_a_huge_folder_is_read_only_so_far_and_says_there_are_more(notes):
    for number in range(READ_ENTRIES + 1):
        (notes / f"{number}.txt").touch()
    where, _ = folder_lines(notes, scratch=False)
    assert where.endswith(f", and over {READ_ENTRIES - SHOWN_NAMES} more.")


def test_names_from_the_users_disk_read_back_whole_and_hold_no_template(home):
    folder = home / "{% notes %}"
    folder.mkdir()
    names = ["{{ 7*7 }}.txt", "{% raw %}", "{#", "two\nlines.md", "it's.md", "<repl>"]
    for name in names:
        (folder / name).touch()
    os.mkdir(os.fsencode(folder / "caf") + b"\xe9")  # a name that isn't UTF-8
    where, how = folder_lines(folder, scratch=False)
    assert "\n" not in where + how
    assert not re.search(r"\{[{%#]", where + how)
    assert sorted(quoted_names(where)) == sorted([*names, "caf\udce9/"])
    assert "click '\\x7b% notes %}' under" in how
