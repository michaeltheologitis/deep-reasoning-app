"""The web app: the Library's, the settings', the folders', the chats', the runs' and the pages' API, the UI's static files at /, and each page's tab at /pages/<id>."""

import asyncio
import sqlite3
import tempfile
from contextlib import asynccontextmanager, closing
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi import Path as InPath
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from deep_reasoning.app_tools import CHANGED_MEANWHILE, granted_afresh
from deep_reasoning.chats import (
    CHAT_ID,
    choose,
    choose_folder,
    create_chat,
    delete_chat,
    find_chat,
    list_chats,
    rename_chat,
    title_new_chat,
)
from deep_reasoning.folders import (
    FolderError,
    chosen_folder,
    data_folder,
    list_folders,
    remove_unless_kept,
    scratch_folder,
    short_path,
    starting_folder,
    working_folder,
)
from deep_reasoning.importing import NOWHERE
from deep_reasoning.library import (
    ChangedMeanwhile,
    LibraryError,
    add_decomposition,
    add_namespace,
    authored,
    change_namespace,
    check_name,
    check_namespace,
    config_and_registry,
    default_namespace,
    delete_decomposition,
    delete_namespace,
    delete_setting,
    delete_tool,
    edit_decomposition,
    engine_config,
    has_namespace,
    loaded,
    open_library,
    panel,
    reading,
)
from deep_reasoning.pages import page_exists, page_text
from deep_reasoning.runs import (
    RUN_ID,
    RunInProgress,
    Runs,
    chat_runs,
    chat_turns,
    delete_orphans,
    end_unfinished,
    read_turn,
)
from deep_reasoning.settings import (
    Provider,
    SettingsError,
    add_server,
    default_folder,
    find_provider,
    forget_key,
    key_names,
    keyless_environment,
    model_lists,
    provider_list,
    remember_pick,
    remove_server,
    run_key,
    save_default_folder,
    save_key,
    starting_model,
    update_server,
)
from deep_reasoning.tool_check import refuse_unbuilt

WEB = Path(__file__).parent / "web"
# Requests naming any other host are refused, so a web page can't reach the API by DNS rebinding.
LOCAL_HOSTS = ["127.0.0.1", "localhost"]
# The SQLite failures of the Library file a person can do something about; the rest are bugs.
UNWRITABLE = ("SQLITE_READONLY", "SQLITE_FULL", "SQLITE_IOERR", "SQLITE_CANTOPEN")
BUSY = ("SQLITE_BUSY", "SQLITE_LOCKED")
MODEL_NAME = r"^[^\x00-\x1f\x7f]{1,200}$"
NO_PAGE = "There is no such page: it may have been deleted with its chat."
# What wasn't done when a tool the panel grants afresh doesn't build.
PANEL_GRANTED = "wasn't granted (granted, it would stop every chat)"
# The app's pages load and send nothing but to the app, whatever an answer holds; inline
# styles are the custom properties the page sets (--depth, --clamp-lines).
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
POLICY = {"Content-Security-Policy": CONTENT_SECURITY_POLICY}

log = structlog.get_logger()

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ModelPick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    name: Annotated[str, StringConstraints(strip_whitespace=True, pattern=MODEL_NAME)]


class ChatChange(BaseModel):
    """What a PATCH of a chat may change: its title; its namespace, model or folder, from
    its next message. A folder sent as null or blank is its scratch folder."""

    model_config = ConfigDict(extra="forbid")
    title: Title | None = None
    namespace: str | None = None
    model: ModelPick | None = None
    folder: str | None = None


class Web(StaticFiles):
    """The UI's files, each sent with the policy."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(POLICY)
        return response


def shown_folder(folder):
    """A folder kept as an absolute path, as the page shows it; None stays None."""
    return None if folder is None else short_path(Path(folder))


def same_origin(
    host: Annotated[str, Header()] = "",
    origin: Annotated[str | None, Header()] = None,
):
    """Refuse a request another site's page sent: a browser names the page's origin, and
    the app's own pages are served from the host they ask."""
    if origin is not None and origin != f"http://{host}":
        raise HTTPException(
            HTTPStatus.FORBIDDEN, "Requests from other sites are refused."
        )


def create_app(library, home):
    """The app over the app file at *library*: the Library, the chats, their runs' events
    and the app's settings, which each request opens afresh. The keys are in *home*, and
    the file's run and scratch folders in its data folder there."""
    library = Path(library)
    home = Path(home)
    data = data_folder(home, library)
    runs = Runs(library, data)

    @asynccontextmanager
    async def lifespan(app):
        with closing(open_library(library)) as connection:
            delete_orphans(connection)
            for run in end_unfinished(connection):
                log.warning("Ended a run the app stopped under", run=run)
        yield
        await runs.close()

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(same_origin)])
    app.state.runs = runs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)

    @app.exception_handler(LibraryError)
    @app.exception_handler(SettingsError)
    @app.exception_handler(FolderError)
    async def refused(request, error):
        return JSONResponse({"detail": str(error)}, status_code=HTTPStatus.BAD_REQUEST)

    # So the page knows to read the Library again and show what is stored now.
    @app.exception_handler(ChangedMeanwhile)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=HTTPStatus.CONFLICT)

    @app.exception_handler(sqlite3.OperationalError)
    async def unusable(request, error):
        name = error.sqlite_errorname
        if name.startswith(UNWRITABLE):
            detail = f"The Library file can't be written: {error}."
        elif name.startswith(BUSY):
            detail = (
                "The Library file is busy: another program is writing it. Try again."
            )
        else:
            raise error
        return JSONResponse({"detail": detail}, status_code=HTTPStatus.BAD_REQUEST)

    @app.exception_handler(RequestValidationError)
    async def malformed(request, error):
        # FastAPI's own answer repeats what was sent; a key sent in the wrong shape mustn't come back.
        errors = [
            {name: value for name, value in each.items() if name != "input"}
            for each in error.errors()
        ]
        return JSONResponse(
            {"detail": jsonable_encoder(errors)},
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(RunInProgress)
    async def busy(request, error):
        detail = f"Chat {error} has a run going."
        return JSONResponse({"detail": detail}, status_code=HTTPStatus.CONFLICT)

    def library_panel(connection):
        return {"file": library.name, **panel(connection)}

    @app.get("/api/library")
    def read_library():
        with closing(open_library(library)) as connection:
            return library_panel(connection)

    @app.post("/api/decompositions", status_code=HTTPStatus.CREATED)
    def post_decomposition(
        namespace: Annotated[str, Body()], yaml: Annotated[str, Body()]
    ):
        with closing(open_library(library)) as connection:
            stored, _ = add_decomposition(connection, namespace, yaml)
            return stored

    # The rest answer with the whole Library; an edit or delete carries the YAML it opened on (was).
    @app.put("/api/decompositions")
    def put_decomposition(
        namespace: Annotated[str, Body()],
        name: Annotated[str, Body()],
        was: Annotated[str, Body()],
        yaml: Annotated[str, Body()],
    ):
        with closing(open_library(library)) as connection:
            edit_decomposition(connection, namespace, name, was, yaml)
            return library_panel(connection)

    @app.delete("/api/decompositions")
    def remove_decomposition(
        namespace: Annotated[str, Body()],
        name: Annotated[str, Body()],
        was: Annotated[str, Body()],
    ):
        with closing(open_library(library)) as connection:
            delete_decomposition(connection, namespace, name, was)
            return library_panel(connection)

    # An imported namespace's, tool's or setting's name may hold a /, hence :path.
    def built_grants(name, yaml, adding):
        """The custom tools namespace *name*, as its YAML *yaml* would make it, grants
        that no namespace grants yet, each built first, as a run builds it but with no
        key, and refused in words when one doesn't build. The build holds no lock, so
        the write checks under its own that these are still the ones it grants."""
        if adding:
            check_name(name)
        namespace, _ = authored(name, yaml)
        with closing(open_library(library)) as connection, reading(connection):
            _, registry = config_and_registry(connection)
            afresh = granted_afresh(connection, registry, namespace)
        built = {tool: (loaded(text), code) for tool, (text, code) in afresh.items()}
        if built:
            environment = keyless_environment(library, home)
            with tempfile.TemporaryDirectory() as work:
                refuse_unbuilt(built, NOWHERE, environment, Path(work), PANEL_GRANTED)
        return partial(grants_the_same, afresh)

    def grants_the_same(afresh, connection, namespace):
        """Raise unless *namespace* grants, as the Library stands under the write lock,
        the custom tools *afresh* that were built."""
        _, registry = config_and_registry(connection)
        if granted_afresh(connection, registry, namespace) != afresh:
            raise LibraryError(CHANGED_MEANWHILE)

    @app.post("/api/namespaces", status_code=HTTPStatus.CREATED)
    def post_namespace(name: Annotated[str, Body()], yaml: Annotated[str, Body()]):
        allowed = built_grants(name, yaml, adding=True)
        with closing(open_library(library)) as connection:
            add_namespace(connection, name, yaml, allowed=allowed)
            return library_panel(connection)

    @app.put("/api/namespaces/{name:path}")
    def put_namespace(
        name: str, yaml: Annotated[str, Body()], was: Annotated[str, Body()]
    ):
        allowed = built_grants(name, yaml, adding=False)
        with closing(open_library(library)) as connection:
            change_namespace(connection, name, yaml, allowed=allowed, was=was)
            return library_panel(connection)

    @app.delete("/api/namespaces/{name:path}")
    def remove_namespace(name: str, decompositions: Annotated[int, Query(ge=0)]):
        with closing(open_library(library)) as connection:
            delete_namespace(connection, name, decompositions)
            return library_panel(connection)

    @app.delete("/api/tools/{name:path}")
    def remove_tool(name: str):
        with closing(open_library(library)) as connection:
            delete_tool(connection, name)
            return library_panel(connection)

    @app.delete("/api/run-settings/{key:path}")
    def remove_setting(key: str):
        with closing(open_library(library)) as connection:
            delete_setting(connection, key)
            return library_panel(connection)

    def known_provider(name: str):
        found = find_provider(library, name)
        if found is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"There is no provider {name}.")
        return found

    def known_server(name: str):
        found = known_provider(name)
        if not found.custom:
            raise HTTPException(
                HTTPStatus.NOT_FOUND, f"There is no custom server {name}."
            )
        return found

    KnownProvider = Annotated[Provider, Depends(known_provider)]
    KnownServer = Annotated[Provider, Depends(known_server)]

    @app.get("/api/providers")
    async def read_providers():
        return provider_list(home, library)

    @app.put("/api/providers/{name}/key")
    async def put_key(
        provider: KnownProvider,
        key: Annotated[str, Body(embed=True)],
    ):
        save_key(home, provider, key)
        return provider_list(home, library)

    @app.delete("/api/providers/{name}/key")
    async def delete_key(provider: KnownProvider):
        forget_key(home, provider)
        return provider_list(home, library)

    @app.post("/api/servers", status_code=HTTPStatus.CREATED)
    async def post_server(
        name: Annotated[str, Body()],
        base_url: Annotated[str, Body()],
        key: Annotated[str | None, Body()] = None,
    ):
        add_server(home, library, name, base_url, key)
        return provider_list(home, library)

    @app.put("/api/servers/{name}")
    async def put_server(
        server: KnownServer,
        base_url: Annotated[str, Body()],
        key: Annotated[str | None, Body()] = None,
    ):
        update_server(home, library, server, base_url, key)
        return provider_list(home, library)

    @app.delete("/api/servers/{name}")
    async def delete_server(server: KnownServer):
        remove_server(home, library, server)
        return provider_list(home, library)

    @app.get("/api/models")
    async def read_models():
        return await model_lists(home, library)

    @app.get("/api/folders")
    def read_folders(path: str = "~"):
        return list_folders(path)

    @app.get("/api/default-folder")
    async def read_default_folder():
        return {"folder": shown_folder(default_folder(library))}

    @app.put("/api/default-folder")
    async def put_default_folder(folder: Annotated[str | None, Body(embed=True)]):
        save_default_folder(library, chosen_folder(folder))
        return await read_default_folder()

    def starting_points():
        """Where a chat that hasn't picked starts: the Library's default namespace, and the
        model new chats start on."""
        return default_namespace(library), starting_model(home, library)

    def shown(chat, starts):
        """A chat as the API gives it: with the id of its run that is going, if one is, and
        the namespace and model its next message runs on."""
        namespace, model = starts
        if chat["model"] is not None:
            model = {"provider": chat["provider"], "name": chat["model"]}
        return {
            "id": chat["id"],
            "title": chat["title"],
            "created": chat["created"],
            "live": runs.live(chat["id"]),
            "namespace": chat["namespace"] or namespace,
            "model": model,
            "folder": shown_folder(chat["folder"]),
            "scratch_folder": short_path(scratch_folder(data, chat["id"])),
        }

    # The chat routes are coroutines, so they run on the event loop with the runs: what
    # they read of a run is whole, and no chat is deleted between this check and a route.
    async def known_chat(chat: Annotated[str, InPath(pattern=CHAT_ID)]):
        found = find_chat(library, chat)
        if found is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"There is no chat {chat}.")
        return found

    KnownChat = Annotated[dict, Depends(known_chat)]

    @app.get("/api/chats")
    async def read_chats():
        starts = starting_points()
        return [shown(chat, starts) for chat in list_chats(library)]

    @app.post("/api/chats", status_code=HTTPStatus.CREATED)
    async def post_chat():
        folder = starting_folder(default_folder(library))
        return shown(create_chat(library, folder), starting_points())

    @app.get("/api/chats/{chat}")
    async def read_chat(chat: KnownChat):
        going = runs.live(chat["id"])
        with closing(open_library(library)) as connection:
            turns = chat_turns(connection, chat["id"], going)
        return shown(chat, starting_points()) | {"turns": turns}

    @app.patch("/api/chats/{chat}")
    async def patch_chat(chat: KnownChat, change: ChatChange):
        if change.namespace is not None:
            with closing(open_library(library)) as connection:
                check_namespace(connection, change.namespace)
        model = change.model
        if model is not None and find_provider(library, model.provider) is None:
            raise SettingsError(f"There is no provider {model.provider}.")
        moved = "folder" in change.model_fields_set
        folder = chosen_folder(change.folder) if moved else None
        if change.title is not None:
            rename_chat(library, chat["id"], change.title)
        provider, name = (model.provider, model.name) if model else (None, None)
        choose(library, chat["id"], change.namespace, provider, name)
        if moved:
            choose_folder(library, chat["id"], folder)
        if model is not None:
            remember_pick(library, provider, name)
        return shown(find_chat(library, chat["id"]), starting_points())

    @app.delete("/api/chats/{chat}", status_code=HTTPStatus.NO_CONTENT)
    async def remove_chat(chat: KnownChat):
        # Read before anything goes, so settings that can't be read delete nothing.
        chosen = [each["folder"] for each in list_chats(library)]
        chosen += [default_folder(library), *runs.folders_in_use(besides=chat["id"])]
        keep = [each for each in chosen if each]
        # The chat goes first: a crash part way through leaves runs that no chat lists,
        # never a chat missing some of its turns.
        delete_chat(library, chat["id"])
        await runs.remove(chat["id"], keep)
        # After its workers exit, as they work in it; off the loop, as it may be large.
        scratch = scratch_folder(data, chat["id"])
        await asyncio.to_thread(remove_unless_kept, scratch, keep)

    @app.post("/api/chats/{chat}/runs", status_code=HTTPStatus.CREATED)
    async def start_run(
        chat: KnownChat, prompt: Annotated[str, Body(embed=True, min_length=1)]
    ):
        # Before the chat's turns are read: a run that is going has no turn.end yet.
        if runs.live(chat["id"]) is not None:
            raise RunInProgress(chat["id"])
        chosen = shown(chat, starting_points())
        namespace, model = chosen["namespace"], chosen["model"]
        with closing(open_library(library)) as connection:
            if not has_namespace(connection, namespace):
                raise LibraryError(
                    f"This chat's namespace {namespace} no longer exists — pick another."
                )
            turns = [
                read_turn(connection, run) for run in chat_runs(connection, chat["id"])
            ]
        earlier = [turn for turn in turns if turn is not None]
        if model is None:
            raise SettingsError("Pick a model for this chat, under the message box.")
        provider = find_provider(library, model["provider"])
        if provider is None:
            raise SettingsError(
                f"There is no provider {model['provider']} any more; pick another model."
            )
        key = run_key(home, provider)
        folder = working_folder(data, chat)
        client = {"base_url": provider.base_url, "api_key_env": provider.key_env}
        keys = {provider.key_env: key}
        # Any provider's key in the shell reaches the worker, so is withheld too.
        names = key_names(library)

        def config_for(run_folder):
            # A run's folder is named by its id.
            run = run_folder.name
            tools = run_folder / "tools"
            config = engine_config(library, namespace, tools, chat["id"], run, names)
            return config | {"model": model["name"], "client": client}

        run = runs.start(
            chat["id"],
            prompt,
            config_for,
            earlier,
            provider.name,
            keys,
            folder,
            names,
        )
        choose(library, chat["id"], namespace, provider.name, model["name"])
        if not earlier:
            title_new_chat(library, chat["id"], prompt)
        return {"run": run}

    def known_run(run: Annotated[str, InPath(pattern=RUN_ID)]):
        if not runs.exists(run):
            raise HTTPException(HTTPStatus.NOT_FOUND, f"There is no run {run}.")
        return run

    @app.post("/api/runs/{run}/stop")
    async def stop_run(run: Annotated[str, Depends(known_run)]):
        if not runs.stop(run):
            raise HTTPException(HTTPStatus.CONFLICT, f"Run {run} has already ended.")
        return {"run": run, "stopping": True}

    @app.get("/api/runs/{run}/events", response_class=EventSourceResponse)
    async def run_events(
        run: Annotated[str, Depends(known_run)],
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[int, Header(ge=0)] = 0,
    ):
        # EventSource sends no Last-Event-ID on its first connect, hence ?after=.
        async for event in runs.events(run, after=max(after, last_event_id)):
            yield ServerSentEvent(data=event, id=str(event["seq"]))

    # Written by the reader: FastAPI's own encoder recurses, and a shared file may nest a
    # record deep.
    @app.get("/api/pages/{page}")
    def read_a_page(page: str):
        with closing(open_library(library)) as connection:
            text = page_text(connection, page)
        if text is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, NO_PAGE)
        return Response(text, media_type="application/json")

    # The tab a page's link opens: page.js reads the page from the API and draws it.
    @app.get("/pages/{page}")
    def page_tab(page: str):
        with closing(open_library(library)) as connection:
            found = page_exists(connection, page)
        if not found:
            return PlainTextResponse(NO_PAGE, status_code=HTTPStatus.NOT_FOUND)
        return FileResponse(WEB / "page.html", headers=POLICY)

    # A mount at / answers every path, so API routes must be added above it.
    app.mount("/", Web(directory=WEB, html=True))
    return app
