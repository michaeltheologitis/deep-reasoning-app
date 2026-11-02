# Deep Reasoning

A minimal browser chat for Dean's `deep_reasoner` engine, in the look of Claude Code.

## Install and run

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and GitHub access to Dean's private
`DeanLight/deep_reasoner_beta` (if `git ls-remote https://github.com/DeanLight/deep_reasoner_beta.git` asks for a
password, run `gh auth login` and let it set up git's credentials). Then:

```
git clone https://github.com/michaeltheologitis/deep-reasoning-app.git
cd deep-reasoning-app
uv run deep-reasoning
```

The first run takes a minute. It opens http://127.0.0.1:8000 in a tab (`--port 8001` for another port,
`--no-browser` for no tab); Ctrl+C stops it, and `git pull` updates it. Then add a key in **settings** (bottom left).

## What it supports

**Chats**
- Chats on the left: new, rename, delete. Saved in the Library file (below) with every message's run, they come back
  after a restart. Deleting a chat wipes its messages from the file.
- Each message is a run, shown live as a tree of sub-agents you open level by level. Esc stops it.
- Each run shows its time, sub-agent count, cost (N/A when the model's price isn't known) and model.
- Follow-ups see the chat's earlier messages and answers as the conversation's own turns, and can ask how one was
  worked out or what its sub-agents did. A new Library has a `historian` namespace, which reads the chat's earlier
  runs (each message's sub-agents, thoughts and cells) with the app's own `history` tool, and root hands such
  questions to a sub-agent there, with the conversation so far: root's worked examples, which every namespace
  inherits, show the hand-off, and the historian's show reading `history` before answering. In an older Library,
  add a `historian` namespace with `tools: [history]` in the panel, with examples like a new Library's, or run
  `reset`.
- Each chat has its own model, namespace and working folder, picked under the message box; a change applies from
  the next message.
- Light and dark themes; works at phone width.

**Models and keys**, in **settings** (bottom left)
- Vendors and their key variables: OpenAI `OPENAI_API_KEY`, OpenRouter `OPENROUTER_API_KEY`, Anthropic
  `DEEP_REASONING_ANTHROPIC_KEY`, Google Gemini `GEMINI_API_KEY`.
- Custom servers: any OpenAI-compatible URL (vLLM, Ollama, …), with an optional key.
- The model menu shows the current models of each vendor you have a key for, and all of a custom server's; typing
  searches the vendor's full list, and any model name is accepted.
- Keys come from settings or your shell. Settings keeps them in `~/.deep-reasoning/.env`, readable only by you, and
  never shows them again. A saved key wins over the shell's.
- A custom server's key goes only to the address you saved the server with; a server of the same name at another
  address, as in a shared Library file, gets no key until you give it one.
- Runs never read or receive `ANTHROPIC_API_KEY`: Claude Code, as a namespace's backbone or tool, would bill that key
  instead of your Claude subscription.

**Working folders**
- The agent's Python runs in the chat's folder, so files it writes are there for the next message.
- A scratch folder by default, in `~/.deep-reasoning` (hover the folder's name for its path), deleted with the chat.
  A folder you pick is never deleted.
- Settings has a default folder for new chats; one that isn't on this computer gives them a scratch folder.

**The Library**
- One SQLite file of namespaces, decompositions, custom tools and run settings, and your chats, with their past
  messages and runs, and settings: `~/.deep-reasoning/library.sqlite`, or another with `--library PATH`. Sharing the
  file shares all of it but your keys, which stay in `.env`, and the runs' working files and scratch folders, which
  each file keeps in a folder of its own in `~/.deep-reasoning`.
- Copy the file while the app is stopped: while a run is going, the newest changes may still be in
  `library.sqlite-wal` beside it. The file stays in SQLite's WAL mode, so open a copy from a folder you can write to.
- What a run prints travels with the file. The app withholds any provider's key from it, but can't know other
  secrets a cell reads from your shell.
- The **decompositions** panel (top right) adds decompositions as YAML, and adds, edits and deletes namespaces.
  It marks each decomposition ✓ or ✗ by whether its turns make a worked example like `deep_reasoner`'s own (the
  task; each assistant turn's `<think>` and one `<repl>`, then the `<observation>` it printed; `FinalAnswer` last, or
  `switch_namespace`, which hands the chat to another namespace's agent and answers with its answer),
  and opening a ✗ says why. The add form warns as you type but still adds it, so a draft can be saved. The code
  isn't run, so an observation isn't checked against it.
- In a chat in the `librarian` namespace, picked under the message box, the agent can add or edit a decomposition in
  a namespace you name, add a namespace or change one's own fields (its system suffix, vars, tools and spawn), and
  write a new tool of its own in Python (or replace one) that it then grants to a namespace, with the app's own
  `library` tool; a chat in a namespace without it can't, even through a sub-agent. A tool it writes is built once
  to check it, only in a chat whose namespace runs the local REPL, and can't take a built-in's or the app's name or
  replace a tool root grants or one built into the engine. The tool never
  changes root, a namespace's REPL or backbone (`repl`, `reasoner`), which namespaces grant `library` or `history`,
  the system prompt or the run settings; never grants a tool that runs Claude Code, which would bill your
  subscription; adds no namespace under one that grants `library` or runs Claude Code; never deletes, and never
  writes a key. A change applies from your next message. These limits guard the tool, not the file: the agent's Python isn't sandboxed, and could open the file
  itself. A new Library has `librarian`; in an older one, add a namespace with `tools: [library]`.
- Each Library change the agent makes shows in the chat, under its answer and where it was made in the tree, and opens
  onto its YAML (or a tool's block and Python) before and after; ask the agent to write the before back to undo it. The decompositions panel and the
  namespace menu under the message box show the change at once.
- `uv run deep-reasoning import CONFIG.yaml` brings in a `deep_reasoner` config, replacing items with the same name.
- `uv run deep-reasoning reset` starts afresh: it deletes the Library file (`--library PATH` for another), with its
  chats and settings, and its run and scratch folders, after listing them and asking you to type `delete`
  (`--yes` skips the question). Your keys in `.env` and the folders you picked stay. Stop the app first. It
  deletes only a Library file: a folder, a symlink or any other file is refused and left as it is.
- The Tools tab lists the app's own tools beside the built-in ones, and every tool in the Library; a tool of yours can't
  be named `history` or `library`.
- Every run builds every tool some namespace grants, whatever namespace its chat is in (only which agents can call a
  tool is per namespace): once any namespace grants a tool, its code runs at the start of every message in every
  chat, and a granted tool that stops building stops every chat until you take it out of the namespaces' tools in
  the panel. A tool no namespace grants never runs.

The previous app, built on OpenHands, is in history at `a80bb2e`; its design docs are on the `design-docs` branch.
