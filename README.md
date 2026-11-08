# Deep Reasoning

A minimal browser chat for Dean's `deep_reasoner` engine, in the look of Claude Code: agents answer each message by
reasoning in Python and handing parts to sub-agents, and you watch the whole run as a tree.

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
`--no-browser` for no tab); Ctrl+C stops it, and `git pull` updates it.

**Add a key** in **settings** (bottom left): OpenAI `OPENAI_API_KEY`, OpenRouter `OPENROUTER_API_KEY`, Anthropic
`DEEP_REASONING_ANTHROPIC_KEY` or Google Gemini `GEMINI_API_KEY`, or add a custom server (any OpenAI-compatible URL:
vLLM, Ollama, …). A key already set in your shell works too. A new chat starts on the model you picked last, or else
on a current model of the first provider with a key; pick another under the message box.

## Concepts in five lines

- A **chat** is a conversation; each message you send starts a **run**, which works out its answer.
- In a run, an **agent** works by writing Python and reading what it prints, and may hand parts to **sub-agents**,
  each an agent of its own; the run's tree shows them all.
- A **namespace** is a kind of agent: its instructions (`system_suffix`), tools, preloaded Python variables (`vars`),
  worked examples, and where it may hand work (`spawn`). Chats start in `root`; every namespace inherits from root,
  and `stats.tests` from `stats`.
- A **worked example**, which the engine calls a **decomposition**, is a solved task shown to the agents of its
  namespace (and those under it), so they work the same way.
- The **Library** holds all of these, edited in the **library** panel (top right) or by the **librarian**, a
  namespace that writes it for you; the **historian** is one that answers questions about earlier messages from what
  was recorded.

## Your first 10 minutes

1. **Ask something:** "What is the sum of the primes below 1000?" and Enter. The answer comes under a line such as
   `▸ ✻ Reasoned for 4.2s · $0.0012 · gpt-…`.
2. **Open the tree:** click that line. It opens onto the agent's thoughts (✻), each Python cell with what it
   printed, and the final answer; click a cell for its whole code. A cell that handed work out opens onto its
   sub-agents, each a row you open in turn.
3. **Ask how it got there:** "How did you get that?". Root hands the question to a sub-agent in `historian`, which
   reads the record of your earlier messages (code, outputs, sub-agents) and answers from it, rather than guess or
   redo the work. Open the tree to see the hand-off.
4. **Make a specialist:** click **+ New chat**, click `root ▾` under the message box, pick `librarian`, and ask "Add
   a namespace called stats for statistics questions: compute with numpy and report the mean and standard
   deviation." It writes the namespace and says what it changed; the **◆ Library** line under its answer opens onto
   the YAML it wrote. Ask it to undo a change and it writes the before back.
5. **Use it:** in a new chat, pick `stats` under the message box and ask away. Or, in any chat, name it: "Hand this
   to the stats namespace: …". Root isn't told which namespaces you have, so it won't pick yours on its own.

**To change every chat**, edit root's `system_suffix`, which every namespace inherits: in the library panel's
**Namespaces** tab, open `root`, **Edit YAML**, add a line at the end of the text under `system_suffix: |`, indented
like the lines above it (unindented, it is no longer part of the text and the YAML is refused), and Save:

```yaml
system_suffix: |
  …the lines already there…
  Answer in British English.
```

It applies from the next message. The librarian never changes root; asked to, it says how to do this.

**One message at a time:** a tab runs one message at a time, across all its chats. A message sent while another
chat's run is going stays in the box, and the hint names the chat whose run is going; open that chat and press Esc
(on a touch screen, **stop**) to stop it. A run another tab started shows here, and stops here, but holds up nothing.

**The agent's Python** runs on your computer by default, unsandboxed, in the chat's working folder, with the app's
own Python packages: numpy, pandas (with tabulate, so `to_markdown()` works) and requests are there; scipy, sympy
and matplotlib are not. Add packages by starting the app with them: `uv run --with scipy --with sympy
deep-reasoning`. A module file in the chat's folder imports too.

## What it supports

**Chats**
- Chats on the left: new, rename, delete. Saved in the Library file with every message's run, they come back after a
  restart; deleting one wipes its messages from the file, and its scratch folder (it says so first if that holds
  files, or a run going in it, in any tab, may yet write some).
- The chat used last is listed first, and a reload opens it; **+ New chat** reuses an empty chat, the one on screen or
  one this tab made, never one another tab is setting up; an unsent message stays with its chat, and one sent while
  its chat loads goes once it has; a title is cut past 200 characters, and one cut to fit shows whole in its tooltip.
- Several tabs on one app keep in step: coming back to a tab shows the chats, messages, runs, picks, settings and
  theme another tab changed, and follows a run another tab started; a chat deleted in another tab leaves the list
  with a note.
- Each message's run shows live as a tree of sub-agents you open level by level; what you open stays open after a
  reload. A sub-agent on another model than the chat's names it on its row.
- Esc stops a run, saying so at once, even pressed just as the message goes, though a cell busy in C (a long
  `time.sleep`) holds it up a few seconds. A run the app stopped under reads failed, and what it left going, cut off.
- Each run shows its time, sub-agent count, cost and model. A call is priced by the server's own figure where it sends
  one (OpenRouter does), else from the price list of the vendor whose address it went to (OpenAI, OpenRouter,
  Anthropic, Google). "cost N/A" means a call was cut off before it finished, or its price isn't known, as for a
  vendor's model on a server of your own; a run that made no call costs $0. Costs read to a hundredth of a cent
  under a dollar ($0.0123; one less than $0.0001 as <$0.0001) and to the cent from a dollar ($1,234.56), so a
  chat's runs add up to its total.
- A provider's error leads with what it means ("The provider has no model x (404)", "Couldn't reach the provider"),
  its own words after.
- An answer holding Markdown shows as Markdown; any other shows exactly as written (`List<String>`, `__init__`,
  `2*3*4`). Raw HTML shows as text, an image as a link (never loaded), and links open in a new tab. A long prompt or
  answer shows its first lines with **show all**; a run that ran out of turns says so.
- Follow-ups see the chat's earlier messages and answers, and can ask how one was worked out, whether it was
  computed, what its sub-agents did or what went wrong: root hands those to the `historian`.
- The agent can show you a page of its own (Markdown, tables, and steps or sub-agents of earlier messages, drawn as
  the chat draws them), linked under its answer and opening in a new tab; pages are deleted with their chat. A new
  Library grants this to the historian, so "show me on a page how you got that" goes there.
- Each chat has its own model, namespace and working folder, picked under the message box (typing filters the
  menus); a change applies from the next message. A folder that is gone, or a model whose provider was removed, is
  marked missing; a namespace named with more than plain ASCII, as only a shared file may hold, shows escaped
  (`'r\u043eot'`), as the Library panel writes it.
- Light and dark themes with text at WCAG AA contrast; phone width, where the chats are a drawer (☰; Esc shuts it)
  and the Library and settings fill the screen, each leaving what is under it out of reach; on a touch screen a
  **send** button sends, Enter makes a new line, a chat picked from the list leaves the keyboard down, and rows and
  tree lines are a finger's height. A skip link leads to the message box.
  No internet needed: the page loads nothing but the app.

**Models and keys**
- The vendors above, and custom servers (any OpenAI-compatible URL, with an optional key).
- The model menu shows the current models of each vendor you have a key for, and all of a custom server's; typing
  searches every provider's whole list of chat models, in any case, and Enter picks the first model listed (an exact
  name first); any other name is accepted, after every listed one, on the chat's own provider first.
  The model shows with its provider.
- Settings keeps keys in `~/.deep-reasoning/.env`, readable only by you, and never shows them again; a saved key wins
  over the shell's.
- A custom server's key goes only to the address you saved it with: a server of the same name elsewhere, as in a
  shared Library file, gets none until you give it one.
- Runs never read or receive `ANTHROPIC_API_KEY`: Claude Code, as a namespace's backbone or tool, would bill it
  instead of your Claude subscription.

**Working folders**
- The agent's Python runs in the chat's folder, so files it writes are there for the next message: by default a
  scratch folder in `~/.deep-reasoning` (hover its name for the path), deleted with the chat; a folder you pick is
  never deleted.
- Settings has a default folder for new chats; one that isn't on this computer is marked missing, and new chats get
  a scratch folder.

**The Library**
- One SQLite file holds the namespaces, worked examples, custom tools and run settings, and your chats with their
  runs and pages, and settings: `~/.deep-reasoning/library.sqlite`, or `--library PATH`. Sharing the file shares all
  of it but your keys (in `.env`) and the runs' working files and scratch folders.
- Copy it while the app is stopped (a running app may hold its newest changes in `library.sqlite-wal`), and open a
  copy from a folder you can write to. What a run prints travels with the file: the app withholds providers' keys
  from it, but can't know other secrets a cell reads from your shell.
- The library panel's tabs: **Decompositions** adds, edits and deletes worked examples as YAML, root's included;
  **Namespaces** adds, edits (as YAML) and deletes namespaces, showing each field and where it comes from, and the
  run settings, read only (one the engine doesn't read can be deleted); **Tools** shows each tool, what agents are
  told it does and a custom tool's Python, and deletes one of yours no namespace grants; **Data** shows each
  namespace's `vars` that are tables (a list of records) as tables, to read (edit them in Namespaces). In its
  editors Tab indents, Esc leaves, and Ctrl+Enter or Cmd+Enter saves.
- Each worked example is marked ✓ or ✗ by whether its turns are shaped like `deep_reasoner`'s own (the task; each
  assistant turn's `<think>` and one `<repl>`, then the `<observation>` it printed; `FinalAnswer` last), and a ✗
  says why; it still saves, as a draft agents are shown from the next message. A new one starts in the namespace of
  the chat on screen. Its code isn't run.
- Saving refuses what would fail every run there: a template (`{{` or `{%`) the engine can't render, that names
  anything but what it fills in (`task`, `vars`, `funcs`, `coroutines`, the run settings'
  `prompt_template_variables`; write `{{ '{{' }}` for a literal `{{`), includes another or reaches into Python
  internals; over 1,000,000 characters; a name with outer spaces or invisible characters; a namespace named oddly,
  with a backbone the engine doesn't know, or with a var whose name isn't a Python name, starts with `_`, is a
  keyword or built-in, or is a tool's; and a namespace granting a tool of yours that no namespace granted yet, if it
  doesn't build (built first, up to a minute).
- A save or delete is refused if what you opened changed meanwhile (the librarian, another tab or an import): your
  draft stays beside what is stored. Any refusal stays shown, and any but that one keeps Save off, until you change
  what you wrote. A delete the panel knows would be refused says why at once. The panel and the
  namespace menu read the Library again when you open the panel or come back to the window.
- Each run tells the agents of every namespace but root which namespace they are in, so "hand this to stats" reaching
  `stats` is done there; of one whose REPL isn't the one above it (root: the local one), where their Python runs;
  and, when it sets `spawn`, where they can hand work. The panel shows that line under its System suffix and marks a
  listed namespace that is gone; the librarian can't store it in a suffix.
- A shared Library's rows the engine can't read are left out, so the rest works, and so are worked examples whose
  template a save would refuse and namespaces whose name could pass for another (`Root`, or one holding anything but
  printable ASCII, as no saved name does); the panel marks each ✗ with why, to fix or delete. Its templates run
  sandboxed, reaching no Python internals.
- Every run builds every tool some namespace grants, whatever its chat's namespace: a granted tool's code runs at the
  start of every message in every chat, and one that stops building stops every chat until you take it out of the
  namespaces' tools (the failed run names the tool and the namespaces granting it). A tool no namespace grants never
  runs, but to check it builds.
- A new Library's root has a `system_suffix`, which every namespace inherits, telling agents about the app and how to
  answer; read, edit or delete it in the panel.

**The librarian**
- In a chat in `librarian`, the agent adds or edits worked examples, adds namespaces and changes their
  `system_suffix`, `vars`, `tools` and `spawn`, and writes Python tools of its own and grants them, with the app's
  `library` tool. A chat in another namespace can't, even through a sub-agent.
- It changes only what you name, keeps every line of a text you didn't ask to remove (saying what it removed),
  leaves `spawn` unset unless you ask, makes team members siblings rather than children of the planner, unless
  they should inherit its suffix and worked examples, and says when a worked example disagrees with its namespace's
  suffix.
- Each change shows under its answer and in the tree, opens onto its YAML (or a tool's Python) before and after, and
  applies from your next message; the panel and namespace menu show it at once. Ask it to undo one and it writes the
  before back from the record.
- A tool it writes, or a tool of yours it grants afresh, is built first as another chat would build it, and refused
  if it doesn't build; that works only in a chat whose namespace runs the local REPL.
- It never changes root, a namespace's REPL or backbone, which namespaces grant `library`, `history` or `page`, the
  system prompt or the run settings; never grants a tool that runs Claude Code (billed to your subscription), or adds
  a namespace under one that grants `library` or runs Claude Code; never replaces a tool root grants or one built
  into the engine, or names one as a built-in or app tool; never deletes; never writes a key. These limits guard the
  tool, not the file: the agent's Python isn't sandboxed, and could open the file itself.
- Any chat can ask about the Library ("what namespaces do I have?", "add a namespace for legal questions"): root hands
  it to a librarian sub-agent, which there only reads, and asked for a change says to pick `librarian` and ask there,
  or how to do it in the panel. A namespace that can't reach `historian` or `librarian` (left out of its `spawn`, or
  deleted) answers such questions itself as best it can, or says it can't; the panel says so under its Hand-offs.

**Commands**
- `uv run deep-reasoning import CONFIG.yaml` brings a `deep_reasoner` config's namespaces, worked examples, tools and
  run settings into the Library (each chat keeps its own model), replacing items of the same name and the root fields
  it sets, and prints what it changed, what it left out and what that changes in the app. It refuses, changing
  nothing, a file that isn't a config, the app's `historian` and `librarian` or one under them, a tool named as a
  built-in, a key anywhere in it, whatever the panel would refuse, an `entry_namespace` naming no namespace it would
  have, a `prompt_template_variables` change taking away a variable the Library's worked examples read, `max_iter` or
  `max_depth` below 1, and a granted tool that doesn't build (built once, from the config alone, with no key). It
  reads only files in the config's folder and below (`_compose`, `namespaces_dir`, `factory_from`); pass
  `--within FOLDER` when a config composes files from a folder above it, as `deep_reasoner`'s examples do. An import
  that fails or changes nothing leaves no Library (only its `.lock`). Import only configs you trust. An import can't
  be undone in the app: copy the Library file first to keep a way back.
- `uv run deep-reasoning reset` deletes the Library file (`--library PATH` for another), with its chats and
  settings, and its run and scratch folders, after listing them and asking you to type `delete` (`--yes` skips it).
  Your keys and the folders you picked stay; it deletes only a Library file.
- One app at a time serves a Library: a second on the same file refuses to start, and `import` and `reset` refuse
  while it runs. The lock is `library.sqlite.lock` beside the file; on a file system without locks it says so and
  runs without. A `--library` in a folder that isn't there is refused.

## Upgrading

A Library made by an older version keeps what it holds: a new version adds no namespaces, worked examples or suffixes
to it. For everything a new Library has, run `reset` (it deletes every chat), or open a new Library beside yours
(`uv run deep-reasoning --library ~/fresh.sqlite --port 8001`) and copy what you want from its panel into yours:
`historian` (`tools: [history, page]`, `spawn: []`) and `librarian` (`tools: [library]`, `spawn: [historian]`) with
their suffixes and worked examples, root's `system_suffix`, and root's hand-off examples, in place of an older
`explain an earlier answer` or `show it on a page`.

If you take `history` out of the historian, delete root's hand-off examples to it, which would send questions where
nothing can read the record; if you take `page` out, delete `show it on a page`, `put a table on a page` and `find
where it really went wrong` from root and historian.

The previous app, built on OpenHands, is in history at `a80bb2e`; its design docs are on the `design-docs` branch.
