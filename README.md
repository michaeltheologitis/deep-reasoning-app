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
- The chat used last is listed first, and a reload opens it. **+ New chat** goes to an empty chat if there is one,
  so empty chats don't pile up. A title is cut past 200 characters. An unsent message stays with its chat.
- Several tabs on one app keep in step: coming back to a tab, or opening a chat, shows the chats, messages, runs,
  names and picks another tab changed, and follows a run another tab started. A chat deleted in another tab leaves
  the list with a note naming it.
- Each message is a run, shown live as a tree of sub-agents you open level by level, waiting for the model until its
  first answer; the nodes you open stay open after a reload. A sub-agent on another model than the chat's shows it on
  its row. Esc stops a run, saying so at once, though a cell busy in C (a long `time.sleep`) holds it up a few seconds.
  One run goes at a time in a tab: a message sent meanwhile is kept, and the hint says which chat's run is going.
- Each run shows its time, sub-agent count, cost and model. A call is priced from the price list of the vendor whose
  address it went to (OpenAI, OpenRouter, Anthropic, Google), or by the server's own figure where it sends one, as
  OpenRouter does; so a vendor's model name on a server of your own costs N/A, as does a run whose only calls never
  finished (its server failed them, or you stopped it during its first), never $0.
- A provider's error leads with what it means ("The provider has no model x (404)", "Couldn't reach the provider"),
  its own words after.
- An answer that holds Markdown (headings, lists, tables, code, links, bold) shows as Markdown; any other shows
  exactly as written, so `List<String>`, `__init__` or `2*3*4` stay as they are; bold works where it sits against
  the letters, as Chinese, Japanese, Korean or Thai often do. Raw HTML always shows as text, an
  image as a link to it (never loaded), and every link opens in a new tab. A run that ran out of turns says so in
  place of an answer, and a long prompt or answer shows its first lines with a **show all** toggle.
- Follow-ups see the chat's earlier messages and answers as the conversation's own turns, and can ask how one was
  worked out, what its sub-agents did or what went wrong. A new Library has a `historian` namespace, which reads the
  chat's earlier runs (each message's sub-agents, thoughts and cells) with the app's own `history` tool, and root
  hands such questions to a sub-agent there, with your message as you wrote it and the conversation so far: root's
  worked examples, which every namespace inherits, show the hand-off, and the historian's show reading `history`
  before answering. `history` lists each sub-agent with its result, seconds and cost (`agents`), what went wrong
  anywhere in a message, the top agent's own cells included, with each agent's namespace (`errors`), finds any text
  you name (`find`), and cuts each text past 2,000 characters, so a huge output can't swamp the model (`full` reads
  one step whole). A namespace whose `spawn` list leaves `historian` out can't hand off, nor can any once
  `historian` is deleted: a run there is shown, in each hand-off's place, the same question answered by saying it
  can't, so the agent says so rather than guess or pass the question to another namespace; a hand-off of your own to
  a namespace it can't reach is shown answered there as best it can. The Library is left as it is; the panel says so
  under the namespace's Hand-offs, and deleting `historian` warns of it (a new Library's `librarian` lists it in its
  `spawn`, so take it out there first). If you take `history` out of it, delete
  root's hand-off examples (open each in the Decompositions tab and Delete it): they send questions where nothing
  can read the history. If you take `page` out of it, delete `show it on a page`, `put a table on a page` and `find
  where it really went wrong` from root and historian.
- An older Library keeps what it has: one made since `history` came reads it at root, with a worked example there,
  and needs nothing. For the historian, run `reset`, which deletes the Library and every chat, or add `historian` in
  the panel (`tools: [history, page]`, `spawn: []`, and the examples a new Library has) and give root the hand-off
  examples a new Library has in place of its own `explain an earlier answer`. If your historian lacks `page`, add it
  to its tools and give root and historian the examples a new Library has. One made after `page` came but before
  `errors` has a `show it on a page` that prints the page's `/pages/…` link: replace it in root and historian with a
  new Library's, and add a new Library's `find where it really went wrong` and `say when there is nothing to show`
  to both.
- The agent can show you something on a page of its own, with the app's `page` tool: Markdown text, tables, and steps
  or sub-agents of the chat's earlier messages, copied from their record and drawn as the chat draws them. A line
  under the answer, and a step in its tree, open the page in a new tab; the answer itself says what was found, and
  never holds the link. A page opened again in the same message is updated in place, so it is linked once. Pages
  are kept in the Library file and deleted with their chat. A new Library grants `page` in `historian`, beside
  `history`, with worked examples of showing an earlier answer or an error on a page, and of making none when
  there is nothing to show, so asking to be shown how an answer was reached goes there too.
- Each chat has its own model, namespace and working folder, picked under the message box (typing in the model or
  namespace menu filters it); a change applies from the next message.
- Light and dark themes, each with text at WCAG AA contrast; works at phone width, where the chats are a drawer
  (☰; Esc shuts it), and with no internet: the page loads nothing but the app.

**Models and keys**, in **settings** (bottom left)
- Vendors and their key variables: OpenAI `OPENAI_API_KEY`, OpenRouter `OPENROUTER_API_KEY`, Anthropic
  `DEEP_REASONING_ANTHROPIC_KEY`, Google Gemini `GEMINI_API_KEY`.
- Custom servers: any OpenAI-compatible URL (vLLM, Ollama, …), with an optional key.
- The model menu shows the current models of each vendor you have a key for, and all of a custom server's; typing
  searches the vendor's full list, but for its speech, image, embedding and other models no chat runs on, and any
  model name is accepted. A name a provider lists exactly comes first, so Enter picks it. Under the message box the
  model shows with the provider that serves it.
- Keys come from settings or your shell. Settings keeps them in `~/.deep-reasoning/.env`, readable only by you, and
  never shows them again. A saved key wins over the shell's.
- A custom server's key goes only to the address you saved the server with; a server of the same name at another
  address, as in a shared Library file, gets no key until you give it one.
- Runs never read or receive `ANTHROPIC_API_KEY`: Claude Code, as a namespace's backbone or tool, would bill that key
  instead of your Claude subscription.

**Working folders**
- The agent's Python runs in the chat's folder, so files it writes are there for the next message.
- A scratch folder by default, in `~/.deep-reasoning` (hover the folder's name for its path), deleted with the chat;
  deleting a chat whose scratch folder holds files says so first.
  A folder you pick is never deleted.
- Settings has a default folder for new chats; one that isn't on this computer is marked missing and gives them a
  scratch folder.

**The Library**
- One SQLite file of namespaces, decompositions, custom tools and run settings, and your chats, with their past
  messages, runs and pages, and settings: `~/.deep-reasoning/library.sqlite`, or another with `--library PATH`. Sharing the
  file shares all of it but your keys, which stay in `.env`, and the runs' working files and scratch folders, which
  each file keeps in a folder of its own in `~/.deep-reasoning`.
- Copy the file while the app is stopped: while a run is going, the newest changes may still be in
  `library.sqlite-wal` beside it. The file stays in SQLite's WAL mode, so open a copy from a folder you can write to.
- What a run prints travels with the file. The app withholds any provider's key from it, but can't know other
  secrets a cell reads from your shell.
- The **library** panel (top right) adds, edits (renaming too) and deletes decompositions, root's included,
  as YAML, and adds, edits and deletes namespaces. It marks each decomposition ✓ or ✗ by whether its turns make a
  worked example like `deep_reasoner`'s own (the task; each assistant turn's `<think>` and one `<repl>`, then the
  `<observation>` it printed; `FinalAnswer` last), and opening a ✗ says why. The form warns as you type but still
  saves it, so a draft can be saved; from the next message, agents in its namespace and those under it are shown
  it as it is, so a new one starts in the namespace of the chat on screen, not root. The code isn't run, so an
  observation isn't checked against it. In its YAML editors Tab indents, Esc leaves the editor (a second closes
  the panel), and Ctrl+Enter or Cmd+Enter saves.
- Saving refuses what would make every run there fail: a decomposition holding `{{` or `{%` (which the engine reads
  as a Jinja template) whose template doesn't parse, names anything but what the engine fills in (`task`, `vars`,
  `funcs`, `coroutines` and the run settings' `prompt_template_variables`; write `{{ '{{' }}` for a literal `{{`),
  includes another template (`{% include %}`, `extends`, `import`, `from`) or reaches into Python internals (an
  attribute or key starting with `_`, or one such as `mro`),
  one over 1,000,000 characters, or one whose name starts or ends with a space or holds a character you can't see;
  in a namespace or one under it, a `reasoner` type the engine doesn't know; a `system_suffix` whose template fails
  as a decomposition's would; and a var that isn't a Python name, starts with `_`, is a keyword or one of Python's
  built-ins (`max`, `type`, …), or has the name of a tool its namespace sees. A
  template is checked, never run, so whether `vars.x` holds what it reads isn't. A stored decomposition that fails
  the check, as a shared Library may hold, is marked ✗ with why. Runs render templates sandboxed, so a shared
  Library's templates reach no Python internals: a worked example that tries fails the run, naming it, and a system prompt or
  suffix that tries is shown to the model as written.
- A save or delete in the panel is refused if what you opened was changed meanwhile (by the librarian, another tab
  or an import): your draft stays, beside what is stored now, and saving again replaces it. One deleted meanwhile
  says so, and its row goes. A delete the panel can tell would be refused (a tool a namespace grants, a namespace
  with namespaces under it or that another's `spawn` names) says why at once, without asking first. The panel, and
  the namespace menu under the message box, read the Library again whenever you open the panel or come back to the
  window. While the app isn't answering, the panel shows the Library as it last read it, and says so.
- The Tools tab shows each tool, with what agents are told it does (for one of the Library's own, as far as its
  Python says without running it) and the Python of the Library's own, and deletes one of the Library's that no
  namespace grants. Run settings can't be changed in the panel, but one the engine doesn't read, as an import may
  bring, can be deleted.
- In a chat in the `librarian` namespace, picked under the message box, the agent can add or edit a decomposition in
  a namespace you name, add a namespace or change one's own fields (its system suffix, vars, tools and spawn), and
  write a new tool of its own in Python (or replace one) that it then grants to a namespace, with the app's own
  `library` tool; a chat in a namespace without it can't, even through a sub-agent. It reads the namespaces (in brief,
  or one whole), the tools with their Python, and what this chat changed; a change touches only the fields it names,
  and says what changed. Changing a text such as a system suffix, it starts from the one stored and keeps every line
  you didn't ask it to remove, saying what it removed; a new namespace's own suffix holds only what it adds to its
  parent's; and it leaves `spawn` unset unless you ask, then keeping `historian` and `librarian` in it. A new
  Library's `librarian` knows what it is for, has worked examples of reading the Library, of writing a worked example
  and of undoing a change, and may hand questions about earlier messages to the historian but not delegate to itself
  (its `spawn`, which you may widen in the panel). A tool it writes, or a tool of yours it grants that no
  namespace granted yet, is built first to check it, as another chat would build it (in a process of its own, from an
  empty folder), and one that doesn't build there is refused: granted, it would stop every chat. That works only in a chat
  whose namespace runs the local REPL. A tool it writes can't take a built-in's or the app's name or replace a tool
  root grants or one built into the engine. The tool never
  changes root, a namespace's REPL or backbone (`repl`, `reasoner`), which namespaces grant `library`, `history` or `page`,
  the system prompt or the run settings; never grants a tool that runs Claude Code, which would bill your
  subscription; adds no namespace under one that grants `library` or runs Claude Code; never deletes, and never
  writes a key. A change applies from your next message. These limits guard the tool, not the file: the agent's Python isn't sandboxed, and could open the file
  itself. A new Library has `librarian`; in an older one, add a namespace with `tools: [library]`.
- Any chat can be asked about the Library ("what namespaces do I have?", "what does the historian's suffix say?",
  "add a namespace for legal questions", "make every chat answer in British English"): a new Library's root hands
  such questions to a `librarian` sub-agent, with your message as you wrote it. Outside a `librarian` chat it only
  reads: asked for a change, it says what it would change and that you pick `librarian` under the message box and ask
  there, or, for what the tool never does, how to do it in the panel (what every chat should do goes in root's
  `system_suffix`, with Edit YAML in the Namespaces tab). A namespace that can't reach `librarian` (left out of its
  `spawn` list, or deleted) answers such questions itself as best it can, as for the historian. A new Library's root
  also has a `system_suffix` saying what app its agents are in, which every namespace inherits; edit or delete it in
  the panel. An older Library keeps what it has: to have the same, give root a new Library's `system_suffix`, `read
  the Library` and `add a namespace with a worked example`, and `librarian` a new Library's `read the Library`.
- Each Library change the agent makes shows in the chat, under its answer and where it was made in the tree, and opens
  onto its YAML (or a tool's block and Python) before and after; ask the librarian to undo it, and it writes the
  before back from the record. The library panel and the
  namespace menu under the message box show the change at once.
- `uv run deep-reasoning import CONFIG.yaml` brings a `deep_reasoner` config's namespaces, decompositions, tools and
  run settings into the Library (each chat keeps its own model), replacing items of the same name and the root fields
  it sets, and prints what it changed and what it left out. It refuses, changing nothing, a file that isn't a config,
  the app's `historian` and `librarian` namespaces or one under them, a tool named as a built-in, a key anywhere in
  it, a `system_suffix` the panel would refuse, and a tool a namespace grants that doesn't build: it builds each such tool once to check it, from its own file
  alone and without your keys, so a file or module beside the config isn't carried in. Import only configs you trust.
- `uv run deep-reasoning reset` starts afresh: it deletes the Library file (`--library PATH` for another), with its
  chats and settings, and its run and scratch folders, after listing them and asking you to type `delete`
  (`--yes` skips the question). Your keys in `.env` and the folders you picked stay. It deletes only a Library file:
  a folder, a symlink or any other file is refused and left as it is.
- One app at a time serves a Library: a second `deep-reasoning` on the same file (by any path) refuses to start, and
  `import` and `reset` refuse while it runs, so stop it first. The lock is the empty `library.sqlite.lock` beside the
  file, let go when the app stops, even in a crash; on a file system without locks it says so and runs without. A
  `--library` in a folder that isn't there is refused rather than made.
- The Tools tab lists the app's own tools beside the built-in ones, and every tool in the Library; a tool of yours can't
  be named `history`, `library` or `page`, nor as a built-in.
- Every run builds every tool some namespace grants, whatever namespace its chat is in (only which agents can call a
  tool is per namespace): once any namespace grants a tool, its code runs at the start of every message in every
  chat, and a granted tool that stops building stops every chat until you take it out of the namespaces' tools in
  the panel; the failed run says which tool, and which namespaces grant it. A tool no namespace grants never runs,
  but once when the librarian writes or is about to grant it, to check it builds.

The previous app, built on OpenHands, is in history at `a80bb2e`; its design docs are on the `design-docs` branch.
