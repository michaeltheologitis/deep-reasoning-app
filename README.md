# Deep Reasoning

A chat app for Dean's `deep_reasoner` engine. It runs in your browser, on your own computer. You ask a question, and
an AI agent answers it by writing and running Python, handing parts of the work to other agents when that helps. Every
answer opens into a tree that shows what each agent thought, ran and found.

You can also build your own **helpers**: agents with their own instructions, examples and tools, such as a statistics
helper, an editor that knows your style, or a small team that splits a job between them. You build them by asking the
built-in **librarian** in plain English, or by editing them yourself in the Library panel.

## Install and run

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and access to Dean's private GitHub repository
`DeanLight/deep_reasoner_beta` (if git asks for a password, run `gh auth login`). Then:

```
git clone https://github.com/michaeltheologitis/deep-reasoning-app.git
cd deep-reasoning-app
uv run deep-reasoning
```

It opens http://127.0.0.1:8000. Ctrl+C stops it, and `git pull` updates it. Use `--port 8001` for another port and
`--no-browser` to skip opening a tab.

## Keys and models

Add an API key in **settings** (bottom left): OpenAI, OpenRouter, Anthropic or Google Gemini, or any
OpenAI-compatible server such as vLLM or Ollama. A key already set in your shell works too (for Anthropic, the
variable is `DEEP_REASONING_ANTHROPIC_KEY`).

Each chat picks its model under the message box:
- **Ordinary chats** start on the model you picked last, or else on a default for your provider (gpt-6-luna on OpenAI).
- **Librarian chats** start on gpt-6.1-terra when you have an OpenAI key. In our tests it built helpers right 89% of
  the time, against 62% for gpt-6-luna, at about 5 cents per change. An older librarian chat keeps the model it
  started on.

You can switch any chat's model. Every answer shows what it cost and how long it took.

## How it works

- Each message you send starts a **run**. An agent writes some Python, reads what it prints, and repeats until it can
  answer.
- An agent can hand parts of the work to **sub-agents**. Click the `✻ Reasoned for…` line under an answer to open the
  tree: every agent, its thinking, its code and what it printed.
- The Python runs **on your computer, unsandboxed**, in the chat's working folder. Pick that folder with the folder
  button under the message box; otherwise the chat gets a scratch folder of its own. Agents can read and write files
  there.
- The Python has numpy, pandas, scipy, statsmodels and requests. For more packages, start the app with them:
  `uv run --with sympy deep-reasoning`.
- Press Esc (or **stop** on a phone) to stop a run. Each tab runs one message at a time.

## Namespaces: the agents you talk to

In the panel and the code, a helper is called a **namespace**. A namespace is a kind of agent, defined by:
- its instructions;
- its tools;
- its worked examples, which are solved tasks it copies the approach of (the engine calls these *decompositions*);
- the other namespaces it may hand work to.

You pick which namespace a chat talks to under the message box. Every Library starts with three.

- **root** is the general-purpose agent that every chat starts with. Every other namespace inherits its instructions,
  so a rule added to root applies everywhere.
- **librarian** builds and changes your helpers. Ask it in plain words: "make me a helper that summarises my sales
  CSV", "make its answers shorter", "undo that". It writes instructions, worked examples and custom Python tools, then
  tells you what it changed. It can't delete anything or change root: you do those in the Library panel, and it tells
  you how.
- **historian** answers questions about earlier messages ("how did you get that number?", "show that as a table") from
  what the run recorded, instead of guessing or redoing the work. Other agents hand such questions to it, so you rarely
  pick it yourself.

Every other namespace is one you or the librarian made. Root isn't told which helpers you have: pick one under the
message box, or say "hand this to the stats helper".

## A quick tour

1. Ask "What is the sum of the primes below 1000?", then click the `✻ Reasoned for…` line to see how it got there.
2. Ask "How did you get that?". Root hands the question to the historian.
3. Start a new chat, pick **librarian**, and ask "Make a helper called stats for statistics questions: compute with
   scipy and report the mean and standard deviation."
4. Start a new chat, pick **stats**, and ask it something.
5. Open **library** (top right) to see, and edit, what was made.

## The Library panel

**library** (top right) lists every namespace with its instructions, tools and worked examples, and lets you add,
edit and delete them. Its **Changes** tab lists recent changes, whoever made them (you, the librarian or an import),
each with what it was before and after.

To make a rule apply to every chat, add it to root:
1. Open **Namespaces**, then **root**, then **Edit YAML**.
2. Add your line under `system_suffix: |`, indented like the lines above it.
3. Save.

## Good to know

- Everything except your keys lives in one file, `~/.deep-reasoning/library.sqlite`: your helpers, chats and
  settings. Copy that file to back it up.
- Keys are kept apart, in `~/.deep-reasoning/.env`, readable only by you.
- Only open a Library file or import a config you trust. Custom tools are Python, and they run on your computer.
- A Library made by an older version keeps its own copies of root, librarian and historian, so it doesn't get newer
  built-in instructions. To get them, start a fresh Library (`--library ~/fresh.sqlite`) or run `reset`.

## Commands

- `uv run deep-reasoning import CONFIG.yaml` brings a `deep_reasoner` config (its namespaces, worked examples, tools
  and settings) into your Library, and prints what changed. It can't be undone in the app, so copy your Library first.
- `uv run deep-reasoning reset` deletes your Library, chats included, after asking you to type `delete`. Your keys and
  folders stay.
- Add `--library PATH` to any command to use another Library file.

The previous app, built on OpenHands, is in history at `a80bb2e`.
