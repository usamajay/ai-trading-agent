# Phase 0 — Set up your computer (beginner guide)

Goal: by the end, you have the tools installed, your MT5 demo account ready, and this project on your PC.
Time: about 1 hour. Do the steps in order. If any step shows red error text, copy it into Claude and ask.

---

## Step 1 — Open PowerShell
Press **Windows key**, type `PowerShell`, click **Windows PowerShell**. A blue/black window opens. This is where you paste commands (right-click pastes). Press **Enter** after each command.

## Step 2 — Install the tools (one command each)
`winget` is Windows' built-in app installer.

```powershell
winget install --id Git.Git -e
winget install --id GitHub.cli -e
winget install --id Microsoft.VisualStudioCode -e
winget install --id astral-sh.uv -e
```

What these are:
- **Git** — saves every version of your code (the "history" that makes the system traceable).
- **GitHub CLI (`gh`)** — lets you download and push to GitHub repos from the command line.
- **VS Code** — the editor where you can look at the code.
- **uv** — installs Python and the project's Python libraries for you.

**Close PowerShell and open it again after every install** so it notices the new tools. Check they work:

```powershell
git --version
gh --version
uv --version
```

Then install Python through uv, and add uv's folder to your PATH:

```powershell
uv python install 3.11
uv python update-shell
```
Close and reopen PowerShell again.

## Step 3 — Install Claude Code
```powershell
winget install Anthropic.ClaudeCode
```
(Alternative: `irm https://claude.ai/install.ps1 | iex`, or use the **Code** tab in the Claude desktop app.)
Close and reopen PowerShell, then run `claude --version`.

## Step 4 — Log in to GitHub from your PC
```powershell
gh auth login
```
Choose: **GitHub.com** → **HTTPS** → **Yes** (authenticate Git) → **Login with a web browser**. Copy the one-time code, press Enter, paste it in the browser, approve.

Tell Git your name (used on every saved version):
```powershell
git config --global user.name "Usama Jamil"
git config --global user.email "usamaandcs@gmail.com"
```

## Step 5 — MetaTrader 5 demo account
Use the existing Exness **Standard Demo** MT5 account (confirmed "Demo" in the Exness app). Have ready: **login number**, **password**, **server** (in MT5: File → Login to Trade Account shows the server). They go in a `.env` file in Phase 1 — never in code, never on GitHub.

## Step 6 — Download the project to your PC
```powershell
mkdir $HOME\Projects -Force
cd $HOME\Projects
gh repo clone usamajay/ai-trading-agent
cd ai-trading-agent
uv sync
uv run pytest
```
`uv sync` installs the Python libraries; `uv run pytest` should end with **3 passed**. If so, your setup works.

## Step 7 — Start Claude Code in the project
```powershell
cd $HOME\Projects\ai-trading-agent
claude
```
Then type:
> Read CLAUDE.md, docs/SPEC.md and docs/PHASE_1_TASKS.md. Explain Phase 1 to me in simple words, then start Task 1.1. Guide me step by step — I'm a beginner.

✅ Phase 0 is done when: `git`, `gh`, `uv`, `claude` all show versions; you have your demo login details; and `uv run pytest` shows 3 passed.
