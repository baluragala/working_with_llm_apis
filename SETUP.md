# Setup Guide

End-to-end setup for the **Working with LLM APIs** workshop on **Linux**, **macOS**, and **Windows** (both **Command Prompt** and **PowerShell**).

If you have never installed Python, Git, or VSCode before — start at step 1.
If you already have them — jump to step 3 (clone).

---

## Contents

1. [Install prerequisites](#1-install-prerequisites)
2. [Verify the installations](#2-verify-the-installations)
3. [Clone the repository](#3-clone-the-repository)
4. [Create and activate a virtual environment](#4-create-and-activate-a-virtual-environment)
5. [Install Python dependencies](#5-install-python-dependencies)
6. [Configure your API keys (`.env`)](#6-configure-your-api-keys-env)
7. [Set up VSCode](#7-set-up-vscode)
8. [Run your first script](#8-run-your-first-script)
9. [Optional: pre-class local-model download](#9-optional-pre-class-local-model-download)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Install prerequisites

You need three things: **Python 3.11** (3.10–3.12 also fine), **Git**, and **Visual Studio Code**.

### Linux (Ubuntu / Debian-based)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

VSCode — download the `.deb` from <https://code.visualstudio.com/Download> and install:

```bash
sudo apt install -y ./code_*.deb
```

> Fedora/Arch: use `dnf install python3.11 git` or `pacman -S python git` and grab VSCode from the same link.

### macOS

Easiest path is [Homebrew](https://brew.sh/). If you don't have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```bash
brew install python@3.11 git
brew install --cask visual-studio-code
```

### Windows

Install each from its official source. Click through with default options unless noted.

| Tool | Link | Notes |
|------|------|-------|
| Python 3.11 | <https://www.python.org/downloads/windows/> | **Tick "Add python.exe to PATH"** on the first installer screen. |
| Git for Windows | <https://git-scm.com/download/win> | Defaults are fine. |
| VSCode | <https://code.visualstudio.com/Download> | Defaults are fine. |

---

## 2. Verify the installations

Open a fresh terminal (close and reopen — important on Windows so PATH updates take effect) and run:

### Linux / macOS (Terminal)

```bash
python3 --version
git --version
code --version
```

### Windows (Command Prompt)

```cmd
python --version
git --version
code --version
```

### Windows (PowerShell)

```powershell
python --version
git --version
code --version
```

Each should print a version number. If `code --version` fails on macOS, open VSCode → press `Cmd+Shift+P` → run **Shell Command: Install 'code' command in PATH**.

---

## 3. Clone the repository

Pick a folder where you keep code, then clone.

### Linux / macOS

```bash
cd ~
git clone https://github.com/baluragala/working_with_llm_apis.git working_with_llm_apis
cd working_with_llm_apis
```

### Windows (Command Prompt)

```cmd
cd %USERPROFILE%
git clone https://github.com/baluragala/working_with_llm_apis.git working_with_llm_apis
cd working_with_llm_apis
```

### Windows (PowerShell)

```powershell
cd $HOME
git clone https://github.com/baluragala/working_with_llm_apis.git working_with_llm_apis
cd working_with_llm_apis
```

---

## 4. Create and activate a virtual environment

A virtual environment (`.venv`) keeps this project's packages separate from the rest of your system.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution-policy error, run this **once** in PowerShell, then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

After activation your prompt should show `(.venv)` at the front. To leave the venv later, run `deactivate`.

---

## 5. Install Python dependencies

With the venv active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls in `google-genai`, `openai`, `huggingface_hub`, `transformers`, `torch`, evaluation libraries, and a few utilities. Expect a few minutes on the first run; `torch` is the largest download.

---

## 6. Configure your API keys (`.env`)

Copy the template into a real `.env` file at the repo root.

### Linux / macOS

```bash
cp 01_secure_setup/.env.example .env
```

### Windows (Command Prompt)

```cmd
copy 01_secure_setup\.env.example .env
```

### Windows (PowerShell)

```powershell
Copy-Item 01_secure_setup\.env.example .env
```

Open `.env` in VSCode and paste your real keys. Where to get them is in [`README.md`](./README.md#get-your-keys).

`.env` is already listed in `.gitignore` — it will not be committed.

---

## 7. Set up VSCode

### a. Open the project

```bash
code .
```

(Run from inside the repo root, with the venv still active.)

### b. Install the Python extension

In VSCode: **Extensions** sidebar (`Ctrl+Shift+X` / `Cmd+Shift+X`) → search **Python** → install the one published by **Microsoft**. It bundles Pylance and Jupyter support.

### c. Select the interpreter

`Ctrl+Shift+P` (`Cmd+Shift+P` on macOS) → type **Python: Select Interpreter** → choose the entry pointing at `.venv` inside this folder. The path will look like:

- Linux/macOS: `./.venv/bin/python`
- Windows: `.\.venv\Scripts\python.exe`

You should now see `Python 3.11.x ('.venv')` in the bottom-right status bar.

### d. Run / debug a script

Open `01_secure_setup/test_connectivity.py`. Two ways to run it:

- **Run** — click the ▶ play button at the top-right of the editor.
- **Debug** — set a breakpoint by clicking left of a line number, then press `F5` and pick **Python File**. Use `F10` to step over and `F11` to step into.

The integrated terminal will pick up the venv automatically — you don't need to `activate` again inside VSCode once the interpreter is selected.

---

## 8. Run your first script

From the terminal (with the venv active) — works the same on every platform:

```bash
python 01_secure_setup/test_connectivity.py
```

It pings each provider you configured a key for and prints whether the call succeeded. If all three providers come back green, you're ready for the workshop.

After that, work through the folders in numeric order:

```
02_gemini  →  03_huggingface  →  04_local_model  →  05_openai  →  06_evaluation
```

Inside each folder, run scripts in numeric-prefix order.

---

## 9. Optional: pre-class local-model download

`04_local_model/` downloads **TinyLlama (~2 GB)** the first time it runs. If your conference / classroom Wi-Fi is slow, do this download at home the night before:

```bash
python 04_local_model/01_download.py
```

Once cached, it loads from disk in seconds.

---

## 10. Troubleshooting

**`python: command not found` on macOS/Linux** — use `python3` instead, or `brew link python@3.11`.

**`python` opens the Microsoft Store on Windows** — Python wasn't added to PATH. Reinstall and tick **"Add python.exe to PATH"** on the first screen, then open a new terminal.

**`Activate.ps1 cannot be loaded because running scripts is disabled`** — see the execution-policy fix in [step 4](#4-create-and-activate-a-virtual-environment).

**`pip install` fails on `torch`** — make sure you're on Python 3.10, 3.11, or 3.12 (not 3.13). Check with `python --version`.

**VSCode keeps using the wrong Python** — re-run **Python: Select Interpreter** and pick the `.venv` entry; close any old terminals (they keep the previous interpreter) and open a new one.

**`ModuleNotFoundError` when running a script** — the venv isn't active. Look for `(.venv)` in your prompt. If missing, re-run the activation command from [step 4](#4-create-and-activate-a-virtual-environment).

**API call returns `401` / `invalid key`** — open `.env`, confirm the key has no surrounding quotes or trailing spaces, save, and re-run.

---

That's it. If something here is wrong or unclear during a session, flag it — this doc gets better every cohort.
