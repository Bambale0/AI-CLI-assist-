"""
AI CLI Assistant v1.4.0
Production-ready CLI wrapper around OpenAI/OpenRouter-like endpoints.
"""

from __future__ import annotations
import os
import sys
import json
import re
import shlex
import argparse
import pathlib
import logging
import http.client
from typing import Dict, List, Sequence, Optional, Any, Tuple

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from yaspin import yaspin
from tqdm import tqdm

load_dotenv()  # читает .env рядом или в home

############################################################################
# Logging
############################################################################
LOG = logging.getLogger("aiclient")
logging.basicConfig(
    level=os.getenv("AI_CLI_LOG_LEVEL", "WARNING").upper(),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    stream=sys.stderr,
)

############################################################################
# Environment & constants
############################################################################
_DEFAULT_MODEL = os.getenv("AI_CLI_MODEL", "moonshotai/kimi-k2")
_DEFAULT_BASE = os.getenv("AI_CLI_API_BASE", "https://openrouter.ai/api/v1")
_API_KEY_ENV = ["OPENAI_API_KEY", "AI_CLI_API_KEY", "OPENROUTER_API_KEY"]


class AICLIOError(RuntimeError):
    """Общее исключение CLI с кодом выхода."""
    exit_code = 1


class SecurityError(AICLIOError):
    """Попытка выйти за пределы work_dir."""
    exit_code = 2


class ConfigError(AICLIOError):
    exit_code = 3


############################################################################
# Chat client
############################################################################
class AiChat:
    """Продуктивный клиент openai-like модели."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_base: str = _DEFAULT_BASE,
        work_dir: pathlib.Path | str = ".",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self.work_dir = pathlib.Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("Work dir: %s", self.work_dir)

        # ---- ключ ----
        self.api_key = self._resolve_api_key()

        # ---- history ----
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------ #

    def _resolve_api_key(self) -> str:
        for var in _API_KEY_ENV:
            if os.getenv(var):
                LOG.debug("Using API key from %s", var)
                return os.getenv(var, "")
        raise ConfigError(
            "No API key provided. "
            f"Set any of {', '.join(_API_KEY_ENV)} environmental variables or "
            "provide via --api-key argument."
        )

    # ------------------------------------------------------------------ #
    # Path security
    # ------------------------------------------------------------------ #
    def secure_path(self, path: str) -> pathlib.Path:
        """Resolves relative path and raises on escape attempt."""
        try:
            candidate = (self.work_dir / path).resolve()
            candidate.relative_to(self.work_dir)
            return candidate
        except ValueError as exc:
            raise SecurityError(f"Path traversal detected: {path}") from exc

    # ------------------------------------------------------------------ #
    # I/O files
    # ------------------------------------------------------------------ #
    def read_file(self, path: str) -> str:
        full = self.secure_path(path)
        if not full.is_file():
            raise AICLIOError(f"File {full} not found")
        try:
            return full.read_text(encoding="utf-8")
        except OSError as exc:
            raise AICLIOError(f"Cannot read: {exc}") from exc

    def save_file(self, path: str, content: str, mkdir: bool = True) -> None:
        full = self.secure_path(path)
        if mkdir:
            full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        LOG.info("Saved %s", full)

    # ------------------------------------------------------------------ #
    # low-level API
    # ------------------------------------------------------------------ #
    def _llm_request(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with requests.Session() as s:
            for n in range(self.max_retries + 1):
                try:
                    rsp = s.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    if rsp.status_code != 200:
                        raise RuntimeError(f"API error {rsp.status_code}: {rsp.text}")
                    return rsp.json()["choices"][0]["message"]["content"].strip()
                except (
                    requests.RequestException,
                    http.client.HTTPException,
                ) as exc:  # noqa
                    if n == self.max_retries:
                        raise AICLIOError("LLM request failed permanently") from exc
                    tqdm.write(f"Retrying ({n+1})...")

    # ------------------------------------------------------------------ #
    # high level API
    # ------------------------------------------------------------------ #
    def chat(self, user_message: str, file_content: Optional[str] = None) -> str:
        text = f"{user_message}\n\n<attachment>\n{file_content}\n</attachment>"
        if file_content is None:
            text = user_message

        messages = list(self.history) + [{"role": "user", "content": text}]
        answer = self._llm_request(messages)
        self.history.append(messages[-1])  # push last one, since _llm_request copies
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def edit_file(
        self, file_path: str, instruction: str
    ) -> Tuple[str, Optional[pathlib.Path]]:
        """Returns new content and full path or raises."""
        original = self.read_file(file_path)
        prompt_instructions = (
            f"Edit file {file_path} by instruction:\n{instruction}\n\n"
            "Current content:\n```\n" + original + "\n```\n"
            "Return ONLY updated file content."
        )
        new_content = self._llm_request(
            [{"role": "user", "content": prompt_instructions}]
        )
        new_content = self._extract_code_block(new_content)
        self.save_file(file_path, new_content, mkdir=True)
        return new_content, self.secure_path(file_path)

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract content between triple-backticks."""
        m = re.search(r"```(?:.*?\n)?(.*)```", text, re.S)
        return m.group(1).strip() if m else text.strip()

    def clear_history(self) -> None:
        self.history.clear()

    def pretty_history(self, limit: int = 12, head: int = 120):
        for idx, entry in enumerate(self.history[-limit:], 1):
            role = "U" if entry["role"] == "user" else "A"
            print(f"{idx:>2}{role}: {entry['content'][:head]}$$")

############################################################################
# CLI layer
############################################################################

_console = Console()

_PROMPT_HISTORY = pathlib.Path.home() / ".ai_cli_history"

COMMANDS = {
    "help",
    "exit",
    "attach",
    "create",
    "edit",
    "save",
    "history",
    "clear",
    "pwd",
    "cd",
    "ls",
}
_COMPLETER = WordCompleter(sorted(COMMANDS), ignore_case=True)


# ------------------------------------------------------------------ #


def _pretty_print(text: str, *, title: str = "AI") -> None:
    if text.startswith("[Error") or text.startswith("Traceback"):
        _console.print(Panel(text, title=f"⚠️ {title}", style="red bold"))
    else:
        _console.print(Panel(Markdown(text), title=f"🤖 {title}", style="cyan"))


def _print_help():
    _console.print(
        """[bold green]Built-in commands[/bold green]
  [cyan]attach <path>[/cyan]      -- attach file content to prompt
  [cyan]create <path> <instr>[/cyan] -- generate new file
  [cyan]edit <path> <instr>[/cyan]   -- edit existing file via AI
  [cyan]save <path>[/cyan]          -- appends last AI answer to file
  [cyan]history[/cyan]              -- show last dialogue snippets
  [cyan]clear[/cyan]                -- wipe dialogue history
  [cyan]pwd[/cyan]                  -- show working dir
  [cyan]cd <path>[/cyan]            -- change work dir
  [cyan]ls[/cyan]                   -- list current directory
  [cyan]exit[/cyan] / [cyan]Ctrl-D[/cyan]         -- quit"""
    )


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #
def main(cmdargs: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ai-cli", description="Interactive AI CLI chat."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=_DEFAULT_MODEL,
        help=f"model name (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-b",
        "--base",
        default=_DEFAULT_BASE,
        help="base url (openrouter / local)",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        default=".",
        help="working directory (default: ./ai_workspace)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="more logs, repeatable"
    )
    parser.add_argument("--api-key", help="explicit API key")
    args = parser.parse_args(cmdargs)

    if args.verbose:
        logging.getLogger().setLevel({1: "INFO", 2: "DEBUG"}.get(args.verbose, "DEBUG"))

    # inject key directly
    if args.api_key:
        os.environ["AI_CLI_API_KEY"] = args.api_key

    model = AiChat(
        model=args.model,
        api_base=args.base,
        work_dir=args.work_dir,
    )

    _console.print(
        Panel.fit(
            f"AI CLI {__import__('ai_cli').__version__}\n"
            f"Model: [cyan]{model.model}[/cyan]\n"
            f"Work dir: [dim]{model.work_dir}[/dim]\n"
            f"Try [green]help[/green] for commands",
            title="💬 AI CLI Assistant",
        )
    )

    try:
        _run_loop(model=model)
    except (KeyboardInterrupt, EOFError):
        _console.print("\n:wave: Goodbye!")
    except SystemExit as e:
        raise e from None
    except Exception:
        LOG.exception("Unhandled exception")
        _console.print(
            Panel.fit(
                traceback.format_exc(),
                title="[red]Fatal[/red]",
                style="red",
            )
        )
        sys.exit(1)


def _run_loop(model: AiChat) -> None:
    HISTORY_FILE = _PROMPT_HISTORY
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    while True:
        user_line = (
            prompt(
                "ai> ",
                completer=_COMPLETER,
                history=FileHistory(str(HISTORY_FILE)),
            )
            .strip()
            .lstrip()
        )
        if not user_line:
            continue

        # Coarse tokenising
        cmd_parts = shlex.split(user_line)
        cmd = cmd_parts[0].lower()
        rest = cmd_parts[1:]

        if cmd == "exit":
            return

        if cmd == "help":
            _print_help()
            continue

        if cmd == "history":
            for idx, msg in enumerate(model.history[-12:], 1):
                role = "U" if msg["role"] == "user" else "A"
                _console.print(f"{idx:>2}{role}| {msg['content'][:120]}…")
            continue

        if cmd == "clear":
            model.clear_history()
            _console.print(":broom: History cleared.")
            continue

        if cmd == "pwd":
            _console.print(model.work_dir)
            continue

        if cmd == "ls":
            for entry in sorted(model.work_dir.iterdir()):
                icon = "📁" if entry.is_dir() else "📄"
                _console.print(f"  {icon} {entry.name}")
            continue

        if cmd == "cd":
            if not rest:
                _console.print("[red]cd needs path[/red]")
                continue
            dest = rest[0]
            try:
                new = model.secure_path(dest)
                if not new.is_dir():
                    _console.print(f"[red]{new} is not directory[/red]")
                    continue
                model.work_dir = new
            except SecurityError as e:
                _console.print(f"[red]{e}[/red]")
            continue

        if cmd == "attach":
            if not rest:
                _console.print("[red]attach needs path[/red]")
                continue
            try:
                content = model.read_file(rest[0])
                _console.print(f"Attached {rest[0]} ({len(content)} chars)")
            except AICLIOError as e:
                _console.print(f"[red]{e}[/red]")
            continue

        if cmd == "create":
            if len(rest) < 2:
                _console.print("[red]create <path> <instruction..>")
                continue
            path = rest[0]
            instruction = " ".join(rest[1:])
            try:
                content, _ = model.edit_file(path, instruction)
                _console.print(Panel(content, title=f"✅ Created {path}"))
            except Exception as e:
                _console.print(Panel(str(e), title=":warning:", style="red"))
            continue

        if cmd == "edit":
            if len(rest) < 2:
                _console.print("[red]edit <path> <instruction..>")
                continue
            path = rest[0]
            instruction = " ".join(rest[1:])
            try:
                content, _ = model.edit_file(path, instruction)
                _console.print(Panel(content, title=f"✏  Edited {path}"))
            except Exception as e:
                _console.print(Panel(str(e), title=":warning:", style="red"))
            continue

        if cmd == "save":
            if not rest:
                _console.print("[red]save needs <path>[/red]")
                continue
            path = rest[0]
            try:
                if not model.history:
                    _console.print("[dim]no answer to save[/dim]")
                    continue
                last = next(
                    (x["content"] for x in reversed(model.history) if x["role"] == "assistant"),
                    "",
                )
                model.save_file(path, last)
                _console.print(f"💾 Saved last answer to {path}")
            except Exception as e:
                _console.print(Panel(str(e), style="red"))
            continue

        # ---------- regular chat
        answer = model.chat(user_line)
        _pretty_print(answer)


############################################################################
# Entrypoint
############################################################################
if __name__ == "__main__":
    main()