"""
MedMARS CLI — Interactive Medical Assistant
============================================
Terminal interface for the MedMARS multi-agent medical system.

Usage:
    python cli.py
"""

import io
import json
import logging
import os
import re
import sys
import time
import warnings
import contextlib
from pathlib import Path

# Ensure project root is on sys.path so local modules (orchestration, etc.) resolve
# regardless of the working directory the command is invoked from.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Suppress noisy library output BEFORE any imports ──
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML

# ── Theme ──
_theme = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "danger": "bold red",
    "success": "bold green",
    "trace": "dim",
})
# Console writes to a dup'd stderr fd so redirect_stderr won't capture it
_console_fd = os.dup(sys.stderr.fileno())
_console_file = os.fdopen(_console_fd, "w")
console = Console(theme=_theme, file=_console_file)

PROJECT_ROOT = Path(__file__).resolve().parent


# ================================================================
# Banner
# ================================================================

BANNER = """[bold cyan]
   ███╗   ███╗ ███████╗ ██████╗  ███╗   ███╗  █████╗  ██████╗  ███████╗
   ████╗ ████║ ██╔════╝ ██╔══██╗ ████╗ ████║ ██╔══██╗ ██╔══██╗ ██╔════╝
   ██╔████╔██║ █████╗   ██║  ██║ ██╔████╔██║ ███████║ ██████╔╝ ███████╗
   ██║╚██╔╝██║ ██╔══╝   ██║  ██║ ██║╚██╔╝██║ ██╔══██║ ██╔══██╗ ╚════██║
   ██║ ╚═╝ ██║ ███████╗ ██████╔╝ ██║ ╚═╝ ██║ ██║  ██║ ██║  ██║ ███████║
   ╚═╝     ╚═╝ ╚══════╝ ╚═════╝  ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝
[/bold cyan]
[bold white]             Multi-Agent Retrieval System for Medicine[/bold white]
"""

HELP_TEXT = """[dim]Commands:
  /image <path> <question>  Analyze a medical image
  /summary                  Show conversation summary
  /new                      Start a new conversation
  /help                     Show this help
  /quit                     Exit

  Tip: Use /new to start a fresh session when switching topics.[/dim]
"""


# ================================================================
# MCP config loader
# ================================================================

def _load_mcp_config() -> dict:
    """Load .mcp.json and return server configs."""
    mcp_path = PROJECT_ROOT / ".mcp.json"
    if not mcp_path.exists():
        return {}
    with open(mcp_path, "r") as f:
        data = json.load(f)
    return data.get("mcpServers", {})


def _get_mcp_tools(server_name: str) -> list[str]:
    """Get tool names from an MCP server module."""
    tools = []
    if server_name == "medical-knowledge":
        try:
            from mcp_medical_server import mcp as mcp_server
            for tool in mcp_server._tools.values():
                tools.append(tool.name)
        except Exception:
            # Fallback: read the file and find @mcp.tool() decorated functions
            try:
                server_path = PROJECT_ROOT / "mcp_medical_server.py"
                content = server_path.read_text()
                tools = re.findall(r'def (\w+)\(.*\).*:\s*\n\s*"""', content)
                # Filter to only those after @mcp.tool()
                tools = re.findall(r'@mcp\.tool\(\)\s*\ndef (\w+)', content)
            except Exception:
                pass
    return tools


# ================================================================
# CLI session
# ================================================================

class MedMARSCLI:
    def __init__(self):
        self.thread_id = f"session_{int(time.time())}"
        self.turn_count = 0
        self.session = PromptSession()
        self.ablation_flags = {
            "use_vision": True,
            "use_diagnosis_agent": True,
            "use_medication_graphrag": True,
            "use_general_vectorrag": True,
        }
        self._run_turn = None
        self._app_with_memory = None

    def _startup(self):
        """Show banner, load MCP config, and load pipeline."""
        console.print(BANNER)

        # ── MCP tools ──
        mcp_servers = _load_mcp_config()
        if mcp_servers:
            console.print("[dim]  MCP Servers:[/dim]")
            for name, config in mcp_servers.items():
                tools = _get_mcp_tools(name)
                tool_str = ", ".join(tools) if tools else "loading..."
                console.print(f"[dim]    {name}: {tool_str}[/dim]")
            console.print()

        # ── Tech stack ──
        console.print("[dim]  Agents:  Mistral-7B (Diagnosis) · Qwen2-VL-7B (Vision) · GPT-4o-mini (Orchestration)[/dim]")
        console.print("[dim]  RAG:     DrugBank GraphRAG · MedQuAD VectorRAG (FAISS + BM25 + RRF)[/dim]")
        console.print("[dim]  Safety:  Dual-layer (Regex + LLM) · Faithfulness NLI check[/dim]")
        console.print()

        # ── Load pipeline ──
        with console.status("[bold cyan]  Initializing pipeline...[/bold cyan]", spinner="dots"):
            buf = io.StringIO()
            err_buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
                from orchestration import run_turn, app_with_memory
                self._run_turn = run_turn
                self._app_with_memory = app_with_memory

        console.print("[success]  Pipeline ready.[/success]")
        console.print()
        console.print(HELP_TEXT)

    def _get_prompt(self) -> HTML:
        return HTML(f"<aaa fg='ansibrightgreen'>You&gt; </aaa>")

    def _show_response(self, response: str):
        """Render response panel."""
        md = Markdown(response)
        console.print(Panel(
            md,
            title="[bold cyan]Assistant[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))

    def _show_safety_response(self, response: str):
        """Render safety exit response in red panel."""
        console.print(Panel(
            response,
            title="[bold red]Safety Alert[/bold red]",
            border_style="red",
            padding=(1, 2),
        ))

    def _cmd_summary(self):
        """Show current conversation summary."""
        try:
            state = self._app_with_memory.get_state(
                {"configurable": {"thread_id": self.thread_id}}
            )
            summary = state.values.get("conversation_summary", "")
            gathering = state.values.get("gathering_rounds", 0)
            if summary:
                console.print(Panel(
                    summary,
                    title="[bold]Conversation Summary[/bold]",
                    border_style="blue",
                    padding=(1, 2),
                ))
                if gathering and gathering > 0:
                    console.print(f"  [info]Gathering round: {gathering}[/info]")
            else:
                console.print("  [dim]No summary yet — start a conversation first.[/dim]")
        except Exception:
            console.print("  [dim]No conversation state found.[/dim]")

    def _cmd_new(self):
        """Start a new conversation — clear screen and reset."""
        self.thread_id = f"session_{int(time.time())}"
        self.turn_count = 0
        # Clear screen AND scrollback buffer, bypassing all fd redirections
        with open("/dev/tty", "w") as tty:
            tty.write("\033[2J\033[3J\033[H")
        console.print(BANNER)
        console.print(HELP_TEXT)
        console.print("[success]  New conversation started.[/success]\n")

    def _cmd_image(self, args: str):
        """Handle /image <path> <question>."""
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            console.print("  [warning]Usage: /image <path> <question>[/warning]")
            return

        image_path, question = parts[0], parts[1]
        if not Path(image_path).exists():
            console.print(f"  [danger]File not found: {image_path}[/danger]")
            return

        self._do_turn(question, user_image=image_path)

    def _do_turn(self, user_text: str, user_image: str = None):
        """Execute a conversation turn."""
        self.turn_count += 1

        captured = io.StringIO()
        response = None
        error = None

        with console.status("[bold cyan]  Thinking...[/bold cyan]", spinner="dots"):
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(io.StringIO()):
                try:
                    response = self._run_turn(
                        user_text=user_text,
                        user_image=user_image,
                        thread_id=self.thread_id,
                        ablation_flags=self.ablation_flags,
                    )
                except KeyboardInterrupt:
                    error = "interrupt"
                except Exception as e:
                    error = str(e)

        if error == "interrupt":
            console.print("\n  [warning]Interrupted.[/warning]")
            return
        if error:
            console.print(f"\n  [danger]Error: {error}[/danger]")
            return

        # Check if safety exit by scanning captured output
        is_safety = "[Safety Exit]" in captured.getvalue()
        if is_safety:
            self._show_safety_response(response)
        else:
            self._show_response(response)

    def run(self):
        """Main loop."""
        self._startup()

        while True:
            try:
                user_input = self.session.prompt(self._get_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            lower = user_input.lower()
            if lower in ("/quit", "/exit"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif lower == "/help":
                console.print(HELP_TEXT)
                continue
            elif lower == "/summary":
                self._cmd_summary()
                continue
            elif lower == "/new":
                self._cmd_new()
                continue
            elif lower.startswith("/image "):
                self._cmd_image(user_input[7:])
                continue
            elif user_input.startswith("/"):
                console.print(f"  [warning]Unknown command: {user_input.split()[0]}[/warning]")
                continue

            self._do_turn(user_input)


# ================================================================
# Entry point
# ================================================================

def main():
    cli = MedMARSCLI()
    cli.run()


if __name__ == "__main__":
    main()
