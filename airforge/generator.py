from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .sketch import Layout


CODEX_TIMEOUT_SECONDS = 240


@dataclass(frozen=True)
class GeneratedFiles:
    html_path: Path
    react_path: Path | None = None


class GenerationError(RuntimeError):
    """Raised when the AI generator cannot produce the requested files."""


def generate_page(
    layout: Layout,
    output_dir: Path,
    export: str = "html",
    sketch_path: Path | None = None,
) -> GeneratedFiles:
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_codex_cli()

    html_path = output_dir / "index.html"
    react_path = output_dir / "LandingPage.jsx" if export == "react" else None
    prompt_path = output_dir / "airforge-prompt.json"
    result_path = output_dir / "codex-result.txt"

    _remove_stale_outputs(html_path, react_path)
    prompt_path.write_text(_prompt_payload(layout, export), encoding="utf-8")

    command = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    if sketch_path is not None and sketch_path.exists():
        command.extend(["--image", str(sketch_path)])
    command.extend(
        [
            "--cd",
            str(output_dir),
            "--skip-git-repo-check",
            "--output-last-message",
            str(result_path),
        ]
    )
    command.append(_codex_prompt(export))

    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            input="",
            text=True,
            capture_output=True,
            timeout=CODEX_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationError(
            f"Codex generation timed out after {CODEX_TIMEOUT_SECONDS} seconds. "
            "Try a simpler sketch or run again."
        ) from exc
    if completed.returncode != 0:
        raise GenerationError(_format_codex_failure(completed))

    _validate_generated_file(html_path, "index.html")
    if react_path is not None:
        _validate_generated_file(react_path, "LandingPage.jsx")

    return GeneratedFiles(html_path=html_path, react_path=react_path)


def _require_codex_cli() -> None:
    if shutil.which("codex") is None:
        raise GenerationError(
            "Codex CLI is required for AI generation, but `codex` was not found on PATH. "
            "Install or log in to Codex CLI, then run AirForge again."
        )


def _remove_stale_outputs(html_path: Path, react_path: Path | None) -> None:
    for path in (html_path, react_path):
        if path is not None and path.exists():
            path.unlink()


def _prompt_payload(layout: Layout, export: str) -> str:
    payload = {
        "task": "Generate a clean responsive landing page from this hand-drawn wireframe.",
        "export": export,
        "canvas": {"width": layout.width, "height": layout.height},
        "blocks": [
            {
                "role": block.role,
                "x": block.x,
                "y": block.y,
                "w": block.w,
                "h": block.h,
            }
            for block in layout.blocks
        ],
        "notes": [
            "Use the attached sketch image as the primary visual reference when present.",
            "Use the detected blocks as layout hints, not as copy or final style.",
            "Do not ask follow-up questions.",
            "Do not create placeholder explanations instead of files.",
        ],
    }
    return json.dumps(payload, indent=2)


def _codex_prompt(export: str) -> str:
    react_instruction = ""
    if export == "react":
        react_instruction = """
Also create LandingPage.jsx exporting a default React component for the same page.
Use plain React and CSS class names only; do not require Tailwind, shadcn, or external assets.
"""

    return f"""You are the AI website generator for AirForge.

Read ./airforge-prompt.json and inspect the attached sketch image if one is provided.

Create a polished, responsive landing page from the sketched layout.

Requirements:
- Write ./index.html.
- The HTML must be complete and browser-ready with inline CSS in a <style> tag.
- Use the wireframe structure from airforge-prompt.json, but generate original page copy and visual design.
- Do not use hardcoded AirForge template content.
- Do not use external JavaScript, package managers, build steps, remote images, or API keys.
- Use semantic HTML and responsive CSS.
- Make the design look like a real modern landing page, not a debug visualization.
- Keep it self-contained so opening index.html works immediately.
{react_instruction}
When done, reply briefly with the file names you created."""


def _validate_generated_file(path: Path, label: str) -> None:
    if not path.exists():
        raise GenerationError(f"Codex completed, but did not create {label}.")
    contents = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(contents) < 200:
        raise GenerationError(f"Codex created {label}, but it is too small to be a usable page.")
    if label == "index.html" and ("<html" not in contents.lower() or "<style" not in contents.lower()):
        raise GenerationError(f"Codex created {label}, but it is not a complete self-contained HTML page.")


def _format_codex_failure(completed: subprocess.CompletedProcess[str]) -> str:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    details = "\n".join(part for part in [stdout, stderr] if part)
    if not details:
        details = "No output was returned by Codex."
    return f"Codex generation failed with exit code {completed.returncode}.\n{details}"
