from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import List

from .sketch import Layout, UIBlock


@dataclass(frozen=True)
class GeneratedFiles:
    html_path: Path
    react_path: Path | None = None


def generate_page(layout: Layout, output_dir: Path, export: str = "html") -> GeneratedFiles:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(render_html(layout), encoding="utf-8")

    react_path = None
    if export == "react":
        react_path = output_dir / "LandingPage.jsx"
        react_path.write_text(render_react(layout), encoding="utf-8")

    return GeneratedFiles(html_path=html_path, react_path=react_path)


def render_html(layout: Layout) -> str:
    sections = _ordered_sections(layout)
    body = "\n".join(_render_html_block(block, idx, layout) for idx, block in enumerate(sections, start=1))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AirForge Landing Page</title>
  <style>
{_css()}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def render_react(layout: Layout) -> str:
    sections = _ordered_sections(layout)
    body = "\n".join(_render_react_block(block, idx, layout) for idx, block in enumerate(sections, start=1))
    return f"""export default function LandingPage() {{
  return (
    <main className="page-shell">
{body}
    </main>
  );
}}
"""


def _ordered_sections(layout: Layout) -> List[UIBlock]:
    blocks = sorted(layout.blocks, key=lambda block: (block.y, block.x))
    if not any(block.role == "navbar" for block in blocks):
        blocks.insert(0, UIBlock("navbar", 0, 0, layout.width, max(54, layout.height // 10)))
    if not any(block.role == "hero" for block in blocks):
        blocks.insert(1, UIBlock("hero", 0, layout.height // 5, layout.width, layout.height // 4))
    return blocks


def _render_html_block(block: UIBlock, idx: int, layout: Layout) -> str:
    role = escape(block.role)
    if role == "navbar":
        return """  <header class="nav">
    <a class="brand" href="#">AirForge</a>
    <nav aria-label="Primary">
      <a href="#features">Features</a>
      <a href="#preview">Preview</a>
      <a href="#contact">Contact</a>
    </nav>
  </header>"""
    if role == "hero":
        image_hint = any(candidate.role == "image" and abs(candidate.cy - block.cy) < layout.height * 0.22 for candidate in layout.blocks)
        media = '<div class="hero-media" aria-label="Product preview"></div>' if image_hint else ""
        return f"""  <section class="hero" id="preview">
    <div class="hero-copy">
      <p class="eyebrow">Generated from your sketch</p>
      <h1>Launch the page you pictured.</h1>
      <p>AirForge converts rough wireframe gestures into a clean responsive landing page you can edit immediately.</p>
      <a class="button" href="#contact">Start building</a>
    </div>
    {media}
  </section>"""
    if role == "button":
        return """  <section class="cta-strip">
    <a class="button secondary" href="#contact">Request a demo</a>
  </section>"""
    if role == "footer":
        return """  <footer class="footer" id="contact">
    <span>AirForge</span>
    <a href="mailto:hello@example.com">hello@example.com</a>
  </footer>"""
    if role == "image":
        return f"""  <section class="image-band" aria-label="Visual section {idx}">
    <div></div>
  </section>"""
    return f"""  <article class="card" id="features">
    <span class="card-kicker">Feature {idx}</span>
    <h2>{_card_title(idx)}</h2>
    <p>{_card_copy(idx)}</p>
  </article>"""


def _render_react_block(block: UIBlock, idx: int, layout: Layout) -> str:
    html = _render_html_block(block, idx, layout)
    html = html.replace('class="', 'className="').replace('for="', 'htmlFor="')
    return "\n".join(f"      {line}" for line in html.splitlines())


def _card_title(idx: int) -> str:
    titles = ["Sketch to structure", "Responsive by default", "Ready to customize", "Fast local preview"]
    return titles[(idx - 1) % len(titles)]


def _card_copy(idx: int) -> str:
    copy = [
        "Draw the main blocks and let the generator infer navigation, hero content, cards, and calls to action.",
        "The output uses fluid grids, balanced spacing, and mobile-friendly section ordering.",
        "Generated markup is plain, readable, and intentionally small so you can keep shaping it.",
        "A local browser preview opens as soon as the thumbs-up gesture triggers generation.",
    ]
    return copy[(idx - 1) % len(copy)]


def _css() -> str:
    return """    :root {
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6872;
      --line: #d8e0e6;
      --surface: #ffffff;
      --soft: #f3f7f8;
      --accent: #0e7c7b;
      --accent-dark: #075f63;
      --warm: #f2b84b;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #fbfcfc 0%, #eef4f5 100%);
      letter-spacing: 0;
    }

    a {
      color: inherit;
      text-decoration: none;
    }

    .nav,
    .hero,
    .cta-strip,
    .image-band,
    .card,
    .footer {
      width: min(1120px, calc(100% - 32px));
      margin-inline: auto;
    }

    .nav {
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      border-bottom: 1px solid var(--line);
    }

    .brand {
      font-weight: 800;
      font-size: 1.05rem;
    }

    .nav nav {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 16px;
      color: var(--muted);
      font-size: 0.94rem;
    }

    .hero {
      min-height: 420px;
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(280px, 0.95fr);
      align-items: center;
      gap: clamp(28px, 5vw, 64px);
      padding: clamp(48px, 8vw, 92px) 0;
    }

    .hero-copy {
      display: grid;
      gap: 20px;
    }

    .eyebrow,
    .card-kicker {
      color: var(--accent-dark);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1,
    h2,
    p {
      margin: 0;
    }

    h1 {
      max-width: 760px;
      font-size: clamp(2.45rem, 7vw, 5.8rem);
      line-height: 0.95;
    }

    .hero p {
      max-width: 620px;
      color: var(--muted);
      font-size: clamp(1rem, 2vw, 1.25rem);
      line-height: 1.65;
    }

    .button {
      display: inline-flex;
      min-height: 46px;
      width: fit-content;
      align-items: center;
      justify-content: center;
      border-radius: 6px;
      padding: 0 18px;
      background: var(--accent);
      color: white;
      font-weight: 800;
      box-shadow: 0 12px 28px rgba(14, 124, 123, 0.22);
    }

    .button.secondary {
      background: var(--ink);
      box-shadow: none;
    }

    .hero-media,
    .image-band div {
      min-height: 300px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(14, 124, 123, 0.16), rgba(242, 184, 75, 0.24)),
        repeating-linear-gradient(90deg, rgba(23, 32, 38, 0.06) 0 1px, transparent 1px 34px),
        var(--surface);
      box-shadow: 0 24px 70px rgba(23, 32, 38, 0.10);
    }

    .cta-strip {
      display: flex;
      justify-content: center;
      padding: 20px 0;
    }

    .image-band {
      padding: 24px 0;
    }

    .card {
      display: inline-grid;
      width: min(352px, calc(100% - 32px));
      min-height: 210px;
      align-content: start;
      gap: 14px;
      margin: 16px clamp(8px, 2vw, 18px);
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      vertical-align: top;
      box-shadow: 0 14px 34px rgba(23, 32, 38, 0.08);
    }

    .card h2 {
      font-size: 1.35rem;
    }

    .card p {
      color: var(--muted);
      line-height: 1.6;
    }

    .footer {
      min-height: 96px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-top: 40px;
      border-top: 1px solid var(--line);
      color: var(--muted);
    }

    @media (max-width: 780px) {
      .nav {
        align-items: flex-start;
        flex-direction: column;
        justify-content: center;
        padding: 18px 0;
      }

      .nav nav {
        justify-content: flex-start;
      }

      .hero {
        grid-template-columns: 1fr;
        min-height: unset;
      }

      .hero-media {
        min-height: 240px;
      }

      .card {
        display: grid;
        width: min(100% - 32px, 1120px);
        margin-inline: auto;
      }

      .footer {
        align-items: flex-start;
        flex-direction: column;
        justify-content: center;
      }
    }"""
