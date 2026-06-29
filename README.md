# AirForge

AirForge is a local MVP that turns rough in-air hand sketches into a simple responsive landing page.

Use your index finger to draw rectangles and lines on a webcam canvas. AirForge detects the sketch, interprets 3-5 rough rectangles as common website sections, writes `generated/index.html`, and opens it in your browser.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
python -m airforge
```

Gesture controls:

- Index finger up: draw
- Pinch thumb and index: pause drawing and erase near the pinch point
- Open palm: clear the canvas
- Thumbs-up: generate `generated/index.html` and open a browser preview
- `g`: generate from the current canvas
- `c`: clear
- `s`: save a debug snapshot to `generated/sketch.png`
- `q` or `Esc`: quit

## Quick generator test without a webcam

```bash
python -m airforge --sample
```

This creates a synthetic wireframe with a navbar, hero, button, cards, and footer, then generates and opens the page.

## React export

The default output is standalone HTML/CSS. To also emit a React component:

```bash
python -m airforge --sample --export react
```

React output is written to `generated/LandingPage.jsx`.
