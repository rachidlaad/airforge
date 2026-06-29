from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np


@dataclass(frozen=True)
class Shape:
    kind: str
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def aspect(self) -> float:
        return self.w / max(self.h, 1)


@dataclass(frozen=True)
class UIBlock:
    role: str
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    blocks: List[UIBlock]


def detect_shapes(canvas: np.ndarray) -> List[Shape]:
    """Find rough rectangles and lines in the drawn canvas."""

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim == 3 else canvas
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes: List[Shape] = []
    canvas_area = canvas.shape[0] * canvas.shape[1]
    min_area = max(700, int(canvas_area * 0.002))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.045 * peri, True)
        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18:
            continue

        extent = area / float(w * h)
        looks_rectangular = len(approx) <= 8 and extent >= 0.18
        if looks_rectangular:
            confidence = min(1.0, 0.45 + extent)
            shapes.append(Shape("rectangle", x, y, w, h, confidence))

    shapes.extend(_detect_lines(thresh, shapes))
    return _dedupe_shapes(shapes)


def interpret_layout(shapes: Iterable[Shape], width: int, height: int) -> Layout:
    """Map simple drawn shapes into landing-page UI roles."""

    rectangles = sorted((s for s in shapes if s.kind == "rectangle"), key=lambda s: (s.y, s.x))
    blocks: List[UIBlock] = []
    used: set[int] = set()

    for idx, shape in enumerate(rectangles):
        if shape.y < height * 0.20 and shape.w > width * 0.45 and shape.h < height * 0.20:
            blocks.append(_block("navbar", shape))
            used.add(idx)
            break

    footer_candidates = [
        (idx, shape)
        for idx, shape in enumerate(rectangles)
        if idx not in used and shape.y > height * 0.68 and shape.w > width * 0.45
    ]
    if footer_candidates:
        idx, shape = max(footer_candidates, key=lambda item: item[1].y)
        blocks.append(_block("footer", shape))
        used.add(idx)

    hero_candidates = [
        (idx, shape)
        for idx, shape in enumerate(rectangles)
        if idx not in used and shape.y < height * 0.58 and shape.area > width * height * 0.03
    ]
    if hero_candidates:
        idx, shape = max(hero_candidates, key=lambda item: item[1].area)
        blocks.append(_block("hero", shape))
        used.add(idx)

    remaining = [(idx, shape) for idx, shape in enumerate(rectangles) if idx not in used]
    for idx, shape in remaining:
        role = _classify_remaining(shape, width, height)
        blocks.append(_block(role, shape))

    return Layout(width=width, height=height, blocks=sorted(blocks, key=lambda block: (block.y, block.x)))


def sample_canvas(width: int = 960, height: int = 540) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    color = (255, 255, 255)
    thickness = 5
    cv2.rectangle(canvas, (85, 42), (875, 96), color, thickness)
    cv2.rectangle(canvas, (95, 135), (860, 285), color, thickness)
    cv2.rectangle(canvas, (135, 235), (310, 274), color, thickness)
    cv2.rectangle(canvas, (105, 330), (310, 445), color, thickness)
    cv2.rectangle(canvas, (380, 330), (585, 445), color, thickness)
    cv2.rectangle(canvas, (655, 330), (860, 445), color, thickness)
    cv2.rectangle(canvas, (85, 482), (875, 522), color, thickness)
    return canvas


def _detect_lines(thresh: np.ndarray, rectangles: List[Shape]) -> List[Shape]:
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=max(60, thresh.shape[1] // 10),
        maxLineGap=14,
    )
    if lines is None:
        return []

    detected: List[Shape] = []
    for raw in lines[:24]:
        x1, y1, x2, y2 = raw[0]
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        if max(w, h) < 60:
            continue
        candidate = Shape("line", x, y, max(w, 2), max(h, 2), 0.55)
        if not any(_overlap_ratio(candidate, rect) > 0.75 for rect in rectangles):
            detected.append(candidate)
    return detected


def _dedupe_shapes(shapes: List[Shape]) -> List[Shape]:
    result: List[Shape] = []
    for shape in sorted(shapes, key=lambda s: s.area, reverse=True):
        if any(_is_duplicate(shape, kept) for kept in result):
            continue
        result.append(shape)
    return sorted(result, key=lambda s: (s.y, s.x))


def _overlap_ratio(a: Shape, b: Shape) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1, min(a.area, b.area))


def _is_duplicate(shape: Shape, kept: Shape) -> bool:
    overlap = _overlap_ratio(shape, kept)
    area_ratio = min(shape.area, kept.area) / max(shape.area, kept.area)
    return overlap > 0.70 and area_ratio > 0.22


def _classify_remaining(shape: Shape, width: int, height: int) -> str:
    if shape.area < width * height * 0.018 and shape.aspect > 1.8:
        return "button"
    if shape.aspect > 2.2 and shape.area > width * height * 0.035:
        return "image"
    if shape.y > height * 0.62 and shape.w > width * 0.38:
        return "footer"
    return "card"


def _block(role: str, shape: Shape) -> UIBlock:
    return UIBlock(role=role, x=shape.x, y=shape.y, w=shape.w, h=shape.h)
