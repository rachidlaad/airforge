from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, Tuple


Point = Tuple[int, int]


@dataclass(frozen=True)
class GestureState:
    name: str
    index_tip: Point
    erase_point: Point | None = None


class HandGestureDetector:
    """Classifies a single MediaPipe hand into MVP drawing gestures."""

    def __init__(self, pinch_ratio: float = 0.055) -> None:
        self.pinch_ratio = pinch_ratio

    def classify(self, landmarks, frame_width: int, frame_height: int) -> GestureState:
        points = self._points(landmarks, frame_width, frame_height)
        index_tip = points[8]
        thumb_tip = points[4]
        pinch_distance = self._distance(index_tip, thumb_tip)
        pinch_threshold = self.pinch_ratio * min(frame_width, frame_height)

        if self._is_open_palm(points):
            return GestureState("clear", index_tip)

        if self._is_thumbs_up(points):
            return GestureState("generate", index_tip)

        if pinch_distance <= pinch_threshold:
            midpoint = (
                int((index_tip[0] + thumb_tip[0]) / 2),
                int((index_tip[1] + thumb_tip[1]) / 2),
            )
            return GestureState("erase", index_tip, midpoint)

        if self._finger_extended(points, 8, 6) and not self._finger_extended(points, 12, 10):
            return GestureState("draw", index_tip)

        return GestureState("idle", index_tip)

    @staticmethod
    def _points(landmarks, width: int, height: int) -> Dict[int, Point]:
        landmark_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        return {
            idx: (int(mark.x * width), int(mark.y * height))
            for idx, mark in enumerate(landmark_list)
        }

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _finger_extended(points: Dict[int, Point], tip: int, pip: int) -> bool:
        return points[tip][1] < points[pip][1]

    def _is_open_palm(self, points: Dict[int, Point]) -> bool:
        fingers_up = [
            self._finger_extended(points, 8, 6),
            self._finger_extended(points, 12, 10),
            self._finger_extended(points, 16, 14),
            self._finger_extended(points, 20, 18),
        ]
        thumb_spread = abs(points[4][0] - points[9][0]) > abs(points[5][0] - points[17][0]) * 0.45
        return all(fingers_up) and thumb_spread

    def _is_thumbs_up(self, points: Dict[int, Point]) -> bool:
        thumb_vertical = points[4][1] < points[3][1] < points[2][1]
        fingers_folded = (
            points[8][1] > points[6][1]
            and points[12][1] > points[10][1]
            and points[16][1] > points[14][1]
            and points[20][1] > points[18][1]
        )
        thumb_above_knuckles = points[4][1] < min(points[5][1], points[9][1], points[13][1], points[17][1])
        return thumb_vertical and fingers_folded and thumb_above_knuckles
