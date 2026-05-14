from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .config import CURVE_CALIBRATION_PATH

CurveLineStyle = Literal["solid", "dashed"]
SerializedCurveControlPayload = dict[str, dict[str, float] | CurveLineStyle]


@dataclass(frozen=True)
class CurveControlPoints:
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    line_style: CurveLineStyle | None = None

    def has_points(self) -> bool:
        return self.start is not None or self.end is not None

    def has_overrides(self) -> bool:
        return self.has_points() or self.line_style is not None

    def with_start(self, point: tuple[float, float] | None) -> "CurveControlPoints":
        return CurveControlPoints(start=point, end=self.end, line_style=self.line_style)

    def with_end(self, point: tuple[float, float] | None) -> "CurveControlPoints":
        return CurveControlPoints(start=self.start, end=point, line_style=self.line_style)

    def with_line_style(self, line_style: CurveLineStyle | None) -> "CurveControlPoints":
        return CurveControlPoints(start=self.start, end=self.end, line_style=line_style)


def merge_curve_control_points(
    primary: CurveControlPoints | None,
    fallback: CurveControlPoints | None,
) -> CurveControlPoints | None:
    if primary is None and fallback is None:
        return None

    primary = primary or CurveControlPoints()
    fallback = fallback or CurveControlPoints()
    merged = CurveControlPoints(
        start=primary.start if primary.start is not None else fallback.start,
        end=primary.end if primary.end is not None else fallback.end,
        line_style=primary.line_style if primary.line_style is not None else fallback.line_style,
    )
    return merged if merged.has_overrides() else None


def load_curve_control_points(
    path: Path = CURVE_CALIBRATION_PATH,
) -> dict[int, dict[str, CurveControlPoints]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    figures_raw = payload.get("figures", payload)
    if not isinstance(figures_raw, dict):
        return {}

    figures: dict[int, dict[str, CurveControlPoints]] = {}
    for figure_key, figure_payload in figures_raw.items():
        try:
            figure_num = int(figure_key)
        except (TypeError, ValueError):
            continue

        materials_raw = figure_payload.get("materials", figure_payload)
        if not isinstance(materials_raw, dict):
            continue

        material_controls: dict[str, CurveControlPoints] = {}
        for material_id, control_payload in materials_raw.items():
            controls = _parse_curve_control_points(control_payload)
            if controls is None:
                continue
            material_controls[str(material_id)] = controls

        if material_controls:
            figures[figure_num] = material_controls

    return figures


def save_curve_control_points(
    figures: dict[int, dict[str, CurveControlPoints]],
    path: Path = CURVE_CALIBRATION_PATH,
) -> None:
    payload = {
        "figures": {
            str(figure_num): {
                "materials": {
                    material_id: _serialize_curve_control_points(controls)
                    for material_id, controls in sorted(material_controls.items())
                    if controls.has_overrides()
                }
            }
            for figure_num, material_controls in sorted(figures.items())
            if any(controls.has_overrides() for controls in material_controls.values())
        }
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_curve_control_points(payload: Any) -> CurveControlPoints | None:
    if not isinstance(payload, dict):
        return None

    start = _parse_endpoint(payload.get("start"))
    end = _parse_endpoint(payload.get("end"))
    line_style = _parse_line_style(payload.get("line_style"))
    controls = CurveControlPoints(start=start, end=end, line_style=line_style)
    return controls if controls.has_overrides() else None


def _parse_endpoint(payload: Any) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None

    stress = payload.get("stress_MPa")
    relaxation = payload.get("relaxation_percent")
    if not isinstance(stress, (int, float)) or not isinstance(relaxation, (int, float)):
        return None

    return float(stress), float(relaxation)


def _serialize_curve_control_points(controls: CurveControlPoints) -> SerializedCurveControlPayload:
    payload: SerializedCurveControlPayload = {}
    if controls.start is not None:
        payload["start"] = {
            "stress_MPa": round(float(controls.start[0]), 1),
            "relaxation_percent": round(float(controls.start[1]), 3),
        }
    if controls.end is not None:
        payload["end"] = {
            "stress_MPa": round(float(controls.end[0]), 1),
            "relaxation_percent": round(float(controls.end[1]), 3),
        }
    if controls.line_style is not None:
        payload["line_style"] = controls.line_style
    return payload


def _parse_line_style(payload: Any) -> CurveLineStyle | None:
    if isinstance(payload, str) and payload in {"solid", "dashed"}:
        return cast(CurveLineStyle, payload)

    return None