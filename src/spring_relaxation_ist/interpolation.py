"""Linear interpolation utilities for relaxation curves."""

from __future__ import annotations

import numpy as np

from .schema import Material, RelaxationCurve


class InterpolationError(ValueError):
    """Raised when interpolation cannot be performed."""


def _get_curve_for_temperature(
    material: Material, temperature_C: float
) -> tuple[RelaxationCurve, list[str]]:
    """
    Find the best matching curve for the given temperature.

    Returns (curve, warnings).
    Raises InterpolationError if no usable curve exists.
    """
    usable = [
        c
        for c in material.curves
        if c.extraction_method != "placeholder" and len(c.points) >= 2
    ]

    if not usable:
        raise InterpolationError(
            f"No usable (non-placeholder) curves available for material "
            f"'{material.material_name}'. "
            "The PDF figures have not been digitised yet."
        )

    temps = [c.temperature_C for c in usable]
    if temperature_C < min(temps) or temperature_C > max(temps):
        raise InterpolationError(
            f"Temperature {temperature_C}°C is outside the available range "
            f"[{min(temps)}, {max(temps)}]°C for material '{material.material_name}'. "
            "Extrapolation is not supported."
        )

    # Exact match first
    exact = [c for c in usable if c.temperature_C == temperature_C]
    if exact:
        warnings: list[str] = []
        if exact[0].extraction_confidence < 0.7:
            warnings.append(
                f"Curve for {temperature_C}°C has low extraction confidence "
                f"({exact[0].extraction_confidence:.2f})."
            )
        return exact[0], warnings

    # Linear interpolation between adjacent temperatures
    lower = max(
        (c for c in usable if c.temperature_C < temperature_C), key=lambda c: c.temperature_C
    )
    upper = min(
        (c for c in usable if c.temperature_C > temperature_C), key=lambda c: c.temperature_C
    )
    # Return the closer one (simplified)
    closer = (
        lower
        if (temperature_C - lower.temperature_C) <= (upper.temperature_C - temperature_C)
        else upper
    )
    return closer, [
        f"No exact curve for {temperature_C}°C; using nearest available at {closer.temperature_C}°C."
    ]


def interpolate_relaxation(
    material: Material,
    temperature_C: float,
    stress_MPa: float,
) -> tuple[float, str, list[str]]:
    """
    Interpolate relaxation percent for a given material, temperature, and stress.

    Returns (relaxation_percent, method_description, warnings).
    Raises InterpolationError on out-of-range or missing data.
    """
    curve, warnings = _get_curve_for_temperature(material, temperature_C)

    stresses = np.array([p.stress_MPa for p in curve.points])
    relaxations = np.array([p.relaxation_percent for p in curve.points])

    if stress_MPa < stresses.min() or stress_MPa > stresses.max():
        raise InterpolationError(
            f"Stress {stress_MPa} MPa is outside the available range "
            f"[{stresses.min():.1f}, {stresses.max():.1f}] MPa "
            f"for material '{material.material_name}' at {curve.temperature_C}°C. "
            "Extrapolation is not supported."
        )

    relaxation = float(np.interp(stress_MPa, stresses, relaxations))
    method = f"linear_interpolation (curve: {curve.curve_id})"
    return relaxation, method, warnings
