from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from math import inf, sqrt
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageDraw, ImageTk

from .config import CURVE_CALIBRATION_PATH, FIGURE_PREVIEWS_DIR
from .curve_calibration import CurveControlPoints, CurveLineStyle, load_curve_control_points, save_curve_control_points
from .figure_digitizer import (
    PREVIEW_CURVE_COLORS,
    SUPPORTED_FIGURE_SPECS,
    FigureAnalysisResult,
    Y_AXIS_MAX,
    _analyze_figure,
    _material_display_name,
    _preview_crop_box,
    _render_figure_preview,
    clear_figure_analysis_cache,
    generate_supported_figure_previews,
)

_DEFAULT_CANVAS_WIDTH = 1180
_DEFAULT_CANVAS_HEIGHT = 820
_CONTROLS_PANEL_WIDTH = 360
_CONTROLS_WRAP_LENGTH = 320
_MIN_ZOOM_SCALE = 0.2
_MAX_ZOOM_SCALE = 4.0
_ZOOM_FACTOR = 1.1
_MARKER_HIT_RADIUS = 16.0


@dataclass(frozen=True)
class _FigureCalibrationView:
    analysis: FigureAnalysisResult
    preview_image: Image.Image
    crop_box: tuple[int, int, int, int]


class _CalibrationTab(ttk.Frame):
    def __init__(
        self,
        master: ttk.Notebook,
        pdf_path: Path,
        figure_view: _FigureCalibrationView,
        initial_overrides: dict[str, CurveControlPoints],
    ) -> None:
        super().__init__(master)
        self._pdf_path = pdf_path
        self._figure_view = figure_view
        self._material_ids = figure_view.analysis.spec.material_ids_top_to_bottom
        self._default_controls = {
            material_id: CurveControlPoints(
                start=_curve_point_to_tuple(figure_view.analysis.curves[material_id].points[0]),
                end=_curve_point_to_tuple(figure_view.analysis.curves[material_id].points[-1]),
            )
            for material_id in self._material_ids
        }
        self._override_controls = {
            material_id: initial_overrides.get(material_id, CurveControlPoints())
            for material_id in self._material_ids
        }
        self._selected_material_id = self._material_ids[0]
        self._pending_point_kind: str | None = None
        self._dragging_marker: tuple[str, str] | None = None
        self._pan_pivot_preview: tuple[float, float] | None = None
        self._drag_moved = False
        self._preview_refresh_after_id: str | None = None
        self._photo_image: ImageTk.PhotoImage | None = None
        self._canvas_image_id: int | None = None
        self._zoom_scale = 1.0
        self._scaled_image_size = figure_view.preview_image.size

        self._instructions_var = tk.StringVar(
            value="Use the mouse wheel to zoom, drag markers directly, or choose Set Start / Set End and click the image."
        )
        self._cursor_coordinates_var = tk.StringVar(value="Cursor: move over the plot to see X/Y")
        self._preview_status_var = tk.StringVar(value="Preview: current parsed curve shown")
        self._zoom_var = tk.StringVar(value="Zoom: 100%")
        self._selected_material_var = tk.StringVar(value=_material_display_name(self._selected_material_id))
        self._start_var = tk.StringVar()
        self._end_var = tk.StringVar()
        self._line_style_var = tk.StringVar(value="auto")

        self._build_layout()
        self._update_value_labels()
        self._redraw_canvas()
        self.after_idle(self._fit_image_to_canvas)

    def get_overrides(self) -> dict[str, CurveControlPoints]:
        return {
            material_id: controls
            for material_id, controls in self._override_controls.items()
            if controls.has_overrides()
        }

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            canvas_frame,
            background="#f3f3f3",
            highlightthickness=0,
            width=_DEFAULT_CANVAS_WIDTH,
            height=_DEFAULT_CANVAS_HEIGHT,
        )
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        ttk.Frame(canvas_frame).grid(row=2, column=1, sticky="ew")
        ttk.Label(
            canvas_frame,
            textvariable=self._cursor_coordinates_var,
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(
            canvas_frame,
            textvariable=self._zoom_var,
            anchor="e",
        ).grid(row=2, column=1, sticky="ew", pady=(6, 0))
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_button_release)
        self._canvas.bind("<Motion>", self._on_canvas_motion)
        self._canvas.bind("<Leave>", self._on_canvas_leave)
        self._canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self._canvas.bind("<Button-4>", self._on_mouse_wheel_linux)
        self._canvas.bind("<Button-5>", self._on_mouse_wheel_linux)
        self._canvas.bind("<ButtonPress-3>", self._on_canvas_pan_start)
        self._canvas.bind("<B3-Motion>", self._on_canvas_pan_drag)
        self._canvas.bind("<ButtonRelease-3>", self._on_canvas_pan_end)

        controls_container = ttk.Frame(self, width=_CONTROLS_PANEL_WIDTH)
        controls_container.grid(row=0, column=1, sticky="ns", padx=(6, 12), pady=12)
        controls_container.grid_propagate(False)
        controls_container.columnconfigure(0, weight=1)
        controls_container.rowconfigure(0, weight=1)

        controls_frame = ttk.Frame(controls_container)
        controls_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame.columnconfigure(0, weight=1)

        ttk.Label(
            controls_frame,
            text=f"Figure {self._figure_view.analysis.spec.figure_num}",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            controls_frame,
            text=f"{int(self._figure_view.analysis.spec.temperature_C)}C",
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))
        ttk.Label(
            controls_frame,
            textvariable=self._instructions_var,
            wraplength=_CONTROLS_WRAP_LENGTH,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(0, 12))
        ttk.Label(
            controls_frame,
            textvariable=self._preview_status_var,
            wraplength=_CONTROLS_WRAP_LENGTH,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(0, 12))

        ttk.Label(controls_frame, text="Legend Items").grid(row=4, column=0, sticky="w")
        self._material_list = tk.Listbox(
            controls_frame,
            exportselection=False,
            height=len(self._material_ids),
            width=34,
        )
        for material_id in self._material_ids:
            self._material_list.insert(tk.END, _material_display_name(material_id))
        self._material_list.selection_set(0)
        self._material_list.bind("<<ListboxSelect>>", self._on_material_selected)
        self._material_list.grid(row=5, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(controls_frame, text="Selected").grid(row=6, column=0, sticky="w")
        ttk.Label(
            controls_frame,
            textvariable=self._selected_material_var,
            font=("Segoe UI", 10, "bold"),
            wraplength=_CONTROLS_WRAP_LENGTH,
        ).grid(row=7, column=0, sticky="w", pady=(0, 12))

        ttk.Label(controls_frame, text="Start Point").grid(row=8, column=0, sticky="w")
        ttk.Label(controls_frame, textvariable=self._start_var, wraplength=_CONTROLS_WRAP_LENGTH).grid(
            row=9,
            column=0,
            sticky="w",
            pady=(0, 8),
        )
        ttk.Label(controls_frame, text="End Point").grid(row=10, column=0, sticky="w")
        ttk.Label(controls_frame, textvariable=self._end_var, wraplength=_CONTROLS_WRAP_LENGTH).grid(
            row=11,
            column=0,
            sticky="w",
            pady=(0, 12),
        )

        ttk.Label(controls_frame, text="Tracing Override").grid(row=12, column=0, sticky="w")
        line_style_frame = ttk.Frame(controls_frame)
        line_style_frame.grid(row=13, column=0, sticky="w", pady=(4, 12))
        ttk.Radiobutton(
            line_style_frame,
            text="Auto",
            value="auto",
            variable=self._line_style_var,
            command=lambda: self._set_line_style_override(None),
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Radiobutton(
            line_style_frame,
            text="Solid",
            value="solid",
            variable=self._line_style_var,
            command=lambda: self._set_line_style_override("solid"),
        ).grid(row=0, column=1, sticky="w", padx=(0, 12))
        ttk.Radiobutton(
            line_style_frame,
            text="Dashed",
            value="dashed",
            variable=self._line_style_var,
            command=lambda: self._set_line_style_override("dashed"),
        ).grid(row=0, column=2, sticky="w")

        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=14, column=0, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        ttk.Button(buttons_frame, text="Set Start", command=self._begin_set_start).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(0, 6),
        )
        ttk.Button(buttons_frame, text="Set End", command=self._begin_set_end).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(0, 6),
        )
        ttk.Button(buttons_frame, text="Clear Start", command=self._clear_start).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(0, 6),
        )
        ttk.Button(buttons_frame, text="Clear End", command=self._clear_end).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(0, 6),
        )
        ttk.Button(
            controls_frame,
            text="Reset Material Overrides",
            command=self._reset_material_overrides,
        ).grid(row=15, column=0, sticky="ew")
        ttk.Button(
            controls_frame,
            text="Fit View",
            command=self._fit_image_to_canvas,
        ).grid(row=16, column=0, sticky="ew", pady=(8, 0))

    def _set_selected_material(self, material_id: str) -> None:
        if material_id not in self._material_ids:
            return

        self._selected_material_id = material_id
        self._selected_material_var.set(_material_display_name(self._selected_material_id))
        selection_index = self._material_ids.index(material_id)
        self._material_list.selection_clear(0, tk.END)
        self._material_list.selection_set(selection_index)
        self._material_list.activate(selection_index)
        self._update_value_labels()
        self._redraw_canvas()

    def _on_material_selected(self, _event: tk.Event[tk.Listbox]) -> None:
        selection = self._material_list.curselection()
        if not selection:
            return

        self._set_selected_material(self._material_ids[int(selection[0])])
        self._pending_point_kind = None
        self._instructions_var.set(
            "Use the mouse wheel to zoom, drag markers directly, or choose Set Start / Set End and click the image."
        )

    def _begin_set_start(self) -> None:
        self._pending_point_kind = "start"
        self._instructions_var.set(
            f"Click the image to place the start point for {_material_display_name(self._selected_material_id)}."
        )

    def _begin_set_end(self) -> None:
        self._pending_point_kind = "end"
        self._instructions_var.set(
            f"Click the image to place the end point for {_material_display_name(self._selected_material_id)}."
        )

    def _clear_start(self) -> None:
        self._override_controls[self._selected_material_id] = self._override_controls[
            self._selected_material_id
        ].with_start(None)
        self._pending_point_kind = None
        self._instructions_var.set("Removed the saved start override for the selected material.")
        self._update_value_labels()
        self._redraw_canvas()
        self._schedule_preview_refresh()

    def _clear_end(self) -> None:
        self._override_controls[self._selected_material_id] = self._override_controls[
            self._selected_material_id
        ].with_end(None)
        self._pending_point_kind = None
        self._instructions_var.set("Removed the saved end override for the selected material.")
        self._update_value_labels()
        self._redraw_canvas()
        self._schedule_preview_refresh()

    def _reset_material_overrides(self) -> None:
        self._override_controls[self._selected_material_id] = CurveControlPoints()
        self._pending_point_kind = None
        self._instructions_var.set("Removed all saved overrides for the selected material.")
        self._update_value_labels()
        self._redraw_canvas()
        self._schedule_preview_refresh()

    def _on_canvas_button_press(self, event: tk.Event[tk.Canvas]) -> None:
        preview_point = self._event_to_preview_coordinates(event)
        if preview_point is None:
            return

        hit = _hit_test_control_marker(
            self._figure_view,
            {
                material_id: self._display_controls(material_id)
                for material_id in self._material_ids
            },
            preview_point[0],
            preview_point[1],
            self._zoom_scale,
        )
        if hit is not None:
            self._dragging_marker = hit
            self._drag_moved = False
            self._pending_point_kind = None
            self._set_selected_material(hit[0])
            self._instructions_var.set(
                f"Selected {hit[1]} point for {_material_display_name(hit[0])}. Drag to move it."
            )
            self._canvas.configure(cursor="fleur")
            return

        if self._pending_point_kind is None:
            return

        point = _preview_to_axis_point(self._figure_view, preview_point[0], preview_point[1])
        if point is None:
            self._instructions_var.set("Click inside the calibrated plot area to place a control point.")
            return

        self._set_control_point(self._selected_material_id, self._pending_point_kind, point)

        self._instructions_var.set(
            f"Saved {self._pending_point_kind} point for {_material_display_name(self._selected_material_id)}."
        )
        self._pending_point_kind = None
        self._update_value_labels()
        self._redraw_canvas()
        self._schedule_preview_refresh()

    def _on_canvas_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._dragging_marker is None:
            return

        preview_point = self._event_to_preview_coordinates(event)
        if preview_point is None:
            return

        point = _preview_to_axis_point(
            self._figure_view,
            preview_point[0],
            preview_point[1],
            clamp_to_plot=True,
        )
        if point is None:
            return

        material_id, point_kind = self._dragging_marker
        self._set_control_point(material_id, point_kind, point)
        self._drag_moved = True
        self._instructions_var.set(
            f"Dragging {point_kind} point for {_material_display_name(material_id)}."
        )
        self._update_value_labels()
        self._redraw_canvas()
        self._schedule_preview_refresh()

    def _on_canvas_button_release(self, _event: tk.Event[tk.Canvas]) -> None:
        if self._dragging_marker is None:
            return

        material_id, point_kind = self._dragging_marker
        if self._drag_moved:
            self._instructions_var.set(
                f"Moved {point_kind} point for {_material_display_name(material_id)}."
            )
        self._dragging_marker = None
        self._drag_moved = False
        if self._pan_pivot_preview is None:
            self._canvas.configure(cursor="")

    def _on_canvas_motion(self, event: tk.Event[tk.Canvas]) -> None:
        preview_point = self._event_to_preview_coordinates(event)
        if preview_point is None:
            self._cursor_coordinates_var.set("Cursor: outside the calibrated plot area")
            self._canvas.configure(cursor="")
            return

        point = _preview_to_axis_point(self._figure_view, preview_point[0], preview_point[1])
        if point is None:
            self._cursor_coordinates_var.set("Cursor: outside the calibrated plot area")
        else:
            self._cursor_coordinates_var.set(_format_axis_coordinates("Cursor", point))

        if self._pan_pivot_preview is not None:
            self._canvas.configure(cursor="fleur")
            return

        if self._dragging_marker is not None:
            self._canvas.configure(cursor="fleur")
            return

        hit = _hit_test_control_marker(
            self._figure_view,
            {
                material_id: self._display_controls(material_id)
                for material_id in self._material_ids
            },
            preview_point[0],
            preview_point[1],
            self._zoom_scale,
        )
        self._canvas.configure(cursor="hand2" if hit is not None else "")

    def _on_canvas_leave(self, _event: tk.Event[tk.Canvas]) -> None:
        self._cursor_coordinates_var.set("Cursor: move over the plot to see X/Y")
        if self._dragging_marker is None and self._pan_pivot_preview is None:
            self._canvas.configure(cursor="")

    def _update_value_labels(self) -> None:
        start_label = _format_control_point_label(
            self._default_controls[self._selected_material_id].start,
            self._override_controls[self._selected_material_id].start,
        )
        end_label = _format_control_point_label(
            self._default_controls[self._selected_material_id].end,
            self._override_controls[self._selected_material_id].end,
        )
        self._start_var.set(start_label)
        self._end_var.set(end_label)
        line_style = self._override_controls[self._selected_material_id].line_style
        self._line_style_var.set(line_style if line_style is not None else "auto")

    def _redraw_canvas(self) -> None:
        preview = self._figure_view.preview_image.copy().convert("RGBA")
        scaled_size = (
            max(1, int(round(preview.width * self._zoom_scale))),
            max(1, int(round(preview.height * self._zoom_scale))),
        )
        if scaled_size != preview.size:
            preview = preview.resize(scaled_size, Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(preview)

        for index, material_id in enumerate(self._material_ids):
            color = PREVIEW_CURVE_COLORS[index % len(PREVIEW_CURVE_COLORS)]
            controls = self._display_controls(material_id)
            selected = material_id == self._selected_material_id
            if controls.start is not None:
                canvas_point = _axis_to_canvas_coordinates(self._figure_view, controls.start, self._zoom_scale)
                self._draw_editor_marker(draw, canvas_point, color, "start", selected)
            if controls.end is not None:
                canvas_point = _axis_to_canvas_coordinates(self._figure_view, controls.end, self._zoom_scale)
                self._draw_editor_marker(draw, canvas_point, color, "end", selected)

        rendered = preview.convert("RGB")
        self._scaled_image_size = rendered.size
        self._zoom_var.set(f"Zoom: {int(round(self._zoom_scale * 100.0))}%")
        self._photo_image = ImageTk.PhotoImage(rendered)
        if self._canvas_image_id is None:
            self._canvas_image_id = self._canvas.create_image(0, 0, anchor="nw", image=self._photo_image)
        else:
            self._canvas.itemconfigure(self._canvas_image_id, image=self._photo_image)

        width, height = rendered.size
        self._canvas.configure(scrollregion=(0, 0, width, height))

    def _schedule_preview_refresh(self) -> None:
        if self._preview_refresh_after_id is not None:
            self.after_cancel(self._preview_refresh_after_id)

        self._preview_status_var.set("Preview: reparsing after 1.0s idle")
        self._preview_refresh_after_id = self.after(1000, self._run_preview_refresh)

    def _run_preview_refresh(self) -> None:
        self._preview_refresh_after_id = None
        self._preview_status_var.set("Preview: reparsing current curve")
        try:
            self._figure_view = _build_figure_view(
                self._pdf_path,
                self._figure_view.analysis.spec,
                self.get_overrides(),
            )
        except Exception as exc:  # noqa: BLE001
            self._preview_status_var.set(f"Preview: parse failed - {exc}")
            return

        self._preview_status_var.set("Preview: parsed curve refreshed")
        self._redraw_canvas()

    def _fit_image_to_canvas(self) -> None:
        canvas_width = max(1, self._canvas.winfo_width())
        canvas_height = max(1, self._canvas.winfo_height())
        self._zoom_scale = _calculate_fit_zoom(
            self._figure_view.preview_image.size,
            (canvas_width, canvas_height),
        )
        self._redraw_canvas()
        self._canvas.xview_moveto(0.0)
        self._canvas.yview_moveto(0.0)

    def _set_zoom(
        self,
        new_zoom_scale: float,
        pivot_preview: tuple[float, float] | None = None,
        widget_point: tuple[float, float] | None = None,
    ) -> None:
        clamped_zoom = min(max(new_zoom_scale, _MIN_ZOOM_SCALE), _MAX_ZOOM_SCALE)
        if abs(clamped_zoom - self._zoom_scale) < 1e-6:
            return

        self._zoom_scale = clamped_zoom
        self._redraw_canvas()
        self.update_idletasks()
        if pivot_preview is None or widget_point is None:
            return

        self._position_view_at_pivot(pivot_preview, widget_point)

    def _position_view_at_pivot(
        self,
        pivot_preview: tuple[float, float],
        widget_point: tuple[float, float],
    ) -> None:
        scaled_x = pivot_preview[0] * self._zoom_scale
        scaled_y = pivot_preview[1] * self._zoom_scale
        canvas_width = max(1, self._canvas.winfo_width())
        canvas_height = max(1, self._canvas.winfo_height())
        total_width, total_height = self._scaled_image_size

        if total_width > canvas_width:
            left = _view_offset_for_pivot(total_width, canvas_width, scaled_x, widget_point[0])
            self._canvas.xview_moveto(left / max(total_width - canvas_width, 1))
        else:
            self._canvas.xview_moveto(0.0)

        if total_height > canvas_height:
            top = _view_offset_for_pivot(total_height, canvas_height, scaled_y, widget_point[1])
            self._canvas.yview_moveto(top / max(total_height - canvas_height, 1))
        else:
            self._canvas.yview_moveto(0.0)

    def _on_canvas_pan_start(self, event: tk.Event[tk.Canvas]) -> None:
        preview_point = self._event_to_preview_coordinates(event)
        if preview_point is None:
            return

        self._pan_pivot_preview = preview_point
        self._canvas.configure(cursor="fleur")

    def _on_canvas_pan_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._pan_pivot_preview is None:
            return

        self._position_view_at_pivot(
            self._pan_pivot_preview,
            (float(event.x), float(event.y)),
        )

    def _on_canvas_pan_end(self, _event: tk.Event[tk.Canvas]) -> None:
        if self._pan_pivot_preview is None:
            return

        self._pan_pivot_preview = None
        if self._dragging_marker is None:
            self._canvas.configure(cursor="")

    def _on_mouse_wheel(self, event: tk.Event[tk.Canvas]) -> None:
        preview_point = self._event_to_preview_coordinates(event)
        if preview_point is None:
            return

        factor = _ZOOM_FACTOR if event.delta > 0 else (1.0 / _ZOOM_FACTOR)
        self._set_zoom(
            self._zoom_scale * factor,
            pivot_preview=preview_point,
            widget_point=(float(event.x), float(event.y)),
        )

    def _on_mouse_wheel_linux(self, event: tk.Event[tk.Canvas]) -> None:
        preview_point = self._event_to_preview_coordinates(event)
        if preview_point is None:
            return

        factor = _ZOOM_FACTOR if getattr(event, "num", 0) == 4 else (1.0 / _ZOOM_FACTOR)
        self._set_zoom(
            self._zoom_scale * factor,
            pivot_preview=preview_point,
            widget_point=(float(event.x), float(event.y)),
        )

    def _event_to_preview_coordinates(self, event: tk.Event[tk.Canvas]) -> tuple[float, float] | None:
        canvas_x = float(self._canvas.canvasx(event.x))
        canvas_y = float(self._canvas.canvasy(event.y))
        return _canvas_to_preview_coordinates(self._figure_view, canvas_x, canvas_y, self._zoom_scale)

    def _display_controls(self, material_id: str) -> CurveControlPoints:
        defaults = self._default_controls[material_id]
        overrides = self._override_controls[material_id]
        return CurveControlPoints(
            start=overrides.start if overrides.start is not None else defaults.start,
            end=overrides.end if overrides.end is not None else defaults.end,
            line_style=overrides.line_style,
        )

    def _set_line_style_override(self, line_style: CurveLineStyle | None) -> None:
        self._override_controls[self._selected_material_id] = self._override_controls[
            self._selected_material_id
        ].with_line_style(line_style)
        if line_style is None:
            self._instructions_var.set("Removed the tracing override for the selected material.")
        else:
            self._instructions_var.set(
                f"Tracing override set to {line_style} for {_material_display_name(self._selected_material_id)}."
            )
        self._update_value_labels()
        self._schedule_preview_refresh()

    def _draw_editor_marker(
        self,
        draw: ImageDraw.ImageDraw,
        point: tuple[float, float],
        color: tuple[int, int, int, int],
        point_kind: str,
        selected: bool,
    ) -> None:
        marker_x = int(round(point[0]))
        marker_y = int(round(point[1]))
        halo_radius = 12 if selected else 9

        if point_kind == "start":
            draw.rectangle(
                (marker_x - halo_radius, marker_y - halo_radius, marker_x + halo_radius, marker_y + halo_radius),
                fill=(255, 255, 255, 230),
                outline=(0, 0, 0, 255),
                width=3,
            )
            draw.rectangle(
                (marker_x - 6, marker_y - 6, marker_x + 6, marker_y + 6),
                fill=color,
                outline=(57, 255, 20, 255),
                width=2,
            )
        else:
            draw.ellipse(
                (marker_x - halo_radius, marker_y - halo_radius, marker_x + halo_radius, marker_y + halo_radius),
                fill=(255, 255, 255, 230),
                outline=(57, 255, 20, 255),
                width=3,
            )
            draw.ellipse(
                (marker_x - 6, marker_y - 6, marker_x + 6, marker_y + 6),
                fill=color,
                outline=(0, 0, 0, 255),
                width=2,
            )

    def _set_control_point(
        self,
        material_id: str,
        point_kind: str,
        point: tuple[float, float],
    ) -> None:
        current = self._override_controls[material_id]
        if point_kind == "start":
            self._override_controls[material_id] = current.with_start(point)
        else:
            self._override_controls[material_id] = current.with_end(point)


def launch_calibration_ui(
    pdf_path: Path,
    calibration_path: Path = CURVE_CALIBRATION_PATH,
    preview_dir: Path = FIGURE_PREVIEWS_DIR,
) -> bool:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Tkinter is not available in this Python environment.") from exc

    root.title("Curve Calibration")
    root.geometry("1680x980")

    saved_controls = load_curve_control_points(calibration_path)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=12, pady=(12, 6))

    tabs: dict[int, _CalibrationTab] = {}
    failures: list[str] = []
    for spec in SUPPORTED_FIGURE_SPECS:
        try:
            figure_view = _build_figure_view(
                pdf_path,
                spec,
                saved_controls.get(spec.figure_num, {}),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Figure {spec.figure_num}: {exc}")
            continue

        tab = _CalibrationTab(notebook, pdf_path, figure_view, saved_controls.get(spec.figure_num, {}))
        notebook.add(tab, text=f"Figure {spec.figure_num}")
        tabs[spec.figure_num] = tab

    if failures:
        messagebox.showwarning(
            title="Calibration warnings",
            message="Some figures could not be opened:\n\n" + "\n".join(failures),
            parent=root,
        )

    if not tabs:
        root.destroy()
        raise RuntimeError("No calibration views could be created.")

    applied = False

    def on_cancel() -> None:
        root.destroy()

    def on_apply() -> None:
        nonlocal applied
        try:
            save_curve_control_points(
                {
                    figure_num: tab.get_overrides()
                    for figure_num, tab in tabs.items()
                },
                calibration_path,
            )
            clear_figure_analysis_cache()
            generate_supported_figure_previews(pdf_path, preview_dir)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                title="Calibration save failed",
                message=str(exc),
                parent=root,
            )
            return

        applied = True
        root.destroy()

    buttons = ttk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=(0, 12))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(buttons, text="Apply", command=on_apply).pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return applied


def _curve_point_to_tuple(point: Any) -> tuple[float, float]:
    return float(point.stress_MPa), float(point.relaxation_percent)


def _build_figure_view(
    pdf_path: Path,
    spec: Any,
    manual_curve_controls: dict[str, CurveControlPoints],
) -> _FigureCalibrationView:
    analysis = _analyze_figure(pdf_path, spec, manual_curve_controls=manual_curve_controls)
    crop_box = _preview_crop_box(
        (analysis.page_image.shape[1], analysis.page_image.shape[0]),
        analysis.plot_bounds,
    )
    return _FigureCalibrationView(
        analysis=analysis,
        preview_image=_render_figure_preview(analysis, show_curve_controls=False),
        crop_box=crop_box,
    )


def _calculate_fit_zoom(
    image_size: tuple[int, int],
    viewport_size: tuple[int, int],
) -> float:
    image_width, image_height = image_size
    viewport_width, viewport_height = viewport_size
    if image_width <= 0 or image_height <= 0:
        return 1.0

    fit_scale = min(viewport_width / image_width, viewport_height / image_height)
    return min(1.0, max(fit_scale, _MIN_ZOOM_SCALE))


def _view_offset_for_pivot(
    total_size: int,
    viewport_size: int,
    scaled_pivot: float,
    widget_pivot: float,
) -> float:
    if total_size <= viewport_size:
        return 0.0

    return min(max(scaled_pivot - widget_pivot, 0.0), total_size - viewport_size)


def _canvas_to_preview_coordinates(
    figure_view: _FigureCalibrationView,
    canvas_x: float,
    canvas_y: float,
    zoom_scale: float,
) -> tuple[float, float] | None:
    if zoom_scale <= 0.0:
        return None

    preview_x = canvas_x / zoom_scale
    preview_y = canvas_y / zoom_scale
    image_width, image_height = figure_view.preview_image.size
    if not (0.0 <= preview_x < image_width and 0.0 <= preview_y < image_height):
        return None

    return preview_x, preview_y


def _axis_to_preview_coordinates(
    figure_view: _FigureCalibrationView,
    point: tuple[float, float],
) -> tuple[float, float]:
    crop_left, crop_top, _, _ = figure_view.crop_box
    preview_x = float(figure_view.analysis.calibration.stress_to_x(point[0]) - crop_left)
    preview_y = float(figure_view.analysis.calibration.relaxation_to_y(point[1]) - crop_top)
    return preview_x, preview_y


def _axis_to_canvas_coordinates(
    figure_view: _FigureCalibrationView,
    point: tuple[float, float],
    zoom_scale: float,
) -> tuple[float, float]:
    preview_x, preview_y = _axis_to_preview_coordinates(figure_view, point)
    return preview_x * zoom_scale, preview_y * zoom_scale


def _hit_test_control_marker(
    figure_view: _FigureCalibrationView,
    controls_by_material: dict[str, CurveControlPoints],
    preview_x: float,
    preview_y: float,
    zoom_scale: float,
) -> tuple[str, str] | None:
    tolerance = _MARKER_HIT_RADIUS / max(zoom_scale, 0.01)
    best_hit: tuple[str, str] | None = None
    best_distance = inf

    for material_id, controls in controls_by_material.items():
        if controls.start is not None:
            distance = _marker_distance(_axis_to_preview_coordinates(figure_view, controls.start), (preview_x, preview_y))
            if distance <= tolerance and distance < best_distance:
                best_hit = (material_id, "start")
                best_distance = distance

        if controls.end is not None:
            distance = _marker_distance(_axis_to_preview_coordinates(figure_view, controls.end), (preview_x, preview_y))
            if distance <= tolerance and distance < best_distance:
                best_hit = (material_id, "end")
                best_distance = distance

    return best_hit


def _marker_distance(
    marker_preview_point: tuple[float, float],
    cursor_preview_point: tuple[float, float],
) -> float:
    delta_x = marker_preview_point[0] - cursor_preview_point[0]
    delta_y = marker_preview_point[1] - cursor_preview_point[1]
    return sqrt((delta_x * delta_x) + (delta_y * delta_y))


def _preview_to_axis_point(
    figure_view: _FigureCalibrationView,
    preview_x: float,
    preview_y: float,
    clamp_to_plot: bool = False,
) -> tuple[float, float] | None:
    image_width, image_height = figure_view.preview_image.size
    if not (0.0 <= preview_x < image_width and 0.0 <= preview_y < image_height):
        return None

    crop_left, crop_top, _, _ = figure_view.crop_box
    page_x = crop_left + preview_x
    page_y = crop_top + preview_y
    plot_left, plot_top, plot_right, plot_bottom = figure_view.analysis.plot_bounds
    if clamp_to_plot:
        page_x = min(max(page_x, float(plot_left)), float(plot_right))
        page_y = min(max(page_y, float(plot_top)), float(plot_bottom))
    elif not (plot_left <= page_x <= plot_right and plot_top <= page_y <= plot_bottom):
        return None

    stress = figure_view.analysis.calibration.x_to_stress(page_x)
    relaxation = figure_view.analysis.calibration.y_to_relaxation(page_y)
    clamped_stress = min(max(float(stress), 0.0), figure_view.analysis.spec.x_max)
    clamped_relaxation = min(max(float(relaxation), 0.0), Y_AXIS_MAX)
    return round(clamped_stress, 1), round(clamped_relaxation, 3)


def _format_control_point_label(
    default_point: tuple[float, float] | None,
    override_point: tuple[float, float] | None,
) -> str:
    if override_point is not None:
        return _format_point("Manual", override_point)
    if default_point is not None:
        return _format_point("Current", default_point)
    return "Not set"


def _format_point(prefix: str, point: tuple[float, float]) -> str:
    return f"{prefix}: {point[0]:.1f} MPa, {point[1]:.3f}%"


def _format_axis_coordinates(prefix: str, point: tuple[float, float]) -> str:
    return f"{prefix}: X {point[0]:.1f} MPa | Y {point[1]:.3f}%"