"""
gui.py — UI Layout and Interaction
====================================
Defines the main application window, control panel, and
Matplotlib canvas embedded inside the PyQt5 window.

Architecture:
    MainWindow
    ├── ControlPanel  (left)  — sliders, dropdowns, buttons
    └── PlotPanel     (right) — 2D heatmap + 1D cross-section
"""

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox,
    QSizePolicy, QCheckBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QStatusBar, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulation import DiffractionSimulator


# ── Colour palette ──────────────────────────────────────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_WIDGET  = "#1c2128"
BORDER     = "#30363d"
ACCENT     = "#58a6ff"
ACCENT2    = "#3fb950"
TEXT_PRI   = "#e6edf3"
TEXT_SEC   = "#8b949e"
WARN       = "#f78166"


class SliderGroup(QWidget):
    """A labelled slider with a live numeric readout."""

    def __init__(self, label: str, vmin: float, vmax: float,
                 default: float, decimals: int = 0,
                 unit: str = "", parent=None):
        super().__init__(parent)
        self.scale   = 10 ** decimals   # integer ↔ float conversion factor
        self.decimals = decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        # Row: label + value display
        top = QHBoxLayout()
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px;")
        self.val_lbl = QLabel(self._fmt(default))
        self.val_lbl.setStyleSheet(
            f"color:{ACCENT}; font-size:12px; font-weight:bold;"
        )
        top.addWidget(self.lbl)
        top.addStretch()
        top.addWidget(self.val_lbl)
        if unit:
            u = QLabel(unit)
            u.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
            top.addWidget(u)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(vmin * self.scale))
        self.slider.setMaximum(int(vmax * self.scale))
        self.slider.setValue(int(default * self.scale))
        self.slider.setStyleSheet(self._slider_style())
        self.slider.valueChanged.connect(
            lambda v: self.val_lbl.setText(self._fmt(v / self.scale))
        )

        layout.addLayout(top)
        layout.addWidget(self.slider)

    def value(self) -> float:
        return self.slider.value() / self.scale

    def connect(self, fn):
        self.slider.valueChanged.connect(fn)

    def _fmt(self, v: float) -> str:
        return f"{v:.{self.decimals}f}"

    @staticmethod
    def _slider_style() -> str:
        return f"""
            QSlider::groove:horizontal {{
                height: 4px; background: {BORDER};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT}; border-radius: 2px;
            }}
        """


class ControlPanel(QFrame):
    """Left panel — all adjustable parameters and controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(
            f"background:{BG_PANEL}; border-right:1px solid {BORDER};"
        )
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 20, 16, 20)
        root.setSpacing(0)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel("Diffraction Simulator")
        title.setStyleSheet(
            f"color:{TEXT_PRI}; font-size:16px; font-weight:bold; "
            f"letter-spacing:0.5px;"
        )
        sub = QLabel("Fourier Optics · Scalar Theory")
        sub.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; margin-bottom:16px;")
        root.addWidget(title)
        root.addWidget(sub)

        sep = self._separator()
        root.addWidget(sep)

        # ── Aperture type ──────────────────────────────────────────────
        root.addSpacing(12)
        root.addWidget(self._section_label("Aperture"))

        self.aperture_combo = QComboBox()
        self.aperture_combo.addItems(DiffractionSimulator.APERTURE_TYPES)
        self.aperture_combo.setStyleSheet(self._combo_style())
        root.addWidget(self.aperture_combo)

        # ── Physics parameters ─────────────────────────────────────────
        root.addSpacing(12)
        root.addWidget(self._section_label("Parameters"))

        self.sl_wavelength = SliderGroup(
            "Wavelength (λ)", 380, 780, 550, decimals=0, unit="nm"
        )
        self.sl_radius = SliderGroup(
            "Aperture radius", 0.05, 0.50, 0.20, decimals=2, unit="au"
        )

        root.addWidget(self.sl_wavelength)
        root.addWidget(self.sl_radius)

        # ── Rectangular-specific params (shown/hidden contextually) ────
        self.rect_group = QGroupBox("Rect / Slit Parameters")
        self.rect_group.setStyleSheet(self._groupbox_style())
        rgl = QVBoxLayout(self.rect_group)
        rgl.setContentsMargins(8, 8, 8, 8)

        self.sl_rect_w = SliderGroup("Width",     0.05, 0.80, 0.30, decimals=2)
        self.sl_rect_h = SliderGroup("Height",    0.05, 0.80, 0.15, decimals=2)
        self.sl_slit_d = SliderGroup("Separation",0.05, 0.80, 0.30, decimals=2)
        self.sl_slit_w = SliderGroup("Slit width",0.01, 0.30, 0.05, decimals=2)

        rgl.addWidget(self.sl_rect_w)
        rgl.addWidget(self.sl_rect_h)
        rgl.addWidget(self.sl_slit_d)
        rgl.addWidget(self.sl_slit_w)
        root.addWidget(self.rect_group)
        self.rect_group.hide()

        # ── Grid resolution ────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(self._section_label("Grid"))
        self.sl_grid = SliderGroup("Resolution N×N", 64, 512, 256, decimals=0)
        root.addWidget(self.sl_grid)

        # ── Display options ────────────────────────────────────────────
        root.addSpacing(12)
        root.addWidget(self._section_label("Display"))

        self.cb_log   = QCheckBox("Logarithmic scale")
        self.cb_apert = QCheckBox("Show aperture overlay")
        self.cb_cross = QCheckBox("Show cross-section")
        self.cb_cross.setChecked(True)

        for cb in (self.cb_log, self.cb_apert, self.cb_cross):
            cb.setStyleSheet(self._checkbox_style())
            root.addWidget(cb)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["inferno", "hot", "plasma", "viridis", "gray"])
        self.cmap_combo.setStyleSheet(self._combo_style())
        root.addWidget(self.cmap_combo)

        root.addSpacing(12)

        # ── Action buttons ─────────────────────────────────────────────
        self.btn_run  = QPushButton("▶  Run Simulation")
        self.btn_save = QPushButton("💾  Save Image")

        self.btn_run.setStyleSheet(self._btn_style(ACCENT))
        self.btn_save.setStyleSheet(self._btn_style(BG_WIDGET, border=BORDER))

        root.addWidget(self.btn_run)
        root.addSpacing(6)
        root.addWidget(self.btn_save)

        # ── Auto-update toggle ─────────────────────────────────────────
        self.cb_auto = QCheckBox("Auto-update on change")
        self.cb_auto.setStyleSheet(self._checkbox_style())
        root.addWidget(self.cb_auto)

        root.addStretch()

        # ── Stats box ─────────────────────────────────────────────────
        self.stats_lbl = QLabel("Run a simulation to see stats.")
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setStyleSheet(
            f"color:{TEXT_SEC}; font-size:10px; "
            f"background:{BG_WIDGET}; padding:8px; border-radius:4px;"
        )
        root.addWidget(self.stats_lbl)

        # ── Wire aperture combo → show/hide extra group ────────────────
        self.aperture_combo.currentTextChanged.connect(self._on_aperture_change)

    # ------------------------------------------------------------------
    def _on_aperture_change(self, text: str):
        show = text in ("Rectangular", "Double Slit")
        self.rect_group.setVisible(show)
        # Show relevant sub-controls
        self.sl_rect_w.setVisible(text == "Rectangular")
        self.sl_rect_h.setVisible(text == "Rectangular")
        self.sl_slit_d.setVisible(text == "Double Slit")
        self.sl_slit_w.setVisible(text == "Double Slit")

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        return sep

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"color:{TEXT_SEC}; font-size:9px; font-weight:bold; "
            f"letter-spacing:1.5px; margin-bottom:4px;"
        )
        return lbl

    @staticmethod
    def _combo_style() -> str:
        return f"""
            QComboBox {{
                background:{BG_WIDGET}; color:{TEXT_PRI};
                border:1px solid {BORDER}; border-radius:4px;
                padding:4px 8px; font-size:12px;
            }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{
                background:{BG_WIDGET}; color:{TEXT_PRI};
                selection-background-color:{ACCENT};
            }}
        """

    @staticmethod
    def _btn_style(bg: str, border: str = ACCENT) -> str:
        return f"""
            QPushButton {{
                background:{bg}; color:{TEXT_PRI};
                border:1px solid {border}; border-radius:4px;
                padding:8px; font-size:12px; font-weight:bold;
            }}
            QPushButton:hover {{ background:{ACCENT}; color:#000; }}
            QPushButton:pressed {{ background:{ACCENT2}; }}
        """

    @staticmethod
    def _checkbox_style() -> str:
        return f"""
            QCheckBox {{ color:{TEXT_PRI}; font-size:11px; spacing:6px; }}
            QCheckBox::indicator {{
                width:14px; height:14px;
                border:1px solid {BORDER}; border-radius:3px;
                background:{BG_WIDGET};
            }}
            QCheckBox::indicator:checked {{
                background:{ACCENT}; border-color:{ACCENT};
            }}
        """

    @staticmethod
    def _groupbox_style() -> str:
        return f"""
            QGroupBox {{
                color:{TEXT_SEC}; font-size:10px;
                border:1px solid {BORDER}; border-radius:4px;
                margin-top:8px; padding-top:4px;
            }}
            QGroupBox::title {{
                subcontrol-origin:margin; left:8px;
                color:{TEXT_SEC};
            }}
        """


class PlotCanvas(FigureCanvas):
    """
    Matplotlib figure embedded in the Qt window.

    Layout (with cross-section visible):
        ┌─────────────────────┬──────────┐
        │   2D Intensity Map  │ 1D cross │
        │                     │ section  │
        └─────────────────────┴──────────┘
    """

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=BG_DARK, tight_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._init_axes()

    def _init_axes(self, show_cross: bool = True):
        self.fig.clear()
        if show_cross:
            gs = gridspec.GridSpec(
                2, 2,
                figure=self.fig,
                left=0.08, right=0.97, top=0.92, bottom=0.10,
                wspace=0.35, hspace=0.45,
                width_ratios=[2, 1],
                height_ratios=[2, 1]
            )
            self.ax2d    = self.fig.add_subplot(gs[0, 0])  # main 2D plot
            self.ax_hcross = self.fig.add_subplot(gs[1, 0])  # horizontal slice
            self.ax_vcross = self.fig.add_subplot(gs[0, 1])  # vertical slice
            self.ax_apert  = self.fig.add_subplot(gs[1, 1])  # aperture preview
        else:
            self.ax2d = self.fig.add_subplot(111)
            self.ax_hcross = None
            self.ax_vcross = None
            self.ax_apert  = None

        for ax in self.fig.axes:
            ax.set_facecolor(BG_WIDGET)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.tick_params(colors=TEXT_SEC, labelsize=8)
            ax.xaxis.label.set_color(TEXT_SEC)
            ax.yaxis.label.set_color(TEXT_SEC)
            ax.title.set_color(TEXT_PRI)

    def render(self, result: dict, params: dict,
               cmap: str = "inferno", log_scale: bool = False,
               show_aperture: bool = False, show_cross: bool = True):
        """Re-draw the figure with new simulation results."""
        self._init_axes(show_cross=show_cross)

        intensity = result["intensity"]
        if log_scale:
            from utils import apply_log_scale
            display = apply_log_scale(intensity)
        else:
            display = intensity

        # ── 2D intensity map ──────────────────────────────────────────
        im = self.ax2d.imshow(
            display, cmap=cmap, origin="lower",
            extent=[result["x_axis"][0], result["x_axis"][-1],
                    result["y_axis"][0], result["y_axis"][-1]],
            aspect="equal", interpolation="bilinear"
        )
        self.fig.colorbar(im, ax=self.ax2d, fraction=0.046, pad=0.04,
                          label="Norm. Intensity")
        ap  = params.get("aperture_type", "")
        lam = params.get("wavelength_nm", "")
        r   = params.get("aperture_radius", "")
        N   = params.get("grid_N", "")
        self.ax2d.set_title(
            f"{ap} | λ={lam} nm | r={r:.2f} | N={N}",
            fontsize=9, pad=4
        )
        self.ax2d.set_xlabel("Frequency u")
        self.ax2d.set_ylabel("Frequency v")

        if show_cross:
            # ── Horizontal cross-section ──────────────────────────────
            freq = result["x_axis"]
            self.ax_hcross.plot(freq, result["cross_x"],
                                color=ACCENT, linewidth=1.2)
            self.ax_hcross.fill_between(freq, result["cross_x"],
                                        alpha=0.15, color=ACCENT)
            self.ax_hcross.set_title("Horizontal Slice", fontsize=8, pad=3)
            self.ax_hcross.set_xlabel("u")
            self.ax_hcross.set_ylabel("I")
            self.ax_hcross.set_xlim(freq[0], freq[-1])
            self.ax_hcross.set_ylim(-0.02, 1.05)

            # ── Vertical cross-section ────────────────────────────────
            self.ax_vcross.plot(result["cross_y"], freq,
                                color=ACCENT2, linewidth=1.2)
            self.ax_vcross.fill_betweenx(freq, result["cross_y"],
                                         alpha=0.15, color=ACCENT2)
            self.ax_vcross.set_title("Vertical Slice", fontsize=8, pad=3)
            self.ax_vcross.set_xlabel("I")
            self.ax_vcross.set_ylabel("v")
            self.ax_vcross.set_xlim(-0.02, 1.05)

            # ── Aperture preview ──────────────────────────────────────
            if show_aperture:
                self.ax_apert.imshow(
                    result["aperture"], cmap="gray", origin="lower",
                    aspect="equal", interpolation="nearest"
                )
                self.ax_apert.set_title("Aperture", fontsize=8, pad=3)
            else:
                self.ax_apert.text(
                    0.5, 0.5, "Enable\naperture\noverlay",
                    ha="center", va="center",
                    color=TEXT_SEC, fontsize=9,
                    transform=self.ax_apert.transAxes
                )
                self.ax_apert.set_title("Aperture", fontsize=8, pad=3)

        self.fig.patch.set_facecolor(BG_DARK)
        self.draw()

    def save(self, path: str):
        self.fig.savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Diffraction Simulator")
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(f"background:{BG_DARK}; color:{TEXT_PRI};")

        self.simulator = DiffractionSimulator()
        self._last_result = None

        # Auto-update timer — fires 400 ms after last slider move
        self._auto_timer = QTimer()
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self._run_simulation)

        self._build_ui()
        self._connect_signals()

        # Run an initial simulation so the window isn't blank
        self._run_simulation()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.ctrl   = ControlPanel()
        self.canvas = PlotCanvas()

        layout.addWidget(self.ctrl)
        layout.addWidget(self.canvas, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_SEC}; font-size:10px; "
            f"border-top:1px solid {BORDER};"
        )
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — press ▶ Run Simulation")

    def _connect_signals(self):
        self.ctrl.btn_run.clicked.connect(self._run_simulation)
        self.ctrl.btn_save.clicked.connect(self._save_image)

        # Auto-update: connect all sliders to a debounced timer
        for sl in (self.ctrl.sl_wavelength, self.ctrl.sl_radius,
                   self.ctrl.sl_grid, self.ctrl.sl_rect_w,
                   self.ctrl.sl_rect_h, self.ctrl.sl_slit_d,
                   self.ctrl.sl_slit_w):
            sl.connect(self._on_param_change)

        self.ctrl.aperture_combo.currentTextChanged.connect(
            self._on_param_change
        )
        self.ctrl.cmap_combo.currentTextChanged.connect(self._on_param_change)
        self.ctrl.cb_log.stateChanged.connect(self._on_param_change)
        self.ctrl.cb_apert.stateChanged.connect(self._on_param_change)
        self.ctrl.cb_cross.stateChanged.connect(self._on_param_change)

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def _on_param_change(self, *_):
        """Called whenever any control changes; triggers auto-update if on."""
        if self.ctrl.cb_auto.isChecked():
            self._auto_timer.start(400)

    def _run_simulation(self):
        """Read parameters, run physics, update plots."""
        ap_type = self.ctrl.aperture_combo.currentText()
        lam     = self.ctrl.sl_wavelength.value()
        radius  = self.ctrl.sl_radius.value()
        N       = int(self.ctrl.sl_grid.value())

        # Clamp N to nearest even number ≥ 64
        N = max(64, N)

        kwargs = dict(
            wavelength_nm  = lam,
            aperture_radius= radius,
            grid_N         = N,
            aperture_type  = ap_type,
        )

        if ap_type == "Rectangular":
            kwargs["rect_width"]  = self.ctrl.sl_rect_w.value()
            kwargs["rect_height"] = self.ctrl.sl_rect_h.value()
        elif ap_type == "Double Slit":
            kwargs["slit_separation"] = self.ctrl.sl_slit_d.value()
            kwargs["slit_width"]      = self.ctrl.sl_slit_w.value()

        self.status.showMessage("Computing…")
        try:
            result = self.simulator.run(**kwargs)
        except Exception as exc:
            self.status.showMessage(f"Error: {exc}")
            return

        self._last_result = result

        self.canvas.render(
            result,
            params       = self.simulator.params,
            cmap         = self.ctrl.cmap_combo.currentText(),
            log_scale    = self.ctrl.cb_log.isChecked(),
            show_aperture= self.ctrl.cb_apert.isChecked(),
            show_cross   = self.ctrl.cb_cross.isChecked(),
        )

        # Update stats panel
        from utils import ring_stats
        s = ring_stats(result["intensity"])
        self.ctrl.stats_lbl.setText(
            f"Peak: {s['peak']:.3f}  |  Mean: {s['mean']:.5f}\n"
            f"Energy proxy: {s['energy']:.1f}\n"
            f"Grid: {N}×{N}  |  λ={lam:.0f} nm"
        )
        self.status.showMessage(
            f"Done — {ap_type}, λ={lam:.0f} nm, r={radius:.2f}, N={N}"
        )

    def _save_image(self):
        if self._last_result is None:
            self.status.showMessage("Nothing to save — run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Diffraction Pattern",
            "diffraction_pattern.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path:
            self.canvas.save(path)
            self.status.showMessage(f"Saved to {path}")
