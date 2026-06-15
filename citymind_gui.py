"""
CityMind — CustomTkinter dashboard
Run: python citymind_gui.py
"""

import io
import math
import random
import sys
import threading
import time
import tkinter as tk

import customtkinter as ctk

from ambulance_placement import AmbulancePlacementSA
from city_model import CityGraph, LocationType
from crime_risk import CrimeRiskPredictor
from csp_layout import CityLayoutCSP
from dynamic_router import DynamicEmergencyRouter, choose_emergency_start_node
from road_network import RoadNetworkOptimizer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ctk.deactivate_automatic_dpi_awareness()
ctk.set_widget_scaling(1.15)
ctk.set_window_scaling(1.15)

BG = "#0a1022"
PANEL = "#111a31"
CARD = "#16233f"
BORDER = "#2a3e63"
ACCENT = "#00e5ff"
GREEN = "#6dff99"
RED = "#ff647a"
YELLOW = "#f7d354"
PURPLE = "#9f6bff"
CYAN = "#26d6ff"
TEXT = "#eaf3ff"
MUTED = "#8aa4cf"

TYPE_COLORS = {
    LocationType.RESIDENTIAL: "#2fb05f",
    LocationType.HOSPITAL: "#ff5f74",
    LocationType.SCHOOL: "#3da7ff",
    LocationType.INDUSTRIAL: "#f59d35",
    LocationType.POWER_PLANT: "#ba5cff",
    LocationType.AMBULANCE_DEPOT: "#00bcd4",
    LocationType.EMPTY: "#1a2a4a",
}
TYPE_SYM = {
    LocationType.RESIDENTIAL: "RES",
    LocationType.HOSPITAL: "HOS",
    LocationType.SCHOOL: "SCH",
    LocationType.INDUSTRIAL: "IND",
    LocationType.POWER_PLANT: "PWR",
    LocationType.AMBULANCE_DEPOT: "AMB",
    LocationType.EMPTY: "...",
}
AMB_COLORS = ["#ffd740", "#ff70d8", "#73ff9c", "#5bc7ff", "#ff9470", "#d292ff"]


def silent(fn, *a, **kw):
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


class CityMindGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CityMind Command Center")
        self.configure(fg_color=BG)
        self.minsize(1280, 760)
        self.after(20, self._maximize_window)

        self.graph = None
        self.optimizer = None
        self.solver = None
        self.coverage = {}
        self.running = False
        self.paused = False
        self._stop_requested = False
        self.router = None
        self.risk_predictor = None
        self.risk_predictions = {}
        self.police_deployment = {}
        self.event_log = []
        self.sim_step = 0
        self.live_route_nodes = []
        self.route_anim_index = -1
        self.route_anim_job = None
        self.route_animation_queue = []
        self.route_unit_label = "EMS"
        self._sa_hist = []
        self.selected_node_id = None
        self.hover_node_id = None
        self._grid_draw_state = (30, 0, 0)
        self.route_history = []
        self.visual_effects = []
        self.active_banner = None
        self.c4_pending_pulses = {}  # {nid: "white"/"red"/None}

        self.show_roads = ctk.BooleanVar(value=True)
        self.show_heat = ctk.BooleanVar(value=True)
        self.show_pop = ctk.BooleanVar(value=True)
        self.show_risk = ctk.BooleanVar(value=True)
        self.show_dynamic_route = ctk.BooleanVar(value=True)

        self._build_ui()
        self.after(120, self._tick_visual_effects)
        self.after(150, self._init_city)

    def _maximize_window(self):
        try:
            self.state("zoomed")
        except tk.TclError:
            self.attributes("-zoomed", True)

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#090f20", corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="CityMind",
            text_color=ACCENT,
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=(24, 0), pady=(14, 2))
        subtitle = ctk.CTkLabel(
            header,
            text="Urban Intelligence Command Center",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=14),
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=(24, 0), pady=(0, 14))

        badges = ctk.CTkFrame(header, fg_color="transparent")
        badges.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 16))
        for text, color in [
            ("C1 Layout", GREEN),
            ("C2 Roads", YELLOW),
            ("C3 Ambulances", RED),
            ("C4 Routing", PURPLE),
            ("C5 Crime", CYAN),
        ]:
            ctk.CTkLabel(
                badges,
                text=text,
                text_color="#07111f",
                fg_color=color,
                corner_radius=8,
                width=96,
                height=28,
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            ).pack(side="left", padx=4, pady=12)

        self.sidebar = ctk.CTkScrollableFrame(
            self,
            width=370,
            fg_color=PANEL,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
        )
        self.sidebar.grid(row=1, column=0, sticky="nsew", padx=(14, 8), pady=10)
        self.sidebar.grid_columnconfigure(0, weight=1)

        self.main = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=0, border_width=0
        )
        self.main.grid(row=1, column=1, sticky="nsew", padx=(8, 14), pady=10)
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=4)
        self.main.grid_columnconfigure(1, weight=2)

        self._build_sidebar_controls()
        self._build_main_panels()

        self.status_var = ctk.StringVar(value="Initializing...")
        self.banner_var = ctk.StringVar(value="")
        status = ctk.CTkFrame(
            self, fg_color="#0e1a33", corner_radius=0, border_width=0, height=34
        )
        status.grid(row=2, column=0, columnspan=2, sticky="nsew")
        ctk.CTkLabel(
            status,
            textvariable=self.status_var,
            text_color=MUTED,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(
            status,
            textvariable=self.banner_var,
            text_color=ACCENT,
            anchor="e",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(fill="x", padx=16, pady=(0, 6))

    def _section_title(self, parent, text):
        ctk.CTkLabel(
            parent,
            text=text,
            text_color=ACCENT,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(14, 2))
        ctk.CTkFrame(parent, fg_color=BORDER, height=1).pack(
            fill="x", padx=12, pady=(0, 8)
        )

    def _build_sidebar_controls(self):
        sb = self.sidebar
        self._section_title(sb, "Challenge 1  City Layout")
        self._c1_seed = self._slider(sb, "Seed", 0, 9999, 42)
        self._c1_rows = self._slider(sb, "Grid Rows", 6, 15, 10)
        self._c1_cols = self._slider(sb, "Grid Cols", 6, 15, 10)
        self.btn_c1 = self._btn(sb, "Generate Layout", GREEN, self._run_c1)
        self.lbl_c1_status = self._kv(sb, "Status", "—")
        self.lbl_c1_nodes = self._kv(sb, "Nodes placed", "—")

        self._section_title(sb, "Challenge 2  Road Network")
        self.btn_c2 = self._btn(sb, "Optimize Roads", YELLOW, self._run_c2)
        self.lbl_c2_roads = self._kv(sb, "Roads built", "—")
        self.lbl_c2_cost = self._kv(sb, "Total cost", "—")
        self.lbl_c2_p1 = self._kv(sb, "Safety path 1", "—")
        self.lbl_c2_p2 = self._kv(sb, "Safety path 2", "—")

        self._section_title(sb, "Challenge 3  Ambulances")
        self._c3_ambs = self._slider(sb, "Ambulances (N)", 1, 6, 3)
        self._c3_T = self._slider(sb, "Init Temp", 10, 200, 80)
        self._c3_cool = self._slider(sb, "Cooling x1000", 980, 999, 995)
        self._c3_iter = self._slider(sb, "Max Iterations", 500, 12000, 5000, 500)
        self.btn_c3 = self._btn(sb, "Place Ambulances (SA)", RED, self._run_c3)
        self.btn_sim = self._btn(
            sb, "Run 20-Step Integrated Simulation", ACCENT, self._run_simulation
        )
        self.prog_var = ctk.DoubleVar(value=0)
        ctk.CTkProgressBar(
            sb, variable=self.prog_var, progress_color=ACCENT, fg_color="#101e38"
        ).pack(fill="x", padx=12, pady=(6, 8))
        self.lbl_c3_iter = self._kv(sb, "Iteration", "—")
        self.lbl_c3_temp = self._kv(sb, "Temperature", "—")
        self.lbl_c3_curr = self._kv(sb, "Current cost", "—")
        self.lbl_c3_best = self._kv(sb, "Best cost", "—")
        self.lbl_c3_worst = self._kv(sb, "Worst response", "—")
        self.lbl_c3_avg = self._kv(sb, "Avg response", "—")
        self.lbl_sim_step = self._kv(sb, "Simulation step", "0/20")

        self._section_title(sb, "Challenge 4  Dynamic Routing")
        self.btn_c4 = self._btn(sb, "Run C4 Dispatch Demo", PURPLE, self._run_c4_dispatch)
        self.lbl_c4_status = self._kv(sb, "C4 status", "—")
        self.lbl_c4_cost = self._kv(sb, "Route cost", "—")

        self._section_title(sb, "Challenge 5  Crime Risk")
        self.btn_c5 = self._btn(sb, "Run C5 Risk Model", CYAN, self._run_c5_risk)
        self.lbl_c5_status = self._kv(sb, "C5 status", "—")
        self.lbl_c5_mix = self._kv(sb, "Risk mix", "—")
        self.lbl_c5_police = self._kv(sb, "Police deployment", "—")

        self._section_title(sb, "Display Layers")
        for text, var in [
            ("Road Network", self.show_roads),
            ("Coverage Heatmap", self.show_heat),
            ("Population Density", self.show_pop),
            ("Crime Risk Heatmap", self.show_risk),
            ("Dynamic Route Animation", self.show_dynamic_route),
        ]:
            ctk.CTkCheckBox(
                sb,
                text=text,
                variable=var,
                command=self._redraw,
                font=ctk.CTkFont(family="Segoe UI", size=13),
                text_color=TEXT,
                fg_color=ACCENT,
                hover_color="#00c6dc",
            ).pack(anchor="w", padx=14, pady=3)

        ctk.CTkLabel(
            sb,
            text="Simulation Controls",
            text_color=MUTED,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(fill="x", padx=14, pady=(10, 2))
        self.btn_pause = self._btn(sb, "⏸  Pause", "#ffd740", self._toggle_pause)
        self.btn_stop = self._btn(sb, "■  Stop & Reset", "#ff647a", self._stop_and_reset)

    def _build_main_panels(self):
        canvas_panel = ctk.CTkFrame(
            self.main, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        canvas_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_panel.grid_rowconfigure(0, weight=1)
        canvas_panel.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_panel,
            bg="#070f1f",
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        telemetry = ctk.CTkFrame(self.main, fg_color="transparent")
        telemetry.grid(row=0, column=1, sticky="nsew")
        telemetry.grid_rowconfigure(3, weight=1)

        context_panel = ctk.CTkFrame(
            telemetry, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        context_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(
            context_panel,
            text="City Operations Context",
            text_color=ACCENT,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        self.city_context_var = ctk.StringVar(value="Generate a layout to begin city planning.")
        ctk.CTkLabel(
            context_panel,
            textvariable=self.city_context_var,
            text_color=MUTED,
            justify="left",
            wraplength=320,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).pack(fill="x", padx=12, pady=(0, 8))

        metrics_panel = ctk.CTkFrame(
            telemetry, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        metrics_panel.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(
            metrics_panel,
            text="Live City Metrics",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        self.city_metrics_var = ctk.StringVar(value="Nodes: -- | Population: -- | Risk(H): --")
        ctk.CTkLabel(
            metrics_panel,
            textvariable=self.city_metrics_var,
            text_color=MUTED,
            justify="left",
            wraplength=320,
            anchor="w",
            font=ctk.CTkFont(family="Consolas", size=12),
        ).pack(fill="x", padx=12, pady=(0, 8))

        self.mission_feed_var = ctk.StringVar(
            value="Mission Feed:\n- Awaiting operations input.\n- Run C4 or 20-step simulation."
        )
        feed_panel = ctk.CTkFrame(
            telemetry, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        feed_panel.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(
            feed_panel,
            text="Mission Feed",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            feed_panel,
            textvariable=self.mission_feed_var,
            text_color=TEXT,
            justify="left",
            wraplength=320,
            anchor="w",
            font=ctk.CTkFont(family="Consolas", size=11),
        ).pack(fill="x", padx=12, pady=(0, 8))

        self.node_focus_var = ctk.StringVar(
            value="Hover on the grid to inspect node details.\nClick a node to pin focus."
        )
        node_panel = ctk.CTkFrame(
            telemetry, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        node_panel.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(
            node_panel,
            text="Node Inspector",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            node_panel,
            textvariable=self.node_focus_var,
            text_color=TEXT,
            justify="left",
            wraplength=320,
            anchor="w",
            font=ctk.CTkFont(family="Consolas", size=12),
        ).pack(fill="x", padx=12, pady=(0, 8))

        lower = ctk.CTkFrame(telemetry, fg_color="transparent")
        lower.grid(row=4, column=0, sticky="nsew")
        lower.grid_rowconfigure(2, weight=1)

        legend_panel = ctk.CTkFrame(
            lower, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        legend_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        row = ctk.CTkFrame(legend_panel, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(6, 8))
        for lt, color in TYPE_COLORS.items():
            if lt == LocationType.EMPTY:
                continue
            ctk.CTkLabel(
                row,
                text=f" {TYPE_SYM[lt]} ",
                fg_color=color,
                corner_radius=8,
                text_color="#041021",
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                width=50,
            ).pack(side="left", padx=3, pady=4)

        chart_panel = ctk.CTkFrame(
            lower, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        chart_panel.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(
            chart_panel,
            text="SA Convergence",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 0))
        self.chart = tk.Canvas(chart_panel, bg=PANEL, highlightthickness=0, height=68)
        self.chart.pack(fill="x", padx=10, pady=(2, 8))

        self.log_panel = ctk.CTkFrame(
            lower, fg_color=PANEL, border_width=1, border_color=BORDER, corner_radius=12
        )
        self.log_panel.grid(row=2, column=0, sticky="nsew")
        ctk.CTkLabel(
            self.log_panel,
            text="Live Event Log",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        self.log_text = ctk.CTkTextbox(
            self.log_panel,
            fg_color="#0a1327",
            border_color=BORDER,
            border_width=1,
            text_color=TEXT,
            activate_scrollbars=True,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
            height=170,
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(2, 16))
        self.log_text.configure(state="disabled")
        self._update_city_context()

    def _slider(self, parent, label, lo, hi, default, step=1):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.pack(fill="x", padx=12, pady=3)
        box.grid_columnconfigure(0, weight=1)

        v = ctk.IntVar(value=default)
        live = ctk.StringVar(value=str(default))
        ctk.CTkLabel(
            box,
            text=label,
            text_color=TEXT,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            box,
            textvariable=live,
            text_color=ACCENT,
            anchor="e",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).grid(row=0, column=1, sticky="e")

        ctk.CTkSlider(
            box,
            from_=lo,
            to=hi,
            number_of_steps=max(1, int((hi - lo) / step)),
            variable=v,
            progress_color=ACCENT,
            fg_color="#0f1d38",
            button_color=ACCENT,
            button_hover_color="#00c7dc",
            command=lambda val: live.set(str(int(round(val)))),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 1))
        return v

    def _btn(self, parent, text, color, cmd):
        btn = ctk.CTkButton(
            parent,
            text=text,
            command=cmd,
            fg_color=color,
            hover_color=color,
            text_color="#041021",
            corner_radius=10,
            height=34,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        )
        btn.pack(fill="x", padx=12, pady=(6, 3))
        return btn

    def _kv(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color=CARD, border_width=1, border_color=BORDER, corner_radius=8)
        row.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(
            row,
            text=f"{label}:",
            text_color=MUTED,
            width=128,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).pack(side="left", padx=(8, 0), pady=4)
        var = ctk.StringVar(value=value)
        ctk.CTkLabel(
            row,
            textvariable=var,
            text_color=ACCENT,
            anchor="w",
            justify="left",
            wraplength=190,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(side="left", fill="x", expand=True, padx=8, pady=4)
        return var

    def _init_city(self):
        self._run_c1()

    def _append_log(self, message):
        self.event_log.append(message)
        if len(self.event_log) > 350:
            self.event_log = self.event_log[-350:]
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        # Insert a trailing blank line so see("end") scrolls past the
        # last real line — prevents the bottom text being clipped by
        # the lower panel border / padding.
        self.log_text.insert("end", "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _push_mission_feed(self, line):
        current = self.mission_feed_var.get().splitlines()
        if not current or current[0] != "Mission Feed:":
            current = ["Mission Feed:"]
        current.append(f"- {line}")
        current = [current[0]] + current[-5:]
        self.mission_feed_var.set("\n".join(current))

    def _show_banner(self, text, hold_ms=1800):
        self.active_banner = text
        self.banner_var.set(text)
        self.after(hold_ms, self._clear_banner)

    def _clear_banner(self):
        self.active_banner = None
        self.banner_var.set("")

    def _spawn_node_pulse(self, node_ids, color, strength=1.0):
        if self.graph is None:
            return
        for nid in node_ids:
            if nid in self.graph.nodes:
                self.visual_effects.append(
                    {
                        "kind": "pulse",
                        "nid": nid,
                        "color": color,
                        "ttl": int(18 * max(0.7, strength)),
                        "age": 0,
                    }
                )

    def _add_route_history(self, route_nodes, color):
        if not route_nodes:
            return
        self.route_history.append(
            {"nodes": list(route_nodes), "color": color, "ttl": 34, "age": 0}
        )
        if len(self.route_history) > 5:
            self.route_history = self.route_history[-5:]

    def _route_step_delay_ms(self):
        """Fast for 20-step sim (AMB), slow and clear for C4 demo."""
        if self.route_unit_label == "AMB":
            return 30
        return 340

    def _toggle_pause(self):
        """Freeze/unfreeze the current animation and simulation thread."""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.configure(text="\u25b6  Resume", fg_color="#6dff99")
            self._show_banner("PAUSED \u2014 explain to professor", hold_ms=60000)
            self._push_mission_feed("Simulation PAUSED.")
        else:
            self.btn_pause.configure(text="\u23f8  Pause", fg_color="#ffd740")
            self._clear_banner()
            self._push_mission_feed("Simulation RESUMED.")
            # Kick animation back into gear if it was mid-route
            if self.live_route_nodes and self.route_anim_job is None:
                self.route_anim_job = self.after(
                    self._route_step_delay_ms(), self._advance_route_animation
                )

    def _stop_and_reset(self):
        """Safely stop all running threads/animations and reset to blank."""
        self._stop_requested = True
        self.paused = False
        self.running = False
        self.btn_pause.configure(text="\u23f8  Pause", fg_color="#ffd740")
        # Cancel any pending after() animation jobs
        self._clear_route_animation()
        self.route_history = []
        self.visual_effects = []
        self.c4_pending_pulses = {}
        self._clear_banner()
        # Reset module state
        self.optimizer = None
        self.solver = None
        self.coverage = {}
        self.risk_predictions = {}
        self.police_deployment = {}
        self.router = None
        self.risk_predictor = None
        self.sim_step = 0
        self.graph = None
        self.event_log = []
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        # Re-enable buttons
        self.btn_c1.configure(state="normal")
        self.btn_c3.configure(state="normal")
        self.btn_sim.configure(state="normal")
        self.prog_var.set(0)
        for lbl in [self.lbl_c1_status, self.lbl_c1_nodes, self.lbl_c2_roads,
                     self.lbl_c2_cost, self.lbl_c2_p1, self.lbl_c2_p2,
                     self.lbl_c3_iter, self.lbl_c3_temp, self.lbl_c3_curr,
                     self.lbl_c3_best, self.lbl_c3_worst, self.lbl_c3_avg,
                     self.lbl_c4_status, self.lbl_c4_cost,
                     self.lbl_c5_status, self.lbl_c5_mix, self.lbl_c5_police]:
            lbl.set("\u2014")
        self.lbl_sim_step.set("0/20")
        self._stop_requested = False
        self.status_var.set("Resetting... regenerating city layout.")
        self._push_mission_feed("All state cleared. Regenerating C1 layout.")
        # Auto-regenerate C1 layout so the grid reappears immediately
        self._run_c1()

    def _tick_visual_effects(self):
        changed = False
        if self.route_history:
            for item in self.route_history:
                item["age"] += 1
            self.route_history = [x for x in self.route_history if x["age"] < x["ttl"]]
            changed = True
        if self.visual_effects:
            for fx in self.visual_effects:
                fx["age"] += 1
            self.visual_effects = [x for x in self.visual_effects if x["age"] < x["ttl"]]
            changed = True
        # Force continuous redraws while C4 civilian pulses are active
        # so the expanding-ring animation runs smoothly at 90ms tick rate.
        if any(v is not None for v in self.c4_pending_pulses.values()):
            changed = True
        if changed:
            self._redraw()
        self.after(90, self._tick_visual_effects)

    def _update_city_context(self):
        if self.graph is None:
            self.city_context_var.set("Generate a layout to begin city planning.")
            self.city_metrics_var.set("Nodes: -- | Population: -- | Risk(H): --")
            return

        phase = "Planning"
        if self.risk_predictions:
            phase = "Risk-Aware Operations"
        elif self.solver:
            phase = "Emergency Deployment"
        elif self.optimizer:
            phase = "Road Coordination"

        total_nodes = len(self.graph.nodes)
        total_pop = sum(n.population_density for n in self.graph.nodes.values())
        high_risk = 0
        if self.risk_predictions:
            high_risk = sum(1 for p in self.risk_predictions.values() if p["label"] == "HIGH")
        roads = len(self.optimizer.built_roads) if self.optimizer else 0
        ambulances = len(self.solver.best_placement) if self.solver and self.solver.best_placement else 0

        self.city_context_var.set(
            f"Current phase: {phase}. Step {self.sim_step}/20. "
            f"Use C2 and C4 to test route resilience after disruptions."
        )
        self.city_metrics_var.set(
            f"Nodes: {total_nodes} | Population: {total_pop} | Risk(H): {high_risk}\n"
            f"Built roads: {roads} | Active ambulances: {ambulances} | Police posts: {len(self.police_deployment)}"
        )

    def _get_node_from_canvas_xy(self, x, y):
        if self.graph is None:
            return None
        cell, ox, oy = self._grid_draw_state
        if cell <= 0:
            return None
        col = int((x - ox) // cell)
        row = int((y - oy) // cell)
        if 0 <= row < self.graph.rows and 0 <= col < self.graph.cols:
            return self.graph.node_id(row, col)
        return None

    def _set_node_focus_text(self, nid):
        if self.graph is None or nid is None:
            self.node_focus_var.set("Hover on the grid to inspect node details.\nClick a node to pin focus.")
            return
        node = self.graph.get_node(nid)
        risk = self.risk_predictions.get(nid, {}).get("label", "N/A")
        self.node_focus_var.set(
            f"Node ID: {nid}\n"
            f"Cell: ({node.row}, {node.col})\n"
            f"Type: {TYPE_SYM.get(node.location_type, 'UNK')}\n"
            f"Population: {node.population_density}\n"
            f"Risk: {risk}\n"
            f"Accessible: {node.accessible}"
        )

    def _on_canvas_motion(self, event):
        nid = self._get_node_from_canvas_xy(event.x, event.y)
        if nid != self.hover_node_id:
            self.hover_node_id = nid
            if self.selected_node_id is None:
                self._set_node_focus_text(nid)
            self._redraw()

    def _on_canvas_leave(self, _event):
        self.hover_node_id = None
        if self.selected_node_id is None:
            self._set_node_focus_text(None)
        self._redraw()

    def _on_canvas_click(self, event):
        nid = self._get_node_from_canvas_xy(event.x, event.y)
        if nid == self.selected_node_id:
            self.selected_node_id = None
            self._set_node_focus_text(self.hover_node_id)
        else:
            self.selected_node_id = nid
            self._set_node_focus_text(nid)
        self._redraw()

    def _run_c1(self):
        if self.running:
            return
        self.status_var.set("Running Challenge 1: CSP layout generation...")
        self.btn_c1.configure(state="disabled")
        self.lbl_c1_status.set("Running...")
        self.update_idletasks()

        seed = self._c1_seed.get()
        rows = self._c1_rows.get()
        cols = self._c1_cols.get()
        ok = False
        try:
            g = CityGraph(rows=rows, cols=cols)
            csp = CityLayoutCSP(g, seed=seed)
            ok = silent(csp.solve)
            if ok:
                self.graph = g
                self.optimizer = None
                self.solver = None
                self.coverage = {}
                self.risk_predictions = {}
                self.police_deployment = {}
                self.c4_pending_pulses = {}
                self.event_log = []
                self.sim_step = 0
                self.route_unit_label = "EMS"
                self._clear_route_animation()
                self.log_text.configure(state="normal")
                self.log_text.delete("1.0", "end")
                self.log_text.configure(state="disabled")
        except Exception as exc:
            self.lbl_c1_status.set(f"Error: {exc}")
            self.status_var.set(f"C1 failed: {exc}")

        self.btn_c1.configure(state="normal")
        if ok:
            counts = {lt: len(self.graph.get_nodes_of_type(lt)) for lt in LocationType}
            info = (
                f"RES:{counts[LocationType.RESIDENTIAL]}  "
                f"HOS:{counts[LocationType.HOSPITAL]}  "
                f"SCH:{counts[LocationType.SCHOOL]}\n"
                f"IND:{counts[LocationType.INDUSTRIAL]}  "
                f"PWR:{counts[LocationType.POWER_PLANT]}  "
                f"AMB:{counts[LocationType.AMBULANCE_DEPOT]}"
            )
            self.lbl_c1_status.set("Done")
            self.lbl_c1_nodes.set(info)
            self.lbl_c2_roads.set("—")
            self.lbl_c2_cost.set("—")
            self.lbl_c2_p1.set("—")
            self.lbl_c2_p2.set("—")
            self.lbl_c3_best.set("—")
            self.lbl_c3_worst.set("—")
            self.lbl_c4_status.set("—")
            self.lbl_c5_status.set("—")
            self.status_var.set("Challenge 1 complete. Ready for Challenge 2.")
            self._append_log(f"C1 complete: generated {rows}x{cols} layout (seed={seed}).")
            self._update_city_context()
        elif self.status_var.get().startswith("C1 failed:") is False:
            self.lbl_c1_status.set("Failed")
            self.status_var.set("CSP failed. Try a different seed.")
        self._redraw()

    def _clear_route_animation(self):
        if self.route_anim_job:
            self.after_cancel(self.route_anim_job)
            self.route_anim_job = None
        self.live_route_nodes = []
        self.route_anim_index = -1
        self.route_animation_queue = []

    def _start_route_animation(self, route_nodes, queue_if_busy=True):
        if not route_nodes:
            self._redraw()
            return
        if queue_if_busy and self.route_anim_job:
            self.route_animation_queue.append(list(route_nodes))
            return
        self.live_route_nodes = list(route_nodes)
        self.route_anim_index = 0
        self._redraw()
        self.route_anim_job = self.after(self._route_step_delay_ms(), self._advance_route_animation)

    def _play_next_queued_route(self):
        if self.route_animation_queue:
            nxt = self.route_animation_queue.pop(0)
            self._start_route_animation(nxt, queue_if_busy=False)
        else:
            self.live_route_nodes = []
            self.route_anim_index = -1

    def _restart_route_animation(self, route_nodes):
        if self.route_anim_job:
            self.after_cancel(self.route_anim_job)
            self.route_anim_job = None
        self.route_animation_queue = []
        self.live_route_nodes = list(route_nodes)
        self.route_anim_index = 0
        self._redraw()
        self.route_anim_job = self.after(self._route_step_delay_ms(), self._advance_route_animation)

    def _advance_route_animation(self):
        if not self.live_route_nodes:
            self.route_anim_job = None
            return
        if self.paused:
            self.route_anim_job = self.after(200, self._advance_route_animation)
            return
        if self.route_anim_index < len(self.live_route_nodes) - 1:
            self.route_anim_index += 1
            # If the C4 unit just reached a target civilian, turn off its pulse
            current_nid = self.live_route_nodes[self.route_anim_index]
            if current_nid in self.c4_pending_pulses and self.c4_pending_pulses[current_nid] == "white":
                self.c4_pending_pulses[current_nid] = None
            self._redraw()
            self.route_anim_job = self.after(self._route_step_delay_ms(), self._advance_route_animation)
        else:
            self.route_anim_job = None
            # Animation finished — now reveal red pulses for unreachable nodes
            if hasattr(self, '_c4_deferred_skipped') and self._c4_deferred_skipped:
                for nid in self._c4_deferred_skipped:
                    if nid in self.c4_pending_pulses:
                        self.c4_pending_pulses[nid] = "red"
                self._c4_deferred_skipped = []
                self._redraw()
                # Auto-clear red pulses after 4 seconds
                self.after(4000, self._clear_c4_red_pulses)
            self._play_next_queued_route()

    def _clear_c4_red_pulses(self):
        """Auto-clear any remaining red C4 pulses after timeout."""
        for nid in list(self.c4_pending_pulses):
            if self.c4_pending_pulses[nid] == "red":
                self.c4_pending_pulses[nid] = None
        self._redraw()

    def _run_c2(self):
        if self.running or self.graph is None:
            self.status_var.set("Run Challenge 1 first.")
            return
        self.status_var.set("Running Challenge 2: MST road optimization...")
        self.btn_c2.configure(state="disabled")
        self.update_idletasks()
        try:
            opt = RoadNetworkOptimizer(self.graph)
            ok = silent(opt.optimize)
            if ok:
                self.optimizer = opt
                self.lbl_c2_roads.set(str(len(opt.built_roads)))
                self.lbl_c2_cost.set(f"{opt.calculate_total_cost():.1f}")
                self.lbl_c2_p1.set(f"{len(opt.path1_edges)} edges")
                self.lbl_c2_p2.set(f"{len(opt.path2_edges)} edges")
                self.status_var.set("Challenge 2 complete. Roads glowing in neon.")
                self._append_log(
                    f"C2 complete: roads={len(opt.built_roads)} total_cost={opt.calculate_total_cost():.2f}"
                )
                self._update_city_context()
            else:
                self.status_var.set("Road optimization failed.")
                self._append_log("C2 failed: optimizer returned False.")
        except Exception as exc:
            self.status_var.set(f"C2 error: {exc}")
            self._append_log(f"C2 error: {exc}")
        self.btn_c2.configure(state="normal")
        self._redraw()

    def _run_c3(self):
        if self.running or self.graph is None:
            self.status_var.set("Run Challenges 1 and 2 first.")
            return
        self.running = True
        self.btn_c3.configure(state="disabled")
        self.prog_var.set(0)
        self._sa_hist = []
        self.chart.delete("all")
        self.status_var.set("Running Challenge 3: simulated annealing...")

        n_ambs = self._c3_ambs.get()
        t_init = self._c3_T.get()
        alpha = self._c3_cool.get() / 1000.0
        max_iter = self._c3_iter.get()

        def cb(it, temp, cost, best):
            self._sa_hist.append((it, cost, best))
            # Schedule widget updates on the main thread to avoid
            # tkinter threading issues (#28, #29).
            self.after(0, lambda: self.prog_var.set(min(1.0, it / max_iter)))
            self.after(0, lambda: self.lbl_c3_iter.set(str(it)))
            self.after(0, lambda: self.lbl_c3_temp.set(f"{temp:.3f}"))
            self.after(0, lambda: self.lbl_c3_curr.set(f"{cost:.3f}"))
            self.after(0, lambda: self.lbl_c3_best.set(f"{best:.3f}"))
            if it % 250 == 0:
                self.after(0, self._draw_chart)

        def run():
            sa = AmbulancePlacementSA(
                self.graph,
                n_ambulances=n_ambs,
                T_init=t_init,
                T_min=0.02,
                alpha=alpha,
                max_iter=max_iter,
                seed=42,
                callback=cb,
            )
            placement = sa.solve()
            self.solver = sa
            self.coverage = sa.coverage_breakdown()
            self.after(0, lambda: self._c3_done(placement))

        threading.Thread(target=run, daemon=True).start()

    def _c3_done(self, placement):
        self.running = False
        self.btn_c3.configure(state="normal")
        self.prog_var.set(1.0)
        self._draw_chart()

        worst = self.solver.best_cost
        finite = [v for v in self.coverage.values() if v < math.inf]
        avg = sum(finite) / max(1, len(finite))
        self.lbl_c3_worst.set(f"{worst:.3f}")
        self.lbl_c3_avg.set(f"{avg:.3f}")
        pos = ", ".join(f"({self.graph.nodes[a].row},{self.graph.nodes[a].col})" for a in placement)
        self.status_var.set(f"Challenge 3 done. Worst response={worst:.3f}")
        self._append_log(f"C3 complete: worst={worst:.3f} placement={pos}")
        self._update_city_context()
        self._redraw()

    def _run_c4_dispatch(self):
        if self.running or self.graph is None:
            self.status_var.set("Run Challenge 1 first.")
            return
        if not self.optimizer:
            self._run_c2()
        self.router = DynamicEmergencyRouter(self.graph)
        self.route_unit_label = "C4"
        self._show_banner("C4 DISPATCH DEMO LIVE")
        self._push_mission_feed("Dispatch team mobilized. Calculating safest civilian path.")
        start = choose_emergency_start_node(self.graph)
        residential = self.graph.get_nodes_of_type(LocationType.RESIDENTIAL)
        if not residential or len(residential) < 1:
            self.lbl_c4_status.set("No civilians")
            return
        # Exactly 5 civilian targets (or all if fewer than 5)
        targets = random.sample(residential, min(5, len(residential)))
        # Spawn white radar pulses on all target civilians (waiting for rescue)
        self.c4_pending_pulses = {nid: "white" for nid in targets}
        self._c4_deferred_skipped = []  # will be populated after routing
        self._redraw()
        # Pick flood edges dynamically from actual MST edges
        available_edges = []
        for u, nbrs in self.graph.adjacency.items():
            for v, _ in nbrs:
                if u < v:
                    available_edges.append((u, v))
        flood_edges = available_edges[:2] if len(available_edges) >= 2 else []
        result = self.router.route_to_nearest_civilians(
            start_id=start,
            civilian_ids=targets,
            flood_events={1: flood_edges} if flood_edges else {},
        )
        self.lbl_c4_status.set("Done" if result["success"] else "Failed")
        self.lbl_c4_cost.set(f"{result['total_cost']:.2f}")
        for msg in result["events"]:
            self._append_log("C4: " + msg)

        route_nodes = []
        for visit in result.get("visited", []):
            if not route_nodes:
                route_nodes.extend(visit["path"])
            else:
                route_nodes.extend(visit["path"][1:])
        self._add_route_history(route_nodes, "#adf7ff")
        # Don't clear rescued pulses here — _advance_route_animation
        # will turn them off as the C4 unit physically reaches each node.
        # Store skipped civilians for deferred red-pulse after animation ends.
        if result.get("skipped_civilians"):
            self._c4_deferred_skipped = list(result["skipped_civilians"])
            self._push_mission_feed(
                f"C4 skipped {len(result['skipped_civilians'])} unreachable civilians."
            )
        self._push_mission_feed(
            f"C4 completed: {len(result.get('visited', []))} rescues, cost {result['total_cost']:.2f}."
        )
        self._restart_route_animation(route_nodes)
        self.status_var.set("Challenge 4 dispatch demo complete.")
        self._update_city_context()

    def _run_c5_risk(self):
        if self.running or self.graph is None:
            self.status_var.set("Run Challenge 1 first.")
            return
        # C5 risk-adjusted edge costs only make sense AFTER C2 has pruned
        # the graph to MST roads.  Running C5 on the full grid wastes
        # computation and the costs get overwritten when C2 runs later.
        if not self.optimizer:
            self._append_log("WARNING: Running C5 before C2. Edge costs will be recalculated when C2 runs.")
            self.status_var.set("Running C5 (note: C2 not yet run — costs may change later).")
        self.risk_predictor = CrimeRiskPredictor(self.graph, seed=42)
        self._show_banner("C5 RISK ANALYSIS IN PROGRESS")
        self._push_mission_feed("Scanning districts for crime-risk hotspots.")
        self.risk_predictions = self.risk_predictor.train_and_predict()
        self.police_deployment = self.risk_predictor.allocate_police_officers(total_officers=10)
        summary = self.risk_predictor.summary()
        counts = summary["counts"]
        self.graph.set_risk_cost_weight(0.35)
        self.lbl_c5_status.set("Done")
        self.lbl_c5_mix.set(f"L:{counts['LOW']} M:{counts['MEDIUM']} H:{counts['HIGH']}")
        self.lbl_c5_police.set(f"{sum(self.police_deployment.values())} officers / {len(self.police_deployment)} posts")
        self._append_log(
            f"C5 complete: acc={summary['test_accuracy']:.2%}, "
            f"L/M/H={counts['LOW']}/{counts['MEDIUM']}/{counts['HIGH']}"
        )
        deploy_desc = []
        for nid, officers in sorted(self.police_deployment.items(), key=lambda x: x[1], reverse=True):
            node = self.graph.nodes[nid]
            deploy_desc.append(f"({node.row},{node.col}) x{officers}")
        if deploy_desc:
            self._append_log("C5 police deployment: " + ", ".join(deploy_desc))
        high_nodes = [nid for nid, pred in self.risk_predictions.items() if pred["label"] == "HIGH"][:18]
        med_nodes = [nid for nid, pred in self.risk_predictions.items() if pred["label"] == "MEDIUM"][:14]
        self._spawn_node_pulse(high_nodes, "#ff4d7f", strength=1.3)
        self._spawn_node_pulse(med_nodes, "#ffd166", strength=0.9)
        self._push_mission_feed(
            f"C5 heatmap ready: HIGH={counts['HIGH']} MEDIUM={counts['MEDIUM']} LOW={counts['LOW']}."
        )
        self.status_var.set("Challenge 5 risk model complete.")
        self._update_city_context()
        self._redraw()

    def _run_simulation(self):
        if self.running or self.graph is None:
            self.status_var.set("Run Challenge 1 first.")
            return
        self.running = True
        self.btn_sim.configure(state="disabled")
        self.status_var.set("Running integrated simulation (20 steps)...")
        self._append_log("Integrated simulation started.")
        self._show_banner("20-STEP CITY SIMULATION RUNNING", hold_ms=2500)
        self._push_mission_feed("Integrated simulation started. Monitoring city-wide response.")

        # Force C3 coverage heatmap (green-to-red) instead of C5 crime risk
        # so the viewer sees ambulance response distances update dynamically.
        self.show_risk.set(False)
        self.show_heat.set(True)

        def run():
            try:
                if not self.optimizer:
                    self.optimizer = RoadNetworkOptimizer(self.graph)
                    silent(self.optimizer.optimize)
                if not self.solver:
                    self.solver = AmbulancePlacementSA(self.graph, n_ambulances=3, seed=42)
                    self.solver.solve()
                    self.coverage = self.solver.coverage_breakdown()

                self.risk_predictor = CrimeRiskPredictor(self.graph, seed=42)
                self.risk_predictions = self.risk_predictor.train_and_predict()
                self.police_deployment = self.risk_predictor.allocate_police_officers(total_officers=10)
                self.graph.set_risk_cost_weight(0.35)
                self.router = DynamicEmergencyRouter(self.graph)

                start = choose_emergency_start_node(self.graph)
                residential = self.graph.get_nodes_of_type(LocationType.RESIDENTIAL)
                # NOTE: GUI uses its own Random(42) instance, separate from
                # SimulationEngine's self.rng.  Step results WILL differ
                # between terminal and GUI runs — this is expected because
                # they take different code paths (GUI draws, pauses, etc.).
                rng = random.Random(42)
                pending_civilians = set()

                # Track active floods so they expire after 3 steps,
                # matching the terminal simulation_engine behaviour.
                active_floods = []   # [(expire_step, [(u,v), ...])]

                for step in range(1, 21):
                    # --- Check for stop request ---
                    if self._stop_requested:
                        return
                    # --- Spin-wait while paused ---
                    while self.paused and not self._stop_requested:
                        time.sleep(0.15)
                    if self._stop_requested:
                        return

                    # --- Unblock expired floods ---
                    for expire_step, flood_edges in list(active_floods):
                        if step >= expire_step:
                            for u, v in flood_edges:
                                self.graph.unblock_road(u, v)
                            active_floods.remove((expire_step, flood_edges))

                    targets = rng.sample(residential, min(10, len(residential))) if residential else []
                    pending_civilians.update(targets)
                    flood_events = {}
                    if step % 4 == 0:
                        edges = []
                        for u, nbrs in self.graph.adjacency.items():
                            for v, _ in nbrs:
                                if u < v:
                                    edges.append((u, v))
                        rng.shuffle(edges)
                        if edges:
                            flood_events[0] = edges[:1]
                        if len(edges) > 1:
                            flood_events[1] = edges[1:2]
                        # Record floods with expiry
                        all_flood = edges[:2] if len(edges) >= 2 else edges[:1]
                        if all_flood:
                            active_floods.append((step + 3, all_flood))

                    result = self.router.route_to_nearest_civilians(
                        start_id=start,
                        civilian_ids=sorted(pending_civilians),
                        flood_events=flood_events,
                    )

                    route_nodes = []
                    for visit in result.get("visited", []):
                        if not route_nodes:
                            route_nodes.extend(visit["path"])
                        else:
                            route_nodes.extend(visit["path"][1:])

                    if result["success"] and result.get("visited"):
                        start = result["visited"][-1]["target"]
                        msg = (
                            f"Step {step:02d}: routed {len(result['visited'])} civilians, "
                            f"cost={result['total_cost']:.2f}"
                        )
                        pending_civilians = set(result.get("remaining_civilians", []))
                        if pending_civilians:
                            msg += f", pending={len(pending_civilians)}"
                    else:
                        msg = f"Step {step:02d}: routing failed, fallback triggered"
                        pending_civilians = set(result.get("remaining_civilians", []))
                    if step % 5 == 0:
                        self.solver = AmbulancePlacementSA(self.graph, n_ambulances=3, seed=step)
                        self.solver.solve()
                        self.coverage = self.solver.coverage_breakdown()
                        self.police_deployment = self.risk_predictor.allocate_police_officers(total_officers=10)
                        msg += f" | ambulance re-optimized | police posts={len(self.police_deployment)}"
                    self.sim_step = step
                    self.after(
                        0, lambda m=msg, s=step, rn=route_nodes: self._on_sim_step(m, s, rn)
                    )
                    # Short sleep — the fast 30ms animation handles pacing
                    time.sleep(0.1)
                self.after(0, self._on_sim_done)
            except Exception as exc:
                if not self._stop_requested:
                    self.after(0, lambda: self._on_sim_error(exc))

        threading.Thread(target=run, daemon=True).start()

    def _on_sim_step(self, message, step, route_nodes=None):
        self.lbl_sim_step.set(f"{step}/20")
        self._append_log(message)
        self._push_mission_feed(f"Step {step:02d}/20 -> {message}")
        if route_nodes:
            self._add_route_history(route_nodes, "#b9f7ff")
            # No radar pulses during sim — keep the display clean
        if step % 5 == 0:
            self._show_banner(f"SIM MILESTONE REACHED: STEP {step}", hold_ms=1300)
        self._update_city_context()
        if self.show_dynamic_route.get():
            self.route_unit_label = "AMB"
            self._start_route_animation(route_nodes or [])
        else:
            self._redraw()

    def _on_sim_done(self):
        self.running = False
        self.btn_sim.configure(state="normal")
        self.status_var.set("Integrated simulation completed successfully.")
        self._append_log("Integrated simulation completed.")
        self._show_banner("SIMULATION COMPLETE")
        self._push_mission_feed("Simulation complete. Review route quality and risk zones.")
        self._update_city_context()

    def _on_sim_error(self, exc):
        self.running = False
        self.btn_sim.configure(state="normal")
        self.status_var.set(f"Simulation failed: {exc}")
        self._append_log(f"ERROR: {exc}")

    def _draw_chart(self):
        hist = self._sa_hist
        c = self.chart
        c.delete("all")
        if len(hist) < 2:
            return
        w = c.winfo_width() or 640
        h = 92
        pl, pr, pt, pb = 32, 10, 6, 16
        currs = [x[1] for x in hist]
        ymax = max(currs) * 1.05 if currs else 1.0
        n = len(hist)

        def tx(i):
            return pl + (i / (n - 1)) * (w - pl - pr)

        def ty(v):
            return pt + (1 - (v / max(ymax, 1e-6))) * (h - pt - pb)

        for frac in (0.5, 1.0):
            y = ty(ymax * frac)
            c.create_line(pl, y, w - pr, y, fill=BORDER, dash=(3, 4))
            c.create_text(pl - 4, y, text=f"{ymax * frac:.1f}", fill=MUTED, anchor="e")

        curr_pts, best_pts = [], []
        for i, item in enumerate(hist):
            curr_pts += [tx(i), ty(item[1])]
            best_pts += [tx(i), ty(item[2])]
        c.create_line(*curr_pts, fill="#385578", width=1, smooth=True)
        c.create_line(*best_pts, fill=ACCENT, width=2, smooth=True)
        c.create_text(w - 12, 8, text="Best", fill=ACCENT, anchor="ne")

    def _grid_metrics(self):
        rows, cols = self.graph.rows, self.graph.cols
        w = max(200, self.canvas.winfo_width())
        h = max(200, self.canvas.winfo_height())
        pad = 14
        usable_w = max(100, w - 2 * pad)
        usable_h = max(100, h - 2 * pad)
        cell = max(28, int(min(usable_w / cols, usable_h / rows)))
        grid_w = cell * cols
        grid_h = cell * rows
        ox = (w - grid_w) // 2
        oy = (h - grid_h) // 2
        return cell, ox, oy

    def _cell_center(self, node, cell, ox, oy):
        return ox + node.col * cell + cell // 2, oy + node.row * cell + cell // 2

    def _draw_neon_line(self, x1, y1, x2, y2, color, width):
        self.canvas.create_line(x1, y1, x2, y2, fill="#051124", width=width + 5, capstyle="round")
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width + 2, capstyle="round")
        self.canvas.create_line(x1, y1, x2, y2, fill="#ffffff", width=max(1, width - 1), capstyle="round")

    def _redraw(self):
        if self.graph is None:
            return
        g = self.graph
        c = self.canvas
        c.delete("all")
        cell, ox, oy = self._grid_metrics()
        self._grid_draw_state = (cell, ox, oy)

        heat = {}
        if self.show_heat.get() and self.coverage:
            finite = [v for v in self.coverage.values() if v < math.inf]
            mx = max(finite) if finite else 1.0
            for nid, dist in self.coverage.items():
                ratio = min(dist / max(mx, 1e-6), 1.0) if dist < math.inf else 1.0
                r = int(255 * min(1.0, ratio * 1.2))
                gch = int(255 * min(1.0, (1.0 - ratio) * 1.2))
                heat[nid] = f"#{r:02x}{gch:02x}2a"
        if self.show_risk.get() and self.risk_predictions:
            for nid, pred in self.risk_predictions.items():
                if pred["label"] == "HIGH":
                    heat[nid] = "#951829"
                elif pred["label"] == "MEDIUM":
                    heat[nid] = "#8f6915"
                else:
                    heat[nid] = "#1a5c3c"

        text_size = max(8, int(cell * 0.18))
        pop_size = max(7, int(cell * 0.16))
        c.create_rectangle(
            ox - 3,
            oy - 3,
            ox + g.cols * cell + 3,
            oy + g.rows * cell + 3,
            outline="#5a76a3",
            width=2,
        )
        for nid, node in g.nodes.items():
            x0 = ox + node.col * cell
            y0 = oy + node.row * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            fill = heat.get(nid, TYPE_COLORS.get(node.location_type, CARD))
            c.create_rectangle(x0, y0, x1, y1, fill=fill, outline=BORDER, width=1)
            c.create_text(
                (x0 + x1) // 2,
                y0 + int(cell * 0.40),
                text=TYPE_SYM.get(node.location_type, "?"),
                fill="#edf5ff" if node.location_type != LocationType.EMPTY else "#6480aa",
                font=("Segoe UI", text_size, "bold"),
            )
            if self.show_pop.get() and node.population_density > 0:
                c.create_text(
                    (x0 + x1) // 2,
                    y0 + int(cell * 0.70),
                    text=str(node.population_density),
                    fill="#91b4e2",
                    font=("Segoe UI", pop_size),
                )

        for trace in self.route_history:
            pts = []
            for nid in trace["nodes"]:
                node = g.nodes.get(nid)
                if node is None:
                    continue
                pts.extend(self._cell_center(node, cell, ox, oy))
            if len(pts) >= 4:
                if trace["age"] < trace["ttl"] * 0.33:
                    width = max(2, int(cell * 0.09))
                elif trace["age"] < trace["ttl"] * 0.66:
                    width = max(2, int(cell * 0.07))
                else:
                    width = max(1, int(cell * 0.05))
                c.create_line(*pts, fill=trace["color"], width=width, dash=(6, 5))

        if self.hover_node_id is not None and self.hover_node_id in g.nodes:
            node = g.nodes[self.hover_node_id]
            hx0 = ox + node.col * cell
            hy0 = oy + node.row * cell
            hx1 = hx0 + cell - 1
            hy1 = hy0 + cell - 1
            c.create_rectangle(hx0, hy0, hx1, hy1, outline="#6ef0ff", width=2)

        if self.selected_node_id is not None and self.selected_node_id in g.nodes:
            node = g.nodes[self.selected_node_id]
            sx0 = ox + node.col * cell
            sy0 = oy + node.row * cell
            sx1 = sx0 + cell - 1
            sy1 = sy0 + cell - 1
            c.create_rectangle(sx0, sy0, sx1, sy1, outline="#ffffff", width=3)

        for fx in self.visual_effects:
            if fx["kind"] != "pulse":
                continue
            node = g.nodes.get(fx["nid"])
            if node is None:
                continue
            cx, cy = self._cell_center(node, cell, ox, oy)
            progress = fx["age"] / max(1, fx["ttl"] - 1)
            radius = int((cell * 0.24) + (cell * 0.62 * progress))
            width = 2 if progress < 0.6 else 1
            c.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline=fx["color"],
                width=width,
            )

        if self.show_roads.get():
            drawn = set()
            for u, nbrs in g.adjacency.items():
                for v, _ in nbrs:
                    edge = (min(u, v), max(u, v))
                    if edge in drawn:
                        continue
                    drawn.add(edge)
                    nu, nv = g.nodes[u], g.nodes[v]
                    x1, y1 = self._cell_center(nu, cell, ox, oy)
                    x2, y2 = self._cell_center(nv, cell, ox, oy)
                    c.create_line(x1, y1, x2, y2, fill="#2a4266", width=max(1, int(cell * 0.08)))

        if self.show_roads.get() and self.optimizer:
            for u, v in self.optimizer.built_roads:
                nu, nv = g.nodes[u], g.nodes[v]
                x1, y1 = self._cell_center(nu, cell, ox, oy)
                x2, y2 = self._cell_center(nv, cell, ox, oy)
                edge = (min(u, v), max(u, v))
                color, width = ACCENT, max(2, int(cell * 0.11))
                if edge in self.optimizer.path1_edges:
                    color, width = "#00b5ff", max(3, int(cell * 0.14))
                elif edge in self.optimizer.path2_edges:
                    color, width = "#ff4d7f", max(3, int(cell * 0.14))
                self._draw_neon_line(x1, y1, x2, y2, color=color, width=width)

        if self.solver and self.solver.best_placement:
            for idx, aid in enumerate(self.solver.best_placement):
                node = g.nodes[aid]
                cx, cy = self._cell_center(node, cell, ox, oy)
                color = AMB_COLORS[idx % len(AMB_COLORS)]
                r = max(9, int(cell * 0.24))
                c.create_oval(cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5, outline=color, width=2)
                c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="#ffffff", width=1)
                c.create_text(cx, cy, text=f"A{idx+1}", fill="#001018", font=("Segoe UI", max(8, int(cell * 0.20)), "bold"))

        if self.show_dynamic_route.get() and self.live_route_nodes:
            all_pts = []
            for nid in self.live_route_nodes:
                node = g.nodes[nid]
                all_pts.extend(self._cell_center(node, cell, ox, oy))
            i = min(max(self.route_anim_index, 0), len(self.live_route_nodes) - 1)
            completed_pts = []
            for nid in self.live_route_nodes[: i + 1]:
                node = g.nodes[nid]
                completed_pts.extend(self._cell_center(node, cell, ox, oy))
            upcoming_pts = []
            for nid in self.live_route_nodes[i:]:
                node = g.nodes[nid]
                upcoming_pts.extend(self._cell_center(node, cell, ox, oy))
            if len(upcoming_pts) >= 4:
                c.create_line(
                    *upcoming_pts,
                    fill="#d6faff",
                    width=max(2, int(cell * 0.09)),
                    dash=(8, 5),
                )
            if len(completed_pts) >= 4:
                self._draw_neon_line(
                    completed_pts[0],
                    completed_pts[1],
                    completed_pts[2],
                    completed_pts[3],
                    color="#00ffd0",
                    width=max(2, int(cell * 0.12)),
                )
                for p in range(2, len(completed_pts) - 2, 2):
                    self._draw_neon_line(
                        completed_pts[p],
                        completed_pts[p + 1],
                        completed_pts[p + 2],
                        completed_pts[p + 3],
                        color="#00ffd0",
                        width=max(2, int(cell * 0.12)),
                    )
            if len(all_pts) >= 4 and len(completed_pts) < 4:
                c.create_line(*all_pts, fill="#d6faff", width=max(2, int(cell * 0.09)), dash=(8, 5))
            marker = g.nodes[self.live_route_nodes[i]]
            mx, my = self._cell_center(marker, cell, ox, oy)
            mr = max(7, int(cell * 0.18))
            c.create_oval(mx - mr - 5, my - mr - 5, mx + mr + 5, my + mr + 5, outline="#00ffd0", width=2)
            c.create_oval(mx - mr, my - mr, mx + mr, my + mr, fill="#00e5ff", outline="#ffffff", width=2)
            c.create_text(
                mx,
                my,
                text=self.route_unit_label,
                fill="#02101f",
                font=("Segoe UI", max(7, int(cell * 0.15)), "bold"),
            )
        # --- C4 pending civilian pulses (white=waiting, red=unreachable) ---
        for nid, pulse_color in self.c4_pending_pulses.items():
            if pulse_color is None:
                continue  # rescued — no pulse
            node = g.nodes.get(nid)
            if node is None:
                continue
            cx, cy = self._cell_center(node, cell, ox, oy)
            fill = "#ffffff" if pulse_color == "white" else "#ff4d7f"
            # Animated expanding ring
            t = (time.time() * 3.0) % 1.0  # repeating 0→1
            radius = int(cell * 0.18 + cell * 0.45 * t)
            c.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                outline=fill, width=2,
            )
            # Inner dot
            c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=fill, outline="")

        # --- Police officer indicators (drawn when C5 risk is active) ---
        if self.police_deployment and self.show_risk.get():
            font_sz = max(7, int(cell * 0.15))
            for nid, officer_count in self.police_deployment.items():
                node = g.nodes.get(nid)
                if node is None:
                    continue
                x0 = ox + node.col * cell
                y0 = oy + node.row * cell
                label = f"P:{officer_count}"
                # Inset into top-right corner with padding from cell edge
                pad = max(3, int(cell * 0.06))
                tw = max(18, int(cell * 0.40))  # badge width
                th = max(12, int(cell * 0.22))  # badge height
                bx1 = x0 + cell - pad - tw
                by1 = y0 + pad
                bx2 = x0 + cell - pad
                by2 = y0 + pad + th
                # Rounded rectangle via overlapping shapes
                cr = max(3, th // 3)  # corner radius
                c.create_oval(bx1, by1, bx1 + cr * 2, by2, fill="#0a192f", outline="#00e5ff", width=1)
                c.create_oval(bx2 - cr * 2, by1, bx2, by2, fill="#0a192f", outline="#00e5ff", width=1)
                c.create_rectangle(bx1 + cr, by1, bx2 - cr, by2, fill="#0a192f", outline="", width=0)
                # Top/bottom border lines to close the shape
                c.create_line(bx1 + cr, by1, bx2 - cr, by1, fill="#00e5ff", width=1)
                c.create_line(bx1 + cr, by2, bx2 - cr, by2, fill="#00e5ff", width=1)
                # Badge text
                c.create_text(
                    (bx1 + bx2) // 2, (by1 + by2) // 2,
                    text=label, fill="#00e5ff",
                    font=("Segoe UI", font_sz, "bold"),
                    anchor="center",
                )

        for r in range(g.rows):
            c.create_text(ox - 10, oy + r * cell + cell // 2, text=str(r), fill=MUTED, font=("Segoe UI", 9))
        for col in range(g.cols):
            c.create_text(ox + col * cell + cell // 2, oy + g.rows * cell + 10, text=str(col), fill=MUTED, font=("Segoe UI", 9))


if __name__ == "__main__":
    app = CityMindGUI()
    app.mainloop()
