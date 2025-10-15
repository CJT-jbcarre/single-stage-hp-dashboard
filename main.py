import math
from typing import Dict, List, Tuple

import numpy as np

from CoolProp.CoolProp import PropsSI

from dash import Dash, Input, Output, State, dcc, html
import plotly.graph_objects as go


def compute_cycle(
    fluid: str,
    source_inlet_c: float,
    sink_outlet_c: float,
    evap_pinch_k: float,
    cond_pinch_k: float,
    superheat_k: float,
    subcool_k: float,
    eta_isentropic: float,
) -> Dict[str, Dict[str, float]]:
    """
    Compute a simple single-stage vapor-compression heat pump cycle with HX pinch.

    Definitions (heating mode):
    - Evaporating temperature Te = source_inlet_c - evap_pinch_k
    - Condensing temperature Tc = sink_outlet_c + cond_pinch_k
    - State points numbering (standard):
        1: Evaporator outlet / compressor inlet (superheated vapor at Pe)
        2s: After isentropic compression to Pc (reference)
        2: Compressor outlet with efficiency eta_isentropic
        3: Condenser outlet / expansion valve inlet (subcooled liquid at Pc)
        4: Expansion valve outlet (two-phase at Pe, isenthalpic from 3)
    Returns a dict of state properties: P [Pa], T [K], h [J/kg], s [J/kg-K], x (quality)
    """

    # Determine evaporating and condensing absolute temperatures
    te_k = float(source_inlet_c) + 273.15 - float(evap_pinch_k)
    tc_k = float(sink_outlet_c) + 273.15 + float(cond_pinch_k)

    # Saturation pressures at evaporating and condensing
    pe = PropsSI("P", "T", te_k, "Q", 1.0, fluid)
    pc = PropsSI("P", "T", tc_k, "Q", 0.0, fluid)

    # State 1: superheated vapor at Pe, T = Te + superheat
    t1 = te_k + float(superheat_k)
    h1 = PropsSI("H", "T", t1, "P", pe, fluid)
    s1 = PropsSI("S", "T", t1, "P", pe, fluid)

    # State 2s: isentropic to Pc
    t2s = PropsSI("T", "P", pc, "S", s1, fluid)
    h2s = PropsSI("H", "P", pc, "S", s1, fluid)

    # State 2: real compression with isentropic efficiency
    eta = float(eta_isentropic)
    h2 = h1 + (h2s - h1) / max(eta, 1e-6)
    t2 = PropsSI("T", "P", pc, "H", h2, fluid)
    s2 = PropsSI("S", "P", pc, "H", h2, fluid)

    # State 3: subcooled liquid at Pc, T = Tc - subcool
    t3 = tc_k - float(subcool_k)
    h3 = PropsSI("H", "T", t3, "P", pc, fluid)
    s3 = PropsSI("S", "T", t3, "P", pc, fluid)

    # State 4: throttling to Pe (isoenthalpic)
    h4 = h3
    t4 = PropsSI("T", "P", pe, "H", h4, fluid)
    s4 = PropsSI("S", "P", pe, "H", h4, fluid)
    # Quality at 4 (may be <0 or >1 numerically; clamp for display)
    try:
        x4 = PropsSI("Q", "P", pe, "H", h4, fluid)
    except Exception:
        x4 = float("nan")

    # Cycle energetics per kg
    w_comp = h2 - h1
    q_cond = h2 - h3
    q_evap = h1 - h4
    cop_heating = q_cond / max(w_comp, 1e-9)
    cop_cooling = q_evap / max(w_comp, 1e-9)

    points = {
        "1": {"P": pe, "T": t1, "h": h1, "s": s1},
        "2": {"P": pc, "T": t2, "h": h2, "s": s2},
        "3": {"P": pc, "T": t3, "h": h3, "s": s3},
        "4": {"P": pe, "T": t4, "h": h4, "s": s4, "x": x4},
        "meta": {
            "Pe": pe,
            "Pc": pc,
            "Te_K": te_k,
            "Tc_K": tc_k,
            "W_comp": w_comp,
            "Q_cond": q_cond,
            "Q_evap": q_evap,
            "COP_h": cop_heating,
            "COP_c": cop_cooling,
        },
    }
    return points


def saturation_dome(fluid: str, t_min_k: float, t_max_k: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays for saturation dome: pressure and saturated liquid/vapor enthalpies."""
    t_vals = np.linspace(t_min_k, t_max_k, n)
    p_vals = np.array([PropsSI("P", "T", t, "Q", 0.5, fluid) for t in t_vals])
    h_l = np.array([PropsSI("H", "T", t, "Q", 0.0, fluid) for t in t_vals])
    h_v = np.array([PropsSI("H", "T", t, "Q", 1.0, fluid) for t in t_vals])
    return p_vals, h_l, h_v


def isotherm_curve(fluid: str, temperature_k: float, p_min: float, p_max: float, n: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Return P-h points for an isotherm (single-phase regions only)."""
    pressures = np.geomspace(max(p_min, 1000.0), p_max, n)
    h_vals = []
    p_vals = []
    for p in pressures:
        try:
            h = PropsSI("H", "T", temperature_k, "P", p, fluid)
        except Exception:
            continue
        h_vals.append(h)
        p_vals.append(p)
    return np.array(p_vals), np.array(h_vals)


def quality_line(fluid: str, quality: float, t_min_k: float, t_max_k: float, n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Return P-h along a constant quality in two-phase region (0..1)."""
    t_vals = np.linspace(t_min_k, t_max_k, n)
    p_vals = []
    h_vals = []
    for t in t_vals:
        try:
            p = PropsSI("P", "T", t, "Q", quality, fluid)
            h = PropsSI("H", "T", t, "Q", quality, fluid)
        except Exception:
            continue
        p_vals.append(p)
        h_vals.append(h)
    return np.array(p_vals), np.array(h_vals)


def build_ph_figure(fluid: str, points: Dict[str, Dict[str, float]]) -> go.Figure:
    meta = points["meta"]
    pe = meta["Pe"]
    pc = meta["Pc"]
    te_k = meta["Te_K"]
    tc_k = meta["Tc_K"]

    # Determine a reasonable temperature span for the dome
    t_trip = max(PropsSI("Ttriple", fluid), te_k * 0.75)
    t_crit = PropsSI("Tcrit", fluid)
    t_max_dome = min(tc_k * 1.05, t_crit * 0.999)
    p_dome, h_l, h_v = saturation_dome(fluid, t_trip, t_max_dome)

    fig = go.Figure()

    # Saturation dome
    fig.add_trace(
        go.Scatter(
            x=h_l / 1000.0,
            y=p_dome / 1e5,
            mode="lines",
            name="Saturated Liquid",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h_v / 1000.0,
            y=p_dome / 1e5,
            mode="lines",
            name="Saturated Vapor",
            line=dict(color="#ff7f0e", width=2),
        )
    )

    # Quality lines (0.1 .. 0.9)
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p_q, h_q = quality_line(fluid, q, t_trip, t_max_dome)
        if len(p_q) > 0:
            fig.add_trace(
                go.Scatter(
                    x=h_q / 1000.0,
                    y=p_q / 1e5,
                    mode="lines",
                    name=f"x={q:.1f}",
                    line=dict(color="#cccccc", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Isotherms
    p_min = max(np.min(p_dome), pe * 0.6)
    p_max = min(max(pc * 1.4, np.max(p_dome)), PropsSI("pcrit", fluid) * 0.999)
    t_lines_c = [
        (te_k - 10) - 273.15,
        te_k - 273.15,
        (te_k + 10) - 273.15,
        (tc_k - 20) - 273.15,
        (tc_k - 10) - 273.15,
        tc_k - 273.15,
    ]
    for t_c in t_lines_c:
        t_k = t_c + 273.15
        p_iso, h_iso = isotherm_curve(fluid, t_k, p_min, p_max)
        if len(p_iso) > 0:
            fig.add_trace(
                go.Scatter(
                    x=h_iso / 1000.0,
                    y=p_iso / 1e5,
                    mode="lines",
                    name=f"T={t_c:.0f}°C",
                    line=dict(color="#999999", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Cycle line including saturation reference points along condenser and evaporator
    h_tc_v = PropsSI("H", "T", tc_k, "Q", 1.0, fluid)
    h_tc_l = PropsSI("H", "T", tc_k, "Q", 0.0, fluid)
    h_te_v = PropsSI("H", "T", te_k, "Q", 1.0, fluid)
    cycle_h = [
        points["1"]["h"],
        points["2"]["h"],
        h_tc_v,  # condenser saturation vapor
        h_tc_l,  # condenser saturation liquid
        points["3"]["h"],
        points["4"]["h"],
        h_te_v,  # evaporator saturation vapor
        points["1"]["h"],
    ]
    cycle_p = [
        points["1"]["P"],
        points["2"]["P"],
        pc,
        pc,
        points["3"]["P"],
        points["4"]["P"],
        pe,
        points["1"]["P"],
    ]
    fig.add_trace(
        go.Scatter(
            x=np.array(cycle_h) / 1000.0,
            y=np.array(cycle_p) / 1e5,
            mode="lines",
            name="Cycle",
            line=dict(color="#2ca02c", width=3),
            hoverinfo="skip",
        )
    )

    # Points with labels (including saturation points made part of the polyline)
    pts_h = [points[str(i)]["h"] for i in [1, 2, 3, 4]] + [h_tc_v, h_tc_l, h_te_v]
    pts_p = [points[str(i)]["P"] for i in [1, 2, 3, 4]] + [pc, pc, pe]
    fig.add_trace(
        go.Scatter(
            x=np.array(pts_h) / 1000.0,
            y=np.array(pts_p) / 1e5,
            mode="markers+text",
            name="Cycle points",
            text=["1", "2", "3", "4", "Tc Q=1", "Tc Q=0", "Te Q=1"],
            textposition="top center",
            textfont=dict(color="#1f2937", size=12),
            marker=dict(size=9, symbol="circle", color="#2ca02c"),
            showlegend=False,
        )
    )

    # Horizontal lines for Pe and Pc
    fig.add_hline(y=pe / 1e5, line=dict(color="#d62728", width=1, dash="dash"))
    fig.add_hline(y=pc / 1e5, line=dict(color="#9467bd", width=1, dash="dash"))

    # Remove separate saturation markers; included in cycle/points traces above

    # Axis bounds based on the cycle only
    cycle_p_bar = np.array([points[str(i)]["P"] for i in [1, 2, 3, 4]]) / 1e5
    cycle_h_kjkg = np.array([points[str(i)]["h"] for i in [1, 2, 3, 4]]) / 1000.0

    p_min_bar = max(1e-6, float(np.min(cycle_p_bar)))
    p_max_bar = max(1e-6, float(np.max(cycle_p_bar)))
    # 10% padding in log space (multiplicative in linear space)
    log_min = math.log10(p_min_bar)
    log_max = math.log10(p_max_bar)
    span_log = max(log_max - log_min, 1e-6)
    pad_log = 0.1 * span_log
    y_range = [log_min - pad_log, log_max + pad_log]

    h_min = float(np.min(cycle_h_kjkg))
    h_max = float(np.max(cycle_h_kjkg))
    h_span = max(h_max - h_min, 1e-6)
    x_min = h_min - 0.1 * h_span
    x_max = h_max + 0.1 * h_span

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Specific enthalpy h [kJ/kg]",
        yaxis_title="Pressure p [bar]",
        xaxis=dict(showgrid=True, zeroline=False, range=[x_min, x_max]),
        yaxis=dict(type="log", showgrid=True, zeroline=False, range=y_range),
        title=f"P-h Diagram with Cycle — {fluid}",
    )

    return fig


def saturation_dome_ts(fluid: str, t_min_k: float, t_max_k: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays for saturation dome on T-s: T and saturated liquid/vapor entropy."""
    t_vals = np.linspace(t_min_k, t_max_k, n)
    s_l = np.array([PropsSI("S", "T", t, "Q", 0.0, fluid) for t in t_vals])
    s_v = np.array([PropsSI("S", "T", t, "Q", 1.0, fluid) for t in t_vals])
    return t_vals, s_l, s_v


def isobar_curve_ts(fluid: str, pressure: float, t_min_k: float, t_max_k: float, n: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Return s-T points for an isobar (single-phase)."""
    t_vals = np.linspace(t_min_k, t_max_k, n)
    s_vals = []
    t_ok = []
    for t in t_vals:
        try:
            s = PropsSI("S", "T", t, "P", pressure, fluid)
        except Exception:
            continue
        s_vals.append(s)
        t_ok.append(t)
    return np.array(s_vals), np.array(t_ok)


def quality_line_ts(fluid: str, quality: float, t_min_k: float, t_max_k: float, n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Return s-T along constant quality."""
    t_vals = np.linspace(t_min_k, t_max_k, n)
    s_vals = []
    t_ok = []
    for t in t_vals:
        try:
            s = PropsSI("S", "T", t, "Q", quality, fluid)
        except Exception:
            continue
        s_vals.append(s)
        t_ok.append(t)
    return np.array(s_vals), np.array(t_ok)


def build_ts_figure(fluid: str, points: Dict[str, Dict[str, float]]) -> go.Figure:
    meta = points["meta"]
    pe = meta["Pe"]
    pc = meta["Pc"]
    te_k = meta["Te_K"]
    tc_k = meta["Tc_K"]

    t_trip = max(PropsSI("Ttriple", fluid), te_k * 0.75)
    t_crit = PropsSI("Tcrit", fluid)
    t_max = min(tc_k * 1.05, t_crit * 0.999)
    t_vals, s_l, s_v = saturation_dome_ts(fluid, t_trip, t_max)

    fig = go.Figure()

    # Saturation dome
    fig.add_trace(
        go.Scatter(
            x=s_l / 1000.0,
            y=t_vals - 273.15,
            mode="lines",
            name="Saturated Liquid",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s_v / 1000.0,
            y=t_vals - 273.15,
            mode="lines",
            name="Saturated Vapor",
            line=dict(color="#ff7f0e", width=2),
        )
    )

    # Quality lines
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s_q, t_q = quality_line_ts(fluid, q, t_trip, t_max)
        if len(s_q) > 0:
            fig.add_trace(
                go.Scatter(
                    x=s_q / 1000.0,
                    y=t_q - 273.15,
                    mode="lines",
                    name=f"x={q:.1f}",
                    line=dict(color="#cccccc", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Isobars
    p_lines = [pe * 0.9, pe, pc * 0.5, pc, pc * 1.1]
    for p in p_lines:
        s_iso, t_iso = isobar_curve_ts(fluid, p, t_trip, t_max)
        if len(s_iso) > 0:
            fig.add_trace(
                go.Scatter(
                    x=s_iso / 1000.0,
                    y=t_iso - 273.15,
                    mode="lines",
                    name=f"p={p/1e5:.1f} bar",
                    line=dict(color="#999999", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Cycle polyline including saturation reference points (Tc Q=1, Tc Q=0, Te Q=1)
    s_tc_v = PropsSI("S", "T", tc_k, "Q", 1.0, fluid)
    s_tc_l = PropsSI("S", "T", tc_k, "Q", 0.0, fluid)
    s_te_v = PropsSI("S", "T", te_k, "Q", 1.0, fluid)
    cycle_s = [
        points["1"]["s"],
        points["2"]["s"],
        s_tc_v,
        s_tc_l,
        points["3"]["s"],
        points["4"]["s"],
        s_te_v,
        points["1"]["s"],
    ]
    cycle_t = [
        points["1"]["T"],
        points["2"]["T"],
        tc_k,
        tc_k,
        points["3"]["T"],
        points["4"]["T"],
        te_k,
        points["1"]["T"],
    ]
    fig.add_trace(
        go.Scatter(
            x=np.array(cycle_s) / 1000.0,
            y=np.array(cycle_t) - 273.15,
            mode="lines",
            name="Cycle",
            line=dict(color="#2ca02c", width=3),
            hoverinfo="skip",
        )
    )
    pts_s = [points[str(i)]["s"] for i in [1, 2, 3, 4]] + [s_tc_v, s_tc_l, s_te_v]
    pts_t = [points[str(i)]["T"] for i in [1, 2, 3, 4]] + [tc_k, tc_k, te_k]
    fig.add_trace(
        go.Scatter(
            x=np.array(pts_s) / 1000.0,
            y=np.array(pts_t) - 273.15,
            mode="markers+text",
            name="Cycle points",
            text=["1", "2", "3", "4", "Tc Q=1", "Tc Q=0", "Te Q=1"],
            textposition="top center",
            textfont=dict(color="#1f2937", size=12),
            marker=dict(size=9, symbol="circle", color="#2ca02c"),
            showlegend=False,
        )
    )

    # Remove separate saturation markers; included in cycle/points traces above

    # Axis bounds from cycle only
    cycle_s_kjkgk = np.array([points[str(i)]["s"] for i in [1, 2, 3, 4]]) / 1000.0
    cycle_t_c = np.array([points[str(i)]["T"] for i in [1, 2, 3, 4]]) - 273.15
    s_min = float(np.min(cycle_s_kjkgk)); s_max = float(np.max(cycle_s_kjkgk))
    t_min = float(np.min(cycle_t_c)); t_max_c = float(np.max(cycle_t_c))
    s_span = max(s_max - s_min, 1e-6); t_span = max(t_max_c - t_min, 1e-6)
    x_min = s_min - 0.1 * s_span; x_max = s_max + 0.1 * s_span
    y_min = t_min - 0.1 * t_span; y_max = t_max_c + 0.1 * t_span

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Specific entropy s [kJ/kg·K]",
        yaxis_title="Temperature T [°C]",
        xaxis=dict(showgrid=True, zeroline=False, range=[x_min, x_max]),
        yaxis=dict(showgrid=True, zeroline=False, range=[y_min, y_max]),
        title=f"T-s Diagram with Cycle — {fluid}",
    )

    return fig

def hx_temperature_profiles(
    points: Dict[str, Dict[str, float]],
    sink_outlet_c: float,
    source_inlet_c: float,
    evap_pinch_k: float,
    cond_pinch_k: float,
    flow_mode: str = "counter",
    fluid: str = "R134a",
) -> Tuple[go.Figure, go.Figure]:
    """Return temperature profile figures for condenser and evaporator.

    This uses simplified piecewise-linear refrigerant profiles and linear secondary-fluid
    profiles satisfying the selected pinch constraints. The x-axis is heat transferred
    per unit mass [kJ/kg].
    """
    t2_c = points["2"]["T"] - 273.15
    t3_c = points["3"]["T"] - 273.15
    t1_c = points["1"]["T"] - 273.15
    t4_c = points["4"]["T"] - 273.15
    tc_c = points["meta"]["Tc_K"] - 273.15
    te_c = points["meta"]["Te_K"] - 273.15

    # Heat duty axis (kJ/kg) constructed from enthalpy changes including saturation points
    h2 = points["2"]["h"]; h3 = points["3"]["h"]; h1 = points["1"]["h"]; h4 = points["4"]["h"]
    h_tc_v = PropsSI("H", "T", points["meta"]["Tc_K"], "Q", 1.0, fluid)
    h_tc_l = PropsSI("H", "T", points["meta"]["Tc_K"], "Q", 0.0, fluid)
    h_te_v = PropsSI("H", "T", points["meta"]["Te_K"], "Q", 1.0, fluid)

    # Condenser: q from 0 to (h2-h3)
    q_cond_total = (h2 - h3) / 1000.0
    q_desh = max(h2 - h_tc_v, 0.0) / 1000.0
    q_lat = max(h_tc_v - h_tc_l, 0.0) / 1000.0
    q_sub = max(h_tc_l - h3, 0.0) / 1000.0
    q_cond_knots = np.array([0.0, q_desh, q_desh + q_lat, q_cond_total])
    t_cond_knots = np.array([t2_c, tc_c, tc_c, t3_c])
    q_cond = np.linspace(0.0, q_cond_total, 200)
    t_ref_cond = np.interp(q_cond, q_cond_knots, t_cond_knots)

    # Evaporator: q from 0 to (h1-h4)
    q_evap_total = (h1 - h4) / 1000.0
    q_boil = max(h_te_v - h4, 0.0) / 1000.0
    q_super = max(h1 - h_te_v, 0.0) / 1000.0
    q_evap_knots = np.array([0.0, q_boil, q_evap_total])
    t_evap_knots = np.array([te_c, te_c, t1_c])
    q_evap = np.linspace(0.0, q_evap_total, 200)
    t_ref_evap = np.interp(q_evap, q_evap_knots, t_evap_knots)

    # Secondary-side temperatures vs heat (assumed linear)
    if flow_mode == "counter":
        sink_inlet = t3_c - cond_pinch_k
        sink_curve = np.linspace(sink_inlet, sink_outlet_c, len(q_cond))
        # Same-direction temperatures (aligned with refrigerant q increasing)
        sink_aligned_same = sink_curve[::-1]
        q_sink_plot = (q_cond_total - q_cond)  # plot from right to left for sink

        source_outlet = t1_c + evap_pinch_k
        source_curve = np.linspace(source_inlet_c, source_outlet, len(q_evap))
        source_aligned_same = source_curve[::-1]
        q_source_plot = (q_evap_total - q_evap)  # plot from right to left for source
    else:
        sink_inlet = min(sink_outlet_c - 0.1, t3_c - cond_pinch_k)
        sink_curve = np.linspace(sink_inlet, sink_outlet_c, len(q_cond))
        sink_aligned_same = sink_curve
        q_sink_plot = q_cond

        source_outlet = max(source_inlet_c + 0.1, t1_c + evap_pinch_k)
        source_curve = np.linspace(source_inlet_c, source_outlet, len(q_evap))
        source_aligned_same = source_curve
        q_source_plot = q_evap

    # Enforce pinch at correct end and set plotting direction
    if flow_mode == "counter":
        # Condenser cold end (q → total)
        sink_aligned = sink_aligned_same
        sink_shift = float(cond_pinch_k) - float(t_ref_cond[-1] - sink_aligned[-1])
        sink_aligned = sink_aligned + sink_shift
        sink_plot_vals = sink_aligned
        q_sink_plot = q_cond[::-1]

        # Evaporator hot end (q → total)
        source_aligned = source_aligned_same
        source_shift = float(evap_pinch_k) - float(source_aligned[-1] - t_ref_evap[-1])
        source_aligned = source_aligned + source_shift
        source_plot_vals = source_aligned
        q_source_plot = q_evap[::-1]
    else:
        # Co-current: both left→right
        sink_aligned = sink_aligned_same
        sink_shift = float(cond_pinch_k) - float(t_ref_cond[-1] - sink_aligned[-1])
        sink_aligned = sink_aligned + sink_shift
        sink_plot_vals = sink_aligned
        q_sink_plot = q_cond

        source_aligned = source_aligned_same
        source_shift = float(evap_pinch_k) - float(source_aligned[-1] - t_ref_evap[-1])
        source_aligned = source_aligned + source_shift
        source_plot_vals = source_aligned
        q_source_plot = q_evap

    # Build figures
    fig_cond = go.Figure()
    fig_cond.add_trace(go.Scatter(x=q_cond, y=t_ref_cond, mode="lines", name="Refrigerant", line=dict(color="#0ea5e9", width=3)))
    fig_cond.add_trace(
        go.Scatter(
            x=np.array([q_desh, q_desh + q_lat]),
            y=np.array([tc_c, tc_c]),
            mode="markers+text",
            text=["Tc (start)", "Tc (end)"],
            textposition="bottom center",
            name="Saturation (Tc)",
            marker=dict(symbol="diamond", size=9, color="#6b21a8"),
            showlegend=False,
        )
    )
    fig_cond.add_trace(go.Scatter(x=q_sink_plot, y=sink_plot_vals, mode="lines", name="Sink fluid", line=dict(color="#10b981", width=3)))
    fig_cond.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Heat transferred q [kJ/kg]",
        yaxis_title="Temperature [°C]",
        title=f"Condenser temperature profile ({flow_mode}-flow)",
    )

    fig_evap = go.Figure()
    fig_evap.add_trace(go.Scatter(x=q_evap, y=t_ref_evap, mode="lines", name="Refrigerant", line=dict(color="#0ea5e9", width=3)))
    fig_evap.add_trace(
        go.Scatter(
            x=np.array([q_boil]),
            y=np.array([te_c]),
            mode="markers+text",
            text=["Te"],
            textposition="bottom center",
            name="Saturation (Te)",
            marker=dict(symbol="diamond", size=9, color="#6b21a8"),
            showlegend=False,
        )
    )
    fig_evap.add_trace(go.Scatter(x=q_source_plot, y=source_plot_vals, mode="lines", name="Source fluid", line=dict(color="#f59e0b", width=3)))
    fig_evap.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Heat transferred q [kJ/kg]",
        yaxis_title="Temperature [°C]",
        title=f"Evaporator temperature profile ({flow_mode}-flow)",
    )

    return fig_cond, fig_evap


def make_app() -> Dash:
    app = Dash(__name__)
    app.title = "Heat Pump Cycle — P-h Diagram"

    fluids = ["R134a", "R410A", "R1234ze", "1234yf"]

    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Refrigerant", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="fluid",
                        options=[{"label": f, "value": f} for f in fluids],
                        value="R134a",
                        clearable=False,
                    ),
                ],
                style={"flex": "1", "minWidth": 240, "paddingRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Source inlet (evaporator) °C", style={"fontWeight": "600"}),
                    dcc.Slider(id="source_inlet", min=-20, max=30, step=1, value=5,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="source_inlet_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 260, "paddingRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Sink outlet (condenser) °C", style={"fontWeight": "600"}),
                    dcc.Slider(id="sink_outlet", min=25, max=65, step=1, value=45,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="sink_outlet_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 260, "paddingRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Evaporator pinch ΔT [K]", style={"fontWeight": "600"}),
                    dcc.Slider(id="evap_pinch", min=2, max=10, step=0.5, value=5,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="evap_pinch_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 240, "paddingRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Condenser pinch ΔT [K]", style={"fontWeight": "600"}),
                    dcc.Slider(id="cond_pinch", min=2, max=15, step=0.5, value=7,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="cond_pinch_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 240},
            ),
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "10px",
            "alignItems": "flex-end",
        },
    )

    refinements = html.Div(
        [
            html.Div(
                [
                    html.Label("Superheat [K]", style={"fontWeight": "600"}),
                    dcc.Slider(id="superheat", min=0, max=20, step=0.5, value=5,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="superheat_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 240, "paddingRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Subcool [K]", style={"fontWeight": "600"}),
                    dcc.Slider(id="subcool", min=0, max=20, step=0.5, value=3,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="subcool_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 240, "paddingRight": "10px"},
            ),
            html.Div(
                [
                    html.Label("Compressor isentropic efficiency", style={"fontWeight": "600"}),
                    dcc.Slider(id="eta_isen", min=0.5, max=0.9, step=0.01, value=0.75,
                               tooltip={"always_visible": False}, marks=None),
                    html.Div(id="eta_isen_val", style={"textAlign": "right", "color": "#555"}),
                ],
                style={"flex": "1", "minWidth": 280},
            ),
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "10px",
            "alignItems": "flex-end",
            "marginTop": "10px",
        },
    )

    kpis = html.Div(
        [
            html.Div(id="kpi_text", style={"fontSize": "16px", "lineHeight": 1.5}),
        ],
        style={
            "background": "#f7f7f9",
            "padding": "10px 12px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "8px",
            "marginTop": "10px",
        },
    )

    app.layout = html.Div(
        [
            html.H2("Interactive Heat Pump Cycle", style={"marginBottom": 0}),
            html.Div("Single-stage cycle", style={"color": "#666", "marginBottom": "12px"}),
            controls,
            refinements,
            html.Div(
                [
                    html.Label("HX flow arrangement", style={"fontWeight": "600", "marginRight": "10px"}),
                    dcc.RadioItems(
                        id="flow_mode",
                        options=[
                            {"label": "Counter-flow", "value": "counter"},
                            {"label": "Co-current", "value": "cocurrent"},
                        ],
                        value="counter",
                        inline=True,
                    ),
                ],
                style={"marginTop": "8px"},
            ),
            kpis,
            dcc.Graph(id="ph_plot", style={"height": "48vh", "marginTop": "10px"}),
            dcc.Graph(id="ts_plot", style={"height": "36vh", "marginTop": "6px"}),
            html.Div(
                [
                    dcc.Graph(id="cond_plot", style={"height": "36vh", "flex": "1", "minWidth": "360px"}),
                    dcc.Graph(id="evap_plot", style={"height": "36vh", "flex": "1", "minWidth": "360px"}),
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "10px"},
            ),
            dcc.Store(id="cycle_store"),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "14px"},
    )

    # Value mirrors for sliders
    @app.callback(
        Output("source_inlet_val", "children"),
        Output("sink_outlet_val", "children"),
        Output("evap_pinch_val", "children"),
        Output("cond_pinch_val", "children"),
        Output("superheat_val", "children"),
        Output("subcool_val", "children"),
        Output("eta_isen_val", "children"),
        Input("source_inlet", "value"),
        Input("sink_outlet", "value"),
        Input("evap_pinch", "value"),
        Input("cond_pinch", "value"),
        Input("superheat", "value"),
        Input("subcool", "value"),
        Input("eta_isen", "value"),
    )
    def mirror_values(src, sink, evap_pinch, cond_pinch, superheat, subcool, eta):  # noqa: D401
        return (
            f"{src:.0f} °C",
            f"{sink:.0f} °C",
            f"{evap_pinch:.1f} K",
            f"{cond_pinch:.1f} K",
            f"{superheat:.1f} K",
            f"{subcool:.1f} K",
            f"{eta:.2f} (–)",
        )

    @app.callback(
        Output("cycle_store", "data"),
        Input("fluid", "value"),
        Input("source_inlet", "value"),
        Input("sink_outlet", "value"),
        Input("evap_pinch", "value"),
        Input("cond_pinch", "value"),
        Input("superheat", "value"),
        Input("subcool", "value"),
        Input("eta_isen", "value"),
        prevent_initial_call=False,
    )
    def update_cycle(fluid, src, sink, evap_pinch, cond_pinch, superheat, subcool, eta):
        try:
            pts = compute_cycle(
                fluid=fluid,
                source_inlet_c=src,
                sink_outlet_c=sink,
                evap_pinch_k=evap_pinch,
                cond_pinch_k=cond_pinch,
                superheat_k=superheat,
                subcool_k=subcool,
                eta_isentropic=eta,
            )
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}
        return pts

    @app.callback(
        Output("ph_plot", "figure"),
        Output("kpi_text", "children"),
        Output("cond_plot", "figure"),
        Output("evap_plot", "figure"),
        Output("ts_plot", "figure"),
        Input("cycle_store", "data"),
        Input("flow_mode", "value"),
        State("fluid", "value"),
        State("sink_outlet", "value"),
        State("source_inlet", "value"),
        State("evap_pinch", "value"),
        State("cond_pinch", "value"),
        prevent_initial_call=False,
    )
    def update_plot(cycle_store, flow_mode, fluid, sink_outlet, source_inlet, evap_pinch, cond_pinch):
        if not cycle_store or "error" in cycle_store:
            fig = go.Figure()
            fig.update_layout(template="plotly_white")
            msg = cycle_store.get("error", "Waiting for inputs...") if isinstance(cycle_store, dict) else "Waiting for inputs..."
            empty = go.Figure(); empty.update_layout(template="plotly_white")
            return fig, html.Div(msg, style={"color": "#b91c1c"}), empty, empty, empty

        fig = build_ph_figure(fluid, cycle_store)
        meta = cycle_store["meta"]
        te = meta["Te_K"] - 273.15
        tc = meta["Tc_K"] - 273.15
        kpi = html.Div(
            [
                html.Span(f"Te={te:.1f}°C, Tc={tc:.1f}°C", style={"marginRight": "12px"}),
                html.Span(f"COP_h={meta['COP_h']:.2f}", style={"marginRight": "12px", "color": "#065f46"}),
                html.Span(f"COP_c={meta['COP_c']:.2f}", style={"marginRight": "12px", "color": "#1d4ed8"}),
                html.Span(f"W={meta['W_comp']/1000:.1f} kJ/kg", style={"marginRight": "12px"}),
                html.Span(f"Qcond={meta['Q_cond']/1000:.1f} kJ/kg", style={"marginRight": "12px"}),
                html.Span(f"Qevap={meta['Q_evap']/1000:.1f} kJ/kg"),
            ]
        )
        cond_fig, evap_fig = hx_temperature_profiles(
            points=cycle_store,
            sink_outlet_c=sink_outlet,
            source_inlet_c=source_inlet,
            evap_pinch_k=evap_pinch,
            cond_pinch_k=cond_pinch,
            flow_mode=flow_mode,
            fluid=fluid,
        )
        ts_fig = build_ts_figure(fluid, cycle_store)
        return fig, kpi, cond_fig, evap_fig, ts_fig

    return app


def main():
    app = make_app()
    app.run(host="0.0.0.0", port=8050, debug=True)


if __name__ == "__main__":
    main()
