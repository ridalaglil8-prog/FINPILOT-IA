"""Plotly chart builders (consistent dark fintech styling)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = [
    "#00D1FF", "#3B82F6", "#8B5CF6", "#22C55E", "#F59E0B",
    "#EF4444", "#14B8A6", "#EC4899", "#64748B", "#A3E635",
]
GRID_COLOR = "rgba(255,255,255,0.07)"
PAPER_COLOR = "rgba(0,0,0,0)"
FONT_COLOR = "#E6EDF7"


def base_layout(y_title: str = "", x_title: str = "", height: int = 420) -> Dict[str, Any]:
    """Shared dark layout. Deliberately does **not** set ``title``: callers
    pass their own ``title=dict(text=...)`` and a duplicate key would raise."""
    return dict(
        height=height,
        margin=dict(l=40, r=20, t=45, b=40),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=PAPER_COLOR,
        font=dict(color=FONT_COLOR, size=13),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, title=x_title),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )


def with_title(title: str) -> Dict[str, Any]:
    """Layout title dict matching :data:`FONT_COLOR` / :data:`base_layout`."""
    return dict(text=title, font=dict(size=15, color=FONT_COLOR))


def allocation_pie(rows: Sequence[Mapping[str, Any]], title: str,
                   height: int = 400, value_key: str = "value") -> go.Figure:
    labels = [str(r.get("ticker", "")) for r in rows]
    values = [float(r.get(value_key, 0.0)) for r in rows]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
                    line=dict(color="#0A1428", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=PAPER_COLOR, font=dict(color=FONT_COLOR),
        title=dict(text=title, font=dict(size=15, color=FONT_COLOR)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        showlegend=True,
    )
    return fig


def allocation_bars(tickers: Sequence[str], weights_pct: Sequence[float],
                    title: str, horizontal: bool = True,
                    color: str = PALETTE[0], height: int = 380) -> go.Figure:
    values = [float(v) for v in weights_pct]
    x, y = (values, list(tickers)) if horizontal else (list(tickers), values)
    orientation = "h" if horizontal else "v"
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation=orientation, marker_color=color,
        text=[f"{v:.1f} %" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f} %<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=height, x_title="%" if horizontal else ""),
                      title=with_title(title))
    fig.update_yaxes(autorange="reversed", title="")
    if not horizontal:
        fig.update_yaxes(title="")
        # headroom so the "outside" label above the tallest bar is not clipped
        top = max(values) if values else 100.0
        fig.update_yaxes(range=[min(min(values), 0.0) * 1.15, top * 1.15])
    else:
        # same headroom for the label to the right of the longest bar
        fig.update_xaxes(range=[min(min(values), 0.0) * 1.15, max(values) * 1.20 if values else 100.0])
    return fig


def wealth_chart(
    wealth: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "",
    y_title: str = "",
    height: int = 440,
    as_pct_from_start: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if as_pct_from_start:
        series = (wealth / float(wealth.iloc[0]) - 1.0) * 100.0
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                 name="portfolio", line=dict(color=PALETTE[0], width=2.2),
                                 fill="tozeroy", fillcolor="rgba(0,209,255,0.10)"))
        if benchmark is not None and len(benchmark):
            b = (benchmark / float(benchmark.iloc[0]) - 1.0) * 100.0
            fig.add_trace(go.Scatter(x=b.index, y=b.values, mode="lines",
                                     name="benchmark", line=dict(color=PALETTE[4], width=1.8,
                                                                 dash="dot")))
        y_title = y_title or "%"
    else:
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, mode="lines",
                                 name="portfolio", line=dict(color=PALETTE[0], width=2.2),
                                 fill="tozeroy", fillcolor="rgba(0,209,255,0.10)"))
        if benchmark is not None and len(benchmark):
            fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, mode="lines",
                                     name="benchmark", line=dict(color=PALETTE[4], width=1.8,
                                                                 dash="dot")))
        y_title = y_title or ""
    fig.update_layout(**base_layout(y_title=y_title, height=height),
                      title=with_title(title))
    return fig


def benchmark_relative_chart(portfolio: pd.Series, benchmark: pd.Series,
                             title: str, height: int = 420) -> go.Figure:
    """Both series rebased to 100 on their first common date."""
    common = portfolio.index.intersection(benchmark.index)
    fig = go.Figure()
    if len(common) < 2:
        fig.update_layout(**base_layout(height=height), title=with_title(title))
        fig.add_annotation(text="insufficient overlapping data", showarrow=False,
                           font=dict(color=FONT_COLOR))
        return fig
    p = portfolio.loc[common]
    b = benchmark.loc[common]
    fig.add_trace(go.Scatter(x=p.index, y=(p / float(p.iloc[0])) * 100, mode="lines",
                             name="portfolio", line=dict(color=PALETTE[0], width=2.2)))
    fig.add_trace(go.Scatter(x=b.index, y=(b / float(b.iloc[0])) * 100, mode="lines",
                             name="benchmark", line=dict(color=PALETTE[4], width=1.8, dash="dot")))
    fig.add_hline(y=100, line_color=GRID_COLOR)
    fig.update_layout(**base_layout(y_title="index (base 100)", height=height),
                      title=with_title(title))
    return fig


def drawdown_chart(drawdowns: pd.Series, title: str, height: int = 340,
                   color: str = "#EF4444") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=drawdowns.index, y=(drawdowns * 100.0).values, mode="lines",
        name="drawdown", fill="tozeroy",
        line=dict(color=color, width=1.6),
        fillcolor="rgba(239,68,68,0.18)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} %<extra></extra>",
    ))
    fig.update_layout(**base_layout(y_title="%", height=height), title=with_title(title))
    return fig


def correlation_heatmap(labels: Sequence[str], matrix: Sequence[Sequence[float]],
                        title: str, height: int = 460) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix, x=list(labels), y=list(labels),
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="ρ", thickness=12),
        hovertemplate="%{y} / %{x}<br>ρ = %{z:.3f}<extra></extra>",
        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix],
        texttemplate="%{text}",
    ))
    fig.update_layout(**base_layout(height=height), title=with_title(title))
    fig.update_xaxes(title="", autorange=None)
    fig.update_yaxes(title="", autorange="reversed")
    return fig


def comparison_bars(frame: pd.DataFrame, title: str, height: int = 420) -> go.Figure:
    """Grouped bars: current vs optimised allocation (%)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="current", x=frame["ticker"], y=frame["current"],
                         marker_color=PALETTE[3],
                         hovertemplate="%{x}<br>%{y:.2f} %<extra>current</extra>"))
    fig.add_trace(go.Bar(name="optimized", x=frame["ticker"], y=frame["optimized"],
                         marker_color=PALETTE[0],
                         hovertemplate="%{x}<br>%{y:.2f} %<extra>optimized</extra>"))
    fig.update_layout(**base_layout(y_title="%", height=height), title=with_title(title),
                      barmode="group")
    return fig


def stress_chart(rows: Sequence[Mapping[str, Any]], title: str, height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="impact", x=[r["ticker"] for r in rows],
        y=[float(r["impact"]) for r in rows],
        marker_color=[PALETTE[3] if float(r["impact"]) >= 0 else PALETTE[5] for r in rows],
        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=height), title=with_title(title))
    return fig


def histogram(returns: pd.Series, title: str, height: int = 360, nbins: int = 40) -> go.Figure:
    values = returns.dropna().to_numpy(dtype=float)
    fig = go.Figure(go.Histogram(
        x=values * 100.0, nbinsx=nbins, marker_color=PALETTE[0],
        hovertemplate="%{x:.2f} %<br>%{y} days<extra></extra>",
    ))
    fig.add_vline(x=float(np.mean(values) * 100.0) if len(values) else 0,
                  line_dash="dash", line_color=PALETTE[4])
    fig.update_layout(**base_layout(y_title="days", x_title="%", height=height),
                      title=with_title(title), bargap=0.05)
    return fig


def rolling_vol_chart(rolling: pd.Series, title: str, height: int = 360) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rolling.index, y=(rolling * 100).values, mode="lines",
                               name="rolling vol", line=dict(color=PALETTE[2], width=2)))
    fig.update_layout(**base_layout(y_title="%", height=height), title=with_title(title))
    return fig


def bar_compare(labels: Sequence[str], values: Sequence[float], title: str,
                height: int = 360, y_title: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(labels), y=[float(v) for v in values],
        marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
        text=[f"{float(v):,.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(**base_layout(y_title=y_title, height=height), title=with_title(title))
    return fig
