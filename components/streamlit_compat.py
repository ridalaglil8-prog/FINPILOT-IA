"""Version-tolerant wrappers around Streamlit element width APIs.

Streamlit replaced ``use_container_width=True`` with ``width="stretch"``.
The old spelling is deprecated with a removal deadline that has already
passed, so a future Streamlit release will drop it entirely and call sites
using it will raise ``TypeError``.  The app must run both on the locally
pinned Streamlit (1.41, old spelling only) and on whatever current release
Streamlit Community Cloud installs (new spelling), so the correct keyword is
chosen **once at import time** by inspecting the installed Streamlit's
signature.

The new style is used only when ``width`` exists *and* its default is
``"stretch"``.  The default check matters: older Streamlit versions already
have a ``width`` parameter on ``st.dataframe``, but there it is a pixel
width (``int | None``), not the layout string.

Call sites use :func:`plotly_chart` / :func:`dataframe` instead of calling
``st`` directly, and behave identically on every supported Streamlit.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping

import streamlit as st

__all__ = ["PLOTLY_WIDTH", "DATAFRAME_WIDTH", "plotly_chart", "dataframe",
           "width_kwargs"]


def width_kwargs(func: Callable[..., Any]) -> Dict[str, Any]:
    """Return the keyword arguments that stretch ``func`` to full width.

    Returns ``{"width": "stretch"}`` when the installed Streamlit supports
    the layout string, otherwise ``{"use_container_width": True}``.
    """
    try:
        param: inspect.Parameter | None = inspect.signature(func).parameters.get("width")
    except (TypeError, ValueError):          # builtins / C functions
        param = None
    if param is not None and param.default == "stretch":
        return {"width": "stretch"}
    return {"use_container_width": True}


PLOTLY_WIDTH: Mapping[str, Any] = width_kwargs(st.plotly_chart)
DATAFRAME_WIDTH: Mapping[str, Any] = width_kwargs(st.dataframe)


def plotly_chart(figure: Any, **kwargs: Any) -> Any:
    """Render a Plotly figure at full container width on any Streamlit."""
    merged = {**PLOTLY_WIDTH, **kwargs}
    return st.plotly_chart(figure, **merged)


def dataframe(frame: Any, **kwargs: Any) -> Any:
    """Render a DataFrame at full container width on any Streamlit."""
    merged = {**DATAFRAME_WIDTH, **kwargs}
    return st.dataframe(frame, **merged)
