"""Tests for the Streamlit element-width compatibility shim.

The app must render full-width charts/tables on the locally pinned Streamlit
(``use_container_width`` only) and on future releases where that keyword is
removed (``width="stretch"``).  The selection is made by inspecting the
installed Streamlit's signature, so both branches are unit-tested here with
stub signatures, plus a check that the chosen keyword really exists on the
Streamlit installed in this environment.
"""
from __future__ import annotations

import inspect

import streamlit as st

from components.streamlit_compat import (
    DATAFRAME_WIDTH,
    PLOTLY_WIDTH,
    plotly_chart,
    width_kwargs,
)


# --- signature branches -----------------------------------------------------

def _new_style(figure, use_container_width=None, *, width="stretch"):
    """Signature of modern Streamlit's st.plotly_chart."""


def _old_style(figure, use_container_width=False):
    """Signature of Streamlit 1.41's st.plotly_chart (no width parameter)."""


def _pixel_width(data, width=None, *, use_container_width=False):
    """Signature of st.dataframe on older Streamlit: width means pixels.

    Regression guard: merely finding a ``width`` parameter is not enough to
    decide the layout-string style is available.
    """


def test_layout_default_selects_stretch():
    assert width_kwargs(_new_style) == {"width": "stretch"}


def test_missing_width_falls_back_to_container_width():
    assert width_kwargs(_old_style) == {"use_container_width": True}


def test_pixel_width_parameter_is_not_mistaken_for_layout():
    assert width_kwargs(_pixel_width) == {"use_container_width": True}


def test_non_stretch_width_default_falls_back():
    def _pixel_only(figure, width=None):
        ...

    assert width_kwargs(_pixel_only) == {"use_container_width": True}


# --- installed Streamlit ----------------------------------------------------

def test_chosen_keywords_exist_on_installed_streamlit():
    for func, chosen in ((st.plotly_chart, PLOTLY_WIDTH), (st.dataframe, DATAFRAME_WIDTH)):
        params = inspect.signature(func).parameters
        for key in chosen:
            assert key in params, (
                f"{func.__name__} does not accept {key!r} on Streamlit "
                f"{st.__version__}; the width shim picked a wrong keyword"
            )


def test_wrapper_injects_width_and_forwards_caller_arguments(monkeypatch):
    seen: dict = {}

    def fake(fig, **kwargs):
        seen.update(kwargs)
        return "rendered"

    monkeypatch.setattr(st, "plotly_chart", fake)
    assert plotly_chart("fig", theme="streamlit") == "rendered"
    assert seen["theme"] == "streamlit"
    assert any(k in seen for k in ("width", "use_container_width")), (
        "wrapper did not pass a width keyword"
    )
