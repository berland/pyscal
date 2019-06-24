# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

import pandas as pd

from pyscal import WaterOilGas, WaterOil, GasOil


def test_wateroilgas_basic():
    wog = WaterOilGas()
    assert isinstance(wog.wateroil, WaterOil)
    assert isinstance(wog.gasoil, GasOil)
    assert isinstance(wog.wateroil.table, pd.DataFrame)
    assert isinstance(wog.gasoil.table, pd.DataFrame)
    wog.wateroil.add_corey_oil()
    wog.wateroil.add_corey_water()
    wog.gasoil.add_corey_oil()
    wog.gasoil.add_corey_gas()

    assert wog.selfcheck()

    sof3 = wog.SOF3()
    assert isinstance(sof3, str)
    assert len(sof3) > 100

    swof = wog.SWOF()
    assert isinstance(swof, str)
    assert len(swof) > 100

    assert len(wog.SGOF()) > 100
    assert len(wog.SGFN()) > 100
    assert len(wog.SWFN()) > 100
    assert len(wog.SLGOF()) > 100

    wog = WaterOilGas


@settings(max_examples=500)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_wateroilgas_multiple(swirr, swl, swcr, sorw, sorg, sgcr):
    """Check that we get get AssertionErrors when invalid input is provided
    and no crashes"""
    try:
        wog = WaterOilGas(
            swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, sorg=sorg, sgcr=sgcr
        )
    except AssertionError:
        # If pytest is run with -s 'a' characters signify assertion errors, and 'V' valid tests.
        print("a", end="")
        return
    print("V", end="")
    wog.wateroil.add_corey_oil()
    wog.wateroil.add_corey_water()
    wog.gasoil.add_corey_oil()
    wog.gasoil.add_corey_gas()
    assert wog.selfcheck()

    sof3 = wog.SOF3()
    assert isinstance(sof3, str)
    assert len(sof3) > 100


def deactivated_test_flow():
    if not os.path.exists("/usr/bin/flow"):
        pytest.skip()
    wog = WaterOilGas(h=0.1)
    wog.wateroil.add_corey_oil()
    wog.wateroil.add_corey_water()
    wog.gasoil.add_corey_oil()
    wog.gasoil.add_corey_gas()
    wog.run_flow_test()
