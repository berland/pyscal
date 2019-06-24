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

from pyscal import SCALrecommendation, WaterOilGas, WaterOil, GasOil


def test_scal_basic():
    low = WaterOilGas()
    base = WaterOilGas()
    high = WaterOilGas()

    low.wateroil.add_corey_oil(1)
    low.wateroil.add_corey_water(1)
    low.gasoil.add_corey_oil(1)
    low.gasoil.add_corey_gas(1)

    base.wateroil.add_corey_oil(2)
    base.wateroil.add_corey_water(2)
    base.gasoil.add_corey_oil(2)
    base.gasoil.add_corey_gas(2)

    high.wateroil.add_corey_oil(2)
    high.wateroil.add_corey_water(2)
    high.gasoil.add_corey_oil(2)
    high.gasoil.add_corey_gas(2)

    assert low.selfcheck()
    assert base.selfcheck()
    assert high.selfcheck()

    rec = SCALrecommendation(low, base, high, tag="test")
    i = rec.interpolate(0.5)
    i.selfcheck()
    print(i.SOF3())

@settings(max_examples=500)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def de_test_wateroilgas_multiple(swirr, swl, swcr, sorw, sorg, sgcr):
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
