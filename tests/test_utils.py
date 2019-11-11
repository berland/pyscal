# -*- coding: utf-8 -*-
"""Test module for pyscal.utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from pyscal import utils, WaterOil
from test_wateroil import check_table


def test_diffjumppoint():
    """Test estimator for the jump in first derivative for some manually set up cases.

    This code is also extensively tested throuth test_addfromtable"""

    df = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [0.3, 0.2], [1, 1]])

    assert utils.estimate_diffjumppoint(df, side="right") == 0.3
    assert utils.estimate_diffjumppoint(df, side="left") == 0.3

    df = pd.DataFrame(columns=["x", "y"], data=[[0, 0], [1, 1]])
    # We don't really care what gets printed from this, just don't crash..
    assert 0 <= utils.estimate_diffjumppoint(df, side="right") <= 1
    assert 0 <= utils.estimate_diffjumppoint(df, side="left") <= 1

    df = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0.1, 0.1],
            [0.2, 0.2],  # Linear until here
            [0.3, 0.4],  # Nonlinear region
            [0.4, 0.45],  # Nonlinear region
            [0.7, 0.7],  # Linear from here
            [0.8, 0.8],
            [1, 1],
        ],
    )
    assert utils.estimate_diffjumppoint(df, side="left") == 0.2
    assert utils.estimate_diffjumppoint(df, side="right") == 0.7

    df = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0.1, 0.0],
            [0.2, 0.0],  # Linear until here
            [0.3, 0.4],  # Nonlinear region
            [0.9, 1],  # Start linear region again
            [1, 1],
        ],
    )
    assert utils.estimate_diffjumppoint(df, side="left") == 0.2
    assert utils.estimate_diffjumppoint(df, side="right") == 0.9


def test_normalize_nonlinpart():
    # Test that we can make normalized functions
    wo = WaterOil(swl=0.1, swcr=0.12, sorw=0.05, h=0.05)
    wo.add_corey_water(nw=2.1, krwend=0.9)
    wo.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = utils.normalize_nonlinpart(wo)

    assert np.isclose(krwn(0), 0)
    assert np.isclose(krwn(1), 0.9)

    # kron is normalized on son
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test with tricky endpoints
    h = 0.01
    wo = WaterOil(swl=h, swcr=h, sorw=h, h=h)
    wo.add_corey_water(nw=2.1, krwend=0.9)
    wo.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = utils.normalize_nonlinpart(wo)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(krwn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test again with zero endpoints:
    wo = WaterOil(swl=0, swcr=0, sorw=0, h=0.01)
    wo.add_corey_water(nw=2.1, krwend=0.9)
    wo.add_corey_oil(now=3, kroend=0.8)
    krwn, kron = utils.normalize_nonlinpart(wo)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(krwn(1), 0.9)
    assert np.isclose(kron(0), 0)
    assert np.isclose(kron(1), 0.8)

    # Test when endpoints are messed up:
    wo = WaterOil(swl=0.1, swcr=0.2, sorw=0.1, h=0.1)
    wo.add_corey_water(nw=2.1, krwend=0.6)
    wo.add_corey_oil(now=3, kroend=0.8)
    wo.swl = 0
    wo.swcr = 0
    wo.sorw = 0
    krwn, kron = utils.normalize_nonlinpart(wo)
    # These go well still, since we are at zero
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(kron(0), 0)
    # These do not match when endpoints are wrong
    assert not np.isclose(krwn(1), 0.6)
    assert not np.isclose(kron(1), 0.8)

    # So fix endpoints!
    wo.swl = wo.table["sw"].min()
    wo.swcr = wo.estimate_swcr()
    wo.sorw = wo.estimate_sorw()
    # Try again
    krwn, kron = utils.normalize_nonlinpart(wo)
    assert np.isclose(krwn(0), 0.0)
    assert np.isclose(kron(0), 0)
    assert np.isclose(krwn(1), 0.6)
    assert np.isclose(kron(1), 0.8)


def test_interpolate_wo():
    ow_low = WaterOil(swl=0.03, swcr=0.09, sorw=0.1)
    ow_high = WaterOil(swl=0.1, swcr=0.13, sorw=0.05)
    ow_low.add_corey_water(nw=1, krwend=0.8)
    ow_high.add_corey_water(nw=2, krwend=0.7)
    ow_low.add_corey_oil(now=3, kroend=0.6)
    ow_high.add_corey_oil(now=2, kroend=0.95)

    for t in np.arange(0, 1, 0.1):
        ow_ip = utils.interpolate_ow(ow_low, ow_high, t)
        check_table(ow_ip.table)
