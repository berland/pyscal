# -*- coding: utf-8 -*-
"""Test module for pyscal.utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st


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


@settings(deadline=1000)
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=0, max_value=0.1),  # dsorw
    st.floats(min_value=0.1, max_value=5),  # nw1
    st.floats(min_value=0.01, max_value=1),  # krwend1
    st.floats(min_value=0.1, max_value=5),  # now1
    st.floats(min_value=0.01, max_value=1),  # kroend1
    st.floats(min_value=0.1, max_value=5),  # nw2
    st.floats(min_value=0.01, max_value=1),  # krwend2
    st.floats(min_value=0.1, max_value=5),  # now2
    st.floats(min_value=0.01, max_value=1),  # kroend2
)
def test_normalize_nonlinpart_hypo(
    swl,
    dswcr,
    dswlhigh,
    sorw,
    dsorw,
    nw1,
    krwend1,
    now1,
    kroend1,
    nw2,
    krwend2,
    now2,
    kroend2,
):
    """Test the normalization code in utils.

    In particular the fill_value argument to scipy has been tuned to
    fulfill this code"""
    ow_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw)
    ow_high = WaterOil(
        swl=swl + dswlhigh, swcr=swl + dswlhigh + dswcr, sorw=max(sorw - 0.01, 0)
    )
    ow_low.add_corey_water(nw=nw1, krwend=krwend1)
    ow_high.add_corey_water(nw=nw2, krwend=krwend2)
    ow_low.add_corey_oil(now=now1, kroend=kroend1)
    ow_high.add_corey_oil(now=now2, kroend=kroend2)

    krwn1, kron1 = utils.normalize_nonlinpart(ow_low)
    assert np.isclose(krwn1(0), 0)
    assert np.isclose(krwn1(1), krwend1)
    assert np.isclose(kron1(0), 0)
    assert np.isclose(kron1(1), kroend1)

    krwn2, kron2 = utils.normalize_nonlinpart(ow_high)
    assert np.isclose(krwn2(0), 0)
    assert np.isclose(krwn2(1), krwend2)
    assert np.isclose(kron2(0), 0)
    assert np.isclose(kron2(1), kroend2)


def test_normalize_nonlinpart():
    """Manual tests for utils.normalize_nonlinpart"""
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


@settings(deadline=1000)
@given(
    st.floats(min_value=0, max_value=0.1),  # swl
    st.floats(min_value=0, max_value=0.0),  # dswcr
    st.floats(min_value=0, max_value=0.1),  # dswlhigh
    st.floats(min_value=0, max_value=0.3),  # sorw
    st.floats(min_value=0, max_value=0.1),  # dsorw
    st.floats(min_value=0, max_value=0.1),  # nw
    st.floats(min_value=0, max_value=0.1),  # krwend
)
def test_interpolate_wo(swl, dswcr, dswlhigh, sorw, dsorw, nw, krwend):
    ow_low = WaterOil(swl=swl, swcr=swl + dswcr, sorw=sorw)
    ow_high = WaterOil(
        swl=swl + dswlhigh, swcr=swl + dswlhigh + dswcr, sorw=max(sorw - 0.01, 0)
    )
    ow_low.add_corey_water(nw=1, krwend=0.8)
    ow_high.add_corey_water(nw=2, krwend=0.7)
    ow_low.add_corey_oil(now=3, kroend=0.6)
    ow_high.add_corey_oil(now=2, kroend=0.95)
    # print(
    #     " ** Low curve (red):\n"
    #     + ow_low.swcomment
    #     + ow_low.krwcomment
    #     + ow_low.krowcomment
    # )
    # print(
    #     " ** High curve (red):\n"
    #     + ow_high.swcomment
    #     + ow_high.krwcomment
    #     + ow_high.krowcomment
    # )

    for t in np.arange(0, 1, 0.1):
        ow_ip = utils.interpolate_ow(ow_low, ow_high, t)
        # print("Interpolation parameter: " + str(t))
        # print(ow_ip.table)
        # ow_ip.plotkrwkrow()
        check_table(ow_ip.table)
