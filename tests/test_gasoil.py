# -*- coding: utf-8 -*-
"""Test module for relperm"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pytest
from hypothesis import given
import hypothesis.strategies as st

import pandas as pd

from pyscal import GasOil


def check_table(df):
    """Check sanity of important columns"""
    assert not df.empty
    assert not df.isnull().values.any()
    assert len(df["sg"].unique()) == len(df)
    assert df["sg"].is_monotonic
    assert (df["sg"] >= 0.0).all()
    assert df["sgn"].is_monotonic
    assert df["son"].is_monotonic_decreasing
    assert df["krog"].is_monotonic_decreasing
    assert df["krg"].is_monotonic
    if "pc" in df:
        assert df["pc"].is_monotonic


@given(
    st.floats(),
    st.floats(),
    st.floats(min_value=0, max_value=1.1),
    st.floats(),
    st.floats(),
    st.text(),
)
def test_gasoil_random(swirr, sgcr, h, swl, sorg, tag):
    """Shoot wildly with arguments, the code should throw ValueError
    or AssertionError when input is invalid, but we don't want other crashes"""
    try:
        go = GasOil(swirr=swirr, sgcr=sgcr, h=h, swl=swl, sorg=sorg, tag=tag)
        assert not go.table.empty
        assert not go.table.isnull().values.any()
    except AssertionError:
        print("a", end="")
        return
    print("V", end="")


@given(
    st.floats(min_value=0, max_value=0.6),
    st.floats(min_value=0, max_value=0.6),
    st.floats(min_value=0.01, max_value=0.2),
    st.floats(min_value=0, max_value=0.6),
    st.floats(min_value=0, max_value=0.6),
)
def test_gasoil_sensible(swirr, sgcr, h, swl, sorg):
    """Shoot wildly with arguments, the code should throw ValueError
    or AssertionError when input is invalid, but we don't want other crashes"""
    try:
        go = GasOil(swirr=swirr, sgcr=sgcr, h=h, swl=swl, sorg=sorg)
        # print(go.table)
        assert not go.table.empty
        assert not go.table.isnull().values.any()
    except AssertionError:
        print("a", end="")
        return
    print("V", end="")


@given(st.floats(), st.floats())
def test_gasoil_corey1(ng, nog):
    go = GasOil()
    try:
        go.add_corey_oil(nog=nog)
        go.add_corey_gas(ng=ng)
    except AssertionError:
        # This happens for "invalid" input
        return

    assert "krog" in go.table
    assert "krg" in go.table
    assert isinstance(go.krgcomment, str)
    check_table(go.table)
    sgofstr = go.SGOF()
    assert len(sgofstr) > 100


@given(st.floats(), st.floats(), st.floats(), st.floats(), st.floats())
def test_gasoil_let1(l, e, t, krgend, krgmax):
    go = GasOil()
    try:
        go.add_LET_oil(l, e, t, krgend)
        go.add_LET_gas(l, e, t, krgend, krgmax)
    except AssertionError:
        # This happens for negative values f.ex.
        return
    assert "krog" in go.table
    assert "krg" in go.table
    assert isinstance(go.krgcomment, str)
    check_table(go.table)
    sgofstr = go.SGOF()
    assert len(sgofstr) > 100
