"""Utility function for pyscal
"""
import logging
import six
import pandas as pd

import pandas as pd
from scipy.interpolate import interp1d

from .wateroil import WaterOil
from .wateroilgas import WaterOilGas
from .constants import SWINTEGERS
from .constants import EPSILON as epsilon


def interpolate_wog(wog_0, wog_1, param, scalesaturations=False):
    """Interpolate between two WaterOilGas objects

    Args:
        param (float): Between 0 and 1. 0 will return wog1, 1 will
            return wog2. Any number between will be a linear interpolation
            in the permeability direction for each saturation point.
        scalesat (bool): If True, the saturations will be scaled as well
            preserving endpoints.
    Returns:
        WaterOilGas object.
    """
    assert 0 <= param <= 1
    assert isinstance(wog_0, WaterOilGas)
    assert isinstance(wog_1, WaterOilGas)
    if param <= 0.5:
        return wog_0
    return wog_1


def interpolate_wo(wo_0, wo_1, param):
    """Ditto, but only for WaterOil"""
    pass


def interpolate_go(go_0, go_1, param):
    pass


def interpolator(
    tableobject, ow_low, ow_high, parameter, sat="sw", kr1="krw", kr2="krow", pc="pc"
):
    """Interpolates between two curves.

    DEPRECATED FUNCTION!

    The interpolation parameter is 0 through 1,
    irrespective of phases or low-base/base-high/low-high.

    Args:
        tabjeobject (WaterOil or GasOil): A partially setup object where
            relperm and pc columns are to be filled with numbers.
        ow_low (WaterOil or GasOil): "Low" case of interpolation (relates
            to interpolation parameter 0). Must be copies, as they
            will be modified.
        ow_high: Ditto, relates to interpolation parameter 1
        parameter (float): Between 0 and 1, what you want to interpolate to.
        sat (str): Name of the saturation column, typically 'sw' or 'sg'
        kr1 (str): Name of the first relperm column ('krw' or 'krg')
        kr2 (str): Name of the second relperm column ('krow' or 'krog')
        pc (str): Name of the capillary pressure column ('pc')

    Returns:
        None, but modifies the first argument.
    """
    logging.warning("This function call is deprecated")

    ow_low.table.rename(columns={kr1: kr1 + "_1"}, inplace=True)
    ow_high.table.rename(columns={kr1: kr1 + "_2"}, inplace=True)
    ow_low.table.rename(columns={kr2: kr2 + "_1"}, inplace=True)
    ow_high.table.rename(columns={kr2: kr2 + "_2"}, inplace=True)
    ow_low.table.rename(columns={pc: pc + "_1"}, inplace=True)
    ow_high.table.rename(columns={pc: pc + "_2"}, inplace=True)

    # Result data container:
    satresult = pd.DataFrame(data=tableobject.table[sat], columns=[sat])

    # Merge swresult with ow_low and ow_high, and interpolate all
    # columns in sw:
    intdf = (
        pd.concat([ow_low.table, ow_high.table, satresult], sort=True)
        .set_index(sat)
        .sort_index()
        .interpolate(method="slinear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    # Normalized saturations does not make sense for the
    # interpolant, remove:
    for col in ["swn", "son", "swnpc", "H", "J"]:
        if col in intdf.columns:
            del intdf[col]

    intdf[kr1] = intdf[kr1 + "_1"] * (1 - parameter) + intdf[kr1 + "_2"] * parameter
    intdf[kr2] = intdf[kr2 + "_1"] * (1 - parameter) + intdf[kr2 + "_2"] * parameter
    if pc + "_1" in ow_low.table.columns and pc + "_2" in ow_high.table.columns:
        intdf[pc] = intdf[pc + "_1"] * (1 - parameter) + intdf[pc + "_2"] * parameter
    else:
        intdf[pc] = 0

    # Slice out the resulting sw values and columns. Slicing on
    # floating point indices is not robust so we need to slice on an
    # integer version of the sw column
    tableobject.table["swint"] = list(
        map(int, list(map(round, tableobject.table[sat] * SWINTEGERS)))
    )
    intdf["swint"] = list(map(int, list(map(round, intdf.index.values * SWINTEGERS))))
    intdf = intdf.reset_index()
    intdf.drop_duplicates("swint", inplace=True)
    intdf.set_index("swint", inplace=True)
    intdf = intdf.loc[tableobject.table["swint"].values]
    intdf = intdf[[sat, kr1, kr2, pc]].reset_index()

    # intdf['swint'] = (intdf['sw'] * SWINTEGERS).astype(int)
    # intdf.drop_duplicates('swint', inplace=True)

    # Populate the WaterOil object
    tableobject.table[kr1] = intdf[kr1]
    tableobject.table[kr2] = intdf[kr2]
    tableobject.table[pc] = intdf[pc]
    tableobject.table.fillna(method="ffill", inplace=True)
    return


def estimate_diffjumppoint(table, xcol=None, ycol=None, side="right"):
    """Estimate the point where the y-data jumps from being linear
    in x to being nonlinear, or where it shift from one linear domain
    to another (for a piecewise linear function)

    If xcol is sw, and ycol is krw, and side is 'right', this
    will typically estimate sorw for you. If side is 'left' it will
    give you swcr.

    Args:
        table (pd.DataFrame): A Dataframe with x and y data
        xcol (string): The name of the column in table containing x-data. If\
            None (default) the first column in table will be used.
        ycol (string): The name of the column in table containing y-data.
            If None (default) the second column in the table will be used.
        side (string): Must be 'left' or 'right'. Decided whether to look from
            the right side of the x-interval or from the left side for the
            linear domain.
    Returns:
        float: The x value where the start-linear domain ends.
    """

    if not xcol:
        xcol = table.columns[0]
    if not ycol:
        ycol = table.columns[1]
    assert isinstance(ycol, six.string_types)
    assert isinstance(xcol, six.string_types)
    if not side:
        raise ValueError("side cannot be None, use left or right")
    side = side.lower()
    assert side in ["left", "right"]

    # Compute the derivative:
    table["_deriv"] = table[ycol].diff() / table[xcol].diff()
    # The first becomes NaN, extrapolate from the second row:
    table.loc[0, "_deriv"] = table["_deriv"].iloc[1]

    # Pick the derivative at the first or last segment:
    iloc = {"left": 0, "right": -1}
    lin_a = table["_deriv"].iloc[iloc[side]]

    # Make a linear extrapolation from the last segment, starting at max x
    table["_linear"] = (table[xcol] - table[xcol].iloc[iloc[side]]) * lin_a + table[
        ycol
    ].iloc[iloc[side]]
    assert table["_linear"].values[iloc[side]] == table[ycol].values[iloc[side]]

    # Compute how much krw deviates from the linear krw:
    table["_lindev"] = (table[ycol] - table["_linear"]).abs()

    # Use the cumulative sum to determine the onset of non-zero deviation
    # starting from sw=1:
    table["_lindevcumsum"] = table["_lindev"].cumsum()

    if side == "right":
        maxcumsum = table["_lindevcumsum"].max()
        linearpart = table[(table["_lindevcumsum"] - maxcumsum).abs() < epsilon]
        return linearpart.iloc[1][xcol]
    else:
        linearpart = table[(table["_lindevcumsum"] < epsilon)]
        if len(linearpart) == 1:
            linearpart = table[(table["_lindevcumsum"].shift(1) < epsilon)]
        return linearpart.iloc[-1][xcol]


def normalize_nonlinpart(curve):
    """
    Make krw and krow functions that evaluates only on the
    (potentially) nonlinear part of the relperm curves, and with
    a normalized argument on that interval, based on real curves.

    For a WaterOil krw curve, the nonlinear part
    is from swcr to sorw. swcr is mapped to zero, and 1 - sorw is mapped
    to 1. Then there is an assumed linear part from sorw to 1 which
    we ignore here.

    For a WaterOil krow curve, the nonlinear part
    is from 1 - sorw (mapped to zero) to swcr (mapped to 1). If swcr > swl,
    there is a linear part from swcr down to swl, ignored here.

    These endpoints must be known the the WaterOil object coming in (the object
    can determine them using functions 'estimate_sorw()' and 'estimate_swcr()'

    If the entire curve is linear, it will not matter for this function, because
    this function only deals with the presumably known endpoints.

    Arguments:
        curve (WaterOil): incoming oilwater curve set (krw and krow)

    Returns:
        tuple of lambda functions. The first will evaluate krw on
            the normalized Sw interval [0,1], the second will
            evaluate krow on the normalized So interval [0,1].
    """
    krw_interp = interp1d(
        curve.table["sw"],
        curve.table["krw"],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )

    # The internal dataframe might contain normalized
    # saturation values, but we do not want to assume they
    # are there or even correct, therefore we effectively
    # recalculate them (here using lambda functions)
    sw_fn = lambda swn: curve.swcr + swn * (1.0 - curve.swcr - curve.sorw)
    krw_fn = lambda swn: krw_interp(sw_fn(swn))

    kro_interp = interp1d(
        1.0 - curve.table["sw"],
        curve.table["krow"],
        kind="linear",
        bounds_error=False,
        fill_value=(1.0, 0.0),
    )
    so_fn = lambda son: curve.sorw + son * (1.0 - curve.sorw - curve.swcr)
    kro_fn = lambda son: kro_interp(so_fn(son))

    return (krw_fn, kro_fn)


def interpolate_ow(ow_low, ow_high, parameter, h=0.01):
    """Interpolates between two oil-water curves.

    The saturation endpoints for the curves must be known
    by the objects. They can be estimated by estimate_sorw() etc.
    or can be set manually for finer control.

    The interpolation algorithm is different left and right
    for saturation endpoints, and saturation endpoints are
    interpolated individually.

    Arguments:
        ow_low (WaterOil): a "low" case
        ow_high (WaterOil): a "high" case
        parameter (float): Between 0 and 1. 0 will return the low case, 1 will return the
            high case. Any number in between will return an interpolated curve
        h (float): Saturation step-size in interpolant. If defaulted, a value
            smaller than in the input curves are used, to preserve information.
    Returns:
        A new oil-water curve

    """
    # Note: A separate function is for both OilWater and for GasOil. At time of implementation
    # it is guessed to be slightly more work to make one function that tackles both, than
    # two separate. This should be reevaluated later when maintenance kicks in.
    assert isinstance(ow_low, WaterOil)
    assert isinstance(ow_high, WaterOil)

    assert 0 <= parameter <= 1
    # Extrapolation is refused, but perhaps later implemented with truncation to (0,1)

    # Constructs functions that works on normalized saturation interval
    krw1, kro1 = normalize_nonlinpart(ow_low)
    krw2, kro2 = normalize_nonlinpart(ow_high)

    # Construct a lambda function that can be applied to both relperm values
    # and endpoints
    weighted_value = lambda a, b: a * parameter + b * (1.0 - parameter)

    # Interpolate saturation endpoints
    swl_new = weighted_value(ow_low.swl, ow_high.swl)
    swcr_new = weighted_value(ow_low.swcr, ow_high.swcr)
    sorw_new = weighted_value(ow_low.sorw, ow_high.sorw)

    # Construct the new WaterOil object, with interpolated
    # endpoints:
    ow_new = WaterOil(swl=swl_new, swcr=swcr_new, sorw=sorw_new, h=h)

    # Add interpolated relperm data:
    ow_new.table["krw"] = weighted_value(
        krw1(ow_new.table["swn"]), krw2(ow_new.table["swn"])
    )
    ow_new.table["krow"] = weighted_value(
        kro1(ow_new.table["son"]), kro2(ow_new.table["son"])
    )

    return ow_new
