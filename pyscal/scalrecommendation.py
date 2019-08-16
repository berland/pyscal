# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
import pandas as pd

from pyscal.constants import SWINTEGERS

from pyscal import WaterOilGas, WaterOil, GasOil


class SCALrecommendation(object):
    """A SCAL recommendation consists of three OilWaterGas objects,
    tagged low, base and high.

    For a continuum to exist within a SCAL recommendation, relperm
    curves are interpolated pointwise in relperm-vs-saturation
    space. Because the individual low, base and high curves might
    cross, interpolation cannot be performed in the parameter space
    (L, E, T, endpoints).

    If endpoints do not vary, and T has opposite direction than L and
    E going from low to h igh, interpolation in L, E and T-space is
    possible without interpolated curves going outside the area
    contained by low and high curves.

    """

    def __init__(self, low, base, high, tag, h=0.01):
        """Set up a SCAL recommendation curve set

        You can choose to provide ready-made WaterOilGas objects, or
        dictionaries with LET_properties for initialization.

        If you have LET-properties in a dictionary, the low, base and
        high objects must be of type dict must containing the
        properties for each curve set, in the following keys:

           Lw, Ew, Tw, Lo, Eo, To,
           Lg, Eg, Tg, Log, Eog, Tog,
           swirr, swl, sorw, sorg, sgcr

        For oil-water only, you may omit the LET parameters for gas and oil-gas

        If low, base and high are of type WaterOilGas objects, these
        will be used as is.
        """

        self.h = h
        self.tag = tag

        # Help users from pyscals predecessor, or users of pyscal 0.1.x
        if isinstance(low, dict):
            raise ValueError("Use PyscalFactory to create SCAL recommendation objects from dictionaries")

        if (
            isinstance(low, WaterOilGas)
            and isinstance(base, WaterOilGas)
            and isinstance(high, WaterOilGas)
        ):

            self.low = low
            self.base = base
            self.high = high
        else:
            raise ValueError

    # User should add capillary pressure explicitly by calling add**
    # on the class objects, or run the following method to add the
    # same to all curves:
    def add_simple_J(self, a=5, b=-1.5, poro_ref=0.25, perm_ref=100, drho=300, g=9.81):
        """Add (identical) simplified J-function to all water-oil
        curves in the SCAL recommendation set"""
        self.low.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )
        self.base.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )
        self.high.wateroil.add_simple_J(
            a=a, b=b, poro_ref=poro_ref, perm_ref=perm_ref, drho=drho, g=g
        )

    def interpolate(self, parameter, parameter2=None, h=0.02):
        """Interpolate between low, base and high
        parameter = -1 reproduces low curve
        parameter = 0 reproduces base curve
        parameter = 1 reproduces high curve

        Interpolation is performed pointwise in the "relperm" direction
        for each saturation point. During interpolation, endpoints are
        hard to handle, and only swirr is attempted preserved.

        Interpolation is linear in relperm-direction, and will thus not be
        linear in log-relperm-direction

        This method returns an WaterOilGasTable object which can be
        realized into printed tables. No attempt is made to
        parametrize the interpolant in L,E,T parameter space.

        If a second parameter is supplied ("parameter2") this is used
        for the gas-oil interpolation. This enables the gas-oil
        interpolant to be fully uncorrelated to the water-oil
        interpolant. CHECK how this affects endpoints and Eclipse
        consistency!!

        """

        if parameter2 is not None:
            gasparameter = parameter2
        else:
            gasparameter = parameter

        # Initialize wateroil and gasoil curves to be filled with
        # interpolated curves:
        interpolant = WaterOilGas()

        if abs(parameter) > 1.0:
            logging.error("Interpolation parameter must be in [-1,1]")
            interpolant.wateroil = None
            raise AssertionError
        elif np.isclose(parameter, 0.0):
            interpolant.wateroil = self.base.wateroil
        elif np.isclose(parameter, -1.0):
            interpolant.wateroil = self.low.wateroil
        elif np.isclose(parameter, 1.0):
            interpolant.wateroil = self.high.wateroil
        elif parameter < 0.0:
            curve1 = copy.deepcopy(self.base.wateroil)
            curve2 = copy.deepcopy(self.low.wateroil)
            param_transf = -parameter  # 0 gives base, 1 gives low.
            swl = (
                curve1.table["sw"][0] * (1 - param_transf)
                + curve2.table["sw"][0] * param_transf
            )
            interpolant.wateroil = WaterOil(
                swirr=swl,
                swl=swl,
                sorw=0.0,
                h=h,
                tag=self.tag + " interpolant at %g" % parameter,
            )
            self.__interpolator(
                interpolant.wateroil,
                curve1,
                curve2,
                param_transf,
                "sw",
                "krw",
                "krow",
                "pc",
            )
        elif parameter > 0.0:
            curve1 = copy.deepcopy(self.base.wateroil)
            curve2 = copy.deepcopy(self.high.wateroil)
            param_transf = parameter  # 0 gives base, 1 gives high
            swl = (
                curve1.table["sw"][0] * (1 - param_transf)
                + curve2.table["sw"][0] * param_transf
            )
            interpolant.wateroil = WaterOil(
                swirr=swl,
                swl=swl,
                sorw=0.0,
                h=h,
                tag=self.tag + " interpolant at %g" % parameter,
            )
            self.__interpolator(
                interpolant.wateroil,
                curve1,
                curve2,
                param_transf,
                "sw",
                "krw",
                "krow",
                "pc",
            )

        # Gas-oil interpolation
        # We need swl from the interpolated WaterOil object.
        swl = interpolant.wateroil.swl
        if abs(gasparameter) > 1.0:
            logging.error("Interpolation parameter must be in [-1,1]")
            interpolant.gasoil = None
            raise AssertionError
        elif np.isclose(gasparameter, 0.0):
            interpolant.gasoil = self.base.gasoil
        elif np.isclose(gasparameter, -1.0):
            interpolant.gasoil = self.low.gasoil
        elif np.isclose(gasparameter, 1.0):
            interpolant.gasoil = self.high.gasoil
        elif gasparameter < 0.0:
            curve1 = copy.deepcopy(self.base.gasoil)
            curve2 = copy.deepcopy(self.low.gasoil)
            gas_param_transf = -1 * gasparameter  # 0 gives base, 1 gives low.
            # We have to use the extreme, not interpolated sgcr.
            sgcr = min(curve1.sgcr, curve2.sgcr)
            interpolant.gasoil = GasOil(
                sgcr=sgcr,
                swl=swl,
                sorg=0.0,
                h=h,
                tag=self.tag + " interpolant at %g" % gasparameter,
            )
            self.__interpolator(
                interpolant.gasoil,
                curve1,
                curve2,
                gas_param_transf,
                "sg",
                "krg",
                "krog",
                "pc",
            )
            interpolant.gasoil.resetsorg()
        elif gasparameter > 0.0:
            curve1 = copy.deepcopy(self.base.gasoil)
            curve2 = copy.deepcopy(self.high.gasoil)
            gas_param_transf = gasparameter  # 0 gives base, 1 gives high
            # We have to use the extreme, not interpolated sgcr.
            sgcr = min(curve1.sgcr, curve2.sgcr)
            interpolant.gasoil = GasOil(
                sgcr=sgcr,
                swl=swl,
                sorg=0.0,
                h=h,
                tag=self.tag + " interpolant at %g" % gasparameter,
            )
            self.__interpolator(
                interpolant.gasoil,
                curve1,
                curve2,
                gas_param_transf,
                "sg",
                "krg",
                "krog",
                "pc",
            )
            interpolant.gasoil.resetsorg()

        return interpolant

    def __interpolator(
        self,
        tableobject,
        curve1,
        curve2,
        parameter,
        sat="sw",
        kr1="krw",
        kr2="krow",
        pc="pc",
    ):
        """Interpolates between two curves using one parameter between 0 and
        1, does not care if it is water-oil or gas-oil.  First
        argument is the wateroil or gasoil object it is to populate.

        0 will return curve1
        1 will return curve2

        No return value, but modifies the object pointed to by first handle

        """

        curve1.table.rename(columns={kr1: kr1 + "_1"}, inplace=True)
        curve2.table.rename(columns={kr1: kr1 + "_2"}, inplace=True)
        curve1.table.rename(columns={kr2: kr2 + "_1"}, inplace=True)
        curve2.table.rename(columns={kr2: kr2 + "_2"}, inplace=True)
        curve1.table.rename(columns={pc: pc + "_1"}, inplace=True)
        curve2.table.rename(columns={pc: pc + "_2"}, inplace=True)

        # Result data container:
        satresult = pd.DataFrame(data=tableobject.table[sat], columns=[sat])

        # Merge swresult with curve1 and curve2, and interpolate all
        # columns in sw:
        intdf = (
            pd.concat([curve1.table, curve2.table, satresult], sort=True)
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
        if pc + "_1" in curve1.table.columns and pc + "_2" in curve2.table.columns:
            intdf[pc] = (
                intdf[pc + "_1"] * (1 - parameter) + intdf[pc + "_2"] * parameter
            )
        else:
            intdf[pc] = 0

        # Slice out the resulting sw values and columns. Slicing on
        # floating point indices is not robust so we need to slice on an
        # integer version of the sw column
        tableobject.table["swint"] = list(
            map(int, list(map(round, tableobject.table[sat] * SWINTEGERS)))
        )
        intdf["swint"] = list(
            map(int, list(map(round, intdf.index.values * SWINTEGERS)))
        )
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
