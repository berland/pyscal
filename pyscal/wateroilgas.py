# -*- coding: utf-8 -*-

import math
import copy
import numpy as np
import pandas as pd

import pyscal
from pyscal.constants import SWINTEGERS as SWINTEGERS


class WaterOilGas(object):

    """A representation of three-phase properties for oil-water-gas

    Use one object for each satnum.

    Holds one WaterOil object and one GasOil object, ensuring
    consistent endpoints.

    Access the class members 'wateroil' and 'gasoil' directly to add curves
    """

    def __init__(
        self, swirr=0, swl=0.0, swcr=0.0, sorw=0.0, sorg=0, sgcr=0, h=0.01, tag=""
    ):
        """Sets up saturation range for water (Sw) and gas (Sg)"""
        assert sgcr + swcr < 1 - h
        assert sgcr + swl < 1 - h
        assert sorg + swcr < 1 - h
        assert sorg + swl < 1 - h
        assert sgcr + swirr < 1 - h
        assert sorg + swirr < 1 - h
        assert sorw + swirr < 1 - h
        assert sorw + swcr < 1 - h
        assert sorw + swl < 1 - h

        self.wateroil = pyscal.WaterOil(
            swirr=swirr, swl=swl, swcr=swcr, sorw=sorw, h=h, tag=tag
        )
        self.gasoil = pyscal.GasOil(
            swirr=swirr, sgcr=sgcr, sorg=sorg, swl=swl, h=h, tag=tag
        )

    def selfcheck(self):
        """Run selfcheck on both wateroil and gasoil.

        Returns true only if both passes"""
        return self.wateroil.selfcheck() and self.gasoil.selfcheck()

    def SWOF(self, header=True, dataincommentrow=True):
        """Return a SWOF string. Delegated to the wateroil object"""
        return self.wateroil.SWOF(header, dataincommentrow)

    def SGOF(self, header=True, dataincommentrow=True):
        """Return a SGOF string. Delegated to the gasoil object."""
        return self.gasoil.SGOF(header, dataincommentrow)

    def SLGOF(self, header=True, dataincommentrow=True):
        """Return a SLGOF string. Delegated to the gasoil object."""
        return self.gasoil.SLGOF(header, dataincommentrow)

    def SGFN(self, header=True, dataincommentrow=True):
        """Return a SGFN string. Delegated to the gasoil object."""
        return self.gasoil.SGFN(header, dataincommentrow)

    def SWFN(self, header=True, dataincommentrow=True):
        """Return a SWFN string. Delegated to the wateroil object."""
        return self.wateroil.SWFN(header, dataincommentrow)

    def SOF3(self, header=True, dataincommentrow=True):
        """Return a SOF3 string, combining data from the wateroil and
        gasoil objects.

        So - the oil saturation ranges from 0 to 1-swl. The saturation points
        from the WaterOil object is used to generate these
        """
        # Copy of the wateroil data:
        table = pd.DataFrame(self.wateroil.table[["sw", "krow"]])
        table["so"] = 1 - table["sw"]

        # Copy of the gasoil data:
        gastable = pd.DataFrame(self.gasoil.table[["sg", "krog"]])
        gastable["so"] = 1 - gastable["sg"] - self.wateroil.swl

        # Merge WaterOil and GasOil on oil saturation, interpolate for
        # missing data (potentially different sg- and sw-grids)
        sof3table = (
            pd.concat([table, gastable], sort=True)
            .set_index("so")
            .sort_index()
            .interpolate(method="slinear")
            .fillna(method="ffill")
            .fillna(method="bfill")
            .reset_index()
        )
        sof3table["soint"] = list(
            map(int, list(map(round, sof3table["so"] * SWINTEGERS)))
        )
        sof3table.drop_duplicates("soint", inplace=True)

        # The 'so' column has been calculated from floating point numbers
        # and the zero value easily becomes a negative zero, circumvent this:
        zerorow = np.isclose(sof3table["so"], 0.0)
        sof3table.loc[zerorow, "so"] = abs(sof3table.loc[zerorow, "so"])

        string = ""
        if header:
            string += "SOF3\n"
        string += "-- " + self.wateroil.tag + "\n"
        string += "-- " + self.gasoil.tag + "\n"
        string += "-- So Krow Krog\n"
        if dataincommentrow:
            string += self.wateroil.swcomment
            string += self.gasoil.sgcomment

        string += sof3table[["so", "krow", "krog"]].to_csv(
            sep=" ", float_format="%1.5f", header=None, index=False
        )
        string += "/\n"
        return string

    def threephaseconsistency(self):
        """Perform consistency checks on produced curves, similar
        to what Eclipse does at startup

        Returns None if no errors catched. Alternatively
        an error description is returned.

        Possible variation of this function would be
        to throw Exceptions.
        """

        # Eclipse errors:

        # 1: Error in saturation table number 1 at the maximum oil
        # saturation (0.9) krow and krog should both be equal the oil
        # relative permeability for a system with oil and connate
        # water only - but in this case they are different (krow=1.0
        # and krog=0.93)

        errors = ""
        if not np.isclose(
            self.wateroil.table["krow"].max(), self.gasoil.table["krog"].max()
        ):
            errors += "Error: max(krow) is not equal to max(krog)\n"

        # 2: Inconsistent end points in saturation table 1 the maximum
        # gas saturation (0.91) plus the connate water saturation
        # (0.10) must not exceed 1.0
        if self.gasoil.table["sg"].max() + self.wateroil.table["sw"].min() > 1.0:
            errors += "Error: Max(sg) + Swl > 1.0\n"

        # 3: Warning: Consistency problem for gas phase endpoint (krgr > krg)
        # in grid cell (i, j, k) for saturation end-points krgr=1.0
        # krg = 0.49.

        # 4: Warning: Consistency problem for oil phase endpoint (sgu > 1-swl)
        # in grid cell (i, j, k) for saturation endpoints sgu=0.78,
        # swl=0.45, (1-swl) = 0.55

        if len(errors):
            return errors
        else:
            return None

    def run_flow_test(self):
        import subprocess

        try:
            self.run_eclipse_test(eclipselauncher="/usr/bin/flow", launcheroptions="")
        except subprocess.CalledProcessError:
            pass

    def run_eclipse_test(
        self, eclipselauncher="/project/res/bin/runeclipse", launcheroptions="-i"
    ):
        """Start the Eclipse simulator on a minimal deck in order to
        test the properties of the current WaterOilGas deck"""
        import tempfile
        import os
        import subprocess
        import re

        ecldeckbeforeprops = """RUNSPEC
DIMENS
  1 1 1 /
OIL
WATER
GAS
START
  1 'JAN' 2100 /
GRID
DX
   10 /
DY
   10 /
DZ
   50 /
TOPS
   1000 /
PORO
   0.3 /
PERMX
   100 /
PERMY
   100 /
PERMZ
   100 /

PROPS
"""
        ecldeckafterprops = """
DENSITY
  800 1000 1.2 /
PVTW
  1 1 0.0001 0.2 0.00001 /
PVDO
   100 1   1
   150 0.9 1 /
PVDG
   100 1 1
   150 0.9 1 /
ROCK
  100 0.0001 /
SOLUTION
EQUIL
   1000    100     1040    0   1010      0 /"""

        # Generate the finished Eclipse deck.
        ecldeck = ecldeckbeforeprops + self.SWOF() + self.SGOF() + ecldeckafterprops

        tmpdir = tempfile.mkdtemp()
        eclfile = os.path.join(tmpdir, "RELPERMTEST.DATA")
        with open(eclfile, "w") as eclfileh:
            eclfileh.write(ecldeck)
        ecloutput = subprocess.check_output([eclipselauncher, launcheroptions, eclfile])
        ecloutputlines = ecloutput.split("\n")
        print([x for x in ecloutputlines if "Error" in x or "ERROR" in x])
