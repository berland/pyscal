"""Command line tool for pyscal"""

import sys
import argparse

import logging

import numpy as np
from matplotlib import pyplot as plt


from pyscal import WaterOilGas, GasWater, PyscalFactory

from pyscal import __version__

logging.basicConfig()
logger = logging.getLogger(__name__)

EPILOG = """
"""


def get_parser():
    """Construct the argparse parser for the command line script.

    Returns:
        argparse.Parser
    """
    parser = argparse.ArgumentParser(
        prog="pyscalplotter",
        description=(
            "pyscalplotter (" + __version__ + ") is a tool to ..."
            "files for relative permeability input from tabulated parameters."
        ),
        epilog=EPILOG,
    )
    parser.add_argument(
        "parametertable",
        help=(
            "CSV or XLSX file with Corey or LET parameters for relperms. "
            "One SATNUM pr row."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print informational messages while processing input",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )
    parser.add_argument(
        "--delta_s",
        default=None,
        type=float,
        help="Saturation table step-length for sw/sg. Default 0.01",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default=None,
        help="Sheet name if reading XLSX file. Defaults to first sheet",
    )
    return parser


def main():
    """Endpoint for pyscals command line utility.

    Translates from argparse API to Pyscal's Python API"""
    parser = get_parser()
    args = parser.parse_args()

    try:
        pyscalplotter_main(
            parametertable=args.parametertable,
            verbose=args.verbose,
            debug=args.debug,
            delta_s=args.delta_s,
            sheet_name=args.sheet_name,
        )
    except ValueError:
        # If ValueErrors, error messages have already been printed
        sys.exit(1)


def pyscalplotter_main(
    parametertable,
    verbose=False,
    debug=False,
    delta_s=None,
    sheet_name=None,
):
    """A "main()" method not relying on argparse. This can be used

    Args:
        parametertable (string): Filename (CSV or XLSX) to load
        verbose (bool): verbose or not
        delta_s (float): Saturation step-length
        sheet_name (string): Which sheet in XLSX file
    """

    def set_logger_levels(loglevel):
        """Set log levels for all modules imported by this script"""
        logger.setLevel(loglevel)
        logging.getLogger("pyscal.factory").setLevel(loglevel)
        logging.getLogger("pyscal.wateroil").setLevel(loglevel)
        logging.getLogger("pyscal.wateroilgas").setLevel(loglevel)
        logging.getLogger("pyscal.gasoil").setLevel(loglevel)
        logging.getLogger("pyscal.utils").setLevel(loglevel)
        logging.getLogger("pyscal.pyscallist").setLevel(loglevel)

    if verbose:
        set_logger_levels(logging.INFO)
    if debug:
        set_logger_levels(logging.DEBUG)

    scalinput_df = PyscalFactory.load_relperm_df(parametertable, sheet_name=sheet_name)

    logger.debug("Input data:\n%s", scalinput_df.to_string(index=False))

    if "SATNUM" not in scalinput_df:
        logger.error("There is no column called SATNUM in the input data")
        raise ValueError
    _, mpl_ax = plt.subplots()
    num_interpolated_curves = 10
    if "CASE" in scalinput_df:
        scalrec_list = PyscalFactory.create_scal_recommendation_list(
            scalinput_df, h=delta_s
        )
        if scalrec_list[1].type == WaterOilGas:
            for tparam in np.arange(-1, 1, 2 / float(num_interpolated_curves)):
                wog_list = scalrec_list.interpolate(
                    int_param_wo, int_param_go, h=delta_s
                )
        elif scalrec_list[1].type == GasWater:
            for tparam in np.arange(-1, 0, 2 / float(num_interpolated_curves)):
                scalrec_list.interpolate(tparam, h=delta_s)[1].plotkrwkrg(
                    mpl_ax=mpl_ax, alpha=0.2, color="red"
                )
            for tparam in np.arange(0, 1, 2 / float(num_interpolated_curves)):
                scalrec_list.interpolate(tparam, h=delta_s)[1].plotkrwkrg(
                    mpl_ax=mpl_ax, alpha=0.2, color="green"
                )
            scalrec_list.interpolate(-1, h=delta_s)[1].plotkrwkrg(
                mpl_ax=mpl_ax, linestyle="--"
            )
            scalrec_list.interpolate(0, h=delta_s)[1].plotkrwkrg(mpl_ax=mpl_ax)
            scalrec_list.interpolate(1, h=delta_s)[1].plotkrwkrg(
                mpl_ax=mpl_ax, linestyle="--"
            )
            plt.title("GasWater, low-base-high")
    else:
        wog_list = PyscalFactory.create_pyscal_list(
            scalinput_df, h=delta_s
        )  # can be both water-oil, water-oil-gas, or gas-water

    plt.show()


if __name__ == "__main__":
    main()
