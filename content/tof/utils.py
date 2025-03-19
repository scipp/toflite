# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

h_over_m = const.h / const.m_n
two_pi = const.pi * 2.0


def speed_to_wavelength(speed: np.ndarray) -> np.ndarray:
    """
    Convert neutron speeds to wavelengths in angstroms.

    Parameters
    ----------
    speed:
        Input speeds in m/s.
    """
    return 1.0e10 * h_over_m / speed


def wavelength_to_speed(wavelength: np.ndarray) -> np.ndarray:
    """
    Convert neutron wavelengths to speeds.

    Parameters
    ----------
    wavelength:
        Input wavelengths in angstroms.
    """
    return 1.0e10 * h_over_m / wavelength


def one_mask(
    masks: MappingProxyType[str, np.ndarray], unit: str | None = None
) -> np.ndarray:
    """
    Combine multiple masks into a single mask.

    Parameters
    ----------
    masks:
        The masks to combine.
    unit:
        The unit of the output mask.
    """
    out = reduce(lambda a, b: a | b, masks.values())
    out.unit = unit
    return out


@dataclass(frozen=True)
class FacilityPulse:
    time: np.ndarray
    wavelength: np.ndarray
    frequency: float





@dataclass
class NeutronData:
    distance: float
    id: np.ndarray
    speed: np.ndarray
    time: np.ndarray
    toa: np.ndarray
    wavelength: np.ndarray
    blocked_by_me: np.ndarray
    blocked_by_others: np.ndarray

    @property
    def pulses(self) -> int:
        """
        The number of pulses in the data.
        """
        return self.id.shape[0]

    @property
    def neutrons(self) -> int:
        """
        The number of neutrons in one pulse.
        """
        return self.id.shape[1]

    @property
    def size(self) -> int:
        """
        The total number of neutrons in the data.
        """
        return self.id.size


@dataclass
class Plot:
    ax: plt.Axes
    fig: plt.Figure
