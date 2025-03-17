# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

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
        The number of neutrons in the data.
        """
        return self.id.shape[1]


@dataclass
class Plot:
    ax: plt.Axes
    fig: plt.Figure
