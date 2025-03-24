# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

m_n = 1.67492750056e-27
h = 6.62607015e-34
h_over_m = h / m_n


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
    birth_time: np.ndarray
    wavelength: np.ndarray
    frequency: float


@dataclass
class NeutronData:
    distance: float
    id: np.ndarray
    speed: np.ndarray
    birth_time: np.ndarray
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
