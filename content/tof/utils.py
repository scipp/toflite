# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

m_over_h = const.m_n / const.h
two_pi = const.pi * 2.0


def speed_to_wavelength(x: np.ndarray) -> np.ndarray:
    """
    Convert neutron speeds to wavelengths in angstroms.

    Parameters
    ----------
    x:
        Input speeds.
    unit:
        The unit of the output wavelengths.
    """
    return 1.0e10 / (m_over_h * x)


def wavelength_to_speed(x: np.ndarray, unit: str = "m/s") -> np.ndarray:
    """
    Convert neutron wavelengths to speeds.

    Parameters
    ----------
    x:
        Input wavelengths in angstroms.
    unit:
        The unit of the output speeds.
    """
    return 1.0e-10 / (m_over_h * x)


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


@dataclass
class Plot:
    ax: plt.Axes
    fig: plt.Figure
