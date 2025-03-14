# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import math
from enum import Enum, auto
from typing import Tuple

# import scipp as sc
import numpy as np

from .reading import ComponentReading, ReadingField
from .utils import two_pi


class Direction(Enum):
    CLOCKWISE = auto()
    ANTICLOCKWISE = auto()


Clockwise = Direction.CLOCKWISE
AntiClockwise = Direction.ANTICLOCKWISE


class Chopper:
    """
    A chopper is a rotating device with cutouts that blocks the beam at certain times.

    Parameters
    ----------
    frequency:
        The frequency of the chopper. Must be positive.
    distance:
        The distance from the source to the chopper.
    name:
        The name of the chopper.
    phase:
        The phase of the chopper. Because the phase offset implemented as a time delay
        on real beamline choppers, it is applied in the opposite direction
        to the chopper rotation direction. For example, if the chopper rotates
        clockwise, a phase of 10 degrees will shift all window angles by 10 degrees
        in the anticlockwise direction, which will result in the windows opening later.
    open:
        The opening angles of the chopper cutouts.
    close:
        The closing angles of the chopper cutouts.
    centers:
        The centers of the chopper cutouts.
    widths:
        The widths of the chopper cutouts.

    Notes
    -----
    Either `open` and `close` or `centers` and `widths` must be provided, but not both.
    """

    def __init__(
        self,
        *,
        frequency: float,
        distance: float,
        name: str,
        phase: float = 0.0,
        open: np.ndarray | None = None,
        close: np.ndarray | None = None,
        centers: np.ndarray | None = None,
        widths: np.ndarray | None = None,
        direction: Direction = Clockwise,
    ):
        if frequency <= 0.0:
            raise ValueError(f"Chopper frequency must be positive, got {frequency}.")
        self.frequency = float(frequency)
        if direction not in (Clockwise, AntiClockwise):
            raise ValueError(
                "Chopper direction must be Clockwise or AntiClockwise"
                f", got {direction}."
            )
        self.direction = direction
        # Check that either open/close or centers/widths are provided, but not both
        if tuple(x for x in (open, close, centers, widths) if x is not None) not in (
            (open, close),
            (centers, widths),
        ):
            raise ValueError(
                "Either open/close or centers/widths must be provided, got"
                f" open={open}, close={close}, centers={centers}, widths={widths}."
            )
        if open is None:
            half_width = widths * 0.5
            open = centers - half_width
            close = centers + half_width

        # self.open = (open if open.dims else open.flatten(to="cutout")).to(
        #     dtype=float, copy=False
        # )
        self.open = np.asarray(open).astype(float)
        self.close = np.asarray(close).astype(float)
        self.distance = float(distance)
        self.phase = float(phase)
        self.name = name
        super().__init__()

    @property
    def omega(self) -> float:
        """
        The angular velocity of the chopper.
        """
        return two_pi * self.frequency

    def open_close_times(
        self, time_limit: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The times at which the chopper opens and closes.

        Parameters
        ----------
        time_limit:
            Determines how many rotations the chopper needs to perform to reach the time
            limit. If not specified, the chopper will perform a single rotation.
            Must be in microseconds.
        """
        nrot = max(int(math.ceil(time_limit * 1.0e-6 * self.frequency)), 1)
        # Start at -1 to catch early openings in case the phase or opening angles are
        # large
        phases = (np.arange(-1, nrot) * two_pi + self.phase).reshape(-1, 1)

        open_times = np.deg2rad(self.open)
        close_times = np.deg2rad(self.close)
        # If the chopper is rotating anti-clockwise, we mirror the openings because the
        # first cutout will be the last to open.
        if self.direction == AntiClockwise:
            open_times, close_times = (
                (two_pi - close_times)[::-1],
                (two_pi - open_times)[::-1],
            )
            #     sc.array(
            #         dims=close_times.dims,
            #         values=(two_pi - close_times).values[::-1],
            #         unit=close_times.unit,
            #     ),
            #     sc.array(
            #         dims=open_times.dims,
            #         values=(two_pi - open_times).values[::-1],
            #         unit=open_times.unit,
            #     ),
            # )
        # Note that the order is important here: we need (phases + open/close) to get
        # the correct dimension order when we flatten.
        open_times = (phases + open_times).ravel() * 1.0e6 / self.omega
        close_times = (phases + close_times).ravel() * 1.0e6 / self.omega
        return open_times, close_times
        # open_times /= self.omega
        # close_times /= self.omega
        # return (
        #     open_times.to(unit=unit, copy=False),
        #     close_times.to(unit=unit, copy=False),
        # )

    def __repr__(self) -> str:
        return (
            f"Chopper(name={self.name}, distance={self.distance}m, "
            f"frequency={self.frequency}Hz, phase={self.phase}deg, "
            f"direction={self.direction.name}, cutouts={len(self.open)})"
        )

    def as_dict(self):
        return {
            "frequency": self.frequency,
            "open": self.open,
            "close": self.close,
            "distance": self.distance,
            "phase": self.phase,
            "name": self.name,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class ChopperReading(ComponentReading):
    """
    Read-only container for the neutrons that reach the chopper.
    """

    distance: float
    name: str
    frequency: float
    open: np.ndarray
    close: np.ndarray
    phase: float
    open_times: float
    close_times: float
    data: NeutronData
    toas: ReadingField
    wavelengths: ReadingField
    birth_times: ReadingField
    speeds: ReadingField

    def __repr__(self) -> str:
        out = f"ChopperReading: '{self.name}'\n"
        out += f"  distance: {self.distance}\n"
        out += f"  frequency: {self.frequency}\n"
        out += f"  phase: {self.phase}\n"
        out += f"  cutouts: {len(self.open)}\n"
        out += "\n".join(
            f"  {key}: {getattr(self, key)}"
            for key in ("toas", "wavelengths", "birth_times", "speeds")
        )
        return out

    def __str__(self) -> str:
        return self.__repr__()
