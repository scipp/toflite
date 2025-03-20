# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from .reading import ComponentReading
from .utils import NeutronData


class Detector:
    """
    A detector component does not block any neutrons, it sees all neutrons passing
    through it.

    Parameters
    ----------
    distance:
        The distance from the source to the detector.
    name:
        The name of the detector.
    """

    def __init__(self, distance: float, name: str):
        self.distance = float(distance)
        self.name = name

    def __repr__(self) -> str:
        return f"Detector(name={self.name}, distance={self.distance}m)"

    def as_dict(self):
        return {"distance": self.distance, "name": self.name}


@dataclass(frozen=True)
class DetectorReading(ComponentReading):
    """
    Read-only container for the neutrons that reach the detector.
    """

    distance: float
    name: str
    data: NeutronData

    def _repr_stats(self) -> str:
        neutrons = self.data.size
        blocked = int(self.data.blocked_by_me.sum() + self.data.blocked_by_others.sum())
        return f"visible={neutrons - blocked}"

    def __repr__(self) -> str:
        return f"""DetectorReading: '{self.name}'
  distance: {self.distance}
  neutrons: {self._repr_stats()}
"""

    def __str__(self) -> str:
        return self.__repr__()
