# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

from .reading import ComponentReading, ReadingField
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
    toas: ReadingField
    wavelengths: ReadingField
    birth_times: ReadingField
    speeds: ReadingField

    def __repr__(self) -> str:
        out = f"DetectorReading: '{self.name}'\n"
        out += f"  distance: {self.distance:c}\n"
        out += "\n".join(
            f"  {key}: {getattr(self, key)}"
            for key in ("toas", "wavelengths", "birth_times", "speeds")
        )
        return out

    def __str__(self) -> str:
        return self.__repr__()
