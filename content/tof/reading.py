# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .utils import Plot, one_mask, NeutronData


@dataclass(frozen=True)
class ReadingField:
    name: str
    values: np.ndarray
    blocked_by_me: np.ndarray
    blocked_by_others: np.ndarray
# class ReadingField:
#     data: sc.DataArray
#     dim: str

    def plot(self, bins: int = 300, **kwargs):
        fig, ax = plt.subplots()

        for i, x in enumerate(self.values):
            ax.hist(x, bins=bins, histtype="step", lw=1.5)
            # ax[1].hist(self.data.wavelength[i], bins=bins, histtype="step", lw=1.5)
        ax.set(xlabel=self.name, ylabel="Counts")
        # ax[1].set(xlabel="Wavelength [Å]", ylabel="Counts")
        # fig.set_size_inches(10, 4)
        # fig.tight_layout()
        return Plot(fig=fig, ax=ax)



        by_pulse = sc.collapse(self.data, keep="event")
        to_plot = {}
        color = {}
        for key, da in by_pulse.items():
            sel = da[~da.masks["blocked_by_others"]]
            to_plot[key] = sel.hist({self.dim: bins})
            if "blocked_by_me" in self.data.masks:
                name = f"blocked-{key}"
                to_plot[name] = (
                    da[da.masks["blocked_by_me"]]
                    .drop_masks(list(da.masks.keys()))
                    .hist({self.dim: to_plot[key].coords[self.dim]})
                )
                color[name] = "gray"
        return pp.plot(to_plot, **{**{"color": color}, **kwargs})

    def min(self):
        mask = ~one_mask(self.data.masks)
        mask.unit = ""
        return (self.data.coords[self.dim] * mask).min()

    def max(self):
        mask = ~one_mask(self.data.masks)
        mask.unit = ""
        return (self.data.coords[self.dim] * mask).max()

    # def __repr__(self) -> str:
    #     mask = ~one_mask(self.data.masks)
    #     mask.unit = ""
    #     coord = self.data.coords[self.dim] * mask
    #     return (
    #         f"{self.dim}: min={coord.min():c}, max={coord.max():c}, "
    #         f"events={int(self.data.sum().value)}"
    #     )

    # def __str__(self) -> str:
    #     return self.__repr__()

    # def __getitem__(self, val):
    #     return self.__class__(data=self.data[val], dim=self.dim)


def _make_reading_field(data: NeutronData, field: str) -> ReadingField:
    return ReadingField(
        name=field,
        values=getattr(data, field),
        blocked_by_me=data.blocked_by_me,
        blocked_by_others=data.blocked_by_others,
    )


class ComponentReading:
    """
    Data reading for a component placed in the beam path. The reading will have a
    record of the arrival times and wavelengths of the neutrons that passed through it.
    """

    @property
    def toa(self) -> ReadingField:
        return _make_reading_field(self.data, field="toa")

    @property
    def wavelength(self) -> ReadingField:
        return _make_reading_field(self.data, field="wavelength")

    @property
    def birth_time(self) -> ReadingField:
        return _make_reading_field(self.data, field="birth_time")

    @property
    def speed(self) -> ReadingField:
        return _make_reading_field(self.data, field="speed")

    def plot(self, bins: int = 300) -> Plot:
        """
        Plot both the toa and wavelength data side by side.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        return self.toa.plot(bins=bins) + self.wavelength.plot(bins=bins)
