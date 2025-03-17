# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
# import plopp as pp
# import scipp as sc

from .utils import Plot, NeutronData


@dataclass(frozen=True)
class ReadingData:
    """
    A data object contains the data (visible or blocked) for a component data
    (e.g. time-of-flight or wavelengths).

    Parameters
    ----------
    data:
        The data to hold.
    dim:
        The dimension label of the data.
    """

    data: dict[str, NeutronData]
    dim: str

    def __len__(self) -> int:
        """
        The number of pulses in the data.
        """
        return len(self.data)

    def __getitem__(self, val: int | slice):
        """
        Get the data for a single pulse or a range of pulses.

        Parameters
        ----------
        val:
            The index or slice of the pulse(s) to get.
        """
        if isinstance(val, int):
            val = slice(val, val + 1)
        # Convert the slice to a list of indices, which can then be used to create
        # DataGroup keys in the form 'pulse:0', 'pulse:1', etc.
        inds = range(len(self))[val]
        return self.__class__(
            data={f"pulse:{ind}": self.data[f"pulse:{ind}"] for ind in inds},
            dim=self.dim,
        )

    def plot(self, bins: int | np.ndarray = 300, **kwargs):
        """
        Plot the neutrons that reach the component as a histogram.

        Parameters
        ----------
        bins:
            The bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots()
        ax[0].hist(getattr(self.data, self.dim), bins=bins, histtype="step", lw=1.5)
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

        return pp.plot(
            {
                key: da.hist({self.dim: bins})
                for key, da in self.data.items()
                if da.size > 0
            },
            **kwargs,
        )

    def __repr__(self) -> str:
        out = f"ReadingData(dim='{self.dim}')\n"
        for name, da in self.data.items():
            out += f"  {name}: events={len(da)}"
            if len(da) > 0:
                coord = da.coords[self.dim]
                out += f", min={coord.min():c}, max={coord.max():c})"
            out += "\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()


def _field_to_string(field: ReadingData) -> str:
    if isinstance(field.data, sc.DataArray):
        return str(len(field))
    data = field.data
    out = [str(len(data["pulse:0"]))]
    npulses = len(data)
    if npulses > 2:
        out.append("...")
    if npulses > 1:
        out.append(str(len(data[f"pulse:{npulses - 1}"])))
    return "[" + ", ".join(out) + "]"


@dataclass(frozen=True)
class ReadingField:
    """
    Contains the data for the visible neutrons of a given field.
    Possible fields are ``toas``, ``wavelengths``, ``birth_times``, and ``speeds``.
    In the case of a :class:`Chopper`, this also contains the data for the blocked
    neutrons.

    Parameters
    ----------
    visible:
        The visible neutrons (those that are not blocked by the component).
    blocked:
        The blocked neutrons (those that are blocked by the component).
    """

    visible: ReadingData
    blocked: ReadingData | None = None

    @property
    def data(self) -> dict:
        """
        The neutrons that reach the component, split up into those that are blocked by
        the component and those that are not.
        """
        out = {"visible": self.visible.data}
        if self.blocked is not None:
            out["blocked"] = self.blocked.data
        return out

    def __getitem__(self, val):
        return self.__class__(
            visible=self.visible[val],
            blocked=self.blocked[val] if self.blocked is not None else None,
        )

    def _repr_string_body(self) -> str:
        out = f"visible={_field_to_string(self.visible)}"
        if self.blocked is not None:
            out += f", blocked={_field_to_string(self.blocked)}"
        return out

    def __repr__(self) -> str:
        return f"ReadingData(dim='{self.visible.dim}', {self._repr_string_body()})"

    def plot(self, bins: int | np.ndarray = 300, **kwargs):
        """
        Plot the data for the neutrons that reach the component, split up into those
        that are blocked by the component and those that are not.

        Parameters
        ----------
        bins:
            The bins to use for histogramming the neutrons.
        """
        if self.blocked is None:
            return self.visible.plot(bins=bins, **kwargs)
        visible = self.visible.data
        blocked = self.blocked.data
        if isinstance(visible, NeutronData):
            visible = {"onepulse": visible}
            blocked = {"onepulse": blocked}
        dim = self.visible.dim
        to_plot = {}
        colors = {}
        edges = bins
        for i, p in enumerate(visible):
            if isinstance(bins, int):
                edges = np.linspace(
                    # dim=dim,
                    start=min(
                        getattr(visible[p], dim).min(), getattr(blocked[p], dim).min()
                    ),
                    stop=max(
                        getattr(visible[p], dim).max(), getattr(blocked[p], dim).max()
                    ),
                    num=bins,
                    # unit=visible[p].coords[dim].unit,
                )
            vk = f"visible-{p}"
            bk = f"blocked-{p}"
            to_plot.update(
                {vk: visible[p].hist({dim: edges}), bk: blocked[p].hist({dim: edges})}
            )
            colors.update({vk: f"C{i}", bk: "gray"})
        fig, ax = plt.subplots()
        for name, array in to_plot.items():
            ax.hist(array, bins=bins, histtype="step", label=name, color=colors[name])
        # out = pp.plot(
        #     to_plot,
        #     **{**{"color": colors}, **kwargs},
        # )
        return Plot(fig=fig, ax=ax)


class ComponentReading:
    """
    Data reading for a component placed in the beam path. The reading will have a
    record of the arrival times and wavelengths of the neutrons that passed through it.
    """

    def plot(self, bins: int = 300) -> Plot:
        """
        Plot both the toa and wavelength data side by side.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)
        self.toas.plot(bins=bins, ax=ax[0])
        self.wavelengths.plot(bins=bins, ax=ax[1])
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)
