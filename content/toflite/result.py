# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .chopper import ChopperReading
from .detector import DetectorReading
from .source import Source
from .utils import Plot


def _add_rays(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray | str,
    cbar: bool = True,
    cmap: str = "gist_rainbow_r",
    vmin: float | None = None,
    vmax: float | None = None,
    cax: plt.Axes | None = None,
    zorder: int = 1,
):
    coll = LineCollection(np.stack((x, y), axis=2), zorder=zorder)
    if isinstance(color, str):
        coll.set_color(color)
    else:
        coll.set_cmap(plt.colormaps[cmap])
        coll.set_array(color)
        coll.set_norm(plt.Normalize(vmin, vmax))
        if cbar:
            cb = plt.colorbar(coll, ax=ax, cax=cax)
            cb.ax.yaxis.set_label_coords(-0.9, 0.5)
            cb.set_label("Wavelength [Å]")
    ax.add_collection(coll)


class Result:
    def __init__(self, source: Source, choppers: dict, detectors: dict):
        self.source = source.as_readonly()
        self.choppers = {}
        for name, chopper in choppers.items():
            self.choppers[name] = ChopperReading(
                distance=chopper["distance"],
                name=chopper["name"],
                frequency=chopper["frequency"],
                open=chopper["open"],
                close=chopper["close"],
                phase=chopper["phase"],
                open_times=chopper["open_times"],
                close_times=chopper["close_times"],
                data=chopper["data"],
            )

        self.detectors = {}
        for name, det in detectors.items():
            self.detectors[name] = DetectorReading(
                distance=det["distance"], name=det["name"], data=det["data"]
            )

    def plot(
        self,
        visible_rays: int = 1000,
        blocked_rays: int = 0,
        figsize: Tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
        cax: plt.Axes | None = None,
        cbar: bool = True,
        cmap: str = "gist_rainbow_r",
    ) -> Plot:
        """
        Plot the time-distance diagram for the instrument, including the rays of
        neutrons that make it to the furthest detector.
        As plotting many lines can be slow, the number of rays to plot can be
        limited by setting ``max_rays``.
        In addition, it is possible to also plot the rays that are blocked by
        choppers along the flight path by setting ``blocked_rays > 0``.

        Parameters
        ----------
        visible_rays:
            Maximum number of rays to plot.
        blocked_rays:
            Number of blocked rays to plot.
        figsize:
            Figure size.
        ax:
            Axes to plot on.
        cax:
            Axes to use for the colorbar.
        cbar:
            Show a colorbar for the wavelength if ``True``.
        cmap:
            Colormap to use for the wavelength colorbar.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        components = sorted(
            chain(self.choppers.values(), self.detectors.values()),
            key=lambda c: c.distance,
        )
        furthest_component = components[-1]

        repeats = [1] + [2] * len(components)

        mask = (
            furthest_component.data.blocked_by_me
            | furthest_component.data.blocked_by_others
        )
        wavelengths = furthest_component.data.wavelength[~mask]
        vmin, vmax = wavelengths.min(), wavelengths.max()

        for i in range(self.source.pulses):
            ids = np.arange(self.source.neutrons)
            # Plot visible rays
            blocked = (
                furthest_component.data.blocked_by_me[i]
                | furthest_component.data.blocked_by_others[i]
            )
            nblocked = blocked.sum()
            inds = np.random.choice(
                ids[~blocked],
                size=min(self.source.neutrons - nblocked, visible_rays),
                replace=False,
            )

            xstart = self.source.data.birth_time[i][inds]
            xend = furthest_component.data.toa[i][inds]
            ystart = np.zeros_like(xstart)
            yend = np.full_like(ystart, furthest_component.distance)

            _add_rays(
                ax=ax,
                x=np.stack((xstart, xend), axis=1),
                y=np.stack((ystart, yend), axis=1),
                color=self.source.data.wavelength[i][inds],
                cbar=cbar and (i == 0),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cax=cax,
            )

            # Plot blocked rays
            inds = np.random.choice(
                ids[blocked], size=min(blocked_rays, nblocked), replace=False
            )
            x = np.repeat(
                np.stack(
                    [self.source.data.birth_time[i][inds]]
                    + [c.data.toa[i][inds] for c in components],
                    axis=1,
                ),
                repeats,
                axis=1,
            )
            y = np.repeat(
                np.stack(
                    [np.zeros_like(x[:, 0])]
                    + [np.full_like(x[:, 0], c.distance) for c in components],
                    axis=1,
                ),
                repeats,
                axis=1,
            )
            for j, c in enumerate(components):
                m_others = c.data.blocked_by_others[i][inds]
                x[:, 2 * j + 1][m_others] = np.nan
                y[:, 2 * j + 1][m_others] = np.nan
                m_me = c.data.blocked_by_me[i][inds]
                x[:, 2 * j + 2][m_me] = np.nan
                y[:, 2 * j + 2][m_me] = np.nan
            _add_rays(ax=ax, x=x, y=y, color="lightgray", zorder=-1)

            # Plot pulse
            time_coord = self.source.data.birth_time[i]
            tmin = time_coord.min()
            ax.plot([tmin, time_coord.max()], [0, 0], color="gray", lw=3)
            ax.text(tmin, 0, "Pulse", ha="left", va="top", color="gray")

        data = furthest_component.data
        if data.blocked_by_me.sum() + data.blocked_by_others.sum() == data.size:
            toa_max = data.toa.max()
        else:
            toa_max = furthest_component.toa.max()
        dx = 0.05 * toa_max
        # Plot choppers
        for ch in self.choppers.values():
            x0 = ch.open_times
            x1 = ch.close_times
            x = np.empty(3 * x0.size, dtype=x0.dtype)
            x[0::3] = x0
            x[1::3] = 0.5 * (x0 + x1)
            x[2::3] = x1
            x = np.concatenate(
                ([[0]] if x[0] > 0 else [x[0:1]])
                + [x]
                + ([[toa_max + dx]] if x[-1] < toa_max else [])
            )
            y = np.full_like(x, ch.distance)
            y[2::3] = None
            inds = np.argsort(x)
            ax.plot(x[inds], y[inds], color="k")
            ax.text(toa_max, ch.distance, ch.name, ha="right", va="bottom", color="k")

        # Plot detectors
        for det in self.detectors.values():
            ax.plot([0, toa_max], [det.distance] * 2, color="gray", lw=3)
            ax.text(0, det.distance, det.name, ha="left", va="bottom", color="gray")

        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Distance [m]")
        ax.set_xlim(0 - dx, toa_max + dx)
        if figsize is None:
            inches = fig.get_size_inches()
            fig.set_size_inches((min(inches[0] * self.source.pulses, 12.0), inches[1]))
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def __repr__(self) -> str:
        out = (
            f"Result:\n  Source: {self.source.pulses} pulses, "
            f"{self.source.neutrons} neutrons per pulse.\n  Choppers:\n"
        )
        for name, ch in self.choppers.items():
            out += f"    {name}: {ch._repr_stats()}\n"
        out += "  Detectors:\n"
        for name, det in self.detectors.items():
            out += f"    {name}: {det._repr_stats()}\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
