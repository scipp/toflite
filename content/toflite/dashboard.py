# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import ipywidgets as ipw

from itertools import chain
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .source import Source
from .chopper import Chopper
from .detector import Detector
from .model import Model


class ChopperWidget(ipw.VBox):
    def __init__(self):
        self.frequency_widget = ipw.FloatText(description="Frequency")
        self.open_widget = ipw.Text(description="Open")
        self.close_widget = ipw.Text(description="Close")
        self.phase_widget = ipw.FloatText(description="Phase")
        self.distance_widget = ipw.FloatText(description="Distance")
        self.name_widget = ipw.Text(description="Name")
        super().__init__(
            [
                self.name_widget,
                self.frequency_widget,
                self.open_widget,
                self.close_widget,
                self.phase_widget,
                self.distance_widget,
            ]
        )


class DetectorWidget(ipw.VBox):
    def __init__(self):
        self.distance_widget = ipw.FloatText(description="Distance")
        self.name_widget = ipw.Text(description="Name")
        super().__init__([self.name_widget, self.distance_widget])


class SourceWidget(ipw.VBox):
    def __init__(self):
        self.facility_widget = ipw.Dropdown(options=["ess"], description="Facility")
        self.neutrons_widget = ipw.IntText(value=100_000, description="Neutrons")
        self.pulses_widget = ipw.IntText(value=1, description="Pulses")
        super().__init__(
            [self.facility_widget, self.neutrons_widget, self.pulses_widget]
        )


class TofWidget:
    def __init__(self):
        self.top_bar = ipw.HBox()

        self.run_button = ipw.Button(description="Run")
        with plt.ioff():
            self.time_distance_fig, self.time_distance_ax = plt.subplots(figsize=(8, 6))
            divider = make_axes_locatable(self.time_distance_ax)
            self.time_distance_cax = divider.append_axes("right", "4%", pad="5%")
            self.toa_wav_fig, self.toa_wav_ax = plt.subplots(1, 2, figsize=(11.5, 3.75))
            self.time_distance_fig.canvas.header_visible = False
            self.toa_wav_fig.canvas.header_visible = False

        self.source_widget = SourceWidget()

        self.choppers_widget = ipw.VBox([ipw.Button(description="Add chopper")])

        self.detectors_widget = ipw.VBox([ipw.Button(description="Add detector")])

        tab_contents = ["Source", "Choppers", "Detectors"]
        children = [self.source_widget, self.choppers_widget, self.detectors_widget]
        self.tab = ipw.Tab()
        self.tab.children = children
        self.tab.titles = tab_contents
        self.top_bar.children = [
            ipw.VBox([self.run_button, self.time_distance_fig.canvas]),
            self.tab,
        ]

        self.choppers_widget.children[0].on_click(self.add_chopper)
        self.detectors_widget.children[0].on_click(self.add_detector)
        self.run_button.on_click(self.run)

        self.main_widget = ipw.VBox([self.top_bar, self.toa_wav_fig.canvas])

    def add_chopper(self, _):
        children = list(self.choppers_widget.children)
        children.append(ChopperWidget())
        self.choppers_widget.children = children

    def add_detector(self, _):
        children = list(self.detectors_widget.children)
        children.append(DetectorWidget())
        self.detectors_widget.children = children

    def run(self, _):
        source = Source(
            facility=self.source_widget.facility_widget.value,
            neutrons=int(self.source_widget.neutrons_widget.value),
            pulses=int(self.source_widget.pulses_widget.value),
        )
        choppers = [
            Chopper(
                frequency=ch.frequency_widget.value,
                open=[float(v) for v in ch.open_widget.value.split(",")],
                close=[float(v) for v in ch.close_widget.value.split(",")],
                phase=ch.phase_widget.value,
                distance=ch.distance_widget.value,
                name=ch.name_widget.value,
            )
            for ch in self.choppers_widget.children[1:]
        ]

        detectors = [
            Detector(distance=det.distance_widget.value, name=det.name_widget.value)
            for det in self.detectors_widget.children[1:]
        ]

        model = Model(source=source, choppers=choppers, detectors=detectors)
        self.results = model.run()
        self.time_distance_ax.clear()
        self.time_distance_cax.clear()
        self.results.plot(ax=self.time_distance_ax, cax=self.time_distance_cax)

        components = sorted(
            chain(self.results.choppers.values(), self.results.detectors.values()),
            key=lambda c: c.distance,
        )

        self.toa_wav_ax[0].clear()
        self.toa_wav_ax[1].clear()
        for p in range(source.pulses):
            for i, c in enumerate(components):
                label = c.name if p == 0 else None
                toa = c.data.toa[p]
                wav = c.data.wavelength[p]
                sel = ~(c.data.blocked_by_me[p] | c.data.blocked_by_others[p])
                self.toa_wav_ax[0].hist(
                    toa[sel],
                    bins=300,
                    histtype="step",
                    lw=1.5,
                    color=f"C{i}",
                    label=label,
                )
                self.toa_wav_ax[1].hist(
                    wav[sel],
                    bins=300,
                    histtype="step",
                    lw=1.5,
                    color=f"C{i}",
                    label=label,
                )

        self.toa_wav_ax[0].set(xlabel="Time-of-arrival [μs]", ylabel="Counts")
        self.toa_wav_ax[1].set(xlabel="Wavelength [Å]", ylabel="Counts")
        self.toa_wav_ax[0].legend()
        self.toa_wav_ax[1].legend()
        self.time_distance_fig.tight_layout()
        self.toa_wav_fig.tight_layout()


def app():
    return TofWidget().main_widget
