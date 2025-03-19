# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .facilities import library as facilities
from .utils import NeutronData, Plot, wavelength_to_speed


def _make_pulses(
    neutrons: int,
    frequency: float,
    pulses: int,
    p_time: np.ndarray,
    p_wav: np.ndarray,
    sampling: int,
    seed: int | None,
    wmin: float | None = None,
    wmax: float | None = None,
):
    """
    Create pulses from time a wavelength probability distributions.
    The distributions should be supplied as DataArrays where the coordinates
    are the values of the distribution, and the values are the probability.
    Note that the time and wavelength distributions are independent. A neutron with
    a randomly selected birth time from ``p_time`` can adopt any wavelength in
    ``p_wav`` (in other words, the two distributions are simply broadcast into a
    square 2D parameter space).

    Parameters
    ----------
    neutrons:
        Number of neutrons per pulse.
    frequency:
        Pulse frequency.
    pulses:
        Number of pulses.
    p_time:
        Time probability distribution for a single pulse.
    p_wav:
        Wavelength probability distribution for a single pulse.
    sampling:
        Number of points used to sample the probability distributions.
    seed:
        Seed for the random number generator.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    """
    sampling = int(sampling)

    tmin = p_time[:, 0].min()
    tmax = p_time[:, 0].max()
    if wmin is None:
        wmin = p_wav[:, 0].min()
    if wmax is None:
        wmax = p_wav[:, 0].max()

    time_interpolator = interp1d(p_time[:, 0], p_time[:, 1], fill_value="extrapolate")
    wav_interpolator = interp1d(p_wav[:, 0], p_wav[:, 1], fill_value="extrapolate")
    x_time = np.linspace(start=tmin, stop=tmax, num=sampling)
    x_wav = np.linspace(start=wmin, stop=wmax, num=sampling)
    p_time = time_interpolator(x_time)
    p_time /= p_time.sum()
    p_wav = wav_interpolator(x_wav)
    p_wav /= p_wav.sum()

    # In the following, random.choice only allows to select from the values listed
    # in the coordinate of the probability distribution arrays. This leads to data
    # grouped into spikes and empty in between because the sampling resolution used
    # in the linear interpolation above is usually kept low for performance.
    # To make the distribution more uniform, we add some random noise to the chosen
    # values, which effectively fills in the gaps between the spikes.
    # Scipy has some methods to sample from a continuous distribution, but they are
    # prohibitively slow.
    # See https://docs.scipy.org/doc/scipy/tutorial/stats/sampling.html for more
    # information.
    dt = 0.5 * (tmax - tmin) / (len(p_time) - 1)
    dw = 0.5 * (wmax - wmin) / (len(p_wav) - 1)

    # Because of the added noise, some values end up being outside the specified range
    # for the birth times and wavelengths. Using naive clipping leads to pile-up on the
    # edges of the range. To avoid this, we remove the outliers and resample until we
    # have the desired number of neutrons.
    n = 0
    times = []
    wavs = []
    ntot = pulses * neutrons
    rng = np.random.default_rng(seed)
    while n < ntot:
        size = ntot - n
        t = rng.choice(x_time, size=size, p=p_time) + rng.normal(scale=dt, size=size)
        w = rng.choice(x_wav, size=size, p=p_wav) + rng.normal(scale=dw, size=size)
        mask = (t >= tmin) & (t <= tmax) & (w >= wmin) & (w <= wmax)
        times.append(t[mask])
        wavs.append(w[mask])
        n += mask.sum()

    birth_time = np.array(np.concatenate(times)).reshape(pulses, neutrons) + (
        np.arange(pulses).reshape(-1, 1) * 1.0e6 / frequency
    )

    wavelength = np.array(np.concatenate(wavs)).reshape(pulses, neutrons)
    speed = wavelength_to_speed(wavelength)
    return {"birth_time": birth_time, "wavelength": wavelength, "speed": speed}


class Source:
    """
    A class that represents a source of neutrons.
    It is defined by the number of neutrons, a wavelength range, and a time range.
    The default way of creating a pulse is to supply the name of a facility
    (e.g. ``'ess'``) and the number of neutrons. This will create a pulse with the
    default time and wavelength ranges for that facility.

    Parameters
    ----------
    facility:
        Name of a pre-defined pulse shape from a neutron facility.
    neutrons:
        Number of neutrons per pulse.
    pulses:
        Number of pulses.
    sampling:
        Number of points used to interpolate the probability distributions.
    wmin:
        Minimum neutron wavelength.
    wmax:
        Maximum neutron wavelength.
    seed:
        Seed for the random number generator.
    """

    def __init__(
        self,
        facility: str,
        neutrons: int = 1_000_000,
        pulses: int = 1,
        sampling: int = 1000,
        wmin: float | None = None,
        wmax: float | None = None,
        seed: int | None = None,
    ):
        self.facility = facility
        self.neutrons = int(neutrons)
        self.pulses = int(pulses)
        self.data = None

        if facility is not None:
            facility_params = facilities[self.facility]
            self.frequency = facility_params.frequency
            pulse_params = _make_pulses(
                neutrons=self.neutrons,
                p_time=facility_params.birth_time,
                p_wav=facility_params.wavelength,
                sampling=sampling,
                frequency=self.frequency,
                pulses=pulses,
                wmin=wmin,
                wmax=wmax,
                seed=seed,
            )
            ntot = self.neutrons * self.pulses
            self.data = NeutronData(
                distance=0,
                id=np.arange(ntot).reshape(self.pulses, self.neutrons),
                speed=pulse_params["speed"],
                birth_time=pulse_params["birth_time"],
                toa=np.zeros(ntot).reshape(self.pulses, self.neutrons),
                wavelength=pulse_params["wavelength"],
                blocked_by_me=np.zeros(ntot, dtype=bool).reshape(
                    self.pulses, self.neutrons
                ),
                blocked_by_others=np.zeros(ntot, dtype=bool).reshape(
                    self.pulses, self.neutrons
                ),
            )

    def __len__(self) -> int:
        return self.data.sizes["pulse"]

    def plot(self, bins: int = 300) -> tuple:
        """
        Plot the pulses of the source.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)

        for i in range(self.pulses):
            ax[0].hist(self.data.birth_time[i], bins=bins, histtype="step", lw=1.5)
            ax[1].hist(self.data.wavelength[i], bins=bins, histtype="step", lw=1.5)
        ax[0].set(xlabel="Time [μs]", ylabel="Counts")
        ax[1].set(xlabel="Wavelength [Å]", ylabel="Counts")
        fig.set_size_inches(10, 4)
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def as_readonly(self):
        return SourceParameters(
            data=self.data,
            facility=self.facility,
            neutrons=self.neutrons,
            frequency=self.frequency,
            pulses=self.pulses,
        )

    def __repr__(self) -> str:
        return (
            f"Source:\n"
            f"  pulses={self.pulses}, neutrons per pulse={self.neutrons}\n"
            f"  frequency={self.frequency}Hz\n  facility='{self.facility}'"
        )


@dataclass(frozen=True)
class SourceParameters:
    """
    Read-only container for the parameters of a source.
    """

    data: NeutronData
    facility: str | None
    neutrons: int
    frequency: float
    pulses: int
