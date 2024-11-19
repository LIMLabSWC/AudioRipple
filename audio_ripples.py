import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.io import wavfile


a0 = 1e-5  # reference amplitude
sr = 44100  # sample rate


def ripple_sound(dur, n, omega, w, delta, phi, f0, fm1, l=70):
    """Synthesizes a ripple sound.

    Args:
        dur (float): Duration of sound in s.
        n (int): Number of sinusoids.
        omega (:obj:`float` or `array`-like): Ripple density in Hz. Must be a single
            value or an array with length `duration * sr`.
        w (:obj:`float` or `array`-like): Ripple drift in Hz. Must be a single
            value or an array with length `duration * sr`.
        delta (:obj:`float` or `array`-like): Normalized ripple depth. Must be a single
            value or an array with length `duration * sr`. Value(s) must be in
            the range [0, 1].
        phi (float): Ripple starting phase in radians.
        f0 (float): Frequency of the lowest sinusoid in Hz.
        fm1 (float): Frequency of the highest sinusoid in Hz.
        

    Returns:
        y (np.array): The waveform.
        a (np.array): The envelope (useful for plotting).

    """
    # create sinusoids
    m = int(dur * sr)  # total number of samples
    shapea = (1, m)
    shapeb = (n, 1)
    t = np.linspace(0, dur, int(m)).reshape(shapea)
    i = np.arange(n).reshape(shapeb)
    f = f0 * (fm1 / f0) ** (i / (n - 1))
    sphi = 2 * np.pi * np.random.random(shapeb)
    s = np.sin(2 * np.pi * f * t + sphi)

    # create envelope
    x = np.log2(f / f0)
    if hasattr(w, "__iter__"):
        wprime = np.cumsum(w) / sr
    else:
        wprime = w * t
    a = 1 + delta * np.sin(2 * np.pi * (wprime + omega * x) + phi)

    # create the waveform
    y = (a * s / np.sqrt(f)).sum(axis=0)

    return y, a


def plot_env(a, ax, labels=False):
    """Plots an envelope onto an axis.

    Args:
        a (np.array): An array with shape (m, n) where m is the total number of samples
            and n in the number of sinusoids and values representing instantaneous
            amplitudes.
        ax (matplotlib.axes._subplots.AxesSubplot): Axis.
        labels (:obj:`bool`, optional): Include labels or not.

    """
    ax.pcolormesh(a, rasterized=True, vmin=0, vmax=2)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    if labels is True:
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency\n($log_2$ scale)")


def main():
    """Create and plot some ripple sounds.

    """
    from matplotlib import rcParams

    figsize = rcParams["figure.figsize"]
    rcParams["figure.figsize"] = [figsize[0], int(figsize[1] / 2)]
    rcParams["font.size"] = 14

    # default parameter values
    np.random.seed(0)
    dur = 0.01
    n = 100  # reduce this if you want to make figures; otherwise it takes forever!
    omega = 0
    w = 8
    delta = 1
    f0 = 256
    fm1 = 4096
    phi = np.pi/2
    args = (phi, f0, fm1)

    # filenames of figures
    fn = "demo_%s_ripples.png"

    # moving ripple sounds
    print("making moving ripple sounds")
    _, axes = plt.subplots(1, 6, constrained_layout=True, sharex="all", sharey="all",figsize=(30,5))
    _ws = [7.5, 4,0,0,4,7.5]
    omega = [0, 0.5,2, -2, -0.5, 0]
    for i, ax in enumerate(axes):
        _omega = omega[i]
        _w = _ws[i]
        _delta = delta
        print(f"sound with omega={_omega:.2f}, w={_w:.2f}, and delta={_delta:.2f}")
        y, a = ripple_sound(dur, n, _omega, _w, _delta, *args)
        print("plotting")
        plot_env(a, ax, ax == axes[0])
        axes[i].set_title(f"w = {_ws[i]}, omega = {omega[i]}")
    print("making a figure")
    plt.savefig(fn % "short_moving_sweep", bbox_inches=0, transparent=True)



if __name__ == "__main__":
    main()
