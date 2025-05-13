import matplotlib.pyplot as plt
import numpy as np

from bmtool.analysis.lfp import gen_aperiodic


def plot_spectrogram(
    sxx_xarray,
    remove_aperiodic=None,
    log_power=False,
    plt_range=None,
    clr_freq_range=None,
    pad=0.03,
    ax=None,
):
    """Plot spectrogram. Determine color limits using value in frequency band clr_freq_range"""
    sxx = sxx_xarray.PSD.values.copy()
    t = sxx_xarray.time.values.copy()
    f = sxx_xarray.frequency.values.copy()

    cbar_label = "PSD" if remove_aperiodic is None else "PSD Residual"
    if log_power:
        with np.errstate(divide="ignore"):
            sxx = np.log10(sxx)
        cbar_label += " dB" if log_power == "dB" else " log(power)"

    if remove_aperiodic is not None:
        f1_idx = 0 if f[0] else 1
        ap_fit = gen_aperiodic(f[f1_idx:], remove_aperiodic.aperiodic_params)
        sxx[f1_idx:, :] -= (ap_fit if log_power else 10**ap_fit)[:, None]
        sxx[:f1_idx, :] = 0.0

    if log_power == "dB":
        sxx *= 10

    if ax is None:
        _, ax = plt.subplots(1, 1)
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [f[0 if f[0] else 1] if log_power else 0.0, plt_range.item()]
    f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
    if clr_freq_range is None:
        vmin, vmax = None, None
    else:
        c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
        vmin, vmax = sxx[c_idx, :].min(), sxx[c_idx, :].max()

    f = f[f_idx]
    pcm = ax.pcolormesh(t, f, sxx[f_idx, :], shading="gouraud", vmin=vmin, vmax=vmax)
    if "cone_of_influence_frequency" in sxx_xarray:
        coif = sxx_xarray.cone_of_influence_frequency
        ax.plot(t, coif)
        ax.fill_between(t, coif, step="mid", alpha=0.2)
    ax.set_xlim(t[0], t[-1])
    # ax.set_xlim(t[0],0.2)
    ax.set_ylim(f[0], f[-1])
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label, pad=pad)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    return sxx
