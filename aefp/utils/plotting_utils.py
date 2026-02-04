import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, MultipleLocator, LogLocator, ScalarFormatter

mpl.rcParams.update({
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
})


LABELS = {
    "ae": "Autoencoder",
    "ae_ft": "Autoencoder (fine-tuned)",
    "cont": "Contrastive encoder",
    "aec": "Functional connectivity",
    "psd": "Spectral",
}


def style_axes(ax, 
               n_major=5, n_minor=4, 
               x_base=None, y_base=None, 
               logx=False, logy=False, 
               sci=False, sci_pow=(-2, 3)):
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
    else:
        if x_base is not None:
            ax.xaxis.set_major_locator(MultipleLocator(x_base))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(n_major))
        ax.xaxis.set_minor_locator(MultipleLocator((ax.xaxis.get_majorticklocs()[1]-
                                                    ax.xaxis.get_majorticklocs()[0])/(n_minor+1)
                                                   if len(ax.xaxis.get_majorticklocs())>1 else 1))
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
    else:
        if y_base is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_base))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(n_major))
        ax.yaxis.set_minor_locator(MultipleLocator((ax.yaxis.get_majorticklocs()[1]-
                                                    ax.yaxis.get_majorticklocs()[0])/(n_minor+1)
                                                   if len(ax.yaxis.get_majorticklocs())>1 else 1))
    if sci:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits(sci_pow)  # e.g., (-2, 3)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="both", style="sci", scilimits=sci_pow, useMathText=True)
    ax.tick_params(which="both", direction="out")
