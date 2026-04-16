"""
Plotting functionalities for flowsheet and column models
and related objects.
"""


import colorsys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PLOT_FONT_SIZE = 10
DEFAULT_MPL_RC_PARAMS = {
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 10 / 72,
    "figure.constrained_layout.h_pad": 8 / 72,
    "font.size": DEFAULT_PLOT_FONT_SIZE,
    "axes.facecolor": "none",
    "axes.linewidth": 0.82,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelcolor": "inherit",
    "ytick.labelcolor": "inherit",
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "lightgray",
    "grid.linewidth": 0.82,
    "lines.linewidth": 1.5,
    "lines.markersize": 6.5,
    "hatch.linewidth": 0.9,
    "font.family": "sans-serif",
    "text.usetex": True,
    "text.latex.preamble": "\n".join([
        r'\usepackage{amsmath}',
        r'\renewcommand{\familydefault}{\sfdefault}',
        r'\usepackage[scaled]{helvet}',
        r'\usepackage[helvet]{sfmath}',
        r'\everymath={\sf}',
    ]),
    "legend.facecolor": "white",
    "axes.prop_cycle": mpl.cycler(
        color=[
            "tab:blue",
            "orange",
            "mediumseagreen",
            "firebrick",
            "darkviolet",
            "saddlebrown",
            "y",
            "c",
            "gray",
            "deeppink",
            "gold",
            "purple",
            "darkcyan",
            "chocolate",
            "mediumslateblue",
            "olive",
            "steelblue",
            "dodgerblue",
            "lightsalmon",
            "darkseagreen",
        ],
        marker=[
            "o", "^", "s", "D", "v", "<", ">", "1", "2", "3",
            "4", "8", "p", "P", "*", "h", "H", "+", "X", "$f$",
        ]
    ),
}
AX_LABEL_TEXTWIDTH = 24


def set_lightness(rgb, lightness):
    """
    Adjust HSL lightness of an RGB color.

    Parameters
    ----------
    rgb : 3-tuple of float
        RGB color tuple.
    lightness : float
        Lightness to which to set the color.

    Returns
    -------
    3-tuple of float
        RGB of adjusted color.
    """
    hue, _, sat = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(hue, min(1, lightness), s=sat)


def locate_nonoverlapping_fixed_xticks(locs, fig, ax):
    """
    Locate visually nonoverlapping fixed x-axis ticks from among
    a sequence of candidates. Prioritize integer values.

    May still need more work: prioritize ints, but
    aim to make ticks as close to evenly spaced as possible.
    """
    locs = np.array(locs)

    # get axes size, in pt (72 pt is roughly 1 inch)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axwidth = 72 * bbox.width

    # get xtick label size
    xta = ax.get_xticklabels()
    xtick_label_size = xta[0].get_fontsize()

    # get minimum value between ticks,
    # based on tick label size and axes size
    min_gap = (
        # this formula is based on experience,
        # rather than on a carefully detailed calculation.
        # adjust the coefficient '1.7' as needed, or consider
        # a different formula
        1.7 * (max(locs) - min(locs)) * xtick_label_size / axwidth
    )

    int_locs = locs[locs == locs.astype(int)]

    ticks = []
    for idx, loc in enumerate(locs):
        if idx == 0:
            # include first loc
            ticks.append(loc)
        elif idx == len(locs) - 1 and loc - locs[0] >= min_gap:
            # include last loc, provided it's far enough from
            # first loc
            ticks.append(loc)
        elif loc - ticks[-1] >= min_gap and locs[-1] - loc >= min_gap:
            # check: is there an integer loc nearby?
            #        we want to prioritize that over this.
            next_int_loc = next(iter(int_locs[int_locs > loc]), None)
            deprioritize_loc = (
                loc not in int_locs
                and next_int_loc is not None
                and next_int_loc - loc < min_gap
            )
            if not deprioritize_loc:
                # check: is there a loc closer to the midpoint
                # between the last ticked loc and the next integer
                # that would be ticked if not for this loc?
                if next_int_loc is not None and loc not in int_locs:
                    midpoint_to_next_int = (ticks[-1] + next_int_loc) / 2
                    locs_to_next_int = locs[
                        np.logical_and(locs > loc, locs < next_int_loc)
                    ]
                    if locs_to_next_int.size > 0:
                        next_loc_nearest_midpoint = locs_to_next_int[
                            np.argmin(
                                abs(locs_to_next_int - midpoint_to_next_int)
                            )
                        ]
                        is_next_loc_closer = (
                            abs(
                                next_loc_nearest_midpoint
                                - midpoint_to_next_int
                            )
                            < abs(loc - midpoint_to_next_int)
                        )
                        if is_next_loc_closer:
                            continue

                # ensure all ticked locs are far enough from each
                # other
                ticks.append(loc)

    return ticks


def set_nonoverlapping_fixed_xticks(fig, ax, locs):
    """
    Set x-axis ticks to visually nonoverlapping locations.
    Useful for plots in which CO2 capture target or ellipsoidal
    confidence level is the x-axis quantity.
    """
    def format_xtick(x, pos):
        return f"{int(x)}" if x == int(x) else f"{x:.1f}"

    ax.set_xticks(
        locate_nonoverlapping_fixed_xticks(
            locs=np.round(locs, 1),
            fig=fig,
            ax=ax,
        )
    )
    ax.xaxis.set_major_formatter(format_xtick)


def _example():
    """
    Example.
    """
    import numpy as np
    ax1_locs = np.array([0, 20, 40, 60, 80, 90, 95, 99])
    ax2_locs = np.array([90, 92.5, 95, 96, 97, 98, 98.8, 99, 99.3])

    with mpl.rc_context(DEFAULT_MPL_RC_PARAMS):
        fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=200, figsize=(7, 4))

        import numpy as np

        for idx, val in enumerate(np.linspace(0.5, 1, 4)):
            ax1.plot(
                ax1_locs,
                np.array(ax1_locs) ** val,
                label=rf"$y=x^{{{val:.2f}}}$",
            )
            ax2.plot(
                ax2_locs,
                (
                    500 ** (idx * 0.1 + ax2_locs - 99)
                    - 0.3 * (ax2_locs - 90 - idx)
                    + (40 + 1 * idx)
                ),
                label=rf"$y=x^{{{val:.2f}}}$",
            )

        ax1.set_xlabel("X-Axis 1")
        ax2.set_xlabel("X-Axis 2")
        ax1.set_ylabel("Y-Axis 1")
        ax2.set_ylabel("Y-Axis 2")

        ax1.legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncols=2)
        ax2.legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncols=2)

        set_nonoverlapping_fixed_xticks(fig, ax1, ax1_locs)
        set_nonoverlapping_fixed_xticks(fig, ax2, ax2_locs)

        plt.show()


def wrap_quantity_str(namestr, unitstr=None, width=AX_LABEL_TEXTWIDTH):
    """
    Wrap string denoting a quantity and its units.

    Parameters
    ----------
    namestr : str
        Description/name of the quantity.
    unitstr : str, optional
        Quantity units.
    width : int, optional
        Wrap width of the string.

    Returns
    -------
    wrapped_str : str
        Wrapped string
    """
    import textwrap

    def simplify_word(word):
        chars_to_remove = list("^_{}$\\")
        seqs_to_remove = ["\\text", "\\mathrm", "\\."]
        for to_remove in seqs_to_remove + chars_to_remove:
            word = word.replace(to_remove, "")
        return word

    namestr_single_line = namestr.replace("\n", " ")
    namestr_word_map = {
        simplify_word(word): word for word in namestr_single_line.split(" ")
    }
    namestr_word_order = [
        (word, simplify_word(word)) for word in namestr_single_line.split(" ")
    ]

    namestr_simp = " ".join(simp_word for _, simp_word in namestr_word_order)
    unitstr_simp = (
        f" ({simplify_word(unitstr)})" if unitstr is not None else ""
    )
    fullstr_simp = f"{namestr_simp}{unitstr_simp}"

    namestr_simp_wrapped = textwrap.fill(namestr_simp, width=width)
    fullstr_simp_wrapped = textwrap.fill(fullstr_simp, width=width)

    namestr_simp_wrapped_lines = namestr_simp_wrapped.split("\n")
    namestr_wrapped_lines = []
    for simp_line in namestr_simp_wrapped_lines:
        line = " ".join(
            namestr_word_map[simp_word] for simp_word in simp_line.split(" ")
        )
        namestr_wrapped_lines.append(line)
    namestr_wrapped = "\n".join(namestr_wrapped_lines)

    put_unit_str_on_newline = (
        fullstr_simp_wrapped
        != f"{namestr_simp_wrapped}{unitstr_simp}"
    )
    unitstr_final = f"({unitstr})" if unitstr is not None else ""
    if put_unit_str_on_newline:
        wrapped_str = f"{namestr_wrapped}\n{unitstr_final}"
    else:
        wrapped_str = (
            f"{namestr_wrapped}{' ' * (len(unitstr_final) > 0)}{unitstr_final}"
        )

    return wrapped_str


def heatmap(data, row_labels=None, col_labels=None, ax=None,
            xlabel=None, ylabel=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adapted from Matplotlib annotated heatmap documentation:

    https://matplotlib.org/stable/gallery/images_contours_and_fields/
    image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is
        plotted. If not provided, use current Axes or create a new one.
        Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.
        Optional.
    cbarlabel
        The label for the colorbar. Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image display for the heatmap.
    cbar : matplotlib.colorbar.Colorbar
        Colorbar for the heatmap image.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, label=cbarlabel, **cbar_kw)

    # Show all ticks and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticks(
            range(data.shape[1]),
            labels=col_labels,
            ha="center",
        )
    if row_labels is not None:
        ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
        im,
        data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        threshold_lightness=0.5,
        **textkw,
        ):
    """
    A function to annotate a heatmap.

    Adapted from Matplotlib annotated heatmap documentation:

    https://matplotlib.org/stable/gallery/images_contours_and_fields/
    image_annotated_heatmap.html

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The AxesImage to be labeled.
    data : array_like, optional
        Data used to annotate.
        If None, the image's data is used.  Optional.
    valfmt : str or matplotlib.ticker.Formatter, optional
        The format of the annotations inside the heatmap.  This should
        either use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors : tuple, optional
        A pair of colors.  The first is used for values below a
        threshold, the second for those above.  Optional.
    threshold_lightness : float or None, optional
        Lightness threshold according to which the colors from
        `textcolors` are applied. If `None`, the lightness of the middle
        of the colormap is used.
    **kwargs
        All other arguments are forwarded to each call to `text` used to
        create the text labels.

    Returns
    -------
    texts : List of matplotlib.text.Text
        Text instances created by the annotation.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold_lightness is None:
        threshold_lightness = colorsys.rgb_to_hls(*im.cmap(0.5)[:3])[1]

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    data_filled = (
        data.filled(fill_value=np.nan) if np.ma.is_masked(data) else data
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cell_lightness = colorsys.rgb_to_hls(
                *im.cmap(im.norm(data_filled[i, j]))[:3]
            )[1]
            use_dark_text_color = (
                cell_lightness >= threshold_lightness
                or (np.ma.is_masked(data) and data.mask[i, j])
            )
            color = textcolors[0 if use_dark_text_color else 1]

            kw.update(color=color)

            # prepare annotation string
            if np.ma.is_masked(data) and data.mask[i, j]:
                annotation_str = "--"
            else:
                annotation_str = valfmt(data[i, j], None)

            text = im.axes.text(j, i, annotation_str, **kw)
            texts.append(text)

    return texts


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def _heatmap_example():
    """
    Annotated heatmap example.
    """
    capture_targets = np.array(
        [90, 92.5, 95, 96, 97, 98, 99, 99.3, 99.5, 99.8, 99.9]
    )
    conf_lvls = np.array([0, 90, 95, 99])
    det_times = (
        np.random.uniform(low=5, high=15, size=(1, len(capture_targets))) / 60
    )
    pyros_times = np.random.uniform(
        low=1700 / 60,
        high=3200 / 60,
        size=(len(conf_lvls) - 1, len(capture_targets)),
    )
    solve_times = np.vstack([det_times, pyros_times]).T
    fig, ax = plt.subplots(figsize=(3.1, 0.3 * capture_targets.size + 0.2))
    ax.grid(False)

    im, _ = heatmap(
        solve_times,
        row_labels=capture_targets,
        col_labels=conf_lvls,
        ax=ax,
        xlabel="Gaussian Confidence Level ($\\%$)",
        ylabel="$\\mathrm{CO}_2$ Capture Target ($\\%$)",
        cmap="plasma_r",
        cbarlabel="Solve Time (wall min)",
    )
    annotate_heatmap(im=im, valfmt="{x:.1f}", fontsize=8)
    fig.savefig("annotated_heatmap_time.png", bbox_inches="tight", dpi=300)

    pyros_iterations = np.random.randint(
        low=4,
        high=8,
        size=(conf_lvls.size - 1, capture_targets.size),
    ).T.astype(float)
    pyros_iterations[-1, -1] = np.nan
    fig, ax = plt.subplots(figsize=(
        1.1 + 0.5 * pyros_iterations.shape[1],
        0.3 * capture_targets.size + 0.2
    ))
    ax.grid(False)
    min_iters = int(np.nanmin(pyros_iterations))
    max_iters = int(np.nanmax(pyros_iterations))
    iter_range = max_iters - min_iters
    im, _ = heatmap(
        pyros_iterations,
        row_labels=capture_targets,
        col_labels=conf_lvls[1:],
        ax=ax,
        xlabel="Gaussian Confidence Level ($\\%$)",
        ylabel="$\\mathrm{CO}_2$ Capture Target ($\\%$)",
        cmap=mpl.colormaps["plasma_r"].resampled(
            iter_range + 1,
        ),
        cbarlabel="PyROS Iterations",
        norm=mpl.colors.BoundaryNorm(
            np.linspace(min_iters - 0.5, max_iters + 0.5, iter_range + 2),
            ncolors=iter_range + 1,
        ),
        cbar_kw=dict(ticks=np.arange(min_iters, max_iters + 1)),
    )
    annotate_heatmap(im=im, valfmt="{x:.0f}", fontsize=8)

    # all done
    plt.show()


if __name__ == "__main__":
    _heatmap_example()
