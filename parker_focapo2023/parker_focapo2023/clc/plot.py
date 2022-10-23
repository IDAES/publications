import matplotlib.pyplot as plt

def plot_incidence_matrix(
    coo,
    show=False,
    save=False,
    transparent=True,
    fname=None,
    pdf=False,
    **kwds,
):
    rc_params_save = plt.rcParams
    try:
        plt.rcParams["font.size"] = 14
        fig, ax = plt.subplots()
        params = dict(markersize=2)
        params.update(kwds)
        ax.spy(
            coo,
            **params,
        )
        ax.tick_params(bottom=False, direction="inout")
        if show:
            plt.show()
        if save:
            if fname is None:
                fname = "matrix"
            if pdf is True:
                extension = ".pdf"
            else:
                extension = ".png"
            fname += extension
            fig.savefig(fname, transparent=transparent)
    finally:
        plt.rcParams.update(rc_params_save)

    return fig, ax
