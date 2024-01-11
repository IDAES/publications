import matplotlib.pyplot as plt


def _plot_time_indexed_variables(
    data, keys, show=False, save=False, fname=None, transparent=False
):
    fig, ax = plt.subplots()
    time = data.get_time_points()
    for i, key in enumerate(keys):
        data_list = data.get_data_from_key(key)
        label = str(data.get_cuid(key))
        ax.plot(time, data_list, label=label)
    ax.legend()

    if show:
        plt.show()
    if save:
        if fname is None:
            fname = "states.png"
        fig.savefig(fname, transparent=transparent)

    return fig, ax


def _step_time_indexed_variables(
    data, keys, show=False, save=False, fname=None, transparent=False
):
    fig, ax = plt.subplots()
    time = data.get_time_points()
    for i, key in enumerate(keys):
        data_list = data.get_data_from_key(key)
        label = str(data.get_cuid(key))
        ax.step(time, data_list, label=label)
    ax.legend()

    if show:
        plt.show()
    if save:
        if fname is None:
            fname = "inputs.png"
        fig.savefig(fname, transparent=transparent)

    return fig, ax
