import pyomo.environ as pyo
import matplotlib.pyplot as plt


def plot_outlet_states_over_time(
        m,
        show=True,
        prefix=None,
        extra_states=None,
        ):
    if prefix is None:
        prefix = ""
    if extra_states is None:
        extra_states = []
    x0 = m.fs.MB.gas_phase.length_domain.first()
    x1 = m.fs.MB.solid_phase.length_domain.last()
    state_slices = [
        m.fs.MB.gas_phase.properties[:, x1].temperature,
        m.fs.MB.gas_phase.properties[:, x1].pressure,
        m.fs.MB.gas_phase.properties[:, x1].flow_mol,
        m.fs.MB.gas_phase.properties[:, x1].mole_frac_comp["CH4"],
        m.fs.MB.gas_phase.properties[:, x1].mole_frac_comp["H2O"],
        m.fs.MB.gas_phase.properties[:, x1].mole_frac_comp["CO2"],
        m.fs.MB.solid_phase.properties[:, x0].temperature,
        m.fs.MB.solid_phase.properties[:, x0].flow_mass,
        m.fs.MB.solid_phase.properties[:, x0].mass_frac_comp["Fe2O3"],
        m.fs.MB.solid_phase.properties[:, x0].mass_frac_comp["Fe3O4"],
        m.fs.MB.solid_phase.properties[:, x0].mass_frac_comp["Al2O3"],
    ]
    state_refs = [pyo.Reference(slice_) for slice_ in state_slices]
    state_refs.extend(extra_states)
    for i, ref in enumerate(state_refs):
        fig, ax = plot_time_indexed_variable(ref)
        fig.savefig(prefix + "state%s.png" % i, transparent=True)
        if show:
            plt.show()


def plot_time_indexed_variable(var):
    time = list(var.index_set())
    data = [var[t].value for t in time]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(time, data)
    if var.is_reference():
        cuid = pyo.ComponentUID(var.referent)
    else:
        cuid = pyo.ComponentUID(var)
    ax.set_title(str(cuid))
    return fig, ax


def plot_inputs_over_time(m, inputs, show=True):
    x0 = m.fs.MB.gas_phase.length_domain.first()
    x1 = m.fs.MB.solid_phase.length_domain.last()
    inputs = [m.find_component(name) for name in inputs]
    for i, var in enumerate(inputs):
        fig, ax = step_time_indexed_variable(var)
        fig.savefig("input%s.png" % i, transparent=True)
        if show:
            plt.show()


def step_time_indexed_variable(var):
    time = list(var.index_set())
    data = [var[t].value for t in time]
    # We actually don't care about the first input value.
    # I think this is only true for an implicit discretization.
    time = time[1:]
    data = data[1:]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.step(time, data, where="pre")
    if var.is_reference():
        cuid = pyo.ComponentUID(var.referent)
    else:
        cuid = pyo.ComponentUID(var)
    ax.set_title(str(cuid))
    return fig, ax


def plot_outlet_data(data, show=True, prefix=None, extra_states=None):
    x0 = 0.0
    x1 = 1.0
    names = [
        "fs.MB.gas_phase.properties[*,%s].temperature" % x1,
        "fs.MB.gas_phase.properties[*,%s].pressure" % x1,
        "fs.MB.gas_phase.properties[*,%s].flow_mol" % x1,
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[CH4]" % x1,
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[H2O]" % x1,
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[CO2]" % x1,
        "fs.MB.solid_phase.properties[*,%s].temperature" % x0,
        "fs.MB.solid_phase.properties[*,%s].flow_mass" % x0,
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Fe2O3]" % x0,
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Fe3O4]" % x0,
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Al2O3]" % x0,
    ]
    if extra_states is not None:
        names.extend(extra_states)
    plot_time_indexed_data(
        data,
        names,
        show=show,
        prefix=prefix,
    )


def plot_time_indexed_data(
        data,
        names,
        show=True,
        prefix=None,
        ):
    """
    data:
    (
        [t0, ...],
        {
            str(cuid): [value0, ...],
        },
    )
    names: list of str(cuids) that we will actually plot.
    """
    if prefix is None:
        prefix = ""
    time, name_map = data
    for i, name in enumerate(names):
        values = name_map[name]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(time, values)
        ax.set_title(name)
        fig.savefig(prefix + "state%s.png" % i, transparent=True)
        if show:
            plt.show()
