from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.common.collections import ComponentSet
from pyomo.dae.flatten import flatten_dae_components
from pyomo.util.subsystems import (
    generate_subsystem_blocks,
    TemporarySubsystemManager,
)
from pyomo.core.expr.visitor import identify_variables

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface


def _filter_duplicates(comps):
    seen = set()
    for comp in comps:
        if id(comp) not in seen:
            seen.add(id(comp))
            yield comp


def get_subsystems_along_time(
        m,
        time,
        flatten_vars=None,
        flatten_cons=None,
        ):
    """
    Arguments
    ---------
    m: Block
        A model containing components indexed by time
    time: Set
        A set indexing components of the model

    """
    if flatten_vars is None:
        scalar_vars, dae_vars = flatten_dae_components(m, time, Var)
    else:
        scalar_vars, dae_vars = flatten_vars
    if flatten_cons is None:
        scalar_cons, dae_cons = flatten_dae_components(m, time, Constraint)
    else:
        scalar_cons, dae_cons = flatten_cons

    con_vars = dict()
    for t in time:
        con_vars[t] = ComponentSet()
        for con in dae_cons:
            if t in con and con.active and con[t].active:
                for var in identify_variables(
                        con[t].expr,
                        include_fixed=False,
                        ):
                    con_vars[t].add(var)

    subsystems = [
        (
            list(_filter_duplicates(
                con[t] for con in dae_cons
                if t in con and con.active and con[t].active
            )),
            list(_filter_duplicates(
                var[t] for var in dae_vars
                if t in var and var[t] in con_vars[t]
            )),
        )
        for t in time
    ]
    return subsystems


def initialize_by_time_element(
        m,
        time,
        solver,
        solve_kwds=None,
        skip_partition=False,
        flatten_vars=None,
        flatten_cons=None,
        time_subsystems=None,
        ):
    if solve_kwds is None:
        solve_kwds = {}
    reslist = []
    for block, inputs in generate_time_element_blocks(
            m,
            time,
            skip_partition=skip_partition,
            flatten_vars=flatten_vars,
            flatten_cons=flatten_cons,
            time_subsystems=time_subsystems,
            ):
        with TemporarySubsystemManager(to_fix=inputs):
            res = solver.solve(block, **solve_kwds)
            reslist.append(res)
    return reslist


def generate_time_element_blocks(
        m,
        time,
        skip_partition=False,
        flatten_vars=None,
        flatten_cons=None,
        time_subsystems=None,
        ):
    if flatten_vars is None:
        scalar_vars, dae_vars = flatten_dae_components(m, time, Var)
    else:
        scalar_vars, dae_vars = flatten_vars
    if flatten_cons is None:
        scalar_cons, dae_cons = flatten_dae_components(m, time, Constraint)
    else:
        scalar_cons, dae_cons = flatten_cons
    # We repeat the above flattening in this function...
    if time_subsystems is None:
        subsystems = get_subsystems_along_time(
            m,
            time,
            flatten_vars=(scalar_vars, dae_vars),
            flatten_cons=(scalar_cons, dae_cons),
        )
    else:
        subsystems = time_subsystems
    if not skip_partition:
        # If we know our time-subsystems are all independent, as is
        # the case in a discretization that only involves adjacent
        # time points, then we can skip this expensive partition step.
        partition = partition_independent_subsystems(subsystems)
    else:
        partition = [[i] for i in range(len(subsystems))]
    time_partition = [[time.at(i+1) for i in subset] for subset in partition]

    combined_subsystems = [
        (
            [con for i in subset for con in subsystems[i][0]],
            [var for i in subset for var in subsystems[i][1]],
        )
        for subset in partition
    ]
    for i, (block, inputs) in enumerate(
            generate_subsystem_blocks(combined_subsystems)
            ):
        t_points = time_partition[i]
        assert len(block.vars) == len(block.cons)
        if i != 0:
            # Initialize with results of previous solve
            # TODO: Should initialize happen in a separate function?
            for var in dae_vars:
                # The time required to look up the vardata objects
                # in these references is actually showing up in a
                # profile. TODO: Potentially refactor to use
                # dicts: time -> vardata here... (Note that this
                # caches valid time points, as opposed to references)
                vlatest = var[latest]
                for t in t_points:
                    var_t = var[t]
                    if not var_t.fixed:
                        var_t.set_value(vlatest.value)
        yield block, inputs
        # I don't think t_points can be empty. TODO: is this correct?
        latest = t_points[-1]


def generate_time_blocks(m, time):
    subsystems = get_subsystems_along_time(m, time)
    for block, inputs in generate_subsystem_blocks(subsystems):
        yield block, inputs


def partition_independent_subsystems(subsystems):
    """
    This method takes a partition of a model into potentially independent
    subsets, and combines these subsets if any of them are mutually
    dependent according to a block triangularization.

    """
    n_subsystem = len(subsystems)
    total_vars = [var for _, variables in subsystems for var in variables]
    total_cons = [con for constraints, _ in subsystems for con in constraints]
    igraph = IncidenceGraphInterface()
    # As of Pyomo 6.5.0, the functionality of block_triangularize has changed.
    #v_b_map, c_b_map = igraph.block_triangularize(total_vars, total_cons)
    v_b_map, c_b_map = igraph.map_nodes_to_block_triangular_indices(
        total_vars, total_cons
    )
    blocks_per_subsystem = [set() for _ in range(n_subsystem)]
    for i in range(n_subsystem):
        constraints, variables = subsystems[i]
        con_blocks = set(c_b_map[con] for con in constraints)
        var_blocks = set(v_b_map[var] for var in variables)
        # Here we require that the user's subsystems be compatible
        # with a block triangularization. I.e. that the union of diagonal
        # blocks is the same for variables and constraints.
        # Why do we require this again? Because it is required for these
        # time blocks to be square systems that we can solve.
        assert con_blocks == var_blocks
        blocks_per_subsystem[i].update(con_blocks)
    n_blocks = len(set(v_b_map.values()))
    subsystems_per_block = [set() for _ in range(n_blocks)]
    for i in range(n_subsystem):
        for b in blocks_per_subsystem[i]:
            subsystems_per_block[b].add(i)

    # These will partition our subsystems
    subsets = [set((i,)) for i in range(n_subsystem)]
    for subsystems in subsystems_per_block:
        # If a block contains multiple subsystems, these subsystems must
        # be solved simultaneously
        s0 = subsystems.pop()
        for s in subsystems:
            # Combine the subsets for these subsystems
            subsets[s0].update(subsets[s])
            subsets[s] = subsets[s0]
    unique_subsets = list(_filter_duplicates(subsets))
    sorted_subsets = [list(sorted(s)) for s in unique_subsets]
    return sorted_subsets
