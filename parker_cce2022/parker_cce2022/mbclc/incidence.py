from parker_cce2022.mbclc.model import (
        make_model,
        make_square_model,
        fix_design_variables,
        fix_dynamic_inputs,
        )

import pyomo.environ as pyo
from parker_cce2022.common.incidence_analysis.interface import (
        IncidenceGraphInterface,
        )
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet

import networkx as nx
import networkx.algorithms.traversal.depth_first_search as nxdfs
"""
This module contains functions useful for analyzing the incidence graph
structure of the MBCLC reduction reactor model.
"""

def get_maximum_matching(m):
    constraints = list(m.component_data_objects(pyo.Constraint, active=True))

    var_set = ComponentSet()
    variables = []
    for con in constraints:
        for var in identify_variables(con.body, include_fixed=False):
            if var not in var_set:
                variables.append(var)
                var_set.add(var)

    igraph = IncidenceGraphInterface()
    con_var_map = igraph.maximum_matching(variables, constraints)

    ncon, nvar = len(constraints), len(variables)
    matching_len = len(con_var_map)
    print("nvar: %s, ncon: %s" % (nvar, ncon))
    print("Cardinality of maximal matching: %s" % matching_len)

    return con_var_map, variables, constraints


def get_block_triangularization(m):
    constraints = list(m.component_data_objects(pyo.Constraint, active=True))

    var_set = ComponentSet()
    variables = []
    for con in constraints:
        for var in identify_variables(con.body, include_fixed=False):
            if var not in var_set:
                variables.append(var)
                var_set.add(var)

    igraph = IncidenceGraphInterface()
    var_block_map, con_block_map, _ = igraph.block_triangularize(
            variables,
            constraints,
            )

    # TODO: Use generate_strongly_connected_components when it is merged
    blocks = set(var_block_map.values())
    n_blocks = len(blocks)
    var_blocks = [[] for b in range(n_blocks)]
    con_blocks = [[] for b in range(n_blocks)]
    for var, b in var_block_map.items():
        var_blocks[b].append(var)
    for con, b in con_block_map.items():
        con_blocks[b].append(con)

    print("Model contains %s strongly connected components" % n_blocks)

    return var_blocks, con_blocks


def print_block_triangularization():
    steady = True
    nxfe = 5
    nxcp = 1
    m = make_square_model(steady=steady, nxfe=nxfe, nxcp=nxcp)
    var_blocks, con_blocks = get_block_triangularization(m)

    for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
        nvar, ncon = len(vars), len(cons)
        assert nvar == ncon
        print()
        print("Block %s, %sx%s:" % (i, ncon, nvar))
        print("  Variables:")
        for var in vars:
            print("    %s" % var.name)
        print("  Constraints:")
        for con in cons:
            print("    %s" % con.name)


def print_maximum_matching():
    steady = True
    nxfe = 5
    nxcp = 1
    m = make_square_model(steady=steady, nxfe=nxfe, nxcp=nxcp)
    matching, _, _ = get_maximum_matching(m)
    for con, var in matching.items():
        print()
        print(con.name)
        print(var.name)


def get_minimal_subsystem_for_variables(system, solve_for):
    """
    Parameters
    ----------
    system:
        Square system of constraints and variables
    variables:
        Variables that we would like to solve for
    """
    igraph = IncidenceGraphInterface()
    vb_map, cb_map, dag_ll = igraph.block_triangularize(*system)
    n_blocks = len(dag_ll)

    var_blocks = [[] for b in range(n_blocks)]
    con_blocks = [[] for b in range(n_blocks)]
    for var, b in vb_map.items():
        var_blocks[b].append(var)
    for con, b in cb_map.items():
        con_blocks[b].append(con)

    # Create reverse order DAG
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_blocks))
    for n in dag.nodes:
        for neighbor in dag_ll[n]:
            # Reverse direction of edge so we can locate dependencies
            # of each variable
            dag.add_edge(neighbor, n)

    subsystems = []
    already_added = set()
    for var in solve_for:
        if var.fixed:
            # TODO: Should this check be performed in this function
            # or in the calling routine?
            continue
        b_idx = vb_map[var]
        # Can I get the dfs from a group of nodes?
        for b in nxdfs.dfs_postorder_nodes(dag, b_idx):
            if b not in already_added:
                already_added.add(b)
                subsystems.append((con_blocks[b], var_blocks[b]))

    return subsystems


if __name__ == "__main__":
    print_maximum_matching()
    print_block_triangularization()
