import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

def generate_trivial_constraints(m, include_fixed=False):
    # Identifies constraint data objects that are trivial (contain
    # no variable) within m.
    for con in m.component_data_objects(pyo.Constraint):
        found_var = False
        try:
            next(identify_variables(con.body, include_fixed=include_fixed))
            found_var = True
        except StopIteration:
            pass
        if not found_var:
            yield con
