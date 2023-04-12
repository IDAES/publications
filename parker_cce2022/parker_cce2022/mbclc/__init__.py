from .model import (
        SpatialDiscretization,
        make_model,
        make_square_model,
        fix_dof,
        add_constraints_for_missing_variables,
        )
from .initialize import (
        add_inlet_objective,
        set_default_design_vars,
        set_default_inlet_conditions,
        set_values_to_inlets,
        set_gas_values_to_inlets,
        set_solid_values_to_inlets,
        set_gas_temperature_to_solid_inlet,
        set_optimal_design_vars,
        set_optimal_inlet_conditions,
        initialize_steady,
        initialize_steady_without_solid_temperature,
        )
from .incidence import (
        get_maximum_matching,
        get_block_triangularization,
        get_minimal_subsystem_for_variables,
        )
