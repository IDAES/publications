import itertools
from scipy import interpolate
import numpy as np

from workspace.common.serialize.data_from_model import (
        apply_function_elementwise,
        _get_structured_variable_data_from_dict,
        )

def interpolate_data_onto_sets(data, set_dict):
    """
    """
    # data is the structure accessed by the "model" key from the
    # top-level dict
    interp_data = []

    for subset in data:
        # subset describes a set of components indexed by the same sets
        sets = subset["sets"]
        indices = subset["indices"]
        variables = subset["variables"]
        # NOTE: If I extend this data structure to include blocks, this
        # function will need to recursively descend into subblocks.

        if sets is None:
            # We cannot interpolate unindexed variables
            interp_data.append(subset)

        elif not any(s in set_dict for s in sets):
            # Variables are not indexed by any set we wish to interpolate
            # No reason to do any of this work.
            interp_data.append(subset)

        else:
            # These indices define the "interpolation points"
            new_indices = [
                    set_dict[name] if name in set_dict else idx_set
                    for name, idx_set in zip(sets, indices)
                    ]
            interp_coords = list(itertools.product(*new_indices))

            # Necessary to generate structured data for interpolated values
            mesh = np.meshgrid(*new_indices, indexing="ij")

            var_dict = {}
            for name, values in variables.items():
                # This will fail if any set is a singleton
                # TODO: function to remove singletons from mesh
                interp_values = interpolate.interpn(
                        indices,
                        values,
                        interp_coords,
                        )
                data_dict = dict(zip(interp_coords, interp_values))
                var_dict[name] = data_dict

            interp_data.append(
                    _get_structured_variable_data_from_dict(
                        sets,
                        new_indices,
                        var_dict,
                        )
                    )

    return interp_data
