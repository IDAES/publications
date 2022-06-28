import numpy as np
from scipy import integrate
from parker_cce2022.common.serialize.data_from_model import (
        apply_function_elementwise,
        _get_structured_variable_data_from_dict,
        )


def integrate_variable_data(data):
    """
    """
    # TODO: Only integrate certain sets
    # TODO: actually test this function...
    sets = data["sets"]
    indices = data["indices"]
    variables = data["variables"]

    if sets is None:
        return {name: 0.0 for name in variables}

    else:
        indices = list(indices)
        variables = dict(variables)
        for name, values in variables.items():
            array = np.array(values)
            for i in range(len(indices)):
                axis = len(indices) - i - 1
                x = indices[axis]
                array = integrate.trapezoid(array, x=x, axis=axis)
            variables[name] = array.tolist()
        return variables
