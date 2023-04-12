import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from parker_cce2022.mbclc_dynopt.model import (
    get_steady_state_data,
)
from parker_cce2022.mbclc_dynopt.run_implicit_function import (
    run_dynamic_optimization as run_implicit_function,
)
from parker_cce2022.mbclc_dynopt.run_full_space import (
    run_dynamic_optimization as run_full_space,
)

class TestNominalSolves(unittest.TestCase):

    nxfe = 10
    samples_per_horizon = 10

    # For this case, using strings involving the inlet ports is okay.
    sp_input_map = {"fs.MB.solid_inlet.flow_mass[*]": 700.0}
    sp_dof_names = ["fs.MB.gas_inlet.flow_mol[*]"]
    sp_state_list = [("fs.MB.solid_phase.reactions[*,0.0].OC_conv", 0.95)]

    rel_tol = 1e-2

    # Predicted control inputs solved for by the dynamic optimization
    # problem.
    pred_input_data = {
        # Note that zero must be displayed as "0.0", not "0", in this string
        # This is for consistency with the dict I create in the solve driver.
        # This would be a lot nicer if I could do the comparison with CUIDs
        # as keys, but I don't do that because I need json serializability.
        # TimeSeriesData would make this easier by supplying a to_serializable
        # method.
        "fs.MB.gas_phase.properties[*,0.0].flow_mol": [
            128.205,
            208.209,
            208.209,
            172.025,
            172.025,
            162.458,
            162.458,
            161.640,
            161.640,
            161.760,
            161.760,
            161.851,
            161.851,
            161.885,
            161.885,
            161.895,
            161.895,
            161.894,
            161.894,
            161.887,
            161.887,
        ],
        "fs.MB.solid_phase.properties[*,1.0].flow_mass": [
            591.400,
            652.081,
            652.081,
            693.400,
            693.400,
            699.519,
            699.519,
            700.106,
            700.106,
            700.084,
            700.084,
            700.046,
            700.046,
            700.030,
            700.030,
            700.025,
            700.025,
            700.024,
            700.024,
            700.023,
            700.023,
        ],
    }

    def test_implicit_function_solve(self):
        ic_scalar_data, ic_dae_data = get_steady_state_data(
            nxfe=self.nxfe,
        )
        _, sp_dae_data = get_steady_state_data(
            nxfe=self.nxfe,
            input_map=self.sp_input_map,
            to_unfix=self.sp_dof_names,
            setpoint_list=self.sp_state_list,
        )
        status, input_data, time = run_implicit_function(
            initial_conditions=ic_dae_data,
            setpoint=sp_dae_data,
            scalar_data=ic_scalar_data,
            nxfe=self.nxfe,
            samples_per_horizon=self.samples_per_horizon,
        )
        input_data.data.pop("tracking_cost[*]")
        self.assertStructuredAlmostEqual(
            input_data.data, self.pred_input_data, reltol=self.rel_tol
        )
        # TODO: Assert that status is correct

    def test_full_space_solve(self):
        ic_scalar_data, ic_dae_data = get_steady_state_data(
            nxfe=self.nxfe,
        )
        _, sp_dae_data = get_steady_state_data(
            nxfe=self.nxfe,
            input_map=self.sp_input_map,
            to_unfix=self.sp_dof_names,
            setpoint_list=self.sp_state_list,
        )
        status, input_data, time = run_full_space(
            initial_conditions=ic_dae_data,
            setpoint=sp_dae_data,
            scalar_data=ic_scalar_data,
            nxfe=self.nxfe,
            samples_per_horizon=self.samples_per_horizon,
        )
        input_data.data.pop("tracking_cost[*]")
        self.assertStructuredAlmostEqual(
            input_data.data, self.pred_input_data, reltol=self.rel_tol
        )


if __name__ == "__main__":
    unittest.main()
