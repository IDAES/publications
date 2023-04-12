import pyomo.common.unittest as unittest
import math

from parker_cce2022.mbclc_sim.full_space import (
    solve_full_space,
)
from parker_cce2022.mbclc_sim.reduced_space import (
    solve_reduced_space,
)


"""
These tests solve the simulation for a nominal instance with both
the full space and implicit function formulations and make sure the
results are as expected.
"""


class TestSolve(unittest.TestCase):

    temperature_gas = 1000.0
    temperature_solid = 1200.0
    nxfe = 10

    rel_tol = 1e-2

    states_to_compare = [
        "fs.MB.gas_phase.properties[0,1.0].temperature",
        "fs.MB.gas_phase.properties[0,1.0].flow_mol",
        "fs.MB.gas_phase.properties[0,1.0].pressure",
        "fs.MB.gas_phase.properties[0,1.0].mole_frac_comp[CH4]",
        "fs.MB.gas_phase.properties[0,1.0].mole_frac_comp[CO2]",
        "fs.MB.gas_phase.properties[0,1.0].mole_frac_comp[H2O]",
        "fs.MB.solid_phase.properties[0,0.0].temperature",
        "fs.MB.solid_phase.properties[0,0.0].flow_mass",
        "fs.MB.solid_phase.properties[0,0.0].mass_frac_comp[Fe2O3]",
        "fs.MB.solid_phase.properties[0,0.0].mass_frac_comp[Fe3O4]",
        "fs.MB.solid_phase.properties[0,0.0].mass_frac_comp[Al2O3]",
    ]
    # These are values taken from a successful full space solve.
    predicted_values = [
        1199.99,
        377.678,
        1.38984,
        0.00067,
        0.33876,
        0.66056,
        1010.74,
        582.925,
        0.02135,
        0.42065,
        0.55799,
    ]

    def test_full_space_solve(self):
        res, m = solve_full_space(
            T=self.temperature_gas,
            solid_temp=self.temperature_solid,
            nxfe=self.nxfe,
        )
        for name, val in zip(self.states_to_compare, self.predicted_values):
            var = m.find_component(name)
            math.isclose(var.value, val, rel_tol=self.rel_tol)

    def test_implicit_function_solve(self):
        res, m = solve_reduced_space(
            T=self.temperature_gas,
            solid_temp=self.temperature_solid,
            nxfe=self.nxfe,
        )
        for name, val in zip(self.states_to_compare, self.predicted_values):
            var = m.find_component(name)
            math.isclose(var.value, val, rel_tol=self.rel_tol)


if __name__ == "__main__":
    unittest.main()
