import pyomo.common.unittest as unittest
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    large_residuals_set,
)
from parker_focapo2023.clc.model import (
    make_model,
    ModelVersion,
)


class TestConstructModel(unittest.TestCase):
    version = ModelVersion.IDAES_1_7

    def test_construct(self):
        m = make_model(
            steady=False,
            version=self.version,
            initialize=False,
            n_samples=2,
            nxfe=4,
        )
        self.assertEqual(degrees_of_freedom(m), 0)

    def test_matching(self):
        m = make_model(
            steady=False,
            version=self.version,
            initialize=False,
            n_samples=2,
            nxfe=4,
        )
        igraph = IncidenceGraphInterface(m)
        matching = igraph.maximum_matching()
        n_var = len(igraph.variables)
        n_con = len(igraph.constraints)
        card_matching = len(matching)
        self.assertEqual(n_var, n_con)
        self.assertLess(card_matching, n_var)

    def test_initialize(self):
        m = make_model(
            steady=False,
            version=self.version,
            n_samples=2,
            nxfe=4,
        )
        self.assertEqual(len(large_residuals_set(m)), 0)


if __name__ == "__main__":
    unittest.main()
