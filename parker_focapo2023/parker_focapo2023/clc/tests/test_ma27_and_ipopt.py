import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.linalg.ma27_interface import (
    MA27,
    LinearSolverStatus,
)
import scipy.sparse as sps
from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix


class TestIpopt(unittest.TestCase):

    def test_ipopt_available(self):
        ipopt = pyo.SolverFactory("ipopt")
        self.assertTrue(ipopt.available())


class TestMa27(unittest.TestCase):

    def test_ma27(self):
        ma27 = MA27()

        dim = 5
        hessian = 2.0*sps.identity(5).tocoo()
        jac = 1.0*sps.identity(5).tocoo()
        kkt = BlockMatrix(2, 2)
        kkt.set_block(0, 0, hessian)
        kkt.set_block(0, 1, jac.transpose())
        kkt.set_block(1, 0, jac.transpose())
        res = ma27.do_symbolic_factorization(kkt)
        self.assertEqual(res.status, LinearSolverStatus.successful)
        res = ma27.do_numeric_factorization(kkt)
        self.assertEqual(res.status, LinearSolverStatus.successful)


if __name__ == "__main__":
    unittest.main()
