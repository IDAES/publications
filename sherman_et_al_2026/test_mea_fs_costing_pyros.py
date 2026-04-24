"""
Tests to confirm key components of the MEA flowsheet runs work.
"""


import logging
import os
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ as pyo

from confidence_ellipsoid import (
    get_pyros_ellipsoidal_set,
    mag_factor,
    sample_ellipsoid,
)
from flowsheets.combined_flowsheet_withcosting_CCS import MEACombinedFlowsheet
from model_utils import SolverWithBackup, SolverWithScaling
from plotting_utils import DEFAULT_MPL_RC_PARAMS, heatmap, annotate_heatmap
from run_combined_flowsheet_pyros_econ_obj import FlowsheetEconomicOptModelData
from combined_flowsheet_pyros_base import (
    build_and_solve_flowsheet_model,
)


class TestPlotting(unittest.TestCase):
    def test_plot_with_rc_params(self):
        with TempfileManager.new_context() as TMP:
            plot_outfile = TMP.create_tempfile(suffix=".png")
            with mpl.rc_context(DEFAULT_MPL_RC_PARAMS):
                fig, ax = plt.subplots()
                arr = np.arange(10)
                ax.plot(arr, arr ** 2)
                ax.set_xlabel("$x$")
                ax.set_ylabel("$x^2$")
                fig.savefig(plot_outfile)
                plt.close(fig)

            # check plot was successfully exported
            self.assertTrue(os.path.exists(plot_outfile))

    def test_plot_heatmap_nan(self):
        fig, ax = plt.subplots()
        im, _ = heatmap(
            data=np.array([[1, 2], [3, 4], [5, np.nan]]),
            row_labels=["1", "2", "3"],
            col_labels=["A", "B"],
            ax=ax,
            xlabel="row val",
            ylabel="column val",
            cmap="plasma_r",
            cbarlabel="Value",
        )
        annotate_heatmap(im)

    def test_plot_heatmap_no_nan(self):
        fig, ax = plt.subplots()
        im, _ = heatmap(
            data=np.array([[1, 2], [3, 4], [5, 6]]),
            row_labels=["1", "2", "3"],
            col_labels=["A", "B"],
            ax=ax,
            xlabel="row val",
            ylabel="column val",
            cmap="plasma_r",
            cbarlabel="Value",
        )
        annotate_heatmap(im)


class TestConfidenceEllipsoid(unittest.TestCase):
    def test_pyros_ellipsoidal_set(self):
        ell_set = get_pyros_ellipsoidal_set(
            mean=[1] * 6,
            cov_mat=np.eye(6),
            level=0.95,
        )
        self.assertTrue(np.isclose(ell_set.gaussian_conf_lvl, 0.95))

    def test_sample_ellipsoidal_set(self):
        mean = np.ones(3)
        cov_mat = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        conf_lvl = 0.95
        samples = sample_ellipsoid(
            mean=mean,
            cov_mat=cov_mat,
            level=conf_lvl,
            rng=np.random.default_rng(123456),
            samples=1000,
        )
        self.assertTrue(samples.shape, (150, 3))

        # check that all sampled points are in the set
        offsets = samples - mean
        lhs_vals = np.sum(
            offsets @ np.linalg.inv(cov_mat) * offsets,
            axis=1,
        )
        self.assertTrue(np.all(lhs_vals <= mag_factor(conf_lvl, 3) ** 2))


class TestSolverWrappers(unittest.TestCase):
    def test_solver_wrappers(self):
        # test the SolverWithBackup class
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0, bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x * (1 - m.x), sense=pyo.maximize)
        m.con = pyo.Constraint(expr=m.x >= 0.2)
        example_logger = logging.getLogger("example")
        example_logger.addHandler(logging.StreamHandler())
        example_logger.setLevel(logging.DEBUG)

        # check solver works
        solver = SolverWithScaling(
            SolverWithBackup(
                pyo.SolverFactory("ipopt"),
                logger=example_logger,
            ),
            logger=example_logger,
        )
        pyo.assert_optimal_termination(solver.solve(m))

        # check sorting works
        sorted_terminations = solver.solver.sort_termination_conditions([
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.infeasible,
        ])
        self.assertEqual(
            sorted_terminations[0],
            (pyo.TerminationCondition.locallyOptimal, 1),
        )
        self.assertEqual(
            sorted_terminations[1],
            (pyo.TerminationCondition.optimal, 2),
        )
        self.assertEqual(
            sorted_terminations[2],
            (pyo.TerminationCondition.feasible, 0),
        )
        self.assertEqual(
            sorted_terminations[3],
            (pyo.TerminationCondition.infeasible, 3),
        )


class TestBuildAndSolveFlowsheetModel(unittest.TestCase):
    def test_build_and_solve_flowsheet_model(self):
        co2_capture_target = 92.1468  # arbitrarily chosen
        fs_res = build_and_solve_flowsheet_model(
            co2_capture_target,
            solver="gams",
            solver_options={"solver": "conopt4"},
            flowsheet_model_data_type=FlowsheetEconomicOptModelData,
            flowsheet_process_block_constructor=MEACombinedFlowsheet,
            keep_model_in_results=True,
        )
        model = fs_res.model_data.original_model

        # check specific components exist
        self.assertEqual(model.fs.co2_capture_target.value, co2_capture_target)
        self.assertTrue(hasattr(model.fs, "k_eq_b_bicarbonate"))
        self.assertTrue(hasattr(model.fs, "k_eq_b_carbamate"))
        self.assertTrue(hasattr(model.fs, "lwm_coeff_1"))
        self.assertTrue(hasattr(model.fs, "lwm_coeff_2"))
        self.assertTrue(hasattr(model.fs, "lwm_coeff_3"))
        self.assertTrue(hasattr(model.fs, "lwm_coeff_4"))
        self.assertTrue(hasattr(model.fs, "lcoc_obj"))
        self.assertTrue(hasattr(model.fs, "levelized_cost_of_target_capture"))


if __name__ == "__main__":
    unittest.main()
