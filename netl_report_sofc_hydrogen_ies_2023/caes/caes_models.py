#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

__author__ = "John Eslick"

import os
import csv

import pandas as pd
import numpy as np

import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir

from idaes.core import FlowsheetBlockData, declare_process_block_class
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
import idaes.models.unit_models as um  # um = unit models
import idaes.core.util as iutil
from idaes.core.util.initialization import propagate_state
import idaes.core.util.scaling as iscale
from idaes.core.solvers import get_solver
from idaes.models_extra.power_generation.properties.natural_gas_PR import (
    get_prop,
    get_rxn,
)
from idaes.models.properties import iapws95
import idaes.logger as idaeslog
from idaes.core.util.tags import svg_tag


@declare_process_block_class("CaesDischargeFlowsheet")
class CaesDischargeFlowsheetData(FlowsheetBlockData):
    def build(self):
        super().build()
        air_species = {"CO2", "Ar", "H2O", "O2", "N2"}
        air_comp = {
            "O2": 0.2074,
            "H2O": 0.0099,
            "CO2": 0.0003,
            "N2": 0.7732,
            "Ar": 0.0092,
        }

        cat_hrsg_comp = {
            "O2": 0.12625,
            "H2O": 0.01147,
            "CO2": 0.00033,
            "N2": 0.85158,
            "Ar": 0.01037,
        }

        ann_hrsg_comp = {
            "O2": 0.02074,
            "H2O": 0.30041,
            "CO2": 0.12721,
            "N2": 0.54501,
            "Ar": 0.00663,
        }

        self.air_prop_params = GenericParameterBlock(
            **get_prop(air_species, ["Vap"]),
            doc="Air property parameters",
        )
        self.valve = um.Valve(
            doc="Throttle valve",
            valve_function_callback=um.ValveFunctionType.linear,
            property_package=self.air_prop_params,
        )
        self.valve.pressure_flow_equation.deactivate()

        self.exhaust_hx = um.HeatExchanger(
            hot_side={"property_package": self.air_prop_params},
            cold_side={"property_package": self.air_prop_params},
        )

        self.hrsg = um.Heater(
            property_package=self.air_prop_params,
        )

        self.trb = um.Turbine(
            property_package=self.air_prop_params,
        )

        self.air02 = Arc(
            source=self.valve.outlet,
            destination=self.exhaust_hx.cold_side_inlet,
        )

        self.air03 = Arc(
            source=self.exhaust_hx.cold_side_outlet,
            destination=self.hrsg.inlet,
        )

        self.air07 = Arc(
            source=self.hrsg.outlet,
            destination=self.trb.inlet,
        )

        self.air08 = Arc(
            source=self.trb.outlet,
            destination=self.exhaust_hx.hot_side_inlet,
        )

        self.valve.inlet.flow_mol[0].fix(5e3)
        self.valve.inlet.pressure.fix(7e6)
        self.valve.inlet.temperature.fix(323)
        for i, v in air_comp.items():
            self.valve.inlet.mole_frac_comp[:, i].fix(v)

        self.exhaust_hx.overall_heat_transfer_coefficient.fix(100)
        self.exhaust_hx.area.fix(2000)

        self.trb.inlet.temperature.fix(900)
        self.trb.inlet.pressure.fix(4e6)
        self.trb.efficiency_isentropic.fix(0.9)
        self.trb.outlet.pressure.fix(101325)

        pyo.TransformationFactory("network.expand_arcs").apply_to(self)


@declare_process_block_class("CaesChargeFlowsheet")
class CaesChargeFlowsheetData(FlowsheetBlockData):
    def build(self):
        super().build()
        air_species = {"CO2", "Ar", "H2O", "O2", "N2"}
        self.air_prop_params = GenericParameterBlock(
            **get_prop(air_species, ["Vap"]),
            doc="Air property parameters",
        )

        self.storage_volume = pyo.Var(initialize=3e5, units=pyo.units.m**3)
        self.compressor_power = pyo.Var(initialize=137e6, units=pyo.units.W)
        self.stored_air = pyo.Var(initialize=1e5, units=pyo.units.mol)
        self.compressor_power.fix()
        self.storage_volume.fix()

        self.cmp1 = um.Compressor(
            doc="Air compressor",
            property_package=self.air_prop_params,
        )

        self.intercooler1 = um.Heater(
            doc="After cooler before storage",
            property_package=self.air_prop_params,
        )

        self.cmp2 = um.Compressor(
            doc="Air compressor",
            property_package=self.air_prop_params,
        )

        self.intercooler2 = um.Heater(
            doc="After cooler before storage",
            property_package=self.air_prop_params,
        )

        self.cmp3 = um.Compressor(
            doc="Air compressor",
            property_package=self.air_prop_params,
        )

        self.aftercooler = um.Heater(
            doc="After cooler before storage",
            property_package=self.air_prop_params,
        )

        self.air01 = Arc(
            source=self.cmp1.outlet,
            destination=self.intercooler1.inlet,
            doc="Air stream from compressor to after cooler",
        )

        self.air02 = Arc(
            source=self.intercooler1.outlet,
            destination=self.cmp2.inlet,
            doc="Air stream from compressor to after cooler",
        )

        self.air03 = Arc(
            source=self.cmp2.outlet,
            destination=self.intercooler2.inlet,
            doc="Air stream from compressor to after cooler",
        )

        self.air04 = Arc(
            source=self.intercooler2.outlet,
            destination=self.cmp3.inlet,
            doc="Air stream from compressor to after cooler",
        )

        self.air05 = Arc(
            source=self.cmp3.outlet,
            destination=self.aftercooler.inlet,
            doc="Air stream from compressor to after cooler",
        )

        air_comp = {
            "O2": 0.2074,
            "H2O": 0.0099,
            "CO2": 0.0003,
            "N2": 0.7732,
            "Ar": 0.0092,
        }

        self.cmp1.inlet.flow_mol[0].fix(4.4e3)
        # self.cmp1.inlet.flow_mol[0].unfix()
        self.cmp1.inlet.temperature.fix(288.15)
        self.cmp1.inlet.pressure.fix(101047)
        for i, v in air_comp.items():
            self.cmp1.inlet.mole_frac_comp[:, i].fix(v)

        self.cmp1_ratio_eqn = pyo.Constraint(
            expr=self.cmp1.ratioP[0] == 1.3 * self.cmp3.ratioP[0]
        )
        self.cmp2_ratio_eqn = pyo.Constraint(
            expr=self.cmp2.ratioP[0] == self.cmp3.ratioP[0]
        )
        self.cmp_power_eqn = pyo.Constraint(
            expr=self.compressor_power
            == self.cmp1.control_volume.work[0]
            + self.cmp2.control_volume.work[0]
            + self.cmp3.control_volume.work[0]
        )
        self.stored_air_eqn = pyo.Constraint(
            expr=self.stored_air
            == self.storage_volume
            / self.aftercooler.control_volume.properties_out[0].vol_mol_phase["Vap"]
        )

        self.cmp1.efficiency_isentropic.fix(0.80)
        self.cmp2.efficiency_isentropic.fix(0.80)
        self.cmp3.efficiency_isentropic.fix(0.80)
        self.cmp3.outlet.pressure.fix(4e6)
        self.aftercooler.outlet.temperature.fix(323)
        self.intercooler1.outlet.temperature.fix(323)
        self.intercooler2.outlet.temperature.fix(323)

        pyo.TransformationFactory("network.expand_arcs").apply_to(self)


if __name__ == "__main__":
    solver_obj = get_solver()

    print("Setting up discharging model...")
    m2 = pyo.ConcreteModel("CAES Discharge Model")
    m2.fs = CaesDischargeFlowsheet(dynamic=False)
    solver_obj.solve(m2, tee=True)

    m2.fs.hrsg.control_volume.heat.fix(282.399e6)
    m2.fs.valve.inlet.flow_mol[0].unfix()
    solver_obj.solve(m2, tee=True)

    print("Data for discharging Surrogate\n")
    print("pressure (kPa)\t flow (kmol/s)\t power (MW)")
    for p in np.linspace(4e6, 7e6, 13):
        m2.fs.valve.inlet.pressure[0].fix(p)
        solver_obj.solve(m2, tee=False)
        print(
            f"{p/1000.0}\t {pyo.value(m2.fs.valve.inlet.flow_mol[0]/1e3)}\t {-pyo.value(m2.fs.trb.control_volume.work[0]/1e6)}"
        )

    print("\n\nSetting up charging model...")
    m = pyo.ConcreteModel("CAES Charge Model")
    m.fs = CaesChargeFlowsheet(dynamic=False)
    m.fs.cmp_power_eqn.deactivate()
    solver_obj.solve(m, tee=True)

    m.fs.cmp1.inlet.flow_mol[0].unfix()
    m.fs.cmp_power_eqn.activate()
    solver_obj.solve(m, tee=True)

    print("Data for charging Surrogate\n")
    print("power (MW)\t vol (m3)\t pressure (kPa)\t flow (kmol/s)\t stored air (Mmol)")
    for cpow in [225]: #np.linspace(100, 650, 12):
        m.fs.compressor_power.fix(cpow * 1e6)
        for vol in [5e5]: #np.linspace(1e5, 1e6, 11):
            m.fs.storage_volume.fix(vol)
            for p in np.linspace(4e6, 7e6, 13):
                m.fs.cmp3.outlet.pressure[0].fix(float(p))
                solver_obj.solve(m, tee=False)
                print(
                    f"{cpow}\t {vol}\t {p/1000.0}\t {pyo.value(m.fs.cmp1.inlet.flow_mol[0]/1e3)}\t {pyo.value(m.fs.stored_air/1e6)}"
                )
