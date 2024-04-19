#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

#  This code was developed at Carnegie Mellon University by Ilayda Akkor and
#  Chrysanthos E. Gounaris, as part of a research project funded by The Dow Chemical
#  Company. In particular, we acknowledge the close collaboration with the following
#  researchers from Dow: Shachit S. Iyer, John Dowdle and Le Wang,
#  who provided key contributions pertaining to the project conceptualization,
#  the research design and the methodology, model design choices and rationales,
#  and the discussions of the results.

# If you find this code useful for your research, please consider citing
# "Mathematical Modeling and Economic Optimization of a Piperazine-Based
# Post-Combustion Carbon Capture Process" by Ilayda Akkor, Shachit S. Iyer,
# John Dowdle, Le Wang and Chrysanthos E. Gounaris

import sys

# Import IDAES Libraries
from idaes.core import FlowsheetBlock
import idaes.logger as idaeslog
from idaes.core.util.model_statistics import degrees_of_freedom

# Import unit and property models
from column.properties.PZ_column_stripper_vapor_properties import (
    PZParameterBlock as vapor_prop,
)
from column.properties.PZ_column_absorber_liquid_properties import (
    PZParameterBlock as liq_prop,
)
from column.properties.PZ_column_absorber_vapor_properties import (
    PZParameterBlock as feed_prop,
)
from column.properties.PZ_column_stripper_liquid_properties import (
    PZParameterBlock as solvent_prop,
)
from PZ_AFS_heat_exchanger_network import heat_exchangers
from economic_model import economic_model
from optimization_cases import *
from column.PZ_solvent_column import PZPackedColumn
from Advanced_Flash_Stripper import PZ_AFS


def flowheet_model(flue_gas, scale, optimization, split_train):
    # import solver
    solver = SolverFactory("gams")
    solver.options = {"solver": "conopt"}
    # create model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    # creat property models
    m.fs.vap_properties = vapor_prop()
    m.fs.liq_properties = liq_prop()
    m.fs.feed_properties = feed_prop()
    m.fs.solvent_properties = solvent_prop()

    # Create an instance of the column in the flowsheet
    m.fs.absorber = PZPackedColumn(
        intercooler=True,
        has_pressure_change=True,
        vapor_phase={"property_package": m.fs.feed_properties},
        liquid_phase={"property_package": m.fs.liq_properties},
    )

    # dummy obj
    m.fs.absorber.obj = Objective(expr=0)

    # Fix column design variables
    m.fs.absorber.diameter.fix(0.66)
    m.fs.absorber.length.fix(12.2)
    # Fix column inlet variables
    m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(20)
    m.fs.absorber.vapor_properties[0, 0].temperature.fix(313)
    m.fs.absorber.vapor_properties[0, 0].pressure.fix(104500)
    m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["PZ"].fix(0)
    if flue_gas == "coal":
        m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["CO2"].fix(0.105)
        m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["H2O"].fix(0.07)
        m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["N2"].fix(0.825)
    elif flue_gas == "ngcc":
        m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["CO2"].fix(0.041)
        m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["H2O"].fix(0.0875)
        m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["N2"].fix(0.8715)

    N = m.fs.absorber.config.finite_elements
    m.fs.absorber.liquid_properties[0, N].flow_mol.fix(100)
    m.fs.absorber.liquid_properties[0, N].temperature.fix(322)
    m.fs.absorber.liquid_properties[0, N].mole_frac_comp["CO2"].fix(0.041871)
    m.fs.absorber.liquid_properties[0, N].mole_frac_comp["H2O"].fix(0.870899)
    m.fs.absorber.liquid_properties[0, N].mole_frac_comp["PZ"].fix(0.08723)
    m.fs.absorber.liquid_properties[0, N].mole_frac_comp["N2"].fix(0)

    # initialize absorber column
    m.fs.absorber.initialize(solver, outlvl=idaeslog.INFO_HIGH)

    # Scale column dimensions up (if scale is not pilot) as part of the initialization
    if scale == "commercial":

        def abs_flooding_ub(m, i):
            if i == 0 or i == N:
                return Constraint.Skip
            else:
                return m.vapor_velocity[i] <= 0.8 * m.flooding_velocity[i]

        m.fs.absorber.flooding_ub_con = Constraint(
            m.fs.absorber.length_domain, rule=abs_flooding_ub
        )

        def abs_flooding_lb(m, i):
            if i == 0 or i == N:
                return Constraint.Skip
            else:
                return m.vapor_velocity[i] >= 0.3 * m.flooding_velocity[i]

        m.fs.absorber.flooding_lb_con = Constraint(
            m.fs.absorber.length_domain, rule=abs_flooding_lb
        )

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(40)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(200)
        m.fs.absorber.diameter.unfix()
        results = solver.solve(m)
        print("Absorber scale up, step 1: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(60)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(300)
        results = solver.solve(m)
        print("Absorber scale up, step 2: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(100)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(500)
        results = solver.solve(m)
        print("Absorber scale up, step 3: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(200)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(1000)
        results = solver.solve(m)
        print("Absorber scale up, step 4: " + results.solver.termination_condition)

        if flue_gas == "coal":
            m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(1000)
            m.fs.absorber.liquid_properties[0, N].flow_mol.fix(5000)
            results = solver.solve(m)
            print("Absorber scale up, step 5: " + results.solver.termination_condition)

            m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(3200)
            m.fs.absorber.liquid_properties[0, N].flow_mol.fix(16000)
            results = solver.solve(m)
            print("Absorber scale up, step 5: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(4200)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(21000)
        results = solver.solve(m)
        print("Absorber scale up, step 6: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(6760)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(33800)
        results = solver.solve(m)
        print("Absorber scale up, step 7: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(6760 * 1.5)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(33800 * 1.5)
        results = solver.solve(m)
        print("Absorber scale up, step 8: " + results.solver.termination_condition)

        m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(13520)
        m.fs.absorber.liquid_properties[0, N].flow_mol.fix(33800 * 2)
        results = solver.solve(m)
        print("Absorber scale up, step 9: " + results.solver.termination_condition)

    # add bypasses
    m.fs.F_bottom = Var(bounds=(0, None))
    m.fs.F_hot = Var(bounds=(0, None))
    m.fs.F_cold = Var(bounds=(0, None))
    m.fs.F_warm = Var(bounds=(0, None))

    m.fs.T_hot = Var(initialize=150 + 273)
    m.fs.T_hot.fix(150 + 273)
    m.fs.T_warm = Var(initialize=124 + 273)
    m.fs.T_warm.fix(124 + 273)

    m.fs.bypass1 = Var(initialize=0.0604)
    m.fs.bypass1.fix(0.0604)
    m.fs.bypass2 = Var(initialize=1 - 0.723)
    m.fs.bypass2.fix(1 - 0.723)

    m.fs.bypass1_con = Constraint(
        expr=m.fs.absorber.liquid_properties[0, 0].flow_mol
        == m.fs.F_bottom + m.fs.F_cold
    )
    m.fs.bypass1_ratio_con = Constraint(
        expr=m.fs.F_cold
        == m.fs.bypass1 * m.fs.absorber.liquid_properties[0, 0].flow_mol
    )
    m.fs.bypass2_con = Constraint(
        expr=m.fs.F_warm == m.fs.F_cold + m.fs.F_bottom - m.fs.F_hot
    )
    m.fs.bypass2_ratio_con = Constraint(
        expr=m.fs.F_bottom - m.fs.F_hot == m.fs.F_bottom * m.fs.bypass2
    )
    results = solver.solve(m.fs)
    print("Bypasses and streams: " + results.solver.termination_condition)

    # Create an instance of the AFS structure
    m.fs.afs = PZ_AFS(
        vapor_phase={"property_package": m.fs.vap_properties},
        liquid_phase={"property_package": m.fs.solvent_properties},
    )

    m.fs.afs.stripper.obj = Objective(expr=0)
    m.fs.afs.flash_tank.obj = Objective(expr=0)

    # Fix column design variables
    m.fs.afs.stripper.diameter.fix(0.44)
    m.fs.afs.stripper.length.fix(4)
    # Fix column and flash pressure
    m.fs.afs.stripper.vapor_properties[0, 0].pressure.fix(670000)
    m.fs.afs.flash_tank.vapor_properties[0].pressure.fix(670000)
    # Initialize AFS
    m.fs.afs.initialize(solver, outlvl=idaeslog.INFO_HIGH)

    # Connectors from absorber to AFS inlets
    m.fs.afs.F_hot.unfix()
    m.fs.afs.F_warm.unfix()
    m.fs.afs.T_warm.unfix()
    m.fs.afs.T_hot.unfix()
    m.fs.afs.reg_bottom_connector = Constraint(expr=m.fs.F_hot == m.fs.afs.F_hot)
    m.fs.afs.reg_top_connector = Constraint(expr=m.fs.F_warm == m.fs.afs.F_warm)
    m.fs.afs.reg_bottom_temp_connector = Constraint(expr=m.fs.T_hot == m.fs.afs.T_hot)
    m.fs.afs.reg_top_temp_connector = Constraint(expr=m.fs.T_warm == m.fs.afs.T_warm)
    for j in m.fs.afs.stripper.liquid_properties.component_list:
        m.fs.afs.abs_x[j].unfix()

    def reg_abs_connect(m, j):
        return m.absorber.liquid_properties[0, 0].mole_frac_comp[j] == m.afs.abs_x[j]

    m.fs.reg_composition_con = Constraint(
        m.fs.afs.flash_tank.vapor_properties[0].component_list, rule=reg_abs_connect
    )

    m.fs.absorber.del_component(m.fs.absorber.obj)
    m.fs.afs.stripper.del_component(m.fs.afs.stripper.obj)
    m.fs.obj = Objective(expr=0)
    results = solver.solve(m.fs)
    print("Connectors from absorber to AFS: " + results.solver.termination_condition)

    m.fs.x_prime = Var(
        m.fs.afs.flash_tank.liquid_properties.component_list,
        bounds=(0, 1),
        doc="Mol fractions of lean stream after make-up",
    )
    m.fs.M_w = Var(initialize=3, doc="Make-up of water added mol/s")
    m.fs.M_pz = Var(initialize=0.01, doc="Make-up of PZ added mol/s")
    m.fs.F_l_prime = Var(
        initialize=100, bounds=(0, None), doc="Flowrate of lean solvent after make-up"
    )

    def new_x(m, j):
        if j == "CO2":
            return (
                m.x_prime[j] * m.F_l_prime
                == m.afs.flash_tank.liquid_properties[0].mole_frac_comp[j]
                * m.afs.flash_tank.liquid_properties[0].flow_mol
            )
        if j == "H2O":
            return (
                m.x_prime[j] * m.F_l_prime
                == m.afs.flash_tank.liquid_properties[0].mole_frac_comp[j]
                * m.afs.flash_tank.liquid_properties[0].flow_mol
                + m.M_w
            )
        elif j == "PZ":
            return (
                m.x_prime[j] * m.F_l_prime
                == m.afs.flash_tank.liquid_properties[0].mole_frac_comp[j]
                * m.afs.flash_tank.liquid_properties[0].flow_mol
                + m.M_pz
            )

    m.fs.new_x_con = Constraint(
        m.fs.afs.flash_tank.liquid_properties.component_list, rule=new_x
    )

    def make_up(m):
        return (
            m.afs.flash_tank.liquid_properties[0].flow_mol + m.M_w + m.M_pz
            == m.F_l_prime
        )

    m.fs.make_up_cons = Constraint(rule=make_up)

    m.fs.F_l_prime.fix(m.fs.absorber.liquid_properties[0, N].flow_mol.value)
    m.fs.x_prime["PZ"].fix(
        m.fs.absorber.liquid_properties[0, N].mole_frac_comp["PZ"].value
    )

    results = solver.solve(m.fs)
    print("Calculation of make-up needed: " + results.solver.termination_condition)

    m.fs.absorber.liquid_properties[0, N].flow_mol.unfix()
    for j in m.fs.afs.flash_tank.liquid_properties.component_list:
        m.fs.absorber.liquid_properties[0, N].mole_frac_comp[j].unfix()

    def connect_to_abs(m, j):
        return m.absorber.liquid_properties[0, N].mole_frac_comp[j] == m.x_prime[j]

    m.fs.connect_to_abs_con = Constraint(
        m.fs.afs.flash_tank.liquid_properties.component_list, rule=connect_to_abs
    )

    def connect_to_abs2(m):
        return m.absorber.liquid_properties[0, N].flow_mol == m.F_l_prime

    m.fs.connect_to_abs_con2 = Constraint(rule=connect_to_abs2)

    results = solver.solve(m.fs)
    print("Close recycle loop: " + results.solver.termination_condition)

    m.fs = heat_exchangers(m.fs, N, intercooler=True)
    if flue_gas == "coal" and scale == "commercial":
        m.fs.T_hot.unfix()
        m.fs.T_rich_1.unfix()
    results = solver.solve(m.fs)
    print("Heat exchangers: " + results.solver.termination_condition)

    if flue_gas == "coal" and scale == "commercial":
        m.fs.T_hot.fix(m.fs.T_hot.value)
    m.fs = economic_model(m.fs, N)
    results = solver.solve(m.fs)
    print("Economy: " + results.solver.termination_condition)
    print(f"DoFs: {degrees_of_freedom(m)}")

    if optimization:
        if scale == "pilot":
            if flue_gas == "coal":
                m.fs, results = optimize_coal_pilot_plant(m.fs, solver, N)
            elif flue_gas == "ngcc":
                m.fs, results = optimize_ngcc_pilot_plant(m.fs, solver, N)
        elif scale == "commercial":
            if flue_gas == "coal":
                m.fs, results = optimize_coal_commercial_plant(
                    m.fs, solver, N, split_train
                )
            elif flue_gas == "ngcc":
                m.fs, results = optimize_ngcc_commercial_plant(
                    m.fs, solver, N, split_train
                )

    # report results
    for v in m.component_data_objects(Var):
        print(v.name + ": " + str(v.value))
    print(results.solver.termination_condition)
    print(f"DoFs: {degrees_of_freedom(m)}")

    # assertions
    m.fs.absorber.check_model()
    m.fs.afs.stripper.check_model()

    # flowsheet level assertions
    def species_mb_check_flowsheet(m, j):
        if j == "CO2":
            return (
                m.absorber.vapor_properties[0, 0].flow_mol_comp[j]
                - m.absorber.vapor_properties[0, N].flow_mol_comp[j]
                - m.afs.stripper.vapor_properties[0, N].flow_mol_comp[j]
            ) / m.absorber.vapor_properties[0, 0].flow_mol_comp[j]
        elif j == "H2O":
            return (
                m.absorber.vapor_properties[0, 0].flow_mol_comp[j]
                + m.M_w
                - m.absorber.vapor_properties[0, N].flow_mol_comp[j]
                - m.afs.stripper.vapor_properties[0, N].flow_mol_comp[j]
            ) / (m.absorber.vapor_properties[0, 0].flow_mol_comp[j] + m.M_w)
        else:
            return (
                m.absorber.vapor_properties[0, 0].flow_mol_comp[j]
                + m.M_pz
                - m.absorber.vapor_properties[0, N].flow_mol_comp[j]
                - m.afs.stripper.vapor_properties[0, N].flow_mol_comp[j]
            ) / (m.absorber.vapor_properties[0, 0].flow_mol_comp[j] + m.M_pz)

    m.fs.species_mb_check_flowsheet = Expression(
        ["CO2", "H2O", "PZ"], rule=species_mb_check_flowsheet
    )

    for j in m.fs.afs.flash_tank.vapor_properties[0].component_list:
        if (
            value(m.fs.species_mb_check_flowsheet[j]) >= 1e-10
            or value(m.fs.species_mb_check_flowsheet[j]) <= -1e-10
        ):
            print(
                "Warning: mass balance for "
                + j
                + " does not close off, "
                + str(value(m.species_mb_check_flowsheet[j]))
            )

    if (
        value(m.fs.afs.flash_tank.x_sum_flash.expr) < 0.9999
        or value(m.fs.afs.flash_tank.x_sum_flash.expr) > 1.0001
    ):
        print("Warning: sum of liquid mole fractions in flash tank don" "t add up to 1")
    if (
        value(m.fs.afs.flash_tank.y_sum_flash.expr) < 0.9999
        or value(m.fs.afs.flash_tank.y_sum_flash.expr) > 1.0001
    ):
        print("Warning: sum of gas mole fractions in flash tank don" "t add up to 1")
    if value(m.fs.tmin_ex1_1.expr) < 0 or value(m.fs.tmin_ex1_2.expr) < 0:
        print("Warning: Tmin violated for exchanger 1")
    if value(m.fs.tmin_ex2_1.expr) < 0 or value(m.fs.tmin_ex2_2.expr) < 0:
        print("Warning: Tmin violated for exchanger 2")
    if value(m.fs.tmin_ex3_1.expr) < 0 or value(m.fs.tmin_ex3_2.expr) < 0:
        print("Warning: Tmin violated for exchanger 3")
    if m.fs.M_w.value < 0:
        print("Make-up of water is negative: " + str(m.M_w.value))
    if m.fs.M_pz.value < 0:
        print("Make-up of PZ is negative: " + str(m.M_pz.value))
    if (
        m.fs.absorber.liquid_properties[0, N].mole_frac_comp["CO2"].value
        / 2
        / m.fs.absorber.liquid_properties[0, N].mole_frac_comp["PZ"].value
        < 0.2
    ):
        print(
            "Low lean loading: "
            + str(
                m.fs.absorber.liquid_properties[0].mole_frac_comp["CO2"].value
                / 2
                / m.fs.absorber.liquid_properties[0, N].mole_frac_comp["PZ"].value
            )
        )

    print(
        "Capture in absorber: ",
        round(
            (
                (
                    m.fs.absorber.vapor_properties[0, 0].flow_mol.value
                    * m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["CO2"].value
                    - m.fs.absorber.vapor_properties[0, N].flow_mol.value
                    * m.fs.absorber.vapor_properties[0, N].mole_frac_comp["CO2"].value
                )
                / (
                    m.fs.absorber.vapor_properties[0, 0].flow_mol.value
                    * m.fs.absorber.vapor_properties[0, 0].mole_frac_comp["CO2"].value
                )
            )
            * 100,
            2,
        ),
    )
    print(
        "Capture: ",
        round(
            100
            * (m.fs.afs.stripper.vapor_properties[0, N].flow_mol_comp["CO2"].value)
            / (m.fs.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"].value),
            2,
        ),
    )

    if split_train:
        print("Total cost (two trains): " + str(2 * m.fs.total_cost.value))
    else:
        print("Total cost: " + str(m.fs.total_cost.value))
    print(
        "cost of capture: "
        + str(
            round(
                m.fs.total_cost.value
                / m.fs.afs.stripper.vapor_properties[0, N].flow_mol_comp["CO2"].value
                / 8000
                / 3600
                / 44
                * (10**6),
                2,
            )
        )
    )


if __name__ == "__main__":
    flue_gas = sys.argv[1]  # 'ngcc' or 'coal
    scale = sys.argv[2]  # 'commercial' or 'pilot
    optimization = eval(sys.argv[3])  # Bool
    split_train = eval(sys.argv[4])  # Bool
    flowheet_model(flue_gas, scale, optimization, split_train)
