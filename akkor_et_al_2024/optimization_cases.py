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

from pyomo.environ import *


def optimize_coal_pilot_plant(m, solver, num_segments):
    m.del_component(m.obj)
    m.objective = Objective(expr=m.total_cost, sense=minimize)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.93
    )
    results = solver.solve(m)
    print("Objective + capture target: " + results.solver.termination_condition)

    m.ex1_1_con = Constraint(expr=m.tmin_ex1_1 >= 1)
    m.ex1_2_con = Constraint(expr=m.tmin_ex1_2 >= 1)
    m.ex2_1_con = Constraint(expr=m.tmin_ex2_1 >= 1)
    m.ex2_2_con = Constraint(expr=m.tmin_ex2_2 >= 1)
    m.ex3_1_con = Constraint(expr=m.tmin_ex3_1 >= 1)
    m.ex3_2_con = Constraint(expr=m.tmin_ex3_2 >= 1)

    m.cooler_1_con = Constraint(expr=m.tmin_cooler_1 >= 1)
    m.cooler_2_con = Constraint(expr=m.tmin_cooler_2 >= 1)

    m.intercooler_1_con = Constraint(expr=m.tmin_intercooler_1 >= 1)
    m.intercooler_2_con = Constraint(expr=m.tmin_intercooler_2 >= 1)

    results = solver.solve(m)
    print("Temperature crossover constraints: " + results.solver.termination_condition)

    m.afs.stripper.flooding_velocity.unfix()
    m.afs.stripper.flooding_velocity_eq.activate()
    results = solver.solve(m)
    print("Stripper flooding calculations: " + results.solver.termination_condition)

    m.absorber.length.unfix()
    results = solver.solve(m)
    print("Free column length: " + results.solver.termination_condition)

    m.absorber.diameter.unfix()

    def absorber_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.absorber.flooding_ub_con = Constraint(
        m.absorber.length_domain, rule=absorber_flooding_ub
    )

    def absorber_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.absorber.flooding_lb_con = Constraint(
        m.absorber.length_domain, rule=absorber_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free absorber diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.T_final.unfix()
    m.T_final.setub(78 + 273)
    m.T_rich_2.unfix()
    m.T_rich_1.unfix()
    m.A1.setlb(0)
    m.A3.setlb(0)
    m.A2.setlb(0)
    results = solver.solve(m)
    print(
        "Free heat exchanger outlet temperatures: "
        + results.solver.termination_condition
    )

    m.bypass2.unfix()
    m.bypass2.setlb(0)
    m.bypass2.setub(1)
    m.bypass1.unfix()
    m.bypass1.setlb(0)
    m.bypass1.setub(1)
    m.T_warm.setlb(120 + 273)
    results = solver.solve(m)
    print("Free bypass ratios: " + results.solver.termination_condition)

    m.T_hot.unfix()
    m.T_hot.setub(156 + 273)
    m.absorber.T_intercooler.unfix()
    m.absorber.liquid_properties[0, num_segments].temperature.unfix()
    m.M_w.setlb(0)
    results = solver.solve(m)
    print(
        "Free heater/cooler outlet temperatures: "
        + results.solver.termination_condition
    )

    m.afs.stripper.diameter.unfix()

    def reg_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.afs.stripper.flooding_ub_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_ub
    )

    def reg_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.afs.stripper.flooding_lb_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free stripper diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.F_l_prime.unfix()
    m.x_prime["PZ"].unfix()
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setlb(0.3)
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setub(0.31)

    results = solver.solve(m)
    print("Free solvent flowrate: " + results.solver.termination_condition)

    m.del_component(m.capture_con)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.94
    )
    results = solver.solve(m)
    print("Update capture constraint: " + results.solver.termination_condition)

    m.del_component(m.capture_con)
    m.T_warm.setlb(None)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.96
    )
    results = solver.solve(m)
    print("Update capture constraint: " + results.solver.termination_condition)

    m.afs.stripper.length.unfix()
    results = solver.solve(m)
    print("Free column length: " + results.solver.termination_condition)

    return m, results


def optimize_ngcc_pilot_plant(m, solver, num_segments):
    m.del_component(m.obj)
    m.objective = Objective(expr=m.total_cost, sense=minimize)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.93
    )
    results = solver.solve(m)
    print("Objective + capture target: " + results.solver.termination_condition)

    m.ex1_1_con = Constraint(expr=m.tmin_ex1_1 >= 1)
    m.ex1_2_con = Constraint(expr=m.tmin_ex1_2 >= 1)
    m.ex2_1_con = Constraint(expr=m.tmin_ex2_1 >= 1)
    m.ex2_2_con = Constraint(expr=m.tmin_ex2_2 >= 1)
    m.ex3_1_con = Constraint(expr=m.tmin_ex3_1 >= 1)
    m.ex3_2_con = Constraint(expr=m.tmin_ex3_2 >= 1)

    m.cooler_1_con = Constraint(expr=m.tmin_cooler_1 >= 1)
    m.cooler_2_con = Constraint(expr=m.tmin_cooler_2 >= 1)

    m.intercooler_1_con = Constraint(expr=m.tmin_intercooler_1 >= 1)
    m.intercooler_2_con = Constraint(expr=m.tmin_intercooler_2 >= 1)

    results = solver.solve(m)
    print("Temperature crossover constraints: " + results.solver.termination_condition)

    m.afs.stripper.flooding_velocity.unfix()
    m.afs.stripper.flooding_velocity_eq.activate()
    results = solver.solve(m)
    print(
        "Stripper flooding velocity calculations: "
        + results.solver.termination_condition
    )

    m.F_l_prime.unfix()
    m.x_prime["PZ"].unfix()
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setlb(0.3)
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setub(0.31)

    results = solver.solve(m)
    print("Free solvent flowrate: " + results.solver.termination_condition)

    m.absorber.length.unfix()
    results = solver.solve(m)
    print("Free column length: " + results.solver.termination_condition)

    m.absorber.diameter.unfix()

    def absorber_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.absorber.flooding_ub_con = Constraint(
        m.absorber.length_domain, rule=absorber_flooding_ub
    )

    def absorber_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.absorber.flooding_lb_con = Constraint(
        m.absorber.length_domain, rule=absorber_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free absorber diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.afs.stripper.diameter.unfix()

    def reg_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.afs.stripper.flooding_ub_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_ub
    )

    def reg_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.afs.stripper.flooding_lb_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free stripper diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.T_hot.unfix()
    m.T_hot.setub(156 + 273)
    m.absorber.T_intercooler.unfix()
    m.absorber.liquid_properties[0, num_segments].temperature.unfix()
    m.M_w.setlb(0)
    m.T_warm.setlb(60 + 273)
    results = solver.solve(m)
    print(
        "Free heater/cooler outlet temperatures: "
        + results.solver.termination_condition
    )

    m.bypass1.unfix()
    m.bypass1.setlb(0)
    m.bypass1.setub(1)
    m.bypass2.unfix()
    m.bypass2.setlb(0)
    m.bypass2.setub(1)
    results = solver.solve(m)
    print("Bypasses: " + results.solver.termination_condition)

    m.del_component(m.capture_con)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.96
    )
    results = solver.solve(m)
    print("Update capture constraint: " + results.solver.termination_condition)

    m.T_final.unfix()
    m.T_final.setub(78 + 273)
    m.T_rich_2.unfix()
    m.T_rich_1.unfix()
    m.A1.setlb(0)
    m.A3.setlb(0)
    m.A2.setlb(0)
    results = solver.solve(m)
    print(
        "Free heat exchanger outlet temperatures: "
        + results.solver.termination_condition
    )

    m.afs.stripper.length.unfix()
    m.T_warm.setlb(None)
    results = solver.solve(m)
    print("Free column lengths: " + results.solver.termination_condition)

    return m, results


def optimize_coal_commercial_plant(m, solver, num_segments, split_train):
    m.del_component(m.obj)
    m.objective = Objective(expr=m.total_cost, sense=minimize)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.93
    )
    results = solver.solve(m)
    print("Objective + capture target: " + results.solver.termination_condition)

    m.ex1_1_con = Constraint(expr=m.tmin_ex1_1 >= 1)
    m.ex1_2_con = Constraint(expr=m.tmin_ex1_2 >= 1)
    m.ex2_1_con = Constraint(expr=m.tmin_ex2_1 >= 1)
    m.ex2_2_con = Constraint(expr=m.tmin_ex2_2 >= 1)
    m.ex3_1_con = Constraint(expr=m.tmin_ex3_1 >= 1)
    m.ex3_2_con = Constraint(expr=m.tmin_ex3_2 >= 1)

    m.cooler_1_con = Constraint(expr=m.tmin_cooler_1 >= 1)
    m.cooler_2_con = Constraint(expr=m.tmin_cooler_2 >= 1)

    m.intercooler_1_con = Constraint(expr=m.tmin_intercooler_1 >= 1)
    m.intercooler_2_con = Constraint(expr=m.tmin_intercooler_2 >= 1)
    results = solver.solve(m)
    print("Temperature crossover constraints: " + results.solver.termination_condition)

    m.T_hot.unfix()
    m.T_hot.setub(156 + 273)
    m.A_heater.setlb(0)
    results = solver.solve(m)
    print("Free heater outlet: " + results.solver.termination_condition)

    m.T_final.unfix()
    m.T_final.setub(78 + 273)
    m.T_rich_2.unfix()
    m.T_rich_1.unfix()
    m.A1.setlb(0)
    m.A3.setlb(0)
    m.A2.setlb(0)

    results = solver.solve(m)
    print(
        "Temperature crossover constraints + free heat exchanger outlets: "
        + results.solver.termination_condition
    )

    m.afs.stripper.flooding_velocity.unfix()
    m.afs.stripper.flooding_velocity_eq.activate()
    results = solver.solve(m)
    print(
        "Stripper flooding velocity calculations: "
        + results.solver.termination_condition
    )

    m.absorber.length.unfix()
    results = solver.solve(m)
    print("Free column length: " + results.solver.termination_condition)

    m.absorber.diameter.unfix()
    m.absorber.del_component(m.absorber.flooding_lb_con)
    m.absorber.del_component(m.absorber.flooding_ub_con)

    def abs_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.absorber.flooding_ub_con = Constraint(
        m.absorber.length_domain, rule=abs_flooding_ub
    )

    def abs_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.absorber.flooding_lb_con = Constraint(
        m.absorber.length_domain, rule=abs_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free absorber diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.afs.stripper.diameter.unfix()

    def reg_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.afs.stripper.flooding_ub_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_ub
    )

    def reg_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.afs.stripper.flooding_lb_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free stripper diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.absorber.T_intercooler.unfix()
    m.absorber.liquid_properties[0, num_segments].temperature.unfix()
    results = solver.solve(m)
    print("Free cooler outlets: " + results.solver.termination_condition)

    m.F_l_prime.unfix()
    m.x_prime["PZ"].unfix()
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setlb(0.3)
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setub(0.31)

    results = solver.solve(m)
    print("Free solvent flowrate: " + results.solver.termination_condition)

    m.bypass1.unfix()
    m.bypass1.setlb(0)
    m.bypass1.setub(1)
    m.bypass2.unfix()
    m.bypass2.setlb(0)
    m.bypass2.setub(1)
    m.T_warm.setlb(110 + 273)
    results = solver.solve(m)
    print("Free bypasses: " + results.solver.termination_condition)

    m.M_w.setlb(0)
    m.del_component(m.capture_con)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.96
    )
    results = solver.solve(m)
    print("Update capture constraint: " + results.solver.termination_condition)

    m.afs.stripper.length.unfix()
    results = solver.solve(m)
    print("Free column length: " + results.solver.termination_condition)

    if not split_train:
        m.T_warm.setlb(None)
        m.absorber.vapor_properties[0, 0].flow_mol.fix(27000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(27500)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(28000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(28300)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29100)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29500)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29700)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(30000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

    if split_train:
        m.T_warm.setlb(None)
        m.absorber.vapor_properties[0, 0].flow_mol.fix(15000)
        results = solver.solve(m)
        print("Split train: " + results.solver.termination_condition)

    return m, results


def optimize_ngcc_commercial_plant(m, solver, num_segments, split_train):
    m.absorber.diameter.fix(m.absorber.diameter.value)
    m.del_component(m.obj)
    m.objective = Objective(expr=m.total_cost, sense=minimize)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.90
    )
    results = solver.solve(m)
    print("Objective + capture target: " + results.solver.termination_condition)

    m.ex1_1_con = Constraint(expr=m.tmin_ex1_1 >= 1)
    m.ex1_2_con = Constraint(expr=m.tmin_ex1_2 >= 1)
    m.ex2_1_con = Constraint(expr=m.tmin_ex2_1 >= 1)
    m.ex2_2_con = Constraint(expr=m.tmin_ex2_2 >= 1)
    m.ex3_1_con = Constraint(expr=m.tmin_ex3_1 >= 1)
    m.ex3_2_con = Constraint(expr=m.tmin_ex3_2 >= 1)

    m.cooler_1_con = Constraint(expr=m.tmin_cooler_1 >= 1)
    m.cooler_2_con = Constraint(expr=m.tmin_cooler_2 >= 1)

    m.intercooler_1_con = Constraint(expr=m.tmin_intercooler_1 >= 1)
    m.intercooler_2_con = Constraint(expr=m.tmin_intercooler_2 >= 1)

    results = solver.solve(m)
    print("Temperature crossover constraints: " + results.solver.termination_condition)

    m.afs.stripper.flooding_velocity.unfix()
    m.afs.stripper.flooding_velocity_eq.activate()
    results = solver.solve(m)
    print(
        "Stripper flooding velocity calculations: "
        + results.solver.termination_condition
    )

    m.T_final.unfix()
    m.T_final.setub(78 + 273)
    m.T_rich_2.unfix()
    m.T_rich_1.unfix()
    m.A1.setlb(0)
    m.A3.setlb(0)
    m.A2.setlb(0)
    results = solver.solve(m)
    print("Free heat exchanger outlets: " + results.solver.termination_condition)

    m.absorber.length.unfix()
    results = solver.solve(m)
    print("Free column length: " + results.solver.termination_condition)

    m.T_hot.unfix()
    m.T_hot.setub(156 + 273)
    m.absorber.T_intercooler.unfix()
    m.absorber.liquid_properties[0, num_segments].temperature.unfix()
    m.M_w.setlb(0)
    results = solver.solve(m)
    print("Free heater/cooler outlets: " + results.solver.termination_condition)

    m.afs.stripper.diameter.unfix()

    def reg_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.afs.stripper.flooding_ub_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_ub
    )

    def reg_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.afs.stripper.flooding_lb_con = Constraint(
        m.afs.stripper.length_domain, rule=reg_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free stripper diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.F_l_prime.unfix()
    m.x_prime["PZ"].unfix()
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setlb(0.3)
    m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ.setub(0.31)

    results = solver.solve(m)
    print("Free solvent flowrate: " + results.solver.termination_condition)

    m.del_component(m.capture_con)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.93
    )
    results = solver.solve(m)
    print("Update capture target: " + results.solver.termination_condition)

    m.absorber.diameter.unfix()
    m.absorber.del_component(m.absorber.flooding_lb_con)
    m.absorber.del_component(m.absorber.flooding_ub_con)

    def abs_flooding_ub(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] <= 0.80 * m.flooding_velocity[i]

    m.absorber.flooding_ub_con = Constraint(
        m.absorber.length_domain, rule=abs_flooding_ub
    )

    def abs_flooding_lb(m, i):
        if i == 0 or i == num_segments:
            return Constraint.Skip
        else:
            return m.vapor_velocity[i] >= 0.40 * m.flooding_velocity[i]

    m.absorber.flooding_lb_con = Constraint(
        m.absorber.length_domain, rule=abs_flooding_lb
    )

    results = solver.solve(m)
    print(
        "Free absorber diameter + add flooding constraints: "
        + results.solver.termination_condition
    )

    m.del_component(m.capture_con)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
        / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
        >= 0.96
    )
    results = solver.solve(m)
    print("Capture constraint: " + results.solver.termination_condition)

    m.bypass1.unfix()
    m.bypass1.setlb(0)
    m.bypass1.setub(1)
    m.bypass2.unfix()
    m.bypass2.setlb(0)
    m.bypass2.setub(1)
    results = solver.solve(m)
    print("Free bypasses: " + results.solver.termination_condition)

    m.afs.stripper.length.unfix()
    results = solver.solve(m)
    print("Free column lengths: " + results.solver.termination_condition)

    if not split_train:
        m.absorber.vapor_properties[0, 0].flow_mol.fix(25000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(27000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29300)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29600)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29900)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(29950)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(30000)
        results = solver.solve(m)
        print("Increase plant scale: " + results.solver.termination_condition)

    if split_train:
        m.absorber.vapor_properties[0, 0].flow_mol.fix(14000)
        results = solver.solve(m)
        print("Split train, step 1: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(14500)
        results = solver.solve(m)
        print("Split train, step 1: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(14900)
        results = solver.solve(m)
        print("Split train, step 1: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(14950)
        results = solver.solve(m)
        print("Split train, step 2: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(14980)
        results = solver.solve(m)
        print("Split train, step 2: " + results.solver.termination_condition)

        m.absorber.vapor_properties[0, 0].flow_mol.fix(15000)
        results = solver.solve(m)
        print("Split train, step 2: " + results.solver.termination_condition)

    return m, results
