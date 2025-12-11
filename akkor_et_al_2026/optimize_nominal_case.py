from pyomo.environ import *


def optimize_plant(m, solver, num_segments):
    # Set economic objective, subject to lower initial capture target
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

    # Adjust tank models for the two-stage setting
    m.del_component(m.condenser_tank_volume_con)
    m.condenser_tank_volume_con = Constraint(
        expr=m.condenser_tank_volume >= (
            m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["H2O"]
            * m.afs.stripper.vapor_properties.params.mw_comp["H2O"]
            / 987.09
            + m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
            * 8.314
            * m.afs.stripper.vapor_properties[0, num_segments].temperature
            / m.afs.stripper.vapor_properties[0, num_segments].pressure
        )
        * 1
        * 60
    )
    results = solver.solve(m)
    print("condenser tank: " + results.solver.termination_condition)

    m.del_component(m.flash_tank_volume_con)
    m.flash_tank_volume_con = Constraint(
        expr=m.flash_tank_volume >= (m.afs.flash_tank.liquid_properties[0].flow_mol * m.flash_tank_liq_mw
              / m.afs.stripper.liquid_properties[0, 0].dens_mass)
        * 1
        * 60,
        doc="Assuming 1 min residence time",
    )
    results = solver.solve(m)
    print("flash tank: " + results.solver.termination_condition)
    
    # Adjust heater model for the two-stage setting
    m.T_sat = Var(initialize=178)
    m.steam_flowrate.setlb(0)
    m.del_component(m.h_steam)
    m.P_steam = Var(initialize=950, bounds=(750, 950))

    m.T_prediction_con = Constraint(expr=m.T_sat == 0.0496 * m.P_steam + 130.7)

    m.h_steam_variation = Param(initialize=1, mutable=True)
    m.h_steam = Expression(expr=m.h_steam_variation * (-3.4169 * m.T_sat + 2630.3) * 10 ** 3)

    m.del_component(m.steam_side_con)

    def steam_side(m):
        return m.Q_heater == m.h_steam * m.steam_flowrate

    m.steam_side_con = Constraint(
        rule=steam_side, doc="Heat calculation on the steam side"
    )

    m.del_component(m.q_lmtd_heater)
    m.tmin_heater_1_new = Expression(expr=m.T_sat + 273 - m.T_rich_2)
    m.tmin_heater_2_new = Expression(expr=m.T_sat + 273 - m.T_hot)

    def q_lmtd_heater(m):
        return m.Q_heater == m.U_heater * m.A_heater * \
               (
                       m.tmin_heater_1_new * m.tmin_heater_2_new * (m.tmin_heater_1_new + m.tmin_heater_2_new) / 2
               ) ** (1 / 3)

    m.q_lmtd_heater = Constraint(rule=q_lmtd_heater)

    results = solver.solve(m)
    print("Add steam outlet temperature: " + results.solver.termination_condition)

    # Adjust cooler model for the two-stage setting
    m.T_cooler_out = Var(initialize=35)
    m.T_cooler_out.setub(40)
    m.cw_cooler.setlb(0)

    m.del_component(m.cw_cooler_con)
    m.cw_cooler_con = Constraint(
        expr=m.Q_cooler == -m.cw_cooler * m.cp_cw * (m.T_cooler_out - 20),
        doc="Heat calculation on cooling water side",
    )

    m.del_component(m.q_lmtd_cooler)
    m.tmin_cooler_2_new = Expression(expr=m.T_lean_2 - (m.T_cooler_out + 273))
    m.q_lmtd_cooler = Constraint(
        expr=-m.Q_cooler
        == m.U_cooler
        * m.A_cooler
        * (m.tmin_cooler_1 * m.tmin_cooler_2_new * (m.tmin_cooler_1 + m.tmin_cooler_2_new) / 2)
        ** (1 / 3)
    )

    results = solver.solve(m)
    print("Add cooler outlet temperature: " + results.solver.termination_condition)

    # Add temperature crossover constraints for heat exchangers
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

    # Add column flooding constraints and free design variables
    m.afs.stripper.flooding_velocity.unfix()
    m.afs.stripper.flooding_velocity_eq.activate()
    results = solver.solve(m)
    print(
        "Stripper flooding velocity calculations: "
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

    # Update capture constraint to 95%
    m.del_component(m.capture_con)
    m.capture_con = Constraint(
        expr=m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
             / m.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"]
             >= 0.95
    )
    results = solver.solve(m)
    print("Capture constraint: " + results.solver.termination_condition)

    # Incrementally approach feed gas flowrate to be processed (15,000 mol/s)
    m.absorber.vapor_properties[0, 0].flow_mol.fix(13600)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14000)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14200)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14500)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14700)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14800)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14900)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14915)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14930)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14950)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14960)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14970)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(14985)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    m.absorber.vapor_properties[0, 0].flow_mol.fix(15000)
    results = solver.solve(m)
    print("Increase feed: " + results.solver.termination_condition)

    return m
