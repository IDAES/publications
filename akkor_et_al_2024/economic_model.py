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
import numpy as np


def economic_model(m, num_segments):
    # Costing correlations for purchased equipments are from Warren D. Seider.
    # Product and Process Design Principles : Synthesis, Analysis, and Evaluation.

    # Cost indexes to correct for time value of money
    m.current_cost_index = Param(initialize=708)
    m.seider_cost_index = Param(initialize=567, doc="Seider")

    # Columns purchased equipment cost
    m.column_vessel_density = Param(initialize=0.284, doc="lb/in3, steel")

    m.column_ref_W = Param(initialize=45000, doc="lb")
    m.column_ref_vessel_cost = Expression(
        expr=exp(
            10.5449
            - 0.4672 * log(m.column_ref_W)
            + 0.05482 * (log(m.column_ref_W)) ** 2
        )
    )

    #   absorber
    m.abs_total_length = Expression(expr=3 * m.absorber.length * 39.37, doc="inch")
    m.abs_outer_diameter = Expression(
        expr=m.absorber.diameter * 39.37 + 1.25,
        doc="inch, assuming a wall thickness of 1.25 inc",
    )
    #       vessel cost
    m.abs_W = Expression(
        expr=np.pi
        * m.abs_outer_diameter
        * (m.abs_total_length + 0.8 * m.absorber.diameter * 39.371)
        * 1.25
        * m.column_vessel_density
    )

    m.abs_vessel_cost = Expression(
        expr=m.column_ref_vessel_cost
        * (m.abs_W / m.column_ref_W) ** 0.6
        * (m.current_cost_index / m.seider_cost_index),
        doc="Scale the reference cost with the sixth-tenths rule and current cost index",
    )
    #       packing cost
    m.abs_packed_volume = Expression(
        expr=0.25 * np.pi * m.absorber.diameter**2 * m.absorber.length * 35.3147,
        doc="ft3",
    )
    m.abs_packing_price = Param(initialize=250, doc="$/ft3")
    m.abs_total_packing_cost = Expression(
        expr=m.abs_packed_volume
        * m.abs_packing_price
        * (m.current_cost_index / m.seider_cost_index)
    )
    #       total cost = vessel + packing
    m.abs_pc = Var()
    m.abs_pc_calculation = Constraint(
        expr=(m.abs_pc == (m.abs_vessel_cost + m.abs_total_packing_cost))
    )

    #   regenerator
    m.reg_total_length = Expression(expr=2 * m.afs.stripper.length * 39.37, doc="inch")
    m.reg_outer_diameter = Expression(
        expr=m.afs.stripper.diameter * 39.37 + 1.25, doc="inch"
    )
    #       vessel cost
    m.reg_W = Expression(
        expr=np.pi
        * m.reg_outer_diameter
        * (m.reg_total_length + 0.8 * m.afs.stripper.diameter * 39.371)
        * 1.25
        * m.column_vessel_density
    )
    m.reg_vessel_cost = Expression(
        expr=m.column_ref_vessel_cost
        * (m.reg_W / m.column_ref_W) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )
    #       packing cost
    m.reg_packed_volume = Expression(
        expr=0.25
        * np.pi
        * m.afs.stripper.diameter**2
        * m.afs.stripper.length
        * 35.3147,
        doc="ft3",
    )
    m.reg_packing_price = Param(initialize=200, doc="$/ft3")
    m.reg_total_packing_cost = Expression(
        expr=m.reg_packed_volume
        * m.reg_packing_price
        * (m.current_cost_index / m.seider_cost_index)
    )
    #       total cost = vessel + packing
    m.reg_pc = Var()
    m.reg_pc_calculation = Constraint(
        expr=(m.reg_pc == m.reg_vessel_cost + m.reg_total_packing_cost)
    )

    # Heat exchangers

    m.hex_1_pc = Var()
    m.hex_2_pc = Var()
    m.hex_3_pc = Var()
    m.hex_tot_pc = Var()

    m.hex_ref_area = Param(initialize=150, doc="ft2")
    m.hex_ref_cost = Expression(
        expr=exp(
            12.0310
            - 0.8709 * log(m.hex_ref_area)
            + 0.09005 * (log(m.hex_ref_area)) ** 2
        ),
        doc="Floating head type exchanger costing",
    )

    m.hex_1_pc_calculation = Constraint(
        expr=m.hex_1_pc
        == m.hex_ref_cost
        * (m.A1 * 10.764 / m.hex_ref_area) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )
    m.hex_2_pc_calculation = Constraint(
        expr=m.hex_2_pc
        == m.hex_ref_cost
        * (m.A2 * 10.764 / m.hex_ref_area) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )
    m.hex_3_pc_calculation = Constraint(
        expr=m.hex_3_pc
        == m.hex_ref_cost
        * (m.A3 * 10.764 / m.hex_ref_area) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    m.hex_tot_pc_calculation = Constraint(
        expr=m.hex_tot_pc == m.hex_1_pc + m.hex_2_pc + m.hex_3_pc
    )

    # Heaters
    m.heater_pc = Var()

    m.heater_ref_cost = Expression(
        expr=exp(
            12.3310
            - 0.8709 * log(m.hex_ref_area)
            + 0.09005 * (log(m.hex_ref_area)) ** 2
        ),
        doc="Kettle vaporizer type heater costing",
    )
    m.heater_pc_calculation = Constraint(
        expr=m.heater_pc
        == m.heater_ref_cost
        * (m.A_heater * 10.764 / m.hex_ref_area) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # Coolers
    m.cooler_pc = Var()

    m.cooler_pc_calculation = Constraint(
        expr=m.cooler_pc
        == m.hex_ref_cost
        * (m.A_cooler * 10.764 / m.hex_ref_area) ** 0.6
        * (m.current_cost_index / m.seider_cost_index),
        doc="Floating head type exchanger costing",
    )

    m.inter_cooler_pc = Var()

    m.inter_cooler_pc_calculation = Constraint(
        expr=m.inter_cooler_pc
        == m.hex_ref_cost
        * (m.A_intercooler * 10.764 / m.hex_ref_area) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # Pumps
    #   Reference base pump cost
    m.pump_ref_s = Param(initialize=400)
    m.pump_ref_cost = Expression(
        expr=exp(
            12.1656 - 1.1448 * log(m.pump_ref_s) + 0.0862 * (log(m.pump_ref_s)) ** 2
        )
    )
    #   Reefrence electric motor cost
    m.pump_ref_P_c = Param(initialize=3, doc="Hp")
    m.pump_ref_electric_motor_cost = Expression(
        expr=exp(
            5.9332
            + 0.16892 * log(m.pump_ref_P_c)
            - 0.110056 * (log(m.pump_ref_P_c)) ** 2
            + 0.071413 * (log(m.pump_ref_P_c)) ** 3
            - 0.0063788 * (log(m.pump_ref_P_c)) ** 4
        )
    )

    # pump 1 (absorber to stripper)
    m.pump_1_flow = Expression(
        expr=m.absorber.liquid_properties[0, 0].flow_mol
        * (
            (
                m.absorber.liquid_properties[0, 0].mole_frac_comp["CO2"]
                * m.absorber.liquid_properties[0, 0].mw_comp["CO2"]
                * 1000
                + m.absorber.liquid_properties[0, 0].mole_frac_comp["H2O"]
                * m.absorber.liquid_properties[0, 0].mw_comp["H2O"]
                * 1000
                + m.absorber.liquid_properties[0, 0].mole_frac_comp["PZ"]
                * m.absorber.liquid_properties[0, 0].mw_comp["PZ"]
                * 1000
            )
            / 1000
        )
        * 60
        * 264.17
        / m.absorber.liquid_properties[0, 0].dens_mass,
        doc="gpm, volumetric",
    )

    m.pump_1_head = Expression(
        expr=(
            (
                m.afs.stripper.vapor_properties[0, num_segments].pressure
                - m.absorber.vapor_properties[0, 0].pressure
            )
            / m.absorber.liquid_properties[0, 0].dens_mass
            / 9.8
            + m.afs.stripper.length
        )
        * 3.28,
        doc="ft",
    )
    m.pump_1_s = Expression(expr=m.pump_1_flow * m.pump_1_head**0.5)

    m.pump_1_base_cost = Expression(
        expr=m.pump_ref_cost
        * (m.pump_1_s / m.pump_ref_s) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # motor 1
    m.pump_1_eta_p = Expression(
        expr=-0.316 + 0.24015 * log(m.pump_1_flow) - 0.01199 * (log(m.pump_1_flow)) ** 2
    )
    m.pump_1_P_b = Expression(
        expr=m.pump_1_flow
        * m.pump_1_head
        * m.absorber.liquid_properties[0, 0].dens_mass
        * 0.008345
        / 33000
        / m.pump_1_eta_p
    )
    m.pump_1_eta_m = Expression(
        expr=0.8 + 0.0319 * log(m.pump_1_P_b) - 0.00182 * (log(m.pump_1_P_b)) ** 2
    )
    m.pump_1_P_c = Expression(expr=m.pump_1_P_b / m.pump_1_eta_m)

    m.pump_1_electric_motor_cost = Expression(
        expr=m.pump_ref_electric_motor_cost
        * (m.pump_1_P_c / m.pump_ref_P_c) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # pump 1: total cost
    m.pump_1_pc = Var()
    m.pump_1_pc_calculation = Constraint(
        expr=m.pump_1_pc == m.pump_1_base_cost + m.pump_1_electric_motor_cost
    )

    # pump 2
    m.pump_2_flow = Expression(
        expr=m.F_l_prime
        * (
            (
                m.x_prime["CO2"]
                * m.absorber.liquid_properties[0, 0].mw_comp["CO2"]
                * 1000
                + m.x_prime["H2O"]
                * m.absorber.liquid_properties[0, 0].mw_comp["H2O"]
                * 1000
                + m.x_prime["PZ"]
                * m.absorber.liquid_properties[0, 0].mw_comp["PZ"]
                * 1000
            )
            / 1000
        )
        * 60
        * 264.17
        / m.absorber.liquid_properties[0, num_segments - 1].dens_mass,
        doc="gpm",
    )
    m.pump_2_head = Expression(expr=m.absorber.length * 3.28, doc="ft")
    m.pump_2_s = Expression(expr=m.pump_2_flow * m.pump_2_head**0.5)

    m.pump_2_base_cost = Expression(
        expr=m.pump_ref_cost
        * (m.pump_2_s / m.pump_ref_s) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # motor 2
    m.pump_2_eta_p = Expression(
        expr=-0.316 + 0.24015 * log(m.pump_2_flow) - 0.01199 * (log(m.pump_2_flow)) ** 2
    )
    m.pump_2_P_b = Expression(
        expr=m.pump_2_flow
        * m.pump_2_head
        * m.absorber.liquid_properties[0, num_segments - 1].dens_mass
        * 0.008345
        / 33000
        / m.pump_2_eta_p
    )
    m.pump_2_eta_m = Expression(
        expr=0.8 + 0.0319 * log(m.pump_2_P_b) - 0.00182 * (log(m.pump_2_P_b)) ** 2
    )
    m.pump_2_P_c = Expression(expr=m.pump_2_P_b / m.pump_2_eta_m)

    m.pump_2_electric_motor_cost = Expression(
        expr=m.pump_ref_electric_motor_cost
        * (m.pump_2_P_c / m.pump_ref_P_c) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # pump 2: total cost
    m.pump_2_pc = Var()
    m.pump_2_pc_calculation = Constraint(
        expr=m.pump_2_pc == m.pump_2_base_cost + m.pump_2_electric_motor_cost
    )

    #   tanks (flash, condenser)

    m.tank_ref_W = Param(initialize=1000, doc="lb")
    m.tank_ref_cost = Expression(
        expr=exp(
            5.6336 + 0.4599 * log(m.tank_ref_W) + 0.00582 * (log(m.tank_ref_W)) ** 2
        )
    )

    m.flash_tank_liq_mw = Expression(
        expr=m.afs.flash_tank.liquid_properties[0].mole_frac_comp["CO2"] * 44.01
        + m.afs.flash_tank.liquid_properties[0].mole_frac_comp["H2O"] * 18.02
        + m.afs.flash_tank.liquid_properties[0].mole_frac_comp["PZ"] * 86.136
    )

    m.flash_tank_volume = Expression(
        expr=(
            m.afs.flash_tank.liquid_properties[0].flow_mol
            * (m.flash_tank_liq_mw / 1000)
            / m.absorber.liquid_properties[0, 0].dens_mass
            + m.afs.flash_tank.vapor_properties[0].flow_mol
            * 8.314
            * m.afs.flash_tank.vapor_properties[0].temperature
            / m.afs.flash_tank.vapor_properties[0].pressure
        )
        * 1
        * 60,
        doc="Assuming 1 min residence time",
    )

    m.L_flash_tank = Expression(
        expr=((m.flash_tank_volume / np.pi) ** (1 / 3)) * 39.37,
        doc="inch, assuming 2:1 D:L ratio",
    )
    m.D_flash_tank = Expression(expr=2 * m.L_flash_tank, doc="inch")
    m.W_flash_tank = Expression(
        expr=np.pi
        * (m.D_flash_tank + 5 / 16)
        * (m.L_flash_tank + 0.8 * m.D_flash_tank)
        * (5 / 16)
        * m.column_vessel_density
    )

    m.flash_tank_pc = Var()

    m.flash_tank_pc_calculation = Constraint(
        expr=m.flash_tank_pc
        == m.tank_ref_cost
        * (m.W_flash_tank / m.tank_ref_W) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    m.condenser_tank_volume = Expression(
        expr=(
            m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["H2O"]
            * (18.02 / 1000)
            / 987.09
            + m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
            * 8.314
            * m.afs.stripper.vapor_properties[0, num_segments].temperature
            / m.afs.stripper.vapor_properties[0, num_segments].pressure
        )
        * 1
        * 60
    )
    m.L_condenser_tank = Expression(
        expr=((m.condenser_tank_volume / np.pi) ** (1 / 3)) * 39.37
    )  # inch
    m.D_condenser_tank = Expression(expr=2 * m.L_condenser_tank)  # inch
    m.W_condenser_tank = Expression(
        expr=np.pi
        * (m.D_condenser_tank + 5 / 16)
        * (m.L_condenser_tank + 0.8 * m.D_condenser_tank)
        * (5 / 16)
        * m.column_vessel_density
    )

    m.condenser_tank_pc = Var()

    m.condenser_tank_pc_calculation = Constraint(
        expr=m.condenser_tank_pc
        == m.tank_ref_cost
        * (m.W_condenser_tank / m.tank_ref_W) ** 0.6
        * (m.current_cost_index / m.seider_cost_index)
    )

    # operational costs
    m.operation_time = Param(
        initialize=8000 * 3600, doc="s/yr, assumed 8000hr operation in a year"
    )

    #   cooling water
    m.F_cw = Expression(
        expr=(m.cw_intercooler + m.cw_cooler) / 1000 * m.operation_time, doc="ton/year"
    )
    m.cw_price = Param(initialize=0.0329, doc="$/ton")
    m.cw_oc = Var()
    m.cw_oc_calculation = Constraint(expr=m.cw_oc == m.F_cw * m.cw_price)

    #   steam
    m.F_steam = Expression(
        expr=m.steam_flowrate / 1000 * m.operation_time, doc="ton/year"
    )
    m.steam_price = Param(initialize=14.5, doc="$/ton")
    m.steam_oc = Var()
    m.steam_oc_calculation = Constraint(expr=m.steam_oc == m.F_steam * m.steam_price)

    #   electricity (for pumps)
    m.elec_price = Param(initialize=0.06 * (1.0 / (3.6 * 1e6)), doc="$/J")
    m.pump_elec_oc = Var()
    m.pump_elec_oc_calculation = Constraint(
        expr=m.pump_elec_oc
        == (m.pump_1_P_c + m.pump_2_P_c) * 745.7 * m.elec_price * m.operation_time,
        doc="1 Hp = 745.7 J/s",
    )

    # solvent renewal cost
    m.solvent_renewal_cost = Var()
    m.MW_solvent = Expression(
        expr=m.absorber.liquid_properties[0, num_segments].mole_frac_comp["PZ"] * 86.136
        + m.absorber.liquid_properties[0, num_segments].mole_frac_comp["H2O"] * 18.02
        + m.absorber.liquid_properties[0, num_segments].mole_frac_comp["CO2"] * 44.01
    )
    m.t_residence = Param(initialize=40 * 60, doc="40 minutes converted into seconds")
    m.pz_price = Param(initialize=3 * 1250)
    m.solvent_renewal_cost_calculation = Constraint(
        expr=m.solvent_renewal_cost
        == m.F_l_prime
        * m.t_residence
        * m.MW_solvent
        * (10 ** (-6))
        * m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ
        * m.pz_price
    )

    # total cost model
    #       total purchased equipment cost
    m.tpc = Var()
    m.tpc_calculation = Constraint(
        expr=m.tpc
        == m.abs_pc
        + m.reg_pc
        + m.hex_tot_pc
        + m.cooler_pc
        + m.inter_cooler_pc
        + m.heater_pc
        + m.flash_tank_pc
        + m.condenser_tank_pc
        + m.pump_1_pc
        + m.pump_2_pc
        + m.solvent_renewal_cost
    )
    #       capital recovery factor
    m.i = Param(initialize=0.08)
    m.n = Param(initialize=25)
    m.crf = Expression(expr=(m.i * (m.i + 1) ** m.n) / ((m.i + 1) ** m.n - 1))
    m.acc = Expression(expr=m.crf * m.tpc)

    #   total operating and maintenance cost
    #       total utility cost
    m.tuc = Var()
    m.tuc_calculation = Constraint(
        expr=m.tuc == m.cw_oc + m.steam_oc + m.pump_elec_oc + m.solvent_renewal_cost / 3
    )
    m.coef_1 = Param(initialize=0.3863)
    m.coef_2 = Param(initialize=1.05)
    m.tomc = Expression(expr=m.coef_1 * m.tpc + m.coef_2 * m.tuc)
    # annualized total cost
    m.total_cost = Var()
    m.total_cost_calculation = Constraint(expr=m.total_cost == m.acc + m.tomc)

    return m
