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

    m.current_cost_index = Param(
        initialize=800,
        doc="CEPCI index for May 2024 preliminary, taken from Aug 2024 issue",
    )
    m.base_cost_index = Param(initialize=567, doc="Seider, costs from 2013")

    m.indirect_labor_factor = Param(initialize=0.2, mutable=True)

    m.installation_material_foundation = Param(initialize=0.05, mutable=True)
    m.installation_labor_foundation = Param(initialize=1.33, mutable=True)

    m.installation_material_steel = Param(initialize=0.05, mutable=True)
    m.installation_labor_steel = Param(initialize=0.5, mutable=True)

    m.installation_material_instruments = Param(initialize=0.06, mutable=True)
    m.installation_labor_instruments = Param(initialize=0.4, mutable=True)

    m.installation_material_piping = Param(initialize=0.45, mutable=True)
    m.installation_labor_piping = Param(initialize=0.50, mutable=True)

    # Purchased equipment costs
    # Columns purchased equipment cost
    m.column_vessel_density = Param(initialize=0.286, doc="lb/in3, steel")

    m.column_ref_W = Param(initialize=45000, doc="lb")
    m.steel_material_factor = Param(initialize=1.7, doc="304 SS shell material")
    m.column_DR_factor = Param(
        initialize=140,
        doc="$/ft2, installed cost for liquid distributors and redistributors",
    )
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
        doc="inch, assuming a wall thickness of 1.25 inch",
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
        expr=m.column_ref_vessel_cost * (m.abs_W / m.column_ref_W) ** 0.6,
        doc="Scale the reference cost with the sixth-tenths rule",
    )

    #       packing cost
    m.abs_packed_volume = Expression(
        expr=0.25 * np.pi * m.absorber.diameter**2 * m.absorber.length * 35.3147,
        doc="ft3",
    )
    m.abs_packing_price = Param(initialize=250, doc="$/ft3")
    m.abs_total_packing_cost = Expression(
        expr=m.abs_packed_volume * m.abs_packing_price
    )
    #       total cost = vessel  material factor + packing + distributors
    m.abs_bare_cost = Var()
    m.abs_bare_cost_calculation = Constraint(
        expr=(
            m.abs_bare_cost
            == (
                m.steel_material_factor * m.abs_vessel_cost
                + m.abs_total_packing_cost
                + m.column_DR_factor * m.absorber.cross_sectional_area * 10.7639
            )
        )
    )

    m.abs_PEC = Expression(
        expr=m.abs_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
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
        expr=m.column_ref_vessel_cost * (m.reg_W / m.column_ref_W) ** 0.6
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
        expr=m.reg_packed_volume * m.reg_packing_price
    )

    m.reg_bare_cost = Var()
    m.reg_bare_cost_calculation = Constraint(
        expr=(
            m.reg_bare_cost
            == m.steel_material_factor * m.reg_vessel_cost
            + m.reg_total_packing_cost
            + m.column_DR_factor * m.afs.stripper.cross_sectional_area * 10.7639
        )
    )

    m.reg_PEC = Expression(
        expr=m.reg_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )

    # Heat exchangers
    m.hex_1_bare_cost = Var()
    m.hex_2_bare_cost = Var()
    m.hex_3_bare_cost = Var()

    m.hex_ref_area = Param(initialize=150, doc="ft2")
    m.hex_ref_cost = Expression(
        expr=exp(
            12.0310
            - 0.8709 * log(m.hex_ref_area)
            + 0.09005 * (log(m.hex_ref_area)) ** 2
        ),
        doc="Floating head type exchanger costing",
    )

    m.hex_1_bare_cost_calculation = Constraint(
        expr=m.hex_1_bare_cost
        == m.steel_material_factor
        * m.hex_ref_cost
        * (m.A1 * 10.764 / m.hex_ref_area) ** 0.6
    )

    m.hex_1_PEC = Expression(
        expr=m.hex_1_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )

    m.hex_2_bare_cost_calculation = Constraint(
        expr=m.hex_2_bare_cost
        == m.steel_material_factor
        * m.hex_ref_cost
        * (m.A2 * 10.764 / m.hex_ref_area) ** 0.6
    )

    m.hex_2_PEC = Expression(
        expr=m.hex_2_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )

    m.hex_3_bare_cost_calculation = Constraint(
        expr=m.hex_3_bare_cost
        == m.steel_material_factor
        * m.hex_ref_cost
        * (m.A3 * 10.764 / m.hex_ref_area) ** 0.6
    )

    m.hex_3_PEC = Expression(
        expr=m.hex_3_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )
    m.hex_tot_PEC = Var()

    m.hex_tot_PEC_calculation = Constraint(
        expr=m.hex_tot_PEC == m.hex_1_PEC + m.hex_2_PEC + m.hex_3_PEC
    )

    # Heaters
    m.heater_bare_cost = Var()

    m.heater_ref_cost = Expression(
        expr=exp(
            12.3310
            - 0.8709 * log(m.hex_ref_area)
            + 0.09005 * (log(m.hex_ref_area)) ** 2
        ),
        doc="Kettle vaporizer type heater costing",
    )

    m.heater_bare_cost_calculation = Constraint(
        expr=m.heater_bare_cost
        == m.steel_material_factor
        * m.heater_ref_cost
        * (m.A_heater * 10.764 / m.hex_ref_area) ** 0.6
    )

    m.heater_PEC = Expression(
        expr=m.heater_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )

    # Coolers
    m.cooler_bare_cost = Var()

    m.cooler_bare_cost_calculation = Constraint(
        expr=m.cooler_bare_cost
        == m.steel_material_factor
        * m.hex_ref_cost
        * (m.A_cooler * 10.764 / m.hex_ref_area) ** 0.6,
        doc="Floating head type exchanger costing",
    )

    m.cooler_PEC = Expression(
        expr=m.cooler_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )

    m.intercooler_bare_cost = Var()

    m.intercooler_bare_cost_calculation = Constraint(
        expr=m.intercooler_bare_cost
        == m.steel_material_factor
        * m.hex_ref_cost
        * (m.A_intercooler * 10.764 / m.hex_ref_area) ** 0.6
    )

    m.intercooler_PEC = Expression(
        expr=m.intercooler_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
        )
    )

    #   tanks (flash, condenser)

    m.tank_ref_W = Param(initialize=1000, doc="lb")
    m.tank_ref_cost = Expression(
        expr=exp(
            5.6336 + 0.4599 * log(m.tank_ref_W) + 0.00582 * (log(m.tank_ref_W)) ** 2
        ),
        doc="horizontal pressure vessel",
    )

    m.flash_tank_liq_mw = Expression(
        expr=sum(
            m.afs.flash_tank.liquid_properties[0].mole_frac_comp[j]
            * m.afs.flash_tank.liquid_properties.params.mw_comp[j]
            for j in ["CO2", "H2O", "PZ"]
        )
    )

    m.flash_tank_volume = Var()
    m.flash_tank_volume_con = Constraint(
        expr=m.flash_tank_volume == (m.afs.flash_tank.liquid_properties[0].flow_mol * m.flash_tank_liq_mw
              / m.afs.stripper.liquid_properties[0, 0].dens_mass)
        * 1
        * 60,
        doc="Assuming 1 min residence time",
    )

    m.D_flash_tank = Var(initialize=m.afs.stripper.diameter* 39.37)
    m.D_flash_tank_con = Constraint(expr=m.D_flash_tank==m.afs.stripper.diameter * 39.37, doc="inch")
    m.L_flash_tank = Var()
    m.L_flash_tank_con = Constraint(
        expr=m.L_flash_tank == 4 * m.flash_tank_volume * 61023.7 / np.pi / m.D_flash_tank**2,
    )

    m.W_flash_tank = Expression(
        expr=np.pi
        * (m.D_flash_tank + 1.25)
        * (m.L_flash_tank + 0.8 * m.D_flash_tank)
        * 1.25
        * m.column_vessel_density
    )

    m.flash_tank_bare_cost = Var()

    m.flash_tank_bare_cost_calculation = Constraint(
        expr=m.flash_tank_bare_cost
        == m.steel_material_factor
        * m.tank_ref_cost
        * (m.W_flash_tank / m.tank_ref_W) ** 0.6
    )

    m.flash_tank_PEC = Expression(
        expr=m.flash_tank_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
        )
    )

    m.condenser_tank_volume = Var()
    m.condenser_tank_volume_con = Constraint(
        expr=m.condenser_tank_volume == (
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

    m.L_condenser_tank = Var()
    m.L_condenser_tank_con = Constraint(
        expr=m.L_condenser_tank == ((m.condenser_tank_volume / np.pi) ** (1 / 3)) * 39.37,
        doc="inch, assuming 2:1 D:L ratio",
    )

    m.D_condenser_tank = Var()
    m.D_condenser_tank_con = Constraint(expr=m.D_condenser_tank == 2 * m.L_condenser_tank)  # inch

    m.W_condenser_tank = Expression(
        expr=np.pi
        * (m.D_condenser_tank + 1.25)
        * (m.L_condenser_tank + 0.8 * m.D_condenser_tank)
        * 1.25
        * m.column_vessel_density
    )

    m.condenser_tank_bare_cost = Var()

    m.condenser_tank_bare_cost_calculation = Constraint(
        expr=m.condenser_tank_bare_cost
        == m.steel_material_factor
        * m.tank_ref_cost
        * (m.W_condenser_tank / m.tank_ref_W) ** 0.6
    )

    m.condenser_PEC = Expression(
        expr=m.condenser_tank_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
        )
    )

    # Pumps
    #   Reference base pump cost
    m.pump_ref_s = Param(initialize=800, doc="(gpm)(ft)^0.5")
    m.pump_ref_cost = Expression(
        expr=exp(
            12.1656 - 1.1448 * log(m.pump_ref_s) + 0.0862 * (log(m.pump_ref_s)) ** 2
        )
    )
    #   Reference electric motor cost
    m.pump_ref_P_c = Param(initialize=10, doc="Hp")
    m.pump_ref_electric_motor_cost = Expression(
        expr=exp(
            5.9332
            + 0.16829 * log(m.pump_ref_P_c)
            - 0.110056 * (log(m.pump_ref_P_c)) ** 2
            + 0.071413 * (log(m.pump_ref_P_c)) ** 3
            - 0.0063788 * (log(m.pump_ref_P_c)) ** 4
        )
    )
    m.pump_efficiency = Param(initialize=0.75, mutable=True)
    m.motor_efficiency = Param(initialize=0.85, mutable=True)

    # pump 1 (absorber to stripper)
    m.pump_1_flow = Expression(
        expr=m.absorber.liquid_properties[0, 0].flow_mol
        * (
            (
                sum(
                    m.absorber.liquid_properties[0, 0].mole_frac_comp[j]
                    * m.absorber.liquid_properties[0, 0].mw_comp[j]
                    for j in ["CO2", "H2O", "PZ"]
                )
            )
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

    m.pump_1_s = Var()
    m.pump_1_s_con = Constraint(expr=m.pump_1_s >= m.pump_1_flow * m.pump_1_head**0.5)  # Adjusted for the two-stage setting

    m.pump_1_base_cost = Expression(
        expr=m.steel_material_factor
        * m.pump_ref_cost
        * (m.pump_1_s / m.pump_ref_s) ** 0.6
    )

    # motor 1
    m.pump_1_P_c = Var()
    m.pump_1_P_c_oper = Var()
    m.pump_1_P_c_con = Constraint(
        expr=m.pump_1_P_c_oper == m.pump_1_flow
        * m.pump_1_head
        * m.absorber.liquid_properties[0, 0].dens_mass
        * 0.008345
        / 33000
        / m.pump_efficiency
        / m.motor_efficiency
    )

    # Adjusted for the two-stage setting
    m.pump_1_P_c_design_con = Constraint(
        expr=m.pump_1_P_c >= m.pump_1_flow
        * m.pump_1_head
        * m.absorber.liquid_properties[0, 0].dens_mass
        * 0.008345
        / 33000
        / m.pump_efficiency
        / m.motor_efficiency
    )

    m.pump_1_electric_motor_cost = Var()
    m.pump_1_electric_motor_cost_con = Constraint(
        expr=m.pump_1_electric_motor_cost ==
             m.pump_ref_electric_motor_cost * (m.pump_1_P_c / m.pump_ref_P_c) ** 0.6
    )

    # pump 1: total cost
    m.pump_1_bare_cost = Var()
    m.pump_1_bare_cost_calculation = Constraint(
        expr=m.pump_1_bare_cost == m.pump_1_base_cost + m.pump_1_electric_motor_cost
    )

    m.pump_1_PEC = Expression(
        expr=m.pump_1_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
        )
    )

    # pump 2
    m.pump_2_flow = Expression(
        expr=m.F_l_prime
        * sum(
            m.x_prime[j] * m.absorber.liquid_properties.params.mw_comp[j]
            for j in ["CO2", "H2O", "PZ"]
        )
        * 60
        * 264.17
        / m.afs.stripper.liquid_properties[0, 0].dens_mass,
        doc="gpm",
    )

    m.pump_2_head = Expression(expr=m.absorber.length * 3.28, doc="ft")

    m.pump_2_s = Var()
    m.pump_2_s_con = Constraint(expr=m.pump_2_s >= m.pump_2_flow * m.pump_2_head**0.5)

    m.pump_2_base_cost = Expression(
        expr=m.steel_material_factor
        * m.pump_ref_cost
        * (m.pump_2_s / m.pump_ref_s) ** 0.6
    )

    # motor 2
    m.pump_2_P_c = Var()
    m.pump_2_P_c_oper = Var()
    m.pump_2_P_c_con = Constraint(
        expr=m.pump_2_P_c_oper == m.pump_2_flow
        * m.pump_2_head
        * m.absorber.liquid_properties[0, num_segments - 1].dens_mass
        * 0.008345
        / 33000
        / m.pump_efficiency
        / m.motor_efficiency
    )

    m.pump_2_P_c_design_con = Constraint(
        expr=m.pump_2_P_c >= m.pump_2_flow
        * m.pump_2_head
        * m.absorber.liquid_properties[0, num_segments - 1].dens_mass
        * 0.008345
        / 33000
        / m.pump_efficiency
        / m.motor_efficiency
    )

    m.pump_2_electric_motor_cost = Var()
    m.pump_2_electric_motor_cost_con = Constraint(
        expr=m.pump_2_electric_motor_cost ==
             m.pump_ref_electric_motor_cost * (m.pump_2_P_c / m.pump_ref_P_c) ** 0.6
    )

    # pump 2: total cost
    m.pump_2_bare_cost = Var()
    m.pump_2_pc_calculation = Constraint(
        expr=m.pump_2_bare_cost == m.pump_2_base_cost + m.pump_2_electric_motor_cost
    )

    m.pump_2_PEC = Expression(
        expr=m.pump_2_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_piping * (1 + m.installation_labor_piping)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
        )
    )

    # blower
    m.blower_ref_P_c = Param(initialize=100, doc="Hp")
    m.blower_ref_cost = Expression(
        expr=exp(7.0187 + 0.79 * log(m.blower_ref_P_c)), doc="centrifugal blower"
    )

    m.blower_efficiency = Param(initialize=0.75, mutable=True)
    m.specific_heat_ratio = Param(initialize=1.4, mutable=True, doc="air")
    m.flue_gas_pressure_in = Param(initialize=101.3, mutable=True, doc="kPa")
    m.flue_gas_volumetric_flowrate = Expression(
        expr=(m.absorber.vapor_properties[0, 0].flow_mol / 2)
        * sum(
            m.absorber.vapor_properties[0, 0].mole_frac_comp[j]
            * m.absorber.vapor_properties.params.mw_comp[j]
            for j in ["CO2", "H2O", "N2", "PZ"]
        )
        / m.absorber.vapor_properties[0, 1].dens_mass
        * 35.315
        * 60,
        doc="ft3/min, two blowers",
    )

    m.blower_P_c = Expression(
        expr=(
            0.00436
            * (m.specific_heat_ratio / (m.specific_heat_ratio - 1))
            * (
                m.flue_gas_volumetric_flowrate
                * m.flue_gas_pressure_in
                * 0.145
                / m.blower_efficiency
            )
            * (
                (
                    m.absorber.vapor_properties[0, 0].pressure
                    / 1000
                    / m.flue_gas_pressure_in
                )
                ** ((m.specific_heat_ratio - 1) / m.specific_heat_ratio)
                - 1
            )
        )
        / m.motor_efficiency
    )

    m.blowers_bare_cost = Expression(
        expr=2
        * m.blower_ref_cost
        * (m.blower_P_c / m.blower_ref_P_c) ** 0.6
    )

    m.blowers_PEC = Expression(
        expr=m.blowers_bare_cost
        * (
            1
            + m.indirect_labor_factor
            + m.installation_material_foundation * (1 + m.installation_labor_foundation)
            + m.installation_material_instruments
            * (1 + m.installation_labor_instruments)
            + m.installation_material_steel * (1 + m.installation_labor_steel)
        )
    )

    m.total_purchased_equipment_cost = Expression(
        expr=m.abs_PEC
        + m.reg_PEC
        + m.hex_tot_PEC
        + m.heater_PEC
        + m.cooler_PEC
        + m.intercooler_PEC
        + m.flash_tank_PEC
        + m.condenser_PEC
        + m.pump_1_PEC
        + m.pump_2_PEC
        + m.blowers_PEC
    )

    # utility costs
    m.operation_time = Param(
        initialize=8000 * 3600, doc="s/yr, assumed 8000hr operation in a year"
    )
    #   cooling water
    m.cw_price = Param(initialize=0.378, mutable=True, doc="$/GJ, Turton et al (2018)")
    m.cw_oc = Var()
    m.cw_oc_calculation = Constraint(
        expr=m.cw_oc
        == (m.cw_intercooler + m.cw_cooler)
        * 4184
        * (35 - 20)
        / 1e9
        * m.operation_time
        * m.cw_price
    )

    #   steam
    m.steam_price = Param(initialize=2.03, mutable=True, doc="$/GJ, Turton et al, 2018")
    m.steam_oc = Var()
    m.steam_oc_calculation = Constraint(
        expr=m.steam_oc
        == m.steam_flowrate * 2.022 * 10**6 / 1e9 * m.operation_time * m.steam_price
    )

    #   electricity (for pumps)
    m.elec_price = Param(
        initialize=0.025 / (3.6 * 1e6), doc="$/J ($25/MWh)", mutable=True
    )

    m.pump_elec_oc = Var()
    m.pump_elec_oc_calculation = Constraint(
        expr=m.pump_elec_oc
        == (m.pump_1_P_c_oper + m.pump_2_P_c_oper) * 745.7 * m.elec_price * m.operation_time,
        doc="1 Hp = 745.7 J/s",
    )

    m.blower_elec_oc = Var()
    m.blower_elec_oc_calculation = Constraint(
        expr=m.blower_elec_oc
        == 2 * m.blower_P_c * 745.7 * m.elec_price * m.operation_time,
        doc="1 Hp = 745.7 J/s",
    )

    # solvent renewal cost
    m.solvent_renewal_cost = Var()
    m.t_residence = Param(initialize=45 * 60, doc="45 minutes converted into seconds")
    m.pz_price = Param(initialize=3 * 1250, mutable=True, doc="$/ton solvent")
    m.solvent_renewal_cost_calculation = Constraint(
        expr=m.solvent_renewal_cost
        == m.F_l_prime
        * m.t_residence
        * sum(
            m.absorber.liquid_properties[0, num_segments].mole_frac_comp[j]
            * m.absorber.liquid_properties.params.mw_comp[j]
            for j in ["CO2", "H2O", "PZ"]
        )
        * 1e-3
        * m.absorber.liquid_properties[0, num_segments - 1].weight_percent_PZ
        * m.pz_price
    )

    # CAPEX
    m.supporting_facility_cost = Param(initialize=0.15, mutable=True)
    m.BEC = Expression(
        expr=(
            (m.total_purchased_equipment_cost * (1 + m.supporting_facility_cost))
            * (m.current_cost_index / m.base_cost_index)
            + m.solvent_renewal_cost
        ),
        doc="Bare Erected Cost",
    )

    m.EPCC_factor = Param(
        initialize=0.2,
        mutable=True,
        doc="Engineering, Procurement and Construction Cost",
    )
    m.EPCC = Expression(
        expr=m.BEC * (1 + m.EPCC_factor),
        doc="Engineering, Procurement and Construction Cost",
    )

    m.process_contingency = Param(initialize=0.2, mutable=True)
    m.project_contingency = Param(initialize=0.15, mutable=True)

    m.TPC = Expression(
        expr=(m.EPCC * (1 + m.process_contingency)) * (1 + m.project_contingency),
        doc="Total Plant Cost",
    )

    m.owner_cost_factor = Param(initialize=0.20, mutable=True)
    m.TOC = Expression(
        expr=m.TPC * (1 + m.owner_cost_factor), doc="Total Overnight Cost"
    )

    #       capital recovery factor
    m.i = Param(initialize=0.08)
    m.n = Param(initialize=25)
    m.crf = Expression(expr=(m.i * (m.i + 1) ** m.n) / ((m.i + 1) ** m.n - 1))
    m.TASC = Expression(
        expr=m.crf * m.TOC,
        doc="Total As-Spent Cost assumed to be same as Annualized Capital Costs ignoring additional financing costs",
    )

    # fixed O&M costs
    # operating labor

    m.operators_per_shift = Expression(
        expr=log10(m.absorber.vapor_properties[0, 0].flow_mol)
    )
    m.labor_burden = Param(initialize=0.3, mutable=True)
    m.labor_rate = Param(initialize=38.50, mutable=True, doc="$/hr")
    m.operating_labor = Expression(
        expr=m.operators_per_shift
        * m.labor_rate
        * (1 + m.labor_burden)
        * m.operation_time
        / 3600
    )

    # administrative labor
    m.administrative_labor_factor = Param(initialize=0.25, mutable=True)
    m.administrative_labor = Expression(
        expr=m.administrative_labor_factor * (m.operating_labor)
    )

    m.maintenance_factor = Param(
        initialize=0.015, mutable=True, doc="maintenance material and labor"
    )

    m.tax_and_insurance_factor = Param(
        initialize=0.02, mutable=True, doc="property taxes and insurance"
    )

    m.fixed_OMC = Expression(
        expr=m.operating_labor
        + m.administrative_labor
        + m.TPC * (m.maintenance_factor + m.tax_and_insurance_factor)
    )

    # variable O&M costs
    m.variable_OMC = Var()
    m.variable_OMC_calculation = Constraint(
        expr=m.variable_OMC
        == m.cw_oc
        + m.steam_oc
        + m.pump_elec_oc
        + m.blower_elec_oc
        + m.solvent_renewal_cost / 3
    )

    m.TOMC = Expression(expr=m.variable_OMC + m.fixed_OMC)

    # annualized total cost
    m.total_cost = Var()
    m.total_cost_calculation = Constraint(expr=m.total_cost == m.TASC + m.TOMC)

    return m
