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


def heat_exchangers(m, num_segments, intercooler):
    # Split flow configuration with two heat exchangers at the bottom
    # and one at the top

    """
                        T_final  <--   [   3   ]   <--   T[N] gas stripper
              T[0] liq absorber  -->   [   3   ]   -->   T_rich_3

    T[0] liq absorber  -->   [   1   ]  -->  T_rich_1   -->  [   2   ]  -->  T_rich_2
           T_lean_2    <--   [   1   ]  <--  T_lean_1   <--  [   2   ]  <--  T_flash

    """

    # Variables for temperatures of streams
    m.T_lean_1 = Var(initialize=131 + 273)
    m.T_lean_2 = Var(initialize=60 + 273)
    m.T_rich_1 = Var(initialize=127 + 273)
    m.T_rich_2 = Var(initialize=144 + 273)
    m.T_final = Var(initialize=78 + 273)
    m.T_rich_3 = Var(initialize=116 + 273)

    # Variables for duty of exchangers
    m.Q_cooler = Var(initialize=-98391)
    m.Q_heater = Var(initialize=62342)
    m.Q_ex1 = Var(initialize=479436)
    m.Q_ex2 = Var(initialize=127743)
    m.Q_ex3 = Var(initialize=26103)

    # Overall heat transfer coefficients
    m.U = Param(initialize=1000, doc="W/m2/K")
    m.U_cooler = Param(initialize=500)
    m.U_heater = Param(initialize=750)

    # Areas
    m.A_cooler = Var()
    m.A1 = Var(initialize=90)
    m.A2 = Var(initialize=28.3)
    m.A3 = Var(initialize=4)
    m.A_heater = Var()

    # calculate T_warm (mixture of T_rich_3 and stream from second bypass)
    m.T_warm.unfix()

    def t_warm(m):
        return (
            m.F_cold * m.T_rich_3 + m.T_rich_1 * (m.F_bottom - m.F_hot)
        ) == m.T_warm * m.F_warm

    m.T_warm_con = Constraint(rule=t_warm)

    # cooler at the top of absorber column
    m.absorber.liquid_properties[0, num_segments].temperature.fix(49 + 273)
    m.cw_cooler = Var(doc="Cooling water flowrate, kg/s")

    def q_cooler(m):
        return m.Q_cooler == m.absorber.liquid_properties[
            0, num_segments
        ].flow_mol / m.absorber.liquid_properties[
            0, num_segments
        ].conc_mol * m.absorber.liquid_properties[
            0, 0
        ].params.cp_vol * (
            m.absorber.liquid_properties[0, num_segments].temperature - m.T_lean_2
        )

    m.Q_cooler_con = Constraint(rule=q_cooler, doc="Heat calculation on solvent side")

    m.cw_cooler_con = Constraint(
        expr=m.Q_cooler == -m.cw_cooler * 4184 * (35 - 20),
        doc="Heat calculation on cooling water side",
    )

    # LMTD calculation for cooler
    m.tmin_cooler_1 = Expression(
        expr=m.absorber.liquid_properties[0, num_segments].temperature - (20 + 273)
    )
    m.tmin_cooler_2 = Expression(expr=m.T_lean_2 - (35 + 273))

    m.q_lmtd_cooler = Constraint(
        expr=-m.Q_cooler
        == m.U_cooler
        * m.A_cooler
        * (m.tmin_cooler_1 * m.tmin_cooler_2 * (m.tmin_cooler_1 + m.tmin_cooler_2) / 2)
        ** (1 / 3)
    )

    if intercooler:
        m.A_intercooler = Var()
        m.cw_intercooler = Var(doc="kg/s")

        m.cw_intercooler_con = Constraint(
            expr=m.absorber.Q_intercooler == m.cw_intercooler * 4184 * (35 - 20),
            doc="Heat calculation on cooling water side",
        )
        # LMTD calculation for intercooler
        m.tmin_intercooler_1 = Expression(expr=m.absorber.T_intercooler - (20 + 273))
        m.tmin_intercooler_2 = Expression(
            expr=m.absorber.liquid_properties[0, num_segments / 2 + 1].temperature
            - (35 + 273)
        )

        m.q_lmtd_intercooler = Constraint(
            expr=m.absorber.Q_intercooler
            == m.U_cooler
            * m.A_intercooler
            * (
                m.tmin_intercooler_1
                * m.tmin_intercooler_2
                * (m.tmin_intercooler_1 + m.tmin_intercooler_2)
                / 2
            )
            ** (1 / 3)
        )

    # exchanger 1
    m.T_rich_1.fix(126 + 273)

    def q_ex1_cold(m):
        return m.Q_ex1 == m.F_bottom / m.absorber.liquid_properties[
            0, 0
        ].conc_mol * m.absorber.liquid_properties[0, 0].params.cp_vol * (
            m.T_rich_1 - m.absorber.liquid_properties[0, 0].temperature
        )

    m.Q_ex1_cold_con = Constraint(
        rule=q_ex1_cold, doc="Heat calculation on the rich solvent side"
    )

    def q_ex1_hot(m):
        return -m.Q_ex1 == m.afs.flash_tank.liquid_properties[
            0
        ].flow_mol / m.absorber.liquid_properties[
            0, num_segments
        ].conc_mol * m.absorber.liquid_properties[
            0, 0
        ].params.cp_vol * (
            m.T_lean_2 - m.T_lean_1
        )

    m.Q_ex1_hot_con = Constraint(
        rule=q_ex1_hot, doc="Heat calculation on the lean solvent side"
    )

    # LMTD calculation for exchanger 1
    m.tmin_ex1_1 = Expression(expr=m.T_lean_1 - m.T_rich_1)
    m.tmin_ex1_2 = Expression(
        expr=m.T_lean_2 - m.absorber.liquid_properties[0, 0].temperature
    )

    def q_lmtd_ex1(m):
        return m.Q_ex1 == m.U * m.A1 * (
            m.tmin_ex1_1 * m.tmin_ex1_2 * (m.tmin_ex1_1 + m.tmin_ex1_2) / 2
        ) ** (1 / 3)

    m.q_lmtd_ex1 = Constraint(rule=q_lmtd_ex1)

    # exchanger 2
    m.T_rich_2.fix(146 + 273)

    def q_ex2_cold(m):
        return m.Q_ex2 == m.F_hot / m.absorber.liquid_properties[
            0, num_segments
        ].conc_mol * m.absorber.liquid_properties[0, 0].params.cp_vol * (
            m.T_rich_2 - m.T_rich_1
        )

    m.Q_ex2_cold_con = Constraint(
        rule=q_ex2_cold, doc="Heat calculation on the rich solvent side"
    )

    def q_ex2_hot(m):
        return -m.Q_ex2 == m.afs.flash_tank.liquid_properties[
            0
        ].flow_mol / m.absorber.liquid_properties[
            0, num_segments
        ].conc_mol * m.absorber.liquid_properties[
            0, 0
        ].params.cp_vol * (
            m.T_lean_1 - m.afs.flash_tank.liquid_properties[0].temperature
        )

    m.Q_ex2_hot_con = Constraint(
        rule=q_ex2_hot, doc="Heat calculation on the lean solvent side"
    )

    # LMTD calculation for exchanger 2
    m.tmin_ex2_1 = Expression(
        expr=m.afs.flash_tank.liquid_properties[0].temperature - m.T_rich_2
    )
    m.tmin_ex2_2 = Expression(expr=m.T_lean_1 - m.T_rich_1)

    def q_lmtd_ex2(m):
        return m.Q_ex2 == m.U * m.A2 * (
            m.tmin_ex2_1 * m.tmin_ex2_2 * (m.tmin_ex2_1 + m.tmin_ex2_2) / 2
        ) ** (1 / 3)

    m.q_lmtd_ex2 = Constraint(rule=q_lmtd_ex2)

    # exchanger 3
    m.T_final.fix(78 + 273)

    def q_ex3_cold(m):
        return m.Q_ex3 == m.F_cold / m.absorber.liquid_properties[
            0, 0
        ].conc_mol * m.absorber.liquid_properties[0, 0].params.cp_vol * (
            m.T_rich_3 - m.absorber.liquid_properties[0, 0].temperature
        )

    m.Q_ex3_cold_con = Constraint(
        rule=q_ex3_cold, doc="Heat calculation on the rich solvent side"
    )

    # For the gas side of exchanger 3, the heat of condensation of water should be considered
    m.equilP_outlet = Expression(
        expr=exp(
            72.55
            - 7206.7 / m.T_final
            - 7.1385 * log(m.T_final)
            + 4.04 * 10 ** (-6) * (m.T_final**2)
        )
        / 1000,
        doc="Equilibrium pressure of water at the outlet temperature",
    )

    m.cp_co2_integrated = Expression(
        expr=(
            18.86 * m.afs.stripper.vapor_properties[0, num_segments].temperature
            + 0.07937
            * (m.afs.stripper.vapor_properties[0, num_segments].temperature ** 2)
            / 2
            - 6.7834
            * 10 ** (-5)
            * (m.afs.stripper.vapor_properties[0, num_segments].temperature ** 3)
            / 3
            + 2.4426
            * 10 ** (-8)
            * (m.afs.stripper.vapor_properties[0, num_segments].temperature ** 4)
            / 4
        )
        - (
            18.86 * m.T_final
            + 0.07937 * (m.T_final**2) / 2
            - 6.7834 * 10 ** (-5) * (m.T_final**3) / 3
            + 2.4426 * 10 ** (-8) * (m.T_final**4) / 4
        ),
        doc="Heat capacity of CO2 integrated over temperature",
    )

    m.cp_h2o_integrated = Expression(
        expr=(
            33.80 * m.afs.stripper.vapor_properties[0, num_segments].temperature
            - 0.00795
            * (m.afs.stripper.vapor_properties[0, num_segments].temperature ** 2)
            / 2
            + 2.8228
            * 10 ** (-5)
            * (m.afs.stripper.vapor_properties[0, num_segments].temperature ** 3)
            / 3
            - 1.3115
            * 10 ** (-8)
            * (m.afs.stripper.vapor_properties[0, num_segments].temperature ** 4)
            / 4
        )
        - (
            33.80 * m.T_final
            - 0.00795 * (m.T_final**2) / 2
            + 2.8228 * 10 ** (-5) * (m.T_final**3) / 3
            - 1.3115 * 10 ** (-8) * (m.T_final**4) / 4
        )
    )

    # Q = sensible heat of CO2 + sensible heat of H2O + heat of condensation * F_condensed
    # heat of condensation is corrected for temperature using Watson's equation where
    # the reduced temperature is 374 and the reference temperature for heat equal to 40,800
    # is 100 C
    # F_condensed = F_gas_in - F_gas_out
    # F_gas_out = F_gas_CO2_in/y_CO2_out
    # F_gas_out = F_gas_CO2_in/(1-y_H2O_out)
    # y_H2O_out = P*/P (P* is the equilibrium pressure of H2O at outlet temperature)
    # since x_H2O_out = 1
    def q_ex3_hot(m):
        return -m.Q_ex3 == -m.afs.stripper.vapor_properties[
            0, num_segments
        ].flow_mol_comp["CO2"] * m.cp_co2_integrated - m.afs.stripper.vapor_properties[
            0, num_segments
        ].flow_mol_comp[
            "H2O"
        ] * m.cp_h2o_integrated - 40800 * (
            (374 - (m.afs.stripper.vapor_properties[0, num_segments].temperature - 273))
            / (374 - 100)
        ) ** 0.38 * (
            m.afs.stripper.vapor_properties[0, num_segments].flow_mol
            - m.afs.stripper.vapor_properties[0, num_segments].flow_mol_comp["CO2"]
            / (
                1
                - m.equilP_outlet
                / (m.afs.stripper.vapor_properties[0, num_segments].pressure / 1000)
            )
        )

    m.Q_ex3_hot_con = Constraint(rule=q_ex3_hot, doc="Heat calculation on the gas side")

    # LMTD calculation for exchanger 3
    m.tmin_ex3_1 = Expression(
        expr=m.afs.stripper.vapor_properties[0, num_segments].temperature - m.T_rich_3
    )
    m.tmin_ex3_2 = Expression(
        expr=m.T_final - m.absorber.liquid_properties[0, 0].temperature
    )

    def q_lmtd_ex3(m):
        return m.Q_ex3 == m.U * m.A3 * (
            m.tmin_ex3_1 * m.tmin_ex3_2 * (m.tmin_ex3_1 + m.tmin_ex3_2) / 2
        ) ** (1 / 3)

    m.q_lmtd_ex3 = Constraint(rule=q_lmtd_ex3)

    # steam heater
    m.T_hot.fix(150 + 273)

    m.steam_flowrate = Var(initialize=0.004, doc="kg/s")
    m.h_steam = Param(
        initialize=2.022 * 10**6,
        doc="The latent heat of steam at T = 178C and P=8.5 barg, in J/kg",
    )
    m.T_steam = Param(initialize=178 + 273)

    def q_heater(m):
        return m.Q_heater == m.F_hot / m.absorber.liquid_properties[
            0, num_segments
        ].conc_mol * m.absorber.liquid_properties[0, 0].params.cp_vol * (
            m.T_hot - m.T_rich_2
        )

    m.Q_heater_con = Constraint(
        rule=q_heater, doc="Heat calculation on the rich solvent side"
    )

    def steam_side(m):
        return m.Q_heater == m.h_steam * m.steam_flowrate

    m.steam_side_con = Constraint(
        rule=steam_side, doc="Heat calculation on the steam side"
    )

    m.tmin_heater_1 = Expression(expr=m.T_steam - m.T_rich_2)
    m.tmin_heater_2 = Expression(expr=m.T_steam - m.T_hot)

    # LMTD calculation for the steam heater
    def q_lmtd_heater(m):
        return m.Q_heater == m.U_heater * m.A_heater * (
            m.tmin_heater_1 * m.tmin_heater_2 * (m.tmin_heater_1 + m.tmin_heater_2) / 2
        ) ** (1 / 3)

    m.q_lmtd_heater = Constraint(rule=q_lmtd_heater)

    return m
