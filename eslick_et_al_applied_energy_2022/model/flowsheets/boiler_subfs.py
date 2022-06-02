##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes".
##############################################################################
"""
Demonstration and test flowsheet for a dynamic flowsheet.

"""

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.network import Arc

# Import IDAES core
from idaes.core import FlowsheetBlock
from idaes.core.util.misc import TagReference
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import copy_port_values as _set_port

# Import IDAES standard unit model
from idaes.power_generation.unit_models.helm import HelmMixer, MomentumMixingType, HelmSplitter
from idaes.generic_models.unit_models import Mixer
import idaes.logger as idaeslog

# Import non-standard unit models
from unit_models import (
    Drum1D,
    Downcomer,
    WaterwallSection,
    WaterFlash,
    BoilerSurrogate,
    HeatExchangerCrossFlow2D,
    SteamHeater,
    HeatExchangerWith3Streams,
    WaterPipe,
    WaterTank
)

def add_unit_models(m):
    fs = m.fs_main.fs_blr
    prop_water = m.fs_main.prop_water
    prop_gas = m.fs_main.prop_gas

    fs.num_mills = pyo.Var()
    fs.num_mills.fix(4)

    # 14 waterwall zones
    fs.ww_zones = pyo.RangeSet(14)

    # boiler based on surrogate
    fs.aBoiler = BoilerSurrogate(default={"dynamic": False,
                               "side_1_property_package": prop_gas,
                               "side_2_property_package": prop_gas,
                               "has_heat_transfer": False,
                               "has_pressure_change": False,
                               "has_holdup": False})

    # model a drum by a WaterFlash, a Mixer and a Drum model
    fs.aFlash = WaterFlash(default={"dynamic": False,
                               "property_package": prop_water,
                               "has_phase_equilibrium": False,
                               "has_heat_transfer": False,
                               "has_pressure_change": False})
    fs.aMixer = HelmMixer(default={"dynamic": False,
                              "property_package": prop_water,
                              "momentum_mixing_type": MomentumMixingType.equality,
                              "inlet_list": ["FeedWater", "SatWater"]})
    fs.aDrum = Drum1D(default={"property_package": prop_water,
                               "has_holdup": True,
                               "has_heat_transfer": True,
                               "has_pressure_change": True,
                               "finite_elements": 4,
                               "inside_diameter": 1.778,
                               "thickness": 0.127})
    fs.blowdown_split = HelmSplitter(
        default={
            "dynamic": False,
            "property_package": prop_water,
            "outlet_list": ["FW_Downcomer", "FW_Blowdown"],
        }
    )
    # downcomer
    fs.aDowncomer = Downcomer(default={
                               "dynamic": False,
                               "property_package": prop_water,
                               "has_holdup": True,
                               "has_heat_transfer": True,
                               "has_pressure_change": True})

    # 14 WaterwallSection units
    fs.Waterwalls = WaterwallSection(fs.ww_zones,
                               default={
                               "has_holdup": True,
                               "property_package": prop_water,
                               "has_equilibrium_reactions": False,
                               "has_heat_of_reaction": False,
                               "has_heat_transfer": True,
                               "has_pressure_change": True})

    # roof superheater
    fs.aRoof = SteamHeater(default={
                               "dynamic": False,
                               "property_package": prop_water,
                               "has_holdup": True,
                               "has_equilibrium_reactions": False,
                               "has_heat_of_reaction": False,
                               "has_heat_transfer": True,
                               "has_pressure_change": True,
                               "single_side_only" : True})

    # platen superheater
    fs.aPlaten = SteamHeater(default={
                               "dynamic": False,
                               "property_package": prop_water,
                               "has_holdup": True,
                               "has_equilibrium_reactions": False,
                               "has_heat_of_reaction": False,
                               "has_heat_transfer": True,
                               "has_pressure_change": True,
                               "single_side_only" : False})

    # 1st reheater
    fs.aRH1 = HeatExchangerCrossFlow2D(default={
                               "tube_side":{"property_package": prop_water, "has_holdup": False,
                                            "has_pressure_change": True},
                               "shell_side":{"property_package": prop_gas, "has_holdup": False,
                                             "has_pressure_change": True},
                               "finite_elements": 4,
                               "flow_type": "counter_current",
                               "tube_arrangement": "in-line",
                               "tube_side_water_phase": "Vap",
                               "has_radiation": True,
                               "radial_elements": 5,
                               "inside_diameter": 2.202*0.0254,
                               "thickness": 0.149*0.0254})

    # 2nd reheater
    fs.aRH2 = HeatExchangerCrossFlow2D(default={
                               "tube_side":{"property_package": prop_water, "has_holdup": False,
                                            "has_pressure_change": True},
                               "shell_side":{"property_package": prop_gas, "has_holdup": False,
                                             "has_pressure_change": True},
                               "finite_elements": 2,
                               "flow_type": "counter_current",
                               "tube_arrangement": "in-line",
                               "tube_side_water_phase": "Vap",
                               "has_radiation": True,
                               "radial_elements": 5,
                               "inside_diameter": 2.217*0.0254,
                               "thickness": 0.1415*0.0254})

    # primary superheater
    fs.aPSH = HeatExchangerCrossFlow2D(default={
                               "tube_side":{"property_package": prop_water, "has_holdup": False,
                                            "has_pressure_change": True},
                               "shell_side":{"property_package": prop_gas, "has_holdup": False,
                                             "has_pressure_change": True},
                               "finite_elements": 5,
                               "flow_type": "counter_current",
                               "tube_arrangement": "in-line",
                               "tube_side_water_phase": "Vap",
                               "has_radiation": True,
                               "radial_elements": 5,
                               "inside_diameter": 1.45*0.0254,
                               "thickness": 0.15*0.0254})

    # economizer
    fs.aECON = HeatExchangerCrossFlow2D(default={
                               "tube_side":{"property_package": prop_water, "has_holdup": False,
                                            "has_pressure_change": True},
                               "shell_side":{"property_package": prop_gas, "has_holdup": False,
                                             "has_pressure_change": True},
                               "finite_elements": 5,
                               "flow_type": "counter_current",
                               "tube_arrangement": "in-line",
                               "tube_side_water_phase": "Liq",
                               "has_radiation": False,
                               "radial_elements": 5,
                               "inside_diameter": 1.452*0.0254,
                               "thickness": 0.149*0.0254})

    # water pipe from economizer outlet to drum
    fs.aPipe = WaterPipe(default={
                               "dynamic": False,
                               "property_package": prop_water,
                               "has_holdup": True,
                               "has_heat_transfer": False,
                               "has_pressure_change": True,
                               "water_phase": 'Liq',
                               "contraction_expansion_at_end": 'None'})

    # a mixer to mix hot primary air with tempering air
    fs.Mixer_PA = Mixer(
        default={
            "dynamic": False,
            "property_package": prop_gas,
            "momentum_mixing_type": MomentumMixingType.equality,
            "inlet_list": ["PA_inlet", "TA_inlet"],
        }
    )

    # attemperator for main steam before platen SH
    fs.Attemp = HelmMixer(
        default={
            "dynamic": False,
            "property_package": prop_water,
            "momentum_mixing_type": MomentumMixingType.equality,
            "inlet_list": ["Steam_inlet", "Water_inlet"],
        }
    )

    # air preheater as three-stream heat exchanger with heat loss to ambient,
    #     side_1: flue gas
    #     side_2:PA (priamry air?)
    #     side_3:PA (priamry air?)
    fs.aAPH = HeatExchangerWith3Streams(
        default={"dynamic": False,
            "side_1_property_package": prop_gas,
            "side_2_property_package": prop_gas,
            "side_3_property_package": prop_gas,
            "has_heat_transfer": True,
            "has_pressure_change": True,
            "has_holdup": False,
            "flow_type_side_2": "counter-current",
            "flow_type_side_3": "counter-current",
        }
    )
    return m


def set_arcs_and_constraints(m):
    # Make arc to connect streams
    fs = m.fs_main.fs_blr
    prop_gas = m.fs_main.prop_gas
    # water/steam streams
    fs.B001 = Arc(source=fs.aECON.tube_outlet,
                  destination=fs.aPipe.inlet)
    fs.B001b = Arc(source=fs.aPipe.outlet,
                   destination=fs.aMixer.FeedWater)
    fs.B009 = Arc(source=fs.aFlash.liq_outlet,
                  destination=fs.aMixer.SatWater)
    fs.B010 = Arc(source=fs.aMixer.outlet,
                  destination=fs.aDrum.inlet)
    fs.B011 = Arc(source=fs.aDrum.outlet,
                  destination=fs.blowdown_split.inlet)
    fs.B011b = Arc(source=fs.blowdown_split.FW_Downcomer,
                  destination=fs.aDowncomer.inlet)
    fs.B007 = Arc(source=fs.aDowncomer.outlet,
                  destination=fs.Waterwalls[1].inlet)

    def ww_arc_rule(b, i):
        return (b.Waterwalls[i].outlet, b.Waterwalls[i+1].inlet)
    fs.ww_arcs = Arc(range(1,14), rule=ww_arc_rule)

    fs.B008  = Arc(source=fs.Waterwalls[14].outlet,
                   destination=fs.aFlash.inlet)
    fs.B002 = Arc(source=fs.aFlash.vap_outlet,
                  destination=fs.aRoof.inlet)
    fs.B003 = Arc(source=fs.aRoof.outlet,
                  destination=fs.aPSH.tube_inlet)
    fs.B004 = Arc(source=fs.aPSH.tube_outlet,
                  destination=fs.Attemp.Steam_inlet)
    fs.B005 = Arc(source=fs.Attemp.outlet,
                  destination=fs.aPlaten.inlet)
    fs.B012 = Arc(source=fs.aRH1.tube_outlet,
                  destination=fs.aRH2.tube_inlet)
    # air/flue_gas streams, Note Mixer_PA.outlet is not connected with
    # aBoiler directly and mills are not included in flowsheet
    fs.PA04 = Arc(source=fs.aAPH.side_2_outlet,
                  destination=fs.Mixer_PA.PA_inlet)
    fs.SA03 = Arc(source=fs.aAPH.side_3_outlet,
                  destination=fs.aBoiler.side_2_inlet)
    fs.FG01 = Arc(source=fs.aBoiler.side_1_outlet,
                  destination=fs.aRH2.shell_inlet)
    fs.FG02 = Arc(source=fs.aRH2.shell_outlet,
                  destination=fs.aRH1.shell_inlet)
    fs.FG03 = Arc(source=fs.aRH1.shell_outlet,
                  destination=fs.aPSH.shell_inlet)
    fs.FG04 = Arc(source=fs.aPSH.shell_outlet,
                  destination=fs.aECON.shell_inlet)
    fs.FG05 = Arc(source=fs.aECON.shell_outlet,
                  destination=fs.aAPH.side_1_inlet)

    # According to Andrew Lee, this must be called after discretization call
    pyo.TransformationFactory("network.expand_arcs").apply_to(fs)

    # species mole fractions of air
    fs.mole_frac_air = pyo.Param(prop_gas.component_list,
                        mutable=False,
                        initialize={'O2': 0.20784,
                                    'N2': 0.783994,
                                    'NO': 0.000001,
                                    'CO2': 0.000337339,
                                    'H2O': 0.0078267,
                                    'SO2': 0.000001},
                        doc='mole fraction of air species')

    #flowsheet level constraints
    # constraint for heat duty of each zone
    @fs.Constraint(fs.config.time, fs.ww_zones, doc="zone heat loss")
    def zone_heat_loss_eqn(b, t, izone):
        return 1e-6*b.aBoiler.out_heat[t,izone] == 1e-6*b.Waterwalls[izone].heat_fireside[t]

    # constraint for heat duty of platen
    @fs.Constraint(fs.config.time, doc="platen SH heat loss")
    def platen_heat_loss_eqn(b, t):
        return 1e-6*b.aBoiler.out_heat[t,15] == 1e-6*b.aPlaten.heat_fireside[t]

    # constraint for heat duty of roof
    @fs.Constraint(fs.config.time, doc="roof heat loss")
    def roof_heat_loss_eqn(b, t):
        return 1e-6*b.aBoiler.out_heat[t,16] == 1e-6*b.aRoof.heat_fireside[t]

    # constraint for wall temperature of each zone
    @fs.Constraint(fs.config.time, fs.ww_zones, doc="zone wall temperature")
    def zone_wall_temp_eqn(b, t, izone):
        return b.aBoiler.in_WWTemp[t,izone] == b.Waterwalls[izone].temp_slag_boundary[t]

    # constraint for platen wall temperature
    @fs.Constraint(fs.config.time, doc="platen wall temperature")
    def platen_wall_temp_eqn(b, t):
        return b.aBoiler.in_WWTemp[t,15] == b.aPlaten.temp_slag_boundary[t]

    # constraint for roof wall temperature
    @fs.Constraint(fs.config.time, doc="roof wall temperature")
    def roof_wall_temp_eqn(b, t):
        return b.aBoiler.in_WWTemp[t,16] == b.aRoof.temp_slag_boundary[t]

    # set RH steam flow as 90% of main steam flow.  This constraint should be removed for entire plant flowsheet, use 92%
    @fs.Constraint(fs.config.time, doc="RH flow")
    def flow_mol_steam_rh_eqn(b, t):
        return b.aRH1.tube_inlet.flow_mol[t] == 0.91*b.aPlaten.outlet.flow_mol[t]

    # PA to APH
    fs.flow_mol_pa = pyo.Var(
        fs.config.time, initialize=1410, doc="mole flow rate of PA to APH"
    )

    # Tempering air to APH
    fs.flow_mol_ta = pyo.Var(
        fs.config.time, initialize=720, doc="mole flow rate of tempering air"
    )

    fs.flow_pa_kg_per_s = pyo.Var(fs.config.time)
    @fs.Constraint(fs.config.time, doc="total primary air mass flow rate")
    def flowrate_pa_total_eqn(b, t):
        return fs.flow_pa_kg_per_s[t] == (b.flow_mol_pa[t] + b.flow_mol_ta[t]) * 0.0287689

    @fs.Expression(fs.config.time, doc="total primary air mass flow rate")
    def flowrate_pa_total(b, t):
        return (b.flow_mol_pa[t] + b.flow_mol_ta[t]) * 0.0287689

    # set the total PA flow rate same as the boiler flowrate_PA
    @fs.Constraint(fs.config.time, doc="Total PA flow")
    def total_pa_mass_flow_eqn(b, t):
        return b.aBoiler.flowrate_PA[t] == b.flowrate_pa_total[t]

    # equation to calculate flow_mol_comp[] of PA to APH
    @fs.Constraint(fs.config.time, prop_gas.component_list, doc="component flow of PA to APH")
    def flow_component_pa_eqn(b, t, j):
        return (
            b.aAPH.side_2_inlet.flow_mol_comp[t, j]
            == b.flow_mol_pa[t] * b.mole_frac_air[j]
        )

    # equation to calculate flow_mol_comp[] of TA (tempering air)
    @fs.Constraint(fs.config.time, prop_gas.component_list, doc="component flow of tempering air")
    def flow_component_ta_eqn(b, t, j):
        return (
            b.Mixer_PA.TA_inlet.flow_mol_comp[t, j]
            == b.flow_mol_ta[t] * b.mole_frac_air[j]
        )

    # set pressure drop of APH for PA the same as that of SA since no meassured
    # pressure at PA outlet
    @fs.Constraint(fs.config.time, doc="Pressure drop of PA of APH")
    def pressure_drop_of_APH_eqn(b, t):
        return b.aAPH.deltaP_side_2[t] == b.aAPH.deltaP_side_3[t]

    # set PA before APH and TA before Mixer_PA temperature the same
    @fs.Constraint(fs.config.time, doc="Same inlet temperature for PA and SA")
    def pa_ta_temperature_identical_eqn(b, t):
        return b.aAPH.side_2_inlet.temperature[t] == b.Mixer_PA.TA_inlet.temperature[t]

    # set blowdown water flow rate at 2 percent of feed water flow
    fs.blowdown_frac = pyo.Var()
    fs.blowdown_frac.fix(0.02)

    @fs.Constraint(fs.config.time, doc="Blowdown water flow fraction")
    def blowdown_flow_fraction_eqn(b, t):
        return b.blowdown_split.FW_Blowdown.flow_mol[t] == b.aECON.tube_inlet.flow_mol[t]*b.blowdown_frac

    #------------- Three additional constraints for validation and optimization flowsheet -------------
    # set tempering air to total PA flow ratio as a function of coal flow rate
    @fs.Constraint(fs.config.time, doc="Fraction of TA as total PA flow")
    def fraction_of_ta_in_total_pa_eqn(b, t):
        return b.flow_mol_ta[t] == b.flow_mol_pa[t]*0.15

    # set total PA flow to coal flow ratio as a function of coal flow rate
    fs.pa2coal_a = pyo.Var(initialize=0.0017486)
    fs.pa2coal_b = pyo.Var(initialize=0.12634)
    fs.pa2coal_c = pyo.Var(initialize=4.5182)
    fs.pa2coal_a.fix()
    fs.pa2coal_b.fix()
    fs.pa2coal_c.fix()
    @fs.Constraint(fs.config.time, doc="PA to coal ratio")
    def pa_to_coal_ratio_eqn(b, t):
        return b.flowrate_pa_total[t] == b.aBoiler.flowrate_coal_raw[t]*\
            (fs.pa2coal_a*b.aBoiler.flowrate_coal_raw[t]**2 - fs.pa2coal_b*b.aBoiler.flowrate_coal_raw[t] + fs.pa2coal_c)

    # set dry O2 percent in flue gas as a function of coal flow rate
    @fs.Constraint(fs.config.time, doc="Steady state dry O2 in flue gas")
    def dry_o2_in_flue_gas_eqn(b, t):
        return b.aBoiler.fluegas_o2_pct_dry[t] == -0.0007652*b.aBoiler.flowrate_coal_raw[t]**3 + \
            0.06744*b.aBoiler.flowrate_coal_raw[t]**2 - 1.9815*b.aBoiler.flowrate_coal_raw[t] + 22.275

    #--------------------- additional constraint for fheat_ww
    fs.a_fheat_ww = pyo.Var(initialize=7.8431E-05)
    fs.b_fheat_ww = pyo.Var(initialize=8.3181E-03)
    fs.c_fheat_ww = pyo.Var(initialize=1.1926)
    fs.a_fheat_ww.fix()
    fs.b_fheat_ww.fix()
    fs.c_fheat_ww.fix()
    @fs.Constraint(fs.config.time, doc="Water wall heat absorption factor")
    def fheat_ww_eqn(b, t):
        return b.aBoiler.fheat_ww[t] == \
            fs.a_fheat_ww*b.aBoiler.flowrate_coal_raw[t]**2 - \
            fs.b_fheat_ww*b.aBoiler.flowrate_coal_raw[t] + \
            fs.c_fheat_ww

    fs.a_fheat_platen = pyo.Var(initialize=7.8431E-05)
    fs.b_fheat_platen = pyo.Var(initialize=8.3181E-03)
    fs.c_fheat_platen = pyo.Var(initialize=1.1926)
    fs.a_fheat_platen.fix()
    fs.b_fheat_platen.fix()
    fs.c_fheat_platen.fix()
    @fs.Constraint(fs.config.time, doc="Platen superheater heat absorption factor")
    def fheat_platen_eqn(b, t):
        return b.aBoiler.fheat_platen[t] == \
            fs.a_fheat_platen*b.aBoiler.flowrate_coal_raw[t]**2 - \
            fs.b_fheat_platen*b.aBoiler.flowrate_coal_raw[t] + \
            fs.c_fheat_platen
    fs.fheat_platen_eqn.deactivate()

    # additional constraint for APH ua_side_2 (PA)
    ua_a = fs.aph2_ua_a = pyo.Var(initialize=150.1)
    ua_b = fs.aph2_ua_b = pyo.Var(initialize=9377.3)
    ua_c = fs.aph2_ua_c = pyo.Var(initialize=63860)
    ua_a.fix()
    ua_b.fix()
    ua_c.fix()
    @fs.Constraint(fs.config.time, doc="UA for APH of PA")
    def ua_side_2_eqn(b, t):
        return b.aAPH.ua_side_2[t]*1e-4 == 1e-4*(-ua_a*b.aBoiler.flowrate_coal_raw[t]**2 + \
                                        ua_b*b.aBoiler.flowrate_coal_raw[t] + ua_c)

    # additional constraint for APH ua_side_3 (SA), reduced by 5% based on Maojian's results
    ua_a = fs.aph3_ua_a = pyo.Var(initialize=0.32849)
    ua_b = fs.aph3_ua_b = pyo.Var(initialize=1.06022)
    ua_a.fix()
    ua_b.fix()
    @fs.Constraint(fs.config.time, doc="UA for APH of SA")
    def ua_side_3_eqn(b, t):
        return b.aAPH.ua_side_3[t]*1e-5 == (ua_a*b.aBoiler.flowrate_coal_raw[t] + ua_b)*0.95

    # boiler efficiency based on enthalpy increase of main and RH steams
    @fs.Expression(fs.config.time, doc="boiler efficiency based on steam")
    def boiler_efficiency_steam(b, t):
        return (b.aPlaten.outlet.flow_mol[t]*(b.aPlaten.outlet.enth_mol[t]-b.aECON.tube_inlet.enth_mol[t]) + \
               b.aRH2.tube_outlet.flow_mol[t]*(b.aRH2.tube_outlet.enth_mol[t]-b.aRH1.tube_inlet.enth_mol[t])) / \
               (b.aBoiler.flowrate_coal_raw[t]*(1-b.aBoiler.mf_H2O_coal_raw[t])*b.aBoiler.hhv_coal_dry)

    # boiler efficiency based on heat absorbed
    @fs.Expression(fs.config.time, doc="boiler efficiency based on heat")
    def boiler_efficiency_heat(b, t):
        return (b.aBoiler.heat_total[t] + b.aRH2.total_heat[t] + b.aRH1.total_heat[t] + b.aPSH.total_heat[t] + b.aECON.total_heat[t]) / \
               (b.aBoiler.flowrate_coal_raw[t]*(1-b.aBoiler.mf_H2O_coal_raw[t])*b.aBoiler.hhv_coal_dry)
    return m

def set_inputs(m):
    fs = m.fs_main.fs_blr
    ############################ fix variables for geometry and design data #####################
    # boiler
    fs.aBoiler.temp_coal[:].fix(338.7)           # always set at 150 F
    fs.aBoiler.mf_C_coal_dry.fix(0.600245)       # fix coal ultimate analysis on dry basis
    fs.aBoiler.mf_H_coal_dry.fix(0.0456951)
    fs.aBoiler.mf_O_coal_dry.fix(0.108121)
    fs.aBoiler.mf_N_coal_dry.fix(0.0116837)
    fs.aBoiler.mf_S_coal_dry.fix(0.0138247)
    fs.aBoiler.mf_Ash_coal_dry.fix(0.220431)
    fs.aBoiler.hhv_coal_dry.fix(2.48579e+007)
    fs.aBoiler.frac_moisture_vaporized[:].fix(0.6)  # assume constant fraction of moisture vaporized in mills

    # drum
    fs.aDrum.length_drum.fix(14.3256)
    fs.aDrum.level[:].fix(0.889);
    fs.aDrum.count_downcomer.fix(8);
    fs.aDrum.diameter_downcomer.fix(0.3556);
    fs.aDrum.temp_amb[:].fix(300)
    fs.aDrum.thickness_insulation.fix(0.15)

    # blowdown split fraction initially set to a small value, will eventually unfixed due to a flowsheet constraint
    fs.blowdown_split.split_fraction[:,"FW_Blowdown"].fix(0.001)

    # downcomer
    fs.aDowncomer.diameter.fix(0.3556)       # 16 inch downcomer OD. assume downcomer thickness of 1 inch
    fs.aDowncomer.height.fix(43.6245)        # based on furnace height of 43.6245
    fs.aDowncomer.count.fix(8)               # 8 downcomers
    fs.aDowncomer.heat_duty[:].fix(0.0)      # assume no heat loss

    # 14 waterwall sections
    for i in fs.ww_zones:
        # 2.5 inch OD, 0.2 inch thickness, 3 inch pitch
        fs.Waterwalls[i].diameter_in.fix(0.05334)
        fs.Waterwalls[i].thk_tube.fix(0.00508)
        fs.Waterwalls[i].thk_fin.fix(0.005)
        fs.Waterwalls[i].thk_slag[:].fix(0.001)
        fs.Waterwalls[i].length_fin.fix(0.0127)
        fs.Waterwalls[i].count.fix(558)

    # water wall section height
    fs.Waterwalls[1].height.fix(7.3533)
    fs.Waterwalls[2].height.fix(3.7467)
    fs.Waterwalls[3].height.fix(1.3)
    fs.Waterwalls[4].height.fix(1.3461)
    fs.Waterwalls[5].height.fix(1.2748)
    fs.Waterwalls[6].height.fix(1.2748)
    fs.Waterwalls[7].height.fix(1.1589)
    fs.Waterwalls[8].height.fix(1.2954)
    fs.Waterwalls[9].height.fix(3.25)
    fs.Waterwalls[10].height.fix(3.5)
    fs.Waterwalls[11].height.fix(3.6465)
    fs.Waterwalls[12].height.fix(3.5052)
    fs.Waterwalls[13].height.fix(5.4864)
    fs.Waterwalls[14].height.fix(5.4864)

    # water wall section projected area
    fs.Waterwalls[1].area_proj_total.fix(317.692)
    fs.Waterwalls[2].area_proj_total.fix(178.329)
    fs.Waterwalls[3].area_proj_total.fix(61.8753)
    fs.Waterwalls[4].area_proj_total.fix(64.0695)
    fs.Waterwalls[5].area_proj_total.fix(60.6759)
    fs.Waterwalls[6].area_proj_total.fix(60.6759)
    fs.Waterwalls[7].area_proj_total.fix(55.1595)
    fs.Waterwalls[8].area_proj_total.fix(61.6564)
    fs.Waterwalls[9].area_proj_total.fix(154.688)
    fs.Waterwalls[10].area_proj_total.fix(166.587)
    fs.Waterwalls[11].area_proj_total.fix(173.56)
    fs.Waterwalls[12].area_proj_total.fix(166.114)
    fs.Waterwalls[13].area_proj_total.fix(178.226)
    fs.Waterwalls[14].area_proj_total.fix(178.226)

    # roof
    fs.aRoof.diameter_in.fix(2.1*0.0254)
    fs.aRoof.thk_tube.fix(0.2*0.0254)
    fs.aRoof.thk_fin.fix(0.004)
    fs.aRoof.thk_slag[:].fix(0.001)
    fs.aRoof.length_fin.fix(0.5*0.0254)
    fs.aRoof.length_tube.fix(8.2534)
    fs.aRoof.count.fix(177)

    # platen superheater
    fs.aPlaten.diameter_in.fix(0.04125)
    fs.aPlaten.thk_tube.fix(0.00635)
    fs.aPlaten.thk_fin.fix(0.004)
    fs.aPlaten.thk_slag[:].fix(0.001)
    fs.aPlaten.length_fin.fix(0.00955)
    fs.aPlaten.length_tube.fix(45.4533)
    fs.aPlaten.count.fix(11*19)

    # RH1
    fs.aRH1.pitch_x.fix(4.5*0.0254)
    fs.aRH1.pitch_y.fix(6.75*0.0254)
    fs.aRH1.length_tube_seg.fix(300*0.0254)    # Use tube length as estimated
    fs.aRH1.nseg_tube.fix(4)
    fs.aRH1.ncol_tube.fix(78)
    fs.aRH1.nrow_inlet.fix(3)
    fs.aRH1.delta_elevation.fix(0.0)
    fs.aRH1.therm_cond_wall = 43.0
    fs.aRH1.emissivity_wall.fix(0.65)
    fs.aRH1.rfouling_tube = 0.00017     #heat transfer resistance due to tube side fouling (water scales)
    fs.aRH1.rfouling_shell = 0.00088    #heat transfer resistance due to shell side fouling (ash deposition)
    fs.aRH1.fcorrection_htc_tube.fix(0.9)
    fs.aRH1.fcorrection_htc_shell.fix(0.9)
    fs.aRH1.fcorrection_dp_tube.fix(5.3281932)
    fs.aRH1.fcorrection_dp_shell.fix(3.569166)

    # RH2
    fs.aRH2.pitch_x.fix(4.5*0.0254)
    fs.aRH2.pitch_y.fix(13.5*0.0254)
    fs.aRH2.length_tube_seg.fix(420*0.0254)    # Use tube length as estimated
    fs.aRH2.nseg_tube.fix(2)
    fs.aRH2.ncol_tube.fix(39)
    fs.aRH2.nrow_inlet.fix(6)
    fs.aRH2.delta_elevation.fix(0.0)
    fs.aRH2.therm_cond_wall = 43.0
    fs.aRH2.emissivity_wall.fix(0.65)
    fs.aRH2.rfouling_tube = 0.00017     #heat transfer resistance due to tube side fouling (water scales)
    fs.aRH2.rfouling_shell = 0.00088    #heat transfer resistance due to shell side fouling (ash deposition)
    fs.aRH2.fcorrection_htc_tube.fix(0.9)
    fs.aRH2.fcorrection_htc_shell.fix(0.9)
    fs.aRH2.fcorrection_dp_tube.fix(5.3281932)
    fs.aRH2.fcorrection_dp_shell.fix(3.569166)

    # PSH
    fs.aPSH.pitch_x.fix(3.75*0.0254)
    fs.aPSH.pitch_y.fix(6.0*0.0254)
    fs.aPSH.length_tube_seg.fix(302.5*0.0254)
    fs.aPSH.nseg_tube.fix(9)
    fs.aPSH.ncol_tube.fix(88)
    fs.aPSH.nrow_inlet.fix(4)
    fs.aPSH.delta_elevation.fix(5.0)
    fs.aPSH.therm_cond_wall = 43.0
    fs.aPSH.rfouling_tube = 0.00017     #heat transfer resistance due to tube side fouling (water scales)
    fs.aPSH.rfouling_shell = 0.00088    #heat transfer resistance due to shell side fouling (ash deposition)
    fs.aPSH.emissivity_wall.fix(0.7)
    fs.aPSH.fcorrection_htc_tube.fix(1.04)
    fs.aPSH.fcorrection_htc_shell.fix(1.04)
    fs.aPSH.fcorrection_dp_tube.fix(15.6)
    fs.aPSH.fcorrection_dp_shell.fix(1.252764)

    # economizer
    fs.aECON.pitch_x.fix(3.75*0.0254)
    fs.aECON.pitch_y.fix(4.0*0.0254)
    fs.aECON.length_tube_seg.fix(302.5*0.0254)
    fs.aECON.nseg_tube.fix(18)
    fs.aECON.ncol_tube.fix(132)
    fs.aECON.nrow_inlet.fix(2)
    fs.aECON.delta_elevation.fix(12)
    fs.aECON.therm_cond_wall = 43.0
    fs.aECON.rfouling_tube = 0.00017
    fs.aECON.rfouling_shell = 0.00088
    fs.aECON.fcorrection_htc_tube.fix(0.849)
    fs.aECON.fcorrection_htc_shell.fix(0.849)
    fs.aECON.fcorrection_dp_tube.fix(27.322)
    fs.aECON.fcorrection_dp_shell.fix(4.4872429)

    # APH
    fs.aAPH.ua_side_2[:].fix(171103.4)
    fs.aAPH.ua_side_3[:].fix(677069.6)
    fs.aAPH.frac_heatloss.fix(0.15)
    fs.aAPH.deltaP_side_1[:].fix(-1000)
    fs.aAPH.deltaP_side_2[:].fix(-1000)
    fs.aAPH.deltaP_side_3[:].fix(-1000)

    # 132 economizer rising tubes of 2 inch O.D. assuming 1.5 inch I.D.
    fs.aPipe.diameter.fix(0.0381)
    fs.aPipe.length.fix(35)
    fs.aPipe.count.fix(132);
    fs.aPipe.elevation_change.fix(20)

    return m


def initialize(m):
    """Initialize unit models"""
    fs = m.fs_main.fs_blr
    prop_gas = m.fs_main.prop_gas
    outlvl = 4
    _log = idaeslog.getLogger(fs.name, outlvl, tag="unit")
    solve_log = idaeslog.getSolveLogger(fs.name, outlvl, tag="unit")
    solver = pyo.SolverFactory("ipopt")
    solver.options = {
            "tol": 1e-7,
            "linear_solver": "ma27",
            "max_iter": 50,
    }

    # set initial condition to steady-state condition
    # no need to call fix_initial_conditions('steady-state')
    # fs.fix_initial_conditions('steady-state')
    if m.dynamic==True:
        fs.aBoiler.set_initial_condition()
        fs.aFlash.set_initial_condition()
        #skip aMixer
        fs.aDrum.set_initial_condition()
        fs.aDowncomer.set_initial_condition()
        for i in fs.ww_zones:
            fs.Waterwalls[i].set_initial_condition()
        fs.aRoof.set_initial_condition()
        fs.aPSH.set_initial_condition()
        #skip Attemp
        fs.aPlaten.set_initial_condition()
        fs.aRH1.set_initial_condition()
        fs.aRH2.set_initial_condition()
        fs.aECON.set_initial_condition()
        fs.aPipe.set_initial_condition()
        fs.aAPH.set_initial_condition()
        #skip Mixer_PA

    # fix operating conditions
    # aBoiler
    fs.aBoiler.mf_H2O_coal_raw[:].fix(0.156343)
    fs.aBoiler.flowrate_coal_raw[:].fix(29)
    fs.aBoiler.SR[:].fix(1.156)
    fs.aBoiler.SR_lf[:].fix(1.0)
    fs.aBoiler.in_2ndAir_Temp[:].fix(634)      # initial guess, unfixed later
    fs.aBoiler.ratio_PA2coal[:].fix(2.4525)
    fs.aBoiler.in_WWTemp[:,1].fix(641)         # initial guess of waterwall slag layer temperatures
    fs.aBoiler.in_WWTemp[:,2].fix(664)
    fs.aBoiler.in_WWTemp[:,3].fix(722)
    fs.aBoiler.in_WWTemp[:,4].fix(735)
    fs.aBoiler.in_WWTemp[:,5].fix(744)
    fs.aBoiler.in_WWTemp[:,6].fix(747)
    fs.aBoiler.in_WWTemp[:,7].fix(746)
    fs.aBoiler.in_WWTemp[:,8].fix(729)
    fs.aBoiler.in_WWTemp[:,9].fix(716)
    fs.aBoiler.in_WWTemp[:,10].fix(698)
    fs.aBoiler.in_WWTemp[:,11].fix(681)
    fs.aBoiler.in_WWTemp[:,12].fix(665)
    fs.aBoiler.in_WWTemp[:,13].fix(632)
    fs.aBoiler.in_WWTemp[:,14].fix(622)
    fs.aBoiler.in_WWTemp[:,15].fix(799)
    fs.aBoiler.in_WWTemp[:,16].fix(643)
    fs.aBoiler.fheat_ww.fix(1.0)
    fs.aBoiler.fheat_platen.fix(0.98)
    fs.aBoiler.side_1_inlet.pressure[:].fix(79868.35677)    # fixed as operating condition
    fs.aBoiler.side_2_inlet.pressure[:].fix(79868.35677)    # fixed as operating condition

    # aDrum
    fs.aDrum.level[:].fix(0.889)                    # drum level, set as drum radius

    # 14 Waterwalls
    fs.Waterwalls[1].heat_fireside[:].fix(24327106)    # initial guess, unfixed later
    fs.Waterwalls[2].heat_fireside[:].fix(19275085)
    fs.Waterwalls[3].heat_fireside[:].fix(12116325)
    fs.Waterwalls[4].heat_fireside[:].fix(13680850)
    fs.Waterwalls[5].heat_fireside[:].fix(13748641)
    fs.Waterwalls[6].heat_fireside[:].fix(14064033)
    fs.Waterwalls[7].heat_fireside[:].fix(12675550)
    fs.Waterwalls[8].heat_fireside[:].fix(12638767)
    fs.Waterwalls[9].heat_fireside[:].fix(28492415)
    fs.Waterwalls[10].heat_fireside[:].fix(26241924)
    fs.Waterwalls[11].heat_fireside[:].fix(22834318)
    fs.Waterwalls[12].heat_fireside[:].fix(17924138)
    fs.Waterwalls[13].heat_fireside[:].fix(10191487)
    fs.Waterwalls[14].heat_fireside[:].fix(7578554)

    # aRoof
    fs.aRoof.heat_fireside[:].fix(6564520)            # initial guess, unfixed later

    # aPlaten
    fs.aPlaten.heat_fireside[:].fix(49796257)          # initial guess, unfixed later

    # fix economizer water inlet, to be linked with last FWH if linked with steam cycle flowsheet
    fs.aECON.tube_inlet.flow_mol[:].fix(10103)
    fs.aECON.tube_inlet.pressure[:].fix(1.35e7)
    fs.aECON.tube_inlet.enth_mol[:].fix(18335.7)

    # fixed RH1 inlet conditions, to be linked with steam cycle flowsheet
    fs.aRH1.tube_inlet.flow_mol[:].fix(10103*0.9)
    fs.aRH1.tube_inlet.enth_mol[:].fix(55879)
    fs.aRH1.tube_inlet.pressure[:].fix(3029886)

    fs.model_check()

    #################### Initialize Units #######################
    # since dynamic model is initialized by copy steady-state model, calling unit model's initialize() function is skiped
    # tear flue gas stream between PSH and ECON
    # Use FG molar composition to set component flow rates
    fs.aECON.shell_inlet.flow_mol_comp[:,"H2O"].value = 748
    fs.aECON.shell_inlet.flow_mol_comp[:,"CO2"].value = 1054
    fs.aECON.shell_inlet.flow_mol_comp[:,"N2"].value = 5377
    fs.aECON.shell_inlet.flow_mol_comp[:,"O2"].value = 194
    fs.aECON.shell_inlet.flow_mol_comp[:,"SO2"].value = 9
    fs.aECON.shell_inlet.flow_mol_comp[:,"NO"].value = 2.6
    fs.aECON.shell_inlet.temperature[:].value = 861.3
    fs.aECON.shell_inlet.pressure[:].value = 79686
    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.aECON.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after economizer initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aPipe.inlet, fs.aECON.tube_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aPipe.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Pipe initialization ///////////////")

    # PA to APH
    fs.flow_mol_pa[:].value = 1410
    for i in prop_gas.component_list:
        fs.aAPH.side_2_inlet.flow_mol_comp[:, i].fix(
            1410 * fs.mole_frac_air[i]
        )
    fs.aAPH.side_2_inlet.temperature[:].fix(324.8)
    fs.aAPH.side_2_inlet.pressure[:].fix(88107.6)

    # SA to APH
    flow_mol_sa = 4716
    for i in prop_gas.component_list:
        fs.aAPH.side_3_inlet.flow_mol_comp[:, i].fix(
            flow_mol_sa * fs.mole_frac_air[i]
        )
    fs.aAPH.side_3_inlet.temperature[:].fix(366.2)
    fs.aAPH.side_3_inlet.pressure[:].fix(87668.4)

    # Tempering air to Mixer_PA
    fs.flow_mol_ta[:].value = 721
    for i in prop_gas.component_list:
        fs.Mixer_PA.TA_inlet.flow_mol_comp[:, i].fix(721 * fs.mole_frac_air[i]
        )
    fs.Mixer_PA.TA_inlet.temperature[:].value = 324.8
    fs.Mixer_PA.TA_inlet.pressure[:].value = 88107.6

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aAPH.side_1_inlet, fs.aECON.shell_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aAPH.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after aAPH initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.Mixer_PA.PA_inlet, fs.aAPH.side_2_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.Mixer_PA.initialize(outlvl=0)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Mixer_PA initialization ///////////////")

    flow_mol_pa = 61.3/0.0287689
    flow_mol_sa = 4716
    for i in prop_gas.component_list:
        fs.aBoiler.side_1_inlet.flow_mol_comp[:,i].value = flow_mol_pa*fs.mole_frac_air[i]
        fs.aBoiler.side_2_inlet.flow_mol_comp[:,i].value = flow_mol_sa*fs.mole_frac_air[i]
    fs.aBoiler.side_1_inlet.temperature[:].value = 338.7
    fs.aBoiler.side_2_inlet.temperature[:].value = 650

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aAPH.side_1_inlet, fs.aECON.shell_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aBoiler.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Boiler initialization ///////////////")

    # tear stream from last boiler water wall section
    fs.aFlash.inlet.flow_mol[:].value = 133444.4
    fs.aFlash.inlet.pressure[:].value = 11469704
    fs.aFlash.inlet.enth_mol[:].value = 27832

    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.aFlash.initialize(outlvl=0)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Flash initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aMixer.FeedWater, fs.aPipe.outlet)
        _set_port(fs.aMixer.SatWater, fs.aFlash.liq_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aMixer.initialize(outlvl=0)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Mixer initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aDrum.inlet, fs.aMixer.outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aDrum.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Drum initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.blowdown_split.inlet, fs.aDrum.outlet)
        dof1 = degrees_of_freedom(fs)
        fs.blowdown_split.initialize()
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after blowdown_split initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aDowncomer.inlet, fs.blowdown_split.FW_Downcomer)
        dof1 = degrees_of_freedom(fs)
        fs.aDowncomer.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("//////////////////after Downcomer initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.Waterwalls[1].inlet, fs.aDowncomer.outlet)
        dof1 = degrees_of_freedom(fs)
        fs.Waterwalls[1].initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff, ' for Waterwall 1')
        print("//////////////////after Zone 1 initialization ///////////////")

    for i in fs.ww_zones:
       if (i>1):
           if m.dynamic==False or m.init_dyn==True:
               _set_port(fs.Waterwalls[i].inlet, fs.Waterwalls[i-1].outlet)
               dof1 = degrees_of_freedom(fs)
               fs.Waterwalls[i].initialize(outlvl=4)
               dof2 = degrees_of_freedom(fs)
               dof_diff = dof2 - dof1
               print('change of dof=', dof_diff, ' for Waterwall number ', i)
               print("////////////////// After all zones initialization ///////////////")

    # tear steam between RH1 and RH2
    fs.aRH2.tube_inlet.flow_mol[:].value = 7351.8
    fs.aRH2.tube_inlet.enth_mol[:].value = 61628
    fs.aRH2.tube_inlet.pressure[:].value = 2974447

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aRH2.shell_inlet, fs.aBoiler.side_1_outlet)
        # use a lower temperature to avoid convegence issue
        fs.aRH2.shell_inlet.temperature[:].value = 1350
        dof1 = degrees_of_freedom(fs)
        fs.aRH2.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("////////////////// After RH2 initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aRH1.shell_inlet, fs.aRH2.shell_outlet)
        # use a lower temperature to avoid convegence issue
        fs.aRH1.shell_inlet.temperature[:].value = 1200
        dof1 = degrees_of_freedom(fs)
        fs.aRH1.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("////////////////// After RH1 initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aRoof.inlet, fs.aFlash.vap_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aRoof.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("////////////////// After Roof initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aPSH.tube_inlet, fs.aRoof.outlet)
        _set_port(fs.aPSH.shell_inlet, fs.aRH1.shell_outlet)
        # use a lower temperature to avoid convegence issue
        fs.aPSH.shell_inlet.temperature[:].value = 1000
        dof1 = degrees_of_freedom(fs)
        fs.aPSH.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("////////////////// After PSH initialization ///////////////")

    # fixed inlet conditions, to be linked with steam cycle
    fs.Attemp.Water_inlet.flow_mol[:].fix(20)
    fs.Attemp.Water_inlet.pressure[:].value = 12227274.0
    fs.Attemp.Water_inlet.enth_mol[:].fix(12767)

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.Attemp.Steam_inlet, fs.aPSH.tube_outlet)
        dof1 = degrees_of_freedom(fs)
        fs.Attemp.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("////////////////// After Attemp initialization ///////////////")

    if m.dynamic==False or m.init_dyn==True:
        _set_port(fs.aPlaten.inlet, fs.Attemp.outlet)
        dof1 = degrees_of_freedom(fs)
        fs.aPlaten.initialize(outlvl=4)
        dof2 = degrees_of_freedom(fs)
        dof_diff = dof2 - dof1
        print('change of dof=', dof_diff)
        print("////////////////// After Platen initialization ///////////////")

    ############################# Unfix variables after initialization ###########################
    # blowdown split fraction
    fs.blowdown_split.split_fraction[:,"FW_Blowdown"].unfix()
    # Waterwalls[14] heat duty
    for i in fs.ww_zones:
       fs.Waterwalls[i].heat_fireside[:].unfix()
    # heat duty to aPlaten and aRoof
    fs.aPlaten.heat_fireside[:].unfix()
    fs.aRoof.heat_fireside[:].unfix()
    # all wall temperatures
    for i in fs.aBoiler.zones:
       fs.aBoiler.in_WWTemp[:,i].unfix()
    # air and gas component flows
    for i in prop_gas.component_list:
        # SA at FD fan outlet and aAPH inlet
        fs.aAPH.side_2_inlet.flow_mol_comp[:,i].unfix()
        # PA at aAPH inlet
        fs.aAPH.side_3_inlet.flow_mol_comp[:,i].unfix()
        # Tempering air at Mixer_PA inlet
        fs.Mixer_PA.TA_inlet.flow_mol_comp[:, i].unfix()
    # SA pressure needs to be unfixed
    fs.aBoiler.side_2_inlet.pressure[:].unfix()
    # inlet flow based on constraint
    fs.aRH1.tube_inlet.flow_mol[:].unfix()

    #fix feed water pressure and enthalpy but allow flow rate to change
    fs.aECON.tube_inlet.flow_mol[:].unfix()
    fs.aBoiler.SR[:].unfix()
    fs.aBoiler.in_2ndAir_Temp[:].unfix()
    fs.aBoiler.ratio_PA2coal[:].unfix()
    # due to a constraint to set SA and PA deltP the same
    fs.aAPH.deltaP_side_2[:].unfix()
    fs.aAPH.ua_side_2.unfix()
    fs.aAPH.ua_side_3.unfix()

    fs.aBoiler.in_2ndAir_Temp[:].unfix()
    fs.aBoiler.fheat_ww.unfix()

    df = degrees_of_freedom(fs)
    print ("***************degree of freedom = ", df, "********************")
    assert df == 0

    if m.dynamic==False or m.init_dyn==True:
            fs.aBoiler.fheat_ww.fix()
            fs.fheat_ww_eqn.deactivate()
            _log.info("//////////////////Solving boiler steady-state problem///////////////")
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver.solve(fs, tee=slc.tee)
            _log.info("Solving boiler steady-state problem: {}".format(idaeslog.condition(res)))

            fs.aBoiler.fheat_ww.unfix()
            fs.fheat_ww_eqn.activate()
            _log.info("//////////////////Solving different fheat_ww ///////////////")
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver.solve(fs, tee=slc.tee)
            _log.info("Solving high coal flow rate problem: {}".format(idaeslog.condition(res)))
            print("feed water flow=", fs.aMixer.FeedWater.flow_mol[0].value)
            print("flash in flow=", fs.aFlash.inlet.flow_mol[0].value)
            print("flash in enth=", fs.aFlash.inlet.enth_mol[0].value)
            print('flowrate_coal_raw=', pyo.value(fs.aBoiler.flowrate_coal_raw[0]))
            print('fraction of moisture in raw coal=', pyo.value(fs.aBoiler.mf_H2O_coal_raw[0]))
            print('flowrate_fluegas=', pyo.value(fs.aBoiler.flowrate_fluegas[0]))
            print('flowrate_PA=', pyo.value(fs.aBoiler.flowrate_PA[0]))
            print('flowrate_SA=', pyo.value(fs.aBoiler.flowrate_SA[0]))
            print('flowrate_TCA=', pyo.value(fs.aBoiler.flowrate_TCA[0]))
            print('ratio_PA2coal=', pyo.value(fs.aBoiler.ratio_PA2coal[0]))
            print('SR=', fs.aBoiler.SR[0].value)
            print('SR_lf=', fs.aBoiler.SR_lf[0].value)
            print('Roof slag thickness =', fs.aRoof.thk_slag[0].value)
            print('boiler ww heat=', fs.aBoiler.heat_total_ww[0].value)
            print('boiler platen heat=', fs.aBoiler.out_heat[0,15].value)
            print('boiler roof heat=', fs.aBoiler.out_heat[0,16].value)
            print('FEGT =', fs.aRH2.shell_inlet.temperature[0].value)
            print('O2 in flue gas =', fs.aBoiler.fluegas_o2_pct_dry[0].value)
            print('main steam flow =', fs.aPlaten.outlet.flow_mol[0].value)
            print('main steam pressure =', fs.aPlaten.outlet.pressure[0].value)
            print('main steam enthalpy =', fs.aPlaten.outlet.enth_mol[0].value)
            print('main steam temperature =', pyo.value(fs.aPlaten.control_volume.properties_out[0].temperature))
            print('RH2 steam flow =', fs.aRH2.tube_outlet.flow_mol[0].value)
            print('RH2 steam pressure =', fs.aRH2.tube_outlet.pressure[0].value)
            print('RH2 steam enthalpy =', fs.aRH2.tube_outlet.enth_mol[0].value)
            print('RH2 steam out temperature =', pyo.value(fs.aRH2.tube.properties[0,0].temperature))
            print('RH1 steam flow =', fs.aRH1.tube_outlet.flow_mol[0].value)
            print('RH1 steam pressure =', fs.aRH1.tube_outlet.pressure[0].value)
            print('RH1 steam enthalpy =', fs.aRH1.tube_outlet.enth_mol[0].value)
            print('RH1 steam out temperature =', pyo.value(fs.aRH1.tube.properties[0,0].temperature))
            print('RH1 steam in temperature =', pyo.value(fs.aRH1.tube.properties[0,1].temperature))
            print("feed water flow=", fs.aECON.tube_inlet.flow_mol[0].value)
            print("feed water pressure=", fs.aECON.tube_inlet.pressure[0].value)
            print("feed water enthalpy=", fs.aECON.tube_inlet.enth_mol[0].value)
            print('feed water temperature =', pyo.value(fs.aECON.tube.properties[0,1].temperature))
            print('Econ outlet water temperature =', pyo.value(fs.aECON.tube.properties[0,0].temperature))
            print('flue gas temp after RH2 =', fs.aRH2.shell_outlet.temperature[0].value)
            print('flue gas temp after RH1 =', fs.aRH1.shell_outlet.temperature[0].value)
            print('flue gas temp after PSH =', fs.aPSH.shell_outlet.temperature[0].value)
            print('flue gas temp after ECON =', fs.aECON.shell_outlet.temperature[0].value)
            print('furnace pressure =', fs.aRH2.shell_inlet.pressure[0].value)
            print('flue gas pres after RH2 =', fs.aRH2.shell_outlet.pressure[0].value)
            print('flue gas pres after RH1 =', fs.aRH1.shell_outlet.pressure[0].value)
            print('flue gas pres after PSH =', fs.aPSH.shell_outlet.pressure[0].value)
            print('flue gas pres after ECON =', fs.aECON.shell_outlet.pressure[0].value)
            for i in fs.ww_zones:
                print ('heat fireside of zone [', i, ']=', fs.Waterwalls[i].heat_fireside[0].value)
                print ('heat duty of zone [', i, ']=', fs.Waterwalls[i].heat_duty[0].value)
            print('heat duty of platen=', fs.aBoiler.out_heat[0,15].value)
            print('heat duty of roof=', fs.aBoiler.out_heat[0,16].value)
            print('heat duty of all water walls=', pyo.value(fs.aBoiler.heat_total_ww[0]))
            for i in fs.aBoiler.zones:
                print('wall temp [', i, ']=', fs.aBoiler.in_WWTemp[0,i].value)
            print('heat duty of drum=', pyo.value(fs.aDrum.heat_duty[0]))
            print('heat flux platen =', pyo.value(fs.aPlaten.heat_flux_interface[0]))
            print('total heat PSH =', pyo.value(fs.aPSH.total_heat[0]))
            print('total heat PSH from enth =', pyo.value((fs.aPSH.tube_outlet.enth_mol[0]-fs.aPSH.tube_inlet.enth_mol[0])*fs.aPSH.tube_outlet.flow_mol[0]))

    return m
