import time

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import IDAES
from idaes.core import FlowsheetBlock

from idaes.core.util import model_serializer as ms
from idaes.core.util import copy_port_values as _set_port

from idaes.generic_models.unit_models import MomentumMixingType

from idaes.generic_models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.power_generation.unit_models.helm import (
    HelmTurbineStage as TurbineStage,
    #HelmTurbineOutletStage as TurbineOutletStage,
    ValveFunctionType,
    #HelmTurbineMultistage as TurbineMultistage,
    HelmMixer as Mixer,
    MomentumMixingType,
    HelmValve as SteamValve,
    HelmValve as WaterValve,
    HelmIsentropicCompressor as WaterPump,
    HelmSplitter as Separator,
    #HelmNtuCondenser as Condenser,
)

from unit_models import WaterTank, FWH0D, Drum, PIDController
from unit_models import HelmTurbineOutletStage as TurbineOutletStage
from unit_models import HelmTurbineMultistage as TurbineMultistage
from unit_models import HelmNtuCondenser as Condenser

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.properties import iapws95

import idaes.logger as idaeslog

def add_unit_models(m):
    """Add unit models."""
    fs = m.fs_main.fs_stc
    prop_water = m.fs_main.prop_water
    # multistage turbine:
    def throttle_valve_function(blk):
        blk.Cv.fix(1) # this valve function includes Cv.
        a = blk.vfa = pyo.Var(initialize=2.8904E-02, doc="Valve function parameter")
        b = blk.vfb = pyo.Var(initialize=3.3497E-02, doc="Valve function parameter")
        c = blk.vfc = pyo.Var(initialize=1.4514E-02, doc="Valve function parameter")
        d = blk.vfd = pyo.Var(initialize=1.4533E-03, doc="Valve function parameter")
        a.fix()
        b.fix()
        c.fix()
        d.fix()
        o = blk.valve_opening
        @blk.Expression(m.fs_main.time)
        def valve_function(bd, t):
            return a*o[t]**3 - b*o[t]**2 + c*o[t] - d

    fs.turb = TurbineMultistage(
        default={
            "dynamic": False,
            "property_package": prop_water,
            "num_parallel_inlet_stages": 1,
            "throttle_valve_function": ValveFunctionType.custom,
            "throttle_valve_function_callback": throttle_valve_function,
            "num_hp": 14,
            "num_ip": 9,
            "num_lp": 5,
            "hp_split_locations": [14],
            "ip_split_locations": [6, 9],
            "lp_split_locations": [2, 4, 5],
            "hp_disconnect": [14],  # 14 is last hp stage, disconnect hp from ip
            "hp_split_num_outlets": {14: 4},
            "ip_split_num_outlets": {9: 3},
        }
    )
    fs.bfp_turb_valve = SteamValve(
        default={"dynamic": False,"property_package": prop_water})

    # feed water pump turbine (stages before outlet stage)
    fs.bfp_turb = TurbineStage(
        default={
            "dynamic": False,
            "property_package": prop_water,
        }
    )

    # feed water pump turbine (outlet stages)
    fs.bfp_turb_os = TurbineOutletStage(
        default={
            "dynamic": False,
            "property_package": prop_water,
        }
    )

    ## Condenser HX
    fs.condenser = Condenser(
        default={
            "dynamic": False,
            "shell": {
                "has_pressure_change": False,
                "property_package": prop_water,
            },
            "tube": {
                "has_pressure_change": False,
                "property_package": prop_water,
            },
        }
    )

    fs.cw_temperature = pyo.Var(fs.config.time)
    @fs.Constraint(fs.config.time)
    def cw_temperature_eqn(b, t):
        return b.cw_temperature[t] == \
            b.condenser.tube.properties_in[t].temperature

    ## Aux Condenser HX
    fs.aux_condenser = Condenser(
        default={
            "dynamic": False,
            "shell": {
                "has_pressure_change": False,
                "property_package": prop_water,
            },
            "tube": {
                "has_pressure_change": False,
                "property_package": prop_water,
            },
        }
    )

    fs.cw_aux_temperature = pyo.Var(fs.config.time)
    @fs.Constraint(fs.config.time)
    def cw_aux_temperature_eqn(b, t):
        return b.cw_aux_temperature[t] == \
            b.aux_condenser.tube.properties_in[t].temperature

    # Condenser Hotwel, modeled as Mixer. The makeup stream is also added even
    # though it is added to hotwell tank. Set momentum_mixing_type to none since
    # aux_condenser outlet pressure usually not equal to the main condenser
    # outlet pressure We impose the constraints to let the mixed pressure equal
    # to the main condenser pressure and makeup
    fs.condenser_hotwell = Mixer(
        default={
            "dynamic": False,
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["main_condensate", "makeup", "aux_condensate"],
            "property_package": prop_water,
        }
    )

    # Water valve for makeup water to hotwell
    fs.makeup_valve = WaterValve(
        default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )

    # Hotwell Tank, needed for dynamic model
    fs.hotwell_tank = WaterTank(
        default={
            "has_holdup": True,
            "property_package": prop_water,
        }
    )

    # Condensate pump
    fs.cond_pump = WaterPump(
        default={
            "dynamic": False,
            "property_package": prop_water,
        }
    )

    # Splitter after condensate pump
    #outlet_1: condensate to FHW1, outlet_2: hotwell water rejection, outlet_3: DA water rejection
    fs.cond_split = Separator(
        default={
            "dynamic": False,
            "num_outlets": 3,
            "property_package": prop_water
        }
    )

    # Control valve for main condensate flow to FWH1, used to control DA level, after cond_split.outlet_1
    fs.cond_valve = WaterValve(
        default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )

    # Hotwell water rejection valve if level is too high, after cond_split.outlet_2
    fs.hotwell_rejection_valve = WaterValve(
        default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )

    # DA water rejection valve if level is too high, after cond_split.outlet_3
    fs.da_rejection_valve = WaterValve(
        default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )

    # FWH1
    fs.fwh1 = FWH0D(
        default={
            "has_desuperheat": False,
            "has_drain_cooling": False,
            "has_drain_mixer": True,
            "condense": {"tube": {"has_pressure_change": True},
                         "shell": {"has_pressure_change": True},
                         "has_holdup": True},
            "property_package": prop_water,
        }
    )

    # pump for fwh1 condensate
    fs.fwh1_drain_pump = WaterPump(
        default={
            "dynamic": False,
            "property_package": prop_water,
        }
    )
    # Mix the FWH1 drain back into the feedwater
    fs.fwh1_drain_return = Mixer(
        default={
            "dynamic": False,
            "inlet_list": ["feedwater", "fwh1_drain"],
            "property_package": prop_water,
            "momentum_mixing_type": MomentumMixingType.equality,
        }
    )
    # FWH2
    fs.fwh2 = FWH0D(
        default={
            "has_desuperheat": False,
            "has_drain_cooling": True,
            "has_drain_mixer": True,
            "condense": {"tube": {"has_pressure_change": True},
                         "shell": {"has_pressure_change": True},
                         "has_holdup": True},
            "cooling": {"dynamic": False, "has_holdup": False},
            "property_package": prop_water,
        }
    )
    # Control valve after fwh2
    fs.fwh2_valve = WaterValve(default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water,
        }
    )
    # FWH3
    fs.fwh3 = FWH0D(
        default={
            "has_desuperheat": False,
            "has_drain_cooling": True,
            "has_drain_mixer": False,
            "condense": {"tube": {"has_pressure_change": True},
                         "shell": {"has_pressure_change": True},
                         "has_holdup": True},
            "cooling": {"dynamic":False, "has_holdup": False},
            "property_package": prop_water,
        }
    )
    # control valve after fwh3
    fs.fwh3_valve = WaterValve(default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )
    # FWH4 deaerator, used MomentumMixingType.equality for momentum_mixing_type
    fs.fwh4_deair = Mixer(
        default={  # general unit model config
            "dynamic": False,
            "momentum_mixing_type": MomentumMixingType.equality,
            "inlet_list": ["steam", "drain", "feedwater"],
            "property_package": prop_water,
        }
    )
    # Deaerator water tank, modeled as 0-D drum
    fs.da_tank = Drum(
        default={"property_package": prop_water,
                 "has_holdup": True,
                 "has_heat_transfer": False,
                 "has_pressure_change": True
        }
    )
    # boiler feed booster pump (pump before BFP)
    fs.booster = WaterPump(
        default={
            "dynamic": False,
            "property_package": prop_water,
        }
    )
    # BFP (boiler feed pump)
    fs.bfp = WaterPump(
        default={
            "dynamic": False,
            "property_package": prop_water,
        }
    )
    fs.split_attemp = Separator(
        default={
            "dynamic": False,
            "property_package": prop_water,
            "outlet_list": ["FeedWater", "Spray"],
        }
    )

    # Attemperator spray valve
    fs.spray_valve = WaterValve(
        default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )

    # FWH5
    fs.fwh5 = FWH0D(
        default={
            "has_desuperheat": True,
            "has_drain_cooling": True,
            "has_drain_mixer": True,
            "condense": {"tube": {"has_pressure_change": True},
                         "shell": {"has_pressure_change": True},
                         "has_holdup": True},
            "desuperheat": {"dynamic": False},
            "cooling": {"dynamic":False, "has_holdup": False},
            "property_package": prop_water,
        }
    )
    # Control valve after fwh5
    fs.fwh5_valve = WaterValve(default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )
    # FWH6
    fs.fwh6 = FWH0D(
        default={
            "has_desuperheat": True,
            "has_drain_cooling": True,
            "has_drain_mixer": False,
            "condense": {"tube": {"has_pressure_change": True},
                         "shell": {"has_pressure_change": True},
                         "has_holdup": True},
            "desuperheat": {"dynamic": False},
            "cooling": {"dynamic": False, "has_holdup": False},
            "property_package": prop_water,
        }
    )
    # Control valve after fwh6
    fs.fwh6_valve = WaterValve(default={
            "dynamic": False,
            "has_holdup": False,
            "phase":"Liq",
            "property_package": prop_water
        }
    )
    fs.temperature_main_steam = pyo.Var(fs.config.time, initialize=810)
    @fs.Constraint(fs.config.time)
    def temperature_main_steam_eqn(b, t):
        return b.temperature_main_steam[t] == b.turb.throttle_valve[1].control_volume.properties_in[t].temperature

    fs.temperature_hot_reheat = pyo.Var(fs.config.time, initialize=810)
    @fs.Constraint(fs.config.time)
    def temperature_hot_reheat_eqn(b, t):
        return b.temperature_hot_reheat[t] == b.turb.ip_stages[1].control_volume.properties_in[t].temperature

    fs.power_output = pyo.Var(fs.config.time, initialize=200, doc='gross power output in MW')
    @fs.Constraint(fs.config.time)
    def power_output_eqn(b, t):
        return b.power_output[t] == -b.turb.power[t]/1e6
    # Controllers
    if m.dynamic==True:
        # FWH level controllers
        fs.fwh2_ctrl = PIDController(default={"pv":fs.fwh2.condense.level,
                                  "mv":fs.fwh2_valve.valve_opening,
                                  "type": 'PI'})
        fs.fwh3_ctrl = PIDController(default={"pv":fs.fwh3.condense.level,
                                  "mv":fs.fwh3_valve.valve_opening,
                                  "type": 'PI'})
        fs.fwh5_ctrl = PIDController(default={"pv":fs.fwh5.condense.level,
                                  "mv":fs.fwh5_valve.valve_opening,
                                  "type": 'PI'})
        fs.fwh6_ctrl = PIDController(default={"pv":fs.fwh6.condense.level,
                                  "mv":fs.fwh6_valve.valve_opening,
                                  "type": 'PI'})
        # DA level controller, cond_split.outlet_1
        fs.da_ctrl = PIDController(default={"pv":fs.da_tank.level,
                                  "mv":fs.cond_valve.valve_opening,
                                  "type": 'PI'})
        # Hotwell makeup water controller
        fs.makeup_ctrl = PIDController(default={"pv":fs.hotwell_tank.level,
                                  "mv":fs.makeup_valve.valve_opening,
                                  "type": 'PI',
                                  "bounded_output": True})
        # Attemp water spray controller
        fs.spray_ctrl = PIDController(default={"pv":fs.temperature_main_steam,
                                  "mv":fs.spray_valve.valve_opening,
                                  "type": 'PID',
                                  "bounded_output": True})
    return m

def set_arcs_and_constraints(m):
    """ Add streams as arcs """
    fs = m.fs_main.fs_stc
    fs.S017 = Arc(
        source=fs.turb.outlet_stage.outlet, destination=fs.condenser.inlet_1
    )
    fs.S053 = Arc(
        source=fs.bfp_turb.outlet, destination=fs.bfp_turb_os.inlet
    )
    fs.S046 = Arc(
        source=fs.bfp_turb_os.outlet, destination=fs.aux_condenser.inlet_1
    )
    fs.S022 = Arc(
        source=fs.condenser.outlet_1,
        destination=fs.condenser_hotwell.main_condensate,
    )
    fs.S047 = Arc(
        source=fs.aux_condenser.outlet_1,
        destination=fs.condenser_hotwell.aux_condensate,
    )
    fs.S050b = Arc(
        source=fs.makeup_valve.outlet,
        destination=fs.condenser_hotwell.makeup,
    )
    fs.S031a = Arc(
        source=fs.fwh1.condense.outlet_2, destination=fs.fwh1_drain_return.feedwater
    )
    fs.S030 = Arc(
        source=fs.fwh1.condense.outlet_1, destination=fs.fwh1_drain_pump.inlet
    )
    fs.S045 = Arc(
        source=fs.fwh1_drain_pump.outlet, destination=fs.fwh1_drain_return.fwh1_drain
    )
    fs.S024 = Arc(
        source=fs.condenser_hotwell.outlet, destination=fs.hotwell_tank.inlet
    )
    fs.S025 = Arc(
        source=fs.hotwell_tank.outlet, destination=fs.cond_pump.inlet
    )
    fs.S016 = Arc(
        source=fs.turb.lp_split[5].outlet_2, destination=fs.fwh1.drain_mix.steam
    )
    fs.S032 = Arc(
        source=fs.fwh2.cooling.outlet_1, destination=fs.fwh2_valve.inlet
    )
    fs.S032b = Arc(
        source=fs.fwh2_valve.outlet, destination=fs.fwh1.drain_mix.drain
    )
    fs.S026 = Arc(
        source=fs.cond_pump.outlet, destination=fs.cond_split.inlet
    )
    fs.S026b = Arc(
        source=fs.cond_split.outlet_1, destination=fs.cond_valve.inlet
    )
    fs.S027 = Arc(
        source=fs.cond_split.outlet_2, destination=fs.hotwell_rejection_valve.inlet
    )
    fs.S028 = Arc(
        source=fs.cond_split.outlet_3, destination=fs.da_rejection_valve.inlet
    )
    fs.S029 = Arc(
        source=fs.cond_valve.outlet, destination=fs.fwh1.condense.inlet_2
    )
    fs.S031b = Arc(
        source=fs.fwh1_drain_return.outlet, destination=fs.fwh2.cooling.inlet_2
    )
    fs.S015 = Arc(
        source=fs.turb.lp_split[4].outlet_2, destination=fs.fwh2.drain_mix.steam
    )
    fs.S034 = Arc(
        source=fs.fwh3.cooling.outlet_1, destination=fs.fwh3_valve.inlet
    )
    fs.S034b = Arc(
        source=fs.fwh3_valve.outlet, destination=fs.fwh2.drain_mix.drain
    )
    fs.S033 = Arc(
        source=fs.fwh2.condense.outlet_2, destination=fs.fwh3.cooling.inlet_2
    )
    fs.S014 = Arc(
        source=fs.turb.lp_split[2].outlet_2, destination=fs.fwh3.condense.inlet_1
    )
    fs.S035 = Arc(
        source=fs.fwh3.condense.outlet_2, destination=fs.fwh4_deair.feedwater
    )
    fs.S043 = Arc(
        source=fs.turb.ip_split[9].outlet_2, destination=fs.fwh4_deair.steam
    )
    fs.S011 = Arc(
        source=fs.turb.ip_split[9].outlet_3, destination=fs.bfp_turb_valve.inlet
    )
    fs.S052= Arc(
        source=fs.bfp_turb_valve.outlet, destination=fs.bfp_turb.inlet
    )
    fs.S036 = Arc(
        source=fs.fwh4_deair.outlet, destination=fs.da_tank.inlet
    )
    fs.S036b = Arc(
        source=fs.da_tank.outlet, destination=fs.booster.inlet
    )
    fs.booster_static_head = pyo.Var(initialize=5.25e4)
    fs.booster_static_head.fix()
    @fs.Constraint(
        fs.config.time,
        doc="Add static head due to elevation change for booster pump")
    def booster_static_head_eqn(b, t):
        return fs.booster.inlet.pressure[t] == fs.da_tank.outlet.pressure[t] + b.booster_static_head
    fs.booster_static_head_eqn.deactivate()

    fs.S038 = Arc(
        source=fs.booster.outlet, destination=fs.bfp.inlet
    )
    fs.S037 = Arc(
        source=fs.bfp.outlet, destination=fs.split_attemp.inlet
    )
    fs.S054 = Arc(
        source=fs.split_attemp.Spray, destination=fs.spray_valve.inlet
    )
    fs.S039 = Arc(
        source=fs.fwh5.cooling.outlet_1, destination=fs.fwh5_valve.inlet
    )
    fs.S039b = Arc(
        source=fs.fwh5_valve.outlet, destination=fs.fwh4_deair.drain
    )
    fs.S051 = Arc(
        source=fs.split_attemp.FeedWater, destination=fs.fwh5.cooling.inlet_2
    )
    fs.S010 = Arc(
        source=fs.turb.ip_split[6].outlet_2, destination=fs.fwh5.desuperheat.inlet_1
    )
    fs.S041 = Arc(
        source=fs.fwh6.cooling.outlet_1, destination=fs.fwh6_valve.inlet
    )
    fs.S041b = Arc(
        source=fs.fwh6_valve.outlet, destination=fs.fwh5.drain_mix.drain
    )
    fs.S040 = Arc(
        source=fs.fwh5.desuperheat.outlet_2, destination=fs.fwh6.cooling.inlet_2
    )
    fs.S006 = Arc(
        source=fs.turb.hp_split[14].outlet_2, destination=fs.fwh6.desuperheat.inlet_1
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(fs)

    # Add and extra port to the turbine outlet that will let me hook it to the
    # condeser heatexchanger which uses T,P,x state variables because it is
    # easier solve the condenser pressure calculation with those vars.

    @fs.Constraint(fs.config.time)
    def constraint_bfp_power(b, t):
        return 0 == 1e-6*fs.bfp.control_volume.work[t] + 1e-6*fs.bfp_turb.control_volume.work[t] + 1e-6*fs.bfp_turb_os.control_volume.work[t]

    @fs.turb.Constraint(fs.config.time)
    def constraint_reheat_flow(b, t):
        return b.ip_stages[1].inlet.flow_mol[t] == b.hp_split[14].outlet_1.flow_mol[t]

    # declare as flowsheet level constraint to avoid using it in unit initialization
    @fs.Constraint(fs.config.time)
    def makeup_water_pressure_constraint(b, t):
        return b.condenser_hotwell.makeup_state[t].pressure*1e-4 == b.condenser_hotwell.aux_condensate_state[t].pressure*1e-4

    @fs.condenser_hotwell.Constraint(fs.config.time)
    def mixer_pressure_constraint(b, t):
        return b.aux_condensate_state[t].pressure*1e-4 == b.mixed_state[t].pressure*1e-4

    # set DA tank outlet enthalpy to saturation enthalpy at inlet - 100 (to make sure DA tank is liquid only)
    # this constraint is very important to avoid flash of DA tank during load ramping down and could causes convergence issue
    @fs.Constraint(fs.config.time)
    def da_outlet_enthalpy_constraint(b, t):
        return 1e-3*b.da_tank.outlet.enth_mol[t] == 1e-3*(b.da_tank.control_volume.properties_in[t].enth_mol_sat_phase["Liq"] - 100)

    # constraints to set FWH control valve outlet pressure equal to downstream shell pressure
    @fs.Constraint(fs.config.time)
    def fwh1_drain_mixer_pressure_eqn(b, t):
        return b.fwh1.drain_mix.drain.pressure[t]*1e-4 == b.fwh1.drain_mix.steam.pressure[t]*1e-4

    @fs.Constraint(fs.config.time)
    def fwh2_drain_mixer_pressure_eqn(b, t):
        return b.fwh2.drain_mix.drain.pressure[t]*1e-4 == b.fwh2.drain_mix.steam.pressure[t]*1e-4

    @fs.Constraint(fs.config.time)
    def fwh5_drain_mixer_pressure_eqn(b, t):
        return b.fwh5.drain_mix.drain.pressure[t]*1e-5 == b.fwh5.drain_mix.steam.pressure[t]*1e-5

    # cond_pump flow-deltaP curve based on plant manual
    fs.cond_pump.pump_a = pyo.Var(initialize=4.6141e-7)
    fs.cond_pump.pump_b = pyo.Var(initialize=3.7188e-3)
    fs.cond_pump.pump_c = pyo.Var(initialize=51.435)
    fs.cond_pump.pump_d = pyo.Var(initialize=1.8214e6)
    fs.cond_pump.pump_a.fix()
    fs.cond_pump.pump_b.fix()
    fs.cond_pump.pump_c.fix()
    fs.cond_pump.pump_d.fix()
    @fs.cond_pump.Constraint(fs.config.time)
    def cond_pump_curve_constraint(b, t):
        return b.deltaP[t]*1e-5 == 1e-5*(-b.pump_a*b.inlet.flow_mol[t]**3 + b.pump_b*b.inlet.flow_mol[t]**2\
                              - b.pump_c*b.inlet.flow_mol[t] + b.pump_d)

    # booster pump flow-deltaP curve based on plant data fitting
    fs.booster.booster_a = pyo.Var(initialize=2.1494e-7)
    fs.booster.booster_b = pyo.Var(initialize=2.8681e-3)
    fs.booster.booster_c = pyo.Var(initialize=30.994)
    fs.booster.booster_d = pyo.Var(initialize=9.9828e5)
    fs.booster.booster_a.fix()
    fs.booster.booster_b.fix()
    fs.booster.booster_c.fix()
    fs.booster.booster_d.fix()
    @fs.booster.Constraint(fs.config.time)
    def booster_pump_curve_constraint(b, t):
        return b.deltaP[t]*1e-5 == 1e-5*(
            -b.booster_a*b.inlet.flow_mol[t]**3 +
            b.booster_b*b.inlet.flow_mol[t]**2
            - b.booster_c*b.inlet.flow_mol[t] +
            b.booster_d)

    # water flow to economizer slightly higher than steam flow in such that there is makeup flow needed to offset blowdown
    @fs.Constraint(fs.config.time)
    def fw_flow_constraint(b, t):
        return 1e-3*b.da_tank.outlet.flow_mol[t] == 1e-3*b.turb.inlet_split.inlet.flow_mol[t]*1.02

    # Add DCA Expressions
    def rule_dca_no_cool(b, t):
        return (
            b.condense.shell.properties_out[t].temperature
            - b.condense.tube.properties_in[t].temperature
        )

    def rule_dca(b, t):
        return (
            b.cooling.shell.properties_out[t].temperature
            - b.cooling.tube.properties_in[t].temperature
        )

    fs.fwh1.dca = pyo.Expression(fs.config.time, rule=rule_dca_no_cool)
    fs.fwh2.dca = pyo.Expression(fs.config.time, rule=rule_dca)
    fs.fwh3.dca = pyo.Expression(fs.config.time, rule=rule_dca)
    fs.fwh5.dca = pyo.Expression(fs.config.time, rule=rule_dca)
    fs.fwh6.dca = pyo.Expression(fs.config.time, rule=rule_dca)

    return m


def set_inputs(m):
    """Set model parameters, fixed vars, and initail values"""
    fs = m.fs_main.fs_stc
    ############## turbine inputs
    fs.turb.turbine_inlet_cf_fix(0.0014) #original is 0.0011
    fs.turb.turbine_outlet_cf_fix(0.047) #used to be 0.0255
    fs.turb.throttle_valve[1].Cv.fix(0.6) # default is 1
    fs.turb.throttle_valve[1].valve_opening.fix(0.725)
    fs.turb.inlet_stage[1].eff_nozzle.fix(0.93)
    fs.turb.inlet_stage[1].blade_reaction.fix(0.91)

    # Set the turbine steam inlet conditions and flow guess for init
    pin = 12161444.9 #1.082e7 #1.22e7 #10544531#
    hin = 62077.55 #62060 #iapws95.htpx(T=810.9, P=pin)
    fs.turb.inlet_split.inlet.enth_mol[:].fix(hin)
    fs.turb.inlet_split.inlet.flow_mol[:].value = 9500
    fs.turb.inlet_split.inlet.flow_mol[:].unfix()
    fs.turb.inlet_split.inlet.pressure[:].fix(pin)
    # Set inlet mixer up for pressure driven flow.
    fs.turb.inlet_mix.use_equal_pressure_constraint()

    # Set inlet of the ip section for initialization, since it is disconnected
    # revised by JM, fix the IP inlet pressure and enthalpy and use 90% of main steam flow as initial guess
    pin = 2764376.3
    hin = 64247.6
    fs.turb.ip_stages[1].inlet.enth_mol[:].fix(hin)
    fs.turb.ip_stages[1].inlet.pressure[:].fix(pin)
    fs.turb.ip_stages[1].inlet.flow_mol[:].value = 9500*0.9

    # Set the efficency and pressure ratios of stages other than inlet and outlet
    for i, s in fs.turb.hp_stages.items():
        s.ratioP.fix(0.930)
        s.efficiency_isentropic.fix(0.932)
    for i, s in fs.turb.ip_stages.items():
        if i <= 6:
            s.ratioP.fix(0.871)
            s.efficiency_isentropic.fix(0.920)
        else:
            s.ratioP.fix(0.832)
            s.efficiency_isentropic.fix(0.965)
    for i, s in fs.turb.lp_stages.items():
        if i <= 2:
            s.ratioP[:].fix(0.68)
            s.efficiency_isentropic[:].fix(0.863)
        elif i <= 4:
            s.ratioP[:].fix(0.65)
            s.efficiency_isentropic[:].fix(0.855)
        else:
            s.ratioP[:].fix(0.31)
            s.efficiency_isentropic[:].fix(0.835)
    fs.turb.outlet_stage.eff_dry.fix(0.94) #typical 0.87 based on Eric's paper
    fs.turb.outlet_stage.design_exhaust_flow_vol.fix(4725) #oritinal 3150
    # Fix steam extraction, to be unfixed after init
    fs.turb.hp_split[14].split_fraction[:, "outlet_2"].fix(0.02) #to FWH6
    fs.turb.hp_split[14].split_fraction[:, "outlet_3"].fix(0.05) #to process plant, used to be 0
    fs.turb.hp_split[14].split_fraction[:, "outlet_4"].fix(0.0)
    fs.turb.ip_split[6].split_fraction[:, "outlet_2"].fix(0.02)
    fs.turb.ip_split[9].split_fraction[:, "outlet_2"].fix(0.01) #fixed for DA old value 0.02
    fs.turb.ip_split[9].split_fraction[:, "outlet_3"].fix(0.04) # to bfp turbine
    fs.turb.lp_split[2].split_fraction[:, "outlet_2"].fix(0.02)
    fs.turb.lp_split[4].split_fraction[:, "outlet_2"].fix(0.02)
    fs.turb.lp_split[5].split_fraction[:, "outlet_2"].fix(0.02)

    # Set boiler feed pump turbine
    fs.bfp_turb_valve.Cv.fix(0.0011)           # should be around 0.001 to pass initialization step
    fs.bfp_turb_valve.valve_opening.fix(0.5)

    # bfp turbine before outlet stage
    fs.bfp_turb.efficiency_isentropic.fix(0.85) #average of LP turbine
    fs.bfp_turb.ratioP[:].fix(0.0726) #0.0726 for LP turbine stages before outlet stage
    # bfp turbine outlet stage
    fs.bfp_turb_os.eff_dry.fix(0.94) #typical 0.87 based on Eric's paper
    fs.bfp_turb_os.design_exhaust_flow_vol.fix(150) #half of the full load value
    fs.bfp_turb_os.flow_coeff.fix(0.002667)  # during the initialization flow_coeff will initially set to 0.0002 first and then reset back

    ###################### main condenser
    fs.condenser.inlet_2.flow_mol.fix(278510)  # design water flow of 79460 gpm, measured around 100000 gpm (A+B)
    fs.condenser.inlet_2.enth_mol.fix(1800) # 24 C
    fs.condenser.inlet_2.pressure.fix(500000)
    fs.condenser.area.fix(11130)
    fs.condenser.overall_heat_transfer_coefficient.fix(3125)
    # aux condenser
    fs.aux_condenser.inlet_2.flow_mol.fix(11140)
    fs.aux_condenser.inlet_2.enth_mol.fix(1800)
    fs.aux_condenser.inlet_2.pressure.fix(500000)
    fs.aux_condenser.area.fix(333.9)
    fs.aux_condenser.overall_heat_transfer_coefficient.fix(3125)

    fs.makeup_valve.Cv.value = 2.0
    fs.makeup_valve.valve_opening.fix(0.35)

    # Since makeup pump is not modeled, it is assumed that the inlet pressure of the makeup valve is 1 bar
    fs.makeup_valve.inlet.flow_mol.fix(220)
    fs.makeup_valve.inlet.enth_mol.fix(1890)
    fs.makeup_valve.inlet.pressure.fix(1e5)

    fs.hotwell_tank.cross_section_area.fix(84)  # Normal volume 15500 gallons (58.67 m^3)  84*0.7=58.8 m^3
    fs.hotwell_tank.level[:].fix(0.7)

    fs.cond_pump.efficiency_isentropic.fix(0.80)
    fs.cond_pump.deltaP[:].value = 1.4e6

    fs.cond_split.split_fraction[:,"outlet_2"].fix(0.001)
    fs.cond_split.split_fraction[:,"outlet_3"].fix(0.001)

    fs.cond_valve.Cv.value = 16.25043860324222
    fs.cond_valve.valve_opening.fix(0.5)

    fs.hotwell_rejection_valve.Cv.value = 10
    fs.hotwell_rejection_valve.valve_opening.fix(0.001)

    fs.da_rejection_valve.Cv.value = 10
    fs.da_rejection_valve.valve_opening.fix(0.001)

    ############## FWHs
    fs.fwh1.condense.area.fix(555.6)
    fs.fwh1.condense.overall_heat_transfer_coefficient.fix(2842/3)
    fs.fwh1.condense.tube.deltaP[:].fix(0)
    # extra inputs for dynamic model
    fs.fwh1.condense.level.fix(0.2667)
    fs.fwh1.condense.id_shell.fix(1.1684)
    fs.fwh1.condense.vol_frac_shell.fix(0.74)
    fs.fwh1.condense.length_tube.fix(7.6708)
    fs.fwh1.condense.tube.volume.fix(1.34475)

    fs.fwh1_drain_pump.deltaP[:].value = 7e5
    fs.fwh1_drain_pump.efficiency_isentropic.fix(0.80)

    fs.fwh2.condense.area.fix(490.1)
    fs.fwh2.cooling.area.fix(59.5)
    fs.fwh2.condense.overall_heat_transfer_coefficient.fix(3261.7/3)
    fs.fwh2.cooling.overall_heat_transfer_coefficient.fix(2030.7/3)
    fs.fwh2.condense.tube.deltaP[:].fix(0)
    # extra inputs for dynamic model
    fs.fwh2.condense.level.fix(0.2667)
    fs.fwh2.condense.id_shell.fix(1.1938)
    fs.fwh2.condense.vol_frac_shell.fix(0.6923)
    fs.fwh2.condense.length_tube.fix(6.985) # currently use total tube length
    fs.fwh2.condense.tube.volume.fix(1.3457)

    fs.fwh2_valve.Cv.value = 5.153556825830106
    fs.fwh2_valve.valve_opening.fix(0.5)

    fs.fwh3.condense.area.fix(567.6)
    fs.fwh3.cooling.area.fix(57.1)
    fs.fwh3.condense.overall_heat_transfer_coefficient.fix(3596.4/3)
    fs.fwh3.cooling.overall_heat_transfer_coefficient.fix(1860.6/3)
    fs.fwh3.condense.tube.deltaP[:].fix(0)
    # extra inputs for dynamic model
    fs.fwh3.condense.level.fix(0.2667)
    fs.fwh3.condense.id_shell.fix(1.1938)
    fs.fwh3.condense.vol_frac_shell.fix(0.6923)
    fs.fwh3.condense.length_tube.fix(6.9596) # currently use total tube length
    fs.fwh3.condense.tube.volume.fix(1.366)

    fs.fwh3_valve.Cv.value = 1.9257762443933564
    fs.fwh3_valve.valve_opening.fix(0.5)

    fs.da_tank.diameter_drum.fix(3.62585)      # 12 feet OD with 5/8 inch thickness
    fs.da_tank.length_drum.fix(19.812)         # 65 feet
    fs.da_tank.count_downcomer.fix(1)          # 1 outlet only
    fs.da_tank.diameter_downcomer.fix(0.3556)  # 16 inch OD with 1 inch thickness based on FWH5 and FWH6 specification
    fs.da_tank.level.fix(2.7432)               # normal level 9 feet from bottom of tank

    fs.booster.efficiency_isentropic.fix(0.80)
    fs.booster.outlet.pressure.fix(1.45e6)     # 200 psig based on plant data

    fs.bfp.efficiency_isentropic.fix(0.80)
    fs.bfp.outlet.pressure.fix(1.35e7)

    fs.split_attemp.split_fraction[:, "Spray"].fix(0.0007)
    fs.spray_valve.Cv.value = 0.18
    fs.spray_valve.valve_opening.fix(0.1)

    fs.fwh5.condense.area.fix(561.1)
    fs.fwh5.desuperheat.area.fix(79.7)
    fs.fwh5.cooling.area.fix(92.9)
    fs.fwh5.condense.overall_heat_transfer_coefficient.fix(3040.5/3)
    fs.fwh5.desuperheat.overall_heat_transfer_coefficient.fix(436.8/3)
    fs.fwh5.cooling.overall_heat_transfer_coefficient.fix(2019.4/3)
    fs.fwh5.condense.tube.deltaP[:].fix(0)
    # extra inputs for dynamic model
    fs.fwh5.condense.level.fix(0.2667)
    fs.fwh5.condense.id_shell.fix(1.3462)
    fs.fwh5.condense.vol_frac_shell.fix(0.6752)
    fs.fwh5.condense.length_tube.fix(6.1976) # currently use total tube length
    fs.fwh5.condense.tube.volume.fix(1.24837)
    #fs.fwh5.cooling.tube.volume.fix(0.20668)
    #fs.fwh5.cooling.shell.volume.fix(0.7557)

    fs.fwh5_valve.Cv.value = 3.5186973415718414
    fs.fwh5_valve.valve_opening.fix(0.5)

    fs.fwh6.condense.area.fix(691.2)
    fs.fwh6.desuperheat.area.fix(114.3)
    fs.fwh6.cooling.area.fix(111.0)
    fs.fwh6.condense.overall_heat_transfer_coefficient.fix(3215.6/3)
    fs.fwh6.desuperheat.overall_heat_transfer_coefficient.fix(782.8/3)
    fs.fwh6.cooling.overall_heat_transfer_coefficient.fix(1900.3/3)
    fs.fwh6.condense.tube.deltaP[:].fix(0)
    # extra inputs for dynamic model
    fs.fwh6.condense.level.fix(0.2667)
    fs.fwh6.condense.id_shell.fix(1.27)
    fs.fwh6.condense.vol_frac_shell.fix(0.6622)
    fs.fwh6.condense.length_tube.fix(8.372) # currently use total tube length
    fs.fwh6.condense.tube.volume.fix(1.5494)
    #fs.fwh6.cooling.tube.volume.fix(0.2489)
    #fs.fwh6.cooling.shell.volume.fix(0.85174)

    fs.fwh6_valve.Cv.value = 1.4219900087199113
    fs.fwh6_valve.valve_opening.fix(0.5)
    if m.dynamic==True:
        fs.fwh2_ctrl.gain_p.fix(-1e-1)
        fs.fwh2_ctrl.gain_i.fix(-1e-1)
        fs.fwh2_ctrl.setpoint.fix(0.2667)
        fs.fwh2_ctrl.mv_ref.fix(0.5)

        fs.fwh3_ctrl.gain_p.fix(-1e-1)
        fs.fwh3_ctrl.gain_i.fix(-1e-1)
        fs.fwh3_ctrl.setpoint.fix(0.2667)
        fs.fwh3_ctrl.mv_ref.fix(0.5)

        fs.fwh5_ctrl.gain_p.fix(-1e-1)
        fs.fwh5_ctrl.gain_i.fix(-1e-1)
        fs.fwh5_ctrl.setpoint.fix(0.2667)
        fs.fwh5_ctrl.mv_ref.fix(0.5)

        fs.fwh6_ctrl.gain_p.fix(-1e-1)
        fs.fwh6_ctrl.gain_i.fix(-1e-1)
        fs.fwh6_ctrl.setpoint.fix(0.2667)
        fs.fwh6_ctrl.mv_ref.fix(0.5)

        fs.da_ctrl.gain_p.fix(1e-1)
        fs.da_ctrl.gain_i.fix(1e-1)
        fs.da_ctrl.setpoint.fix(2.7432)  # normal level at 2.7432 m (9 ft)
        fs.da_ctrl.mv_ref.fix(0.5)

        fs.makeup_ctrl.gain_p.fix(0.01)   # from 0.5
        fs.makeup_ctrl.gain_i.fix(0.001)   # from 1.0
        fs.makeup_ctrl.setpoint.fix(0.7)  # normal level at 2 m (assumed) screen shot 27.7" (0.704 m)
        fs.makeup_ctrl.mv_ref.fix(0.35)    # use the valve for regulation rather than intermitent valve

        fs.spray_ctrl.gain_p.fix(-5e-2) #increased from 0.1
        fs.spray_ctrl.gain_i.fix(-1e-4) #increased from 1e-4
        fs.spray_ctrl.gain_d.fix(-1e-4) #increased from 1e-4
        fs.spray_ctrl.setpoint.fix(810)
        fs.spray_ctrl.mv_ref.fix(0.1)  #decreased from 0.15

        t0 = fs.config.time.first()
        fs.fwh2_ctrl.integral_of_error[t0].fix(0)
        fs.fwh3_ctrl.integral_of_error[t0].fix(0)
        fs.fwh5_ctrl.integral_of_error[t0].fix(0)
        fs.fwh6_ctrl.integral_of_error[t0].fix(0)
        fs.da_ctrl.integral_of_error[t0].fix(0)
        fs.makeup_ctrl.integral_of_error[t0].fix(0)
        fs.spray_ctrl.integral_of_error[t0].fix(0)
        fs.spray_ctrl.derivative_of_error[t0].fix(0)
    return m


def _add_heat_transfer_correlation(fs):
    _add_u_eq(fs.fwh6.condense)
    _add_u_eq(fs.fwh5.condense)
    _add_u_eq(fs.fwh3.condense)
    _add_u_eq(fs.fwh2.condense)
    _add_u_eq(fs.fwh1.condense)

def initialize(m):
    """ Initialize the units """
    fs = m.fs_main.fs_stc
    t0 = 0
    if m.dynamic==True:
        t0 = fs.config.time.first()
    start_time = time.time()
    outlvl = 4
    _log = idaeslog.getLogger(fs.name, outlvl, tag="unit")
    solve_log = idaeslog.getSolveLogger(fs.name, outlvl, tag="unit")
    print("Starting steam cycle initialization")

    optarg={"tol":1e-7,"linear_solver":"ma27","max_iter":100}
    solver = pyo.SolverFactory("ipopt")
    solver.options = optarg

    # set initial condition for dynamic unit models
    if m.dynamic==True:
        fs.hotwell_tank.set_initial_condition()
        fs.da_tank.set_initial_condition()
        fs.fwh1.set_initial_condition()
        fs.fwh2.set_initial_condition()
        fs.fwh3.set_initial_condition()
        fs.fwh5.set_initial_condition()
        fs.fwh6.set_initial_condition()

    # Initialize Turbine
    fs.turb.outlet_stage.control_volume.properties_out[:].pressure.fix(6000)
    if m.dynamic==False or m.init_dyn==True:
        fs.turb.initialize(
            outlvl=outlvl,
            optarg=solver.options,
            calculate_outlet_cf=False,  # original is True which causes dynamic model dof<0
            calculate_inlet_cf=False)  # changed from True

    _set_port(fs.bfp_turb_valve.inlet, fs.turb.ip_split[9].outlet_3)
    if m.dynamic==False or m.init_dyn==True:
        fs.bfp_turb_valve.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(fs.bfp_turb.inlet, fs.bfp_turb_valve.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.bfp_turb.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(fs.bfp_turb_os.inlet, fs.bfp_turb.outlet)
    fs.bfp_turb_os.control_volume.properties_out[:].pressure.fix(6000)
    if m.dynamic==False or m.init_dyn==True:
        fs.bfp_turb_os.initialize(calculate_cf=False, outlvl=outlvl, optarg=solver.options)
    fs.bfp_turb_os.control_volume.properties_out[:].pressure.unfix()

    # initialize condenser
    _set_port(fs.condenser.inlet_1, fs.turb.outlet_stage.outlet)
    fs.turb.outlet_stage.control_volume.properties_out[:].pressure.unfix()
    if m.dynamic==False or m.init_dyn==True:
        fs.condenser.initialize(unfix='pressure')

    # initialize aux_condenser
    _set_port(fs.aux_condenser.inlet_1, fs.bfp_turb_os.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.aux_condenser.initialize(unfix='pressure')

    # initialize makeup valve
    if m.dynamic==False or m.init_dyn==True:
        fs.makeup_valve.Cv.fix()
        fs.makeup_valve.initialize()
        fs.makeup_valve.Cv.unfix()

    # initialize hotwell
    _set_port(fs.condenser_hotwell.main_condensate, fs.condenser.outlet_1)
    _set_port(fs.condenser_hotwell.aux_condensate, fs.aux_condenser.outlet_1)
    _set_port(fs.condenser_hotwell.makeup, fs.makeup_valve.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.condenser_hotwell.initialize(outlvl=outlvl, optarg=solver.options)
    fs.condenser_hotwell.main_condensate.unfix()

    # initialize hotwell tank
    _set_port(fs.hotwell_tank.inlet, fs.condenser_hotwell.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.hotwell_tank.initialize()

    # initialize condensate pump
    _set_port(fs.cond_pump.inlet, fs.hotwell_tank.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.cond_pump.deltaP.fix()
        fs.cond_pump.cond_pump_curve_constraint.deactivate()
        fs.cond_pump.initialize(outlvl=outlvl, optarg=solver.options)
        fs.cond_pump.deltaP.unfix()
        fs.cond_pump.cond_pump_curve_constraint.activate()

    # initialize condensate split
    _set_port(fs.cond_split.inlet, fs.cond_pump.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.cond_split.split_fraction[:,"outlet_2"].fix()
        fs.cond_split.split_fraction[:,"outlet_3"].fix()
        fs.cond_split.initialize(outlvl=outlvl, optarg=solver.options)
        fs.cond_split.split_fraction[:,"outlet_2"].unfix()
        fs.cond_split.split_fraction[:,"outlet_3"].unfix()

    # initialize cond_valve
    _set_port(fs.cond_valve.inlet, fs.cond_split.outlet_1)
    if m.dynamic==False or m.init_dyn==True:
        fs.cond_valve.Cv.fix()
        fs.cond_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.cond_valve.Cv.unfix()

    # initialize hotwell_rejection_valve
    _set_port(fs.hotwell_rejection_valve.inlet, fs.cond_split.outlet_2)
    if m.dynamic==False or m.init_dyn==True:
        fs.hotwell_rejection_valve.Cv.fix()
        fs.hotwell_rejection_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.hotwell_rejection_valve.Cv.unfix()

    # initialize da_rejection_valve
    _set_port(fs.da_rejection_valve.inlet, fs.cond_split.outlet_3)
    if m.dynamic==False or m.init_dyn==True:
        fs.da_rejection_valve.Cv.fix()
        fs.da_rejection_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.da_rejection_valve.Cv.unfix()

    # set some initial inlet values and initialize fwh1
    fs.fwh1.drain_mix.drain.flow_mol[:] = 1000
    fs.fwh1.drain_mix.drain.pressure[:] = fs.turb.lp_split[5].outlet_2.pressure[t0].value #102042.4
    fs.fwh1.drain_mix.drain.enth_mol[:] = 6117
    _set_port(fs.fwh1.condense.inlet_2, fs.cond_valve.outlet)
    _set_port(fs.fwh1.drain_mix.steam, fs.turb.lp_split[5].outlet_2)

    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.fwh1.initialize(outlvl=outlvl, optarg=solver.options)
        dof2 = degrees_of_freedom(fs)
        dof = dof2-dof1
        print('change of dof=', dof)

    # initialize fwh1 drain pump
    _set_port(fs.fwh1_drain_pump.inlet, fs.fwh1.condense.outlet_1)
    fs.fwh1_drain_pump.control_volume.properties_out[:].pressure.fix(
        fs.fwh1.condense.tube.properties_out[t0].pressure.value
    )
    if m.dynamic==False or m.init_dyn==True:
        fs.fwh1_drain_pump.initialize(outlvl=outlvl, optarg=solver.options)
    fs.fwh1_drain_pump.control_volume.properties_out[:].pressure.unfix()

    # initialize mixer to add fwh1 drain to feedwater
    _set_port(fs.fwh1_drain_return.feedwater, fs.fwh1.condense.outlet_2)
    _set_port(fs.fwh1_drain_return.fwh1_drain, fs.fwh1_drain_pump.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.fwh1_drain_return.initialize(outlvl=outlvl, optarg=solver.options)
    fs.fwh1_drain_return.feedwater.unfix()
    fs.fwh1_drain_return.fwh1_drain.unfix()

    # set some initial inlet values and initialize fwh2
    fs.fwh2.drain_mix.drain.flow_mol[:] = 684
    fs.fwh2.drain_mix.drain.pressure[:] = fs.turb.lp_split[4].outlet_2.pressure[t0].value #3.55770e05
    fs.fwh2.drain_mix.drain.enth_mol[:] = 9106
    _set_port(fs.fwh2.cooling.inlet_2, fs.fwh1.condense.outlet_2)
    _set_port(fs.fwh2.drain_mix.steam, fs.turb.lp_split[4].outlet_2)
    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.fwh2.initialize(outlvl=outlvl, optarg=solver.options)
        dof2 = degrees_of_freedom(fs)
        dof = dof2-dof1
        print('change of dof=', dof)

    # initialize fwh2_valve
    _set_port(fs.fwh2_valve.inlet, fs.fwh2.cooling.outlet_1)
    if m.dynamic==False or m.init_dyn==True:
        # use a lower flow rate to avoid too low exit pressure
        fs.fwh2_valve.inlet.flow_mol[:].value = fs.fwh2_valve.inlet.flow_mol[t0].value*0.75
        fs.fwh2_valve.Cv.fix()
        fs.fwh2_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.fwh2_valve.Cv.unfix()

    # set some initial inlet values and initialize fwh3
    _set_port(fs.fwh3.cooling.inlet_2, fs.fwh2.condense.outlet_2)
    _set_port(fs.fwh3.condense.inlet_1, fs.turb.lp_split[2].outlet_2)
    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.fwh3.initialize(outlvl=outlvl, optarg=solver.options)
        dof2 = degrees_of_freedom(fs)
        dof = dof2-dof1
        print('change of dof=', dof)

    # initialize fwh3_valve
    _set_port(fs.fwh3_valve.inlet, fs.fwh3.cooling.outlet_1)
    if m.dynamic==False or m.init_dyn==True:
        fs.fwh3_valve.Cv.fix()
        fs.fwh3_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.fwh3_valve.Cv.unfix()

    # set some initial inlet values and initialize fwh4
    fs.fwh4_deair.drain.flow_mol[:] = 10
    fs.fwh4_deair.drain.pressure[:] = 1.28e6
    fs.fwh4_deair.drain.enth_mol[:] = 13631
    _set_port(fs.fwh4_deair.feedwater, fs.fwh3.condense.outlet_2)
    _set_port(fs.fwh4_deair.steam, fs.turb.ip_split[9].outlet_2)
    if m.dynamic==False or m.init_dyn==True:
        fs.fwh4_deair.initialize(outlvl=outlvl, optarg=solver.options)
    fs.fwh4_deair.feedwater.unfix()
    fs.fwh4_deair.steam.unfix()
    fs.fwh4_deair.drain.unfix()

    # da_tank initialization
    _set_port(fs.da_tank.inlet, fs.fwh4_deair.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.da_tank.initialize(outlvl=outlvl, optarg=solver.options)

    # init booster
    _set_port(fs.booster.inlet, fs.fwh4_deair.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.booster.booster_pump_curve_constraint.deactivate()
        fs.booster.initialize(outlvl=outlvl, optarg=solver.options)
        fs.booster.booster_pump_curve_constraint.activate()

    # init bfp
    _set_port(fs.bfp.inlet, fs.booster.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.bfp.initialize(outlvl=outlvl, optarg=solver.options)

    # split_attemp
    _set_port(fs.split_attemp.inlet, fs.bfp.outlet)
    if m.dynamic==False or m.init_dyn==True:
        fs.split_attemp.initialize(outlvl=outlvl, optarg=solver.options)

    # spray_valve
    _set_port(fs.spray_valve.inlet, fs.split_attemp.Spray)
    if m.dynamic==False or m.init_dyn==True:
        fs.spray_valve.Cv.fix()
        fs.spray_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.spray_valve.Cv.unfix()

    # init fwh5
    fs.fwh5.drain_mix.drain.flow_mol[:] = 50
    fs.fwh5.drain_mix.drain.pressure[:] = 3.47e6
    fs.fwh5.drain_mix.drain.enth_mol[:] = 15000
    _set_port(fs.fwh5.cooling.inlet_2, fs.split_attemp.FeedWater)
    _set_port(fs.fwh5.desuperheat.inlet_1, fs.turb.ip_split[6].outlet_2)
    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.fwh5.initialize(outlvl=outlvl, optarg=solver.options)
        dof2 = degrees_of_freedom(fs)
        dof = dof2-dof1
        print('change of dof=', dof)

    # initialize fwh5_valve
    _set_port(fs.fwh5_valve.inlet, fs.fwh5.cooling.outlet_1)
    if m.dynamic==False or m.init_dyn==True:
        fs.fwh5_valve.Cv.fix()
        fs.fwh5_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.fwh5_valve.Cv.unfix()

    # set some initial inlet values and initialize fwh3
    _set_port(fs.fwh6.cooling.inlet_2, fs.fwh5.desuperheat.outlet_2)
    _set_port(fs.fwh6.desuperheat.inlet_1, fs.turb.hp_split[14].outlet_2)
    if m.dynamic==False or m.init_dyn==True:
        dof1 = degrees_of_freedom(fs)
        fs.fwh6.initialize(outlvl=outlvl, optarg=solver.options)
        dof2 = degrees_of_freedom(fs)
        dof = dof2-dof1
        print('change of dof=', dof)

    # initialize fwh6_valve
    _set_port(fs.fwh6_valve.inlet, fs.fwh6.cooling.outlet_1)
    if m.dynamic==False or m.init_dyn==True:
        fs.fwh6_valve.Cv.fix()
        fs.fwh6_valve.initialize(outlvl=outlvl, optarg=solver.options)
        fs.fwh6_valve.Cv.unfix()

    # Unfix stream connections prior to Solve
    fs.turb.lp_split[5].split_fraction[:, "outlet_2"].unfix()
    fs.turb.lp_split[4].split_fraction[:, "outlet_2"].unfix()
    fs.turb.lp_split[2].split_fraction[:, "outlet_2"].unfix()
    fs.turb.ip_split[6].split_fraction[:, "outlet_2"].unfix()
    fs.turb.hp_split[14].split_fraction[:, "outlet_2"].unfix()
    fs.turb.ip_split[9].split_fraction[:, "outlet_3"].unfix()
    # let steam extraction for DA be calculated based on a constraint to set outlet FW close to saturation
    fs.turb.ip_split[9].split_fraction[:, "outlet_2"].unfix()

    # set the Us for all FWHs to the original values
    fs.fwh1.condense.overall_heat_transfer_coefficient.fix(2842)
    fs.fwh2.condense.overall_heat_transfer_coefficient.fix(3261.7)
    fs.fwh2.cooling.overall_heat_transfer_coefficient.fix(2030.7)
    fs.fwh3.condense.overall_heat_transfer_coefficient.fix(3596.4)
    fs.fwh3.cooling.overall_heat_transfer_coefficient.fix(1860.6)
    fs.fwh5.condense.overall_heat_transfer_coefficient.fix(3040.5)
    fs.fwh5.desuperheat.overall_heat_transfer_coefficient.fix(436.8)
    fs.fwh5.cooling.overall_heat_transfer_coefficient.fix(2019.4)
    fs.fwh6.condense.overall_heat_transfer_coefficient.fix(3215.6)
    fs.fwh6.desuperheat.overall_heat_transfer_coefficient.fix(782.8)
    fs.fwh6.cooling.overall_heat_transfer_coefficient.fix(1900.3)

    # unfix booster pump outlet pressure since it is specified by pump curve
    fs.booster.outlet.pressure.unfix()
    # let's fix rejection flows and rejection valve Cv and openings and let the code calculate the back pressure
    fs.cond_split.split_fraction[:,"outlet_2"].fix()
    fs.cond_split.split_fraction[:,"outlet_3"].fix()
    fs.hotwell_rejection_valve.Cv.fix()
    fs.da_rejection_valve.Cv.fix()
    fs.hotwell_rejection_valve.valve_opening.fix()
    fs.da_rejection_valve.valve_opening.fix()

    # fix makeup stream inlet pressure and enthalpy but unfix the flow
    fs.makeup_valve.inlet.pressure.fix()
    fs.makeup_valve.inlet.enth_mol.fix()
    fs.makeup_valve.inlet.flow_mol.unfix()

    # unfix spray split fraction, fix back pressure, valve Cv and opening, let code calculate spray flow rate
    fs.split_attemp.split_fraction[:,"Spray"].unfix()
    fs.spray_valve.outlet.pressure.fix(1.1e7)
    fs.spray_valve.valve_opening.fix()
    fs.spray_valve.Cv.fix()

    # fix all other valves' Cv's and unfix their openings
    fs.fwh2_valve.Cv.fix()
    fs.fwh3_valve.Cv.fix()
    fs.fwh5_valve.Cv.fix()
    fs.fwh6_valve.Cv.fix()
    fs.cond_valve.Cv.fix()
    fs.makeup_valve.Cv.fix()
    fs.bfp_turb_valve.Cv.fix()
    fs.fwh2_valve.valve_opening.unfix()
    fs.fwh3_valve.valve_opening.unfix()
    fs.fwh5_valve.valve_opening.unfix()
    fs.fwh6_valve.valve_opening.unfix()
    fs.makeup_valve.valve_opening.unfix()
    fs.cond_valve.valve_opening.unfix()
    fs.bfp_turb_valve.valve_opening.unfix()

    if m.dynamic==False:
        dof = degrees_of_freedom(fs)
        _log.info("Degrees of freedom before solving= {}".format(dof))
        assert dof == 0
        # Solve the full model
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(fs, tee=slc.tee)
            _log.info("Solving bounded problem: {}".format(idaeslog.condition(res)))
            _log.info("Steam cycle initialized in {:.1f}s".format(time.time() - start_time))

        _log.info("Adding heat transfer coefficent correations...")
        _add_heat_transfer_correlation(fs)
    else:
        _add_heat_transfer_correlation(fs)

        fs.fwh2.condense.level.unfix()
        fs.fwh3.condense.level.unfix()
        fs.fwh5.condense.level.unfix()
        fs.fwh6.condense.level.unfix()
        fs.hotwell_tank.level.unfix()
        fs.da_tank.level.unfix()

        fs.fwh2_valve.valve_opening.unfix()
        fs.fwh3_valve.valve_opening.unfix()
        fs.fwh5_valve.valve_opening.unfix()
        fs.fwh6_valve.valve_opening.unfix()
        fs.makeup_valve.valve_opening.unfix()
        fs.cond_valve.valve_opening.unfix()
        fs.spray_valve.valve_opening.unfix()

    return m

def _add_u_eq(blk, uex=0.8):
    """Add heat transfer coefficent adjustment for mass flow rate.  This is
    based on knowing the heat transfer coefficent at a particular flow and
    assuming the heat transfer coefficent is porportial to mass to some parameter

    Args:
        blk: Heat exchanger block to add correlation to
        uex: Correlation parameter value (defalut 0.8)

    Returns:
        None
    """
    ti = blk.flowsheet().config.time
    blk.U0 = pyo.Var(ti)
    blk.f0 = pyo.Var(ti)
    blk.uex = pyo.Var(ti, initialize=uex)
    for t in ti:
        blk.U0[t].value = blk.overall_heat_transfer_coefficient[t].value
        blk.f0[t].value = blk.tube.properties_in[t].flow_mol.value
    blk.overall_heat_transfer_coefficient.unfix()
    blk.U0.fix()
    blk.uex.fix()
    blk.f0.fix()
    @blk.Constraint(ti)
    def U_eq(b, t):
        return (
            b.overall_heat_transfer_coefficient[t] ==
            b.U0[t]*(b.tube.properties_in[t].flow_mol/b.f0[t])**b.uex[t]
        )
