import time as wall_clock
import os
import pyomo.environ as pyo
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.generic_models.properties import iapws95
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale
from idaes.core.util import copy_port_values
from idaes.power_generation.properties import FlueGasParameterBlock
from unit_models import PIDController
import flowsheets.boiler_subfs as blr
import flowsheets.steam_cycle_subfs as stc
from idaes.core.util.dyn_utils import copy_values_at_time, copy_non_time_indexed_values
import matplotlib.pyplot as plt
import idaes.core.plugins
import idaes.core.util.model_serializer as ms
import idaes.core.plugins

import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)

def _new_solve(self, model, **kwargs):
    self.options["nlp_scaling_method"] = "user-scaling"
    self.options["linear_solver"] = "ma27"
    self.options["tol"] = 5e-3
    self.options['ma27_pivtol'] = 0.05
    self.options['ma27_pivtolmax'] = 0.9
    if kwargs.get("tee", True):
        print("THIS IPOPT SOLVER HAS BEEN MONKEY PATCHED FOR SCALING")
    lin_elim = kwargs.pop("linear_eliminate", False)
    if lin_elim:
        linear_eliminate = pyo.TransformationFactory("simple_equality_eliminator")
        linear_eliminate.apply_to(model, max_iter=15, reversible=True)
    #iscale.constraint_autoscale_large_jac(model, min_scale=1e-6)
    try:
        res = self._old_solve(model, **kwargs)
        if lin_elim:
            linear_eliminate.revert()
    except:
        if lin_elim:
            linear_eliminate.revert()
        raise
    return res

def monkey_patch_ipopt():
    from pyomo.solvers.plugins.solvers.IPOPT import IPOPT
    IPOPT._old_solve = IPOPT.solve
    IPOPT.solve = _new_solve

def undo_patch_ipopt():
    IPOPT.solve = IPOPT._old_solve

monkey_patch_ipopt()


def set_scaling_factors(m):
    """ Set scaling factors for variables and expressions. These are used for
    variable scaling and used by the framework to scale constraints.

    Args:
        m: plant model to set scaling factors for.

    Returns:
        None
    """
    #First set boiler scaling factors
    fs = m.fs_main.fs_blr

    iscale.set_scaling_factor(fs.flow_mol_ta, 1e-2)

    for i, ww in fs.Waterwalls.items():
        iscale.set_scaling_factor(ww.heat_fireside, 1e-7)
        if i < 4:
            iscale.set_scaling_factor(ww.heat_flux_conv, 1e-4)
        else:
            iscale.set_scaling_factor(ww.heat_flux_conv, 1e-5)
        iscale.set_scaling_factor(ww.diameter_in, 100)
        iscale.set_scaling_factor(ww.control_volume.material_holdup, 1e-4)
        iscale.set_scaling_factor(ww.control_volume.energy_holdup, 1e-8)
        iscale.set_scaling_factor(ww.energy_holdup_slag, 1e-3)
        iscale.set_scaling_factor(ww.energy_holdup_metal, 1e-6)
        iscale.set_scaling_factor(ww.N_Re, 1e-6)
        iscale.set_scaling_factor(ww.pitch, 1e3)
        for j, c in ww.hconv_lo_eqn.items():
            iscale.constraint_scaling_transform(c, 1e-2)

    iscale.set_scaling_factor(fs.aRoof.heat_fireside, 1e-6)
    iscale.set_scaling_factor(fs.aRoof.heat_flux_conv, 1e-4)
    iscale.set_scaling_factor(fs.aRoof.hconv, 1e-3)
    iscale.set_scaling_factor(fs.aRoof.deltaP, 1e-3)
    iscale.set_scaling_factor(fs.aRoof.diameter_in, 100)
    iscale.set_scaling_factor(fs.aRoof.N_Re, 1e-6)

    iscale.set_scaling_factor(fs.aPlaten.heat_fireside, 1e-7)
    iscale.set_scaling_factor(fs.aPlaten.heat_flux_conv, 1e-4)
    iscale.set_scaling_factor(fs.aPlaten.hconv, 1e-3)
    iscale.set_scaling_factor(fs.aPlaten.deltaP, 1e-3)
    iscale.set_scaling_factor(fs.aPlaten.diameter_in, 100)
    iscale.set_scaling_factor(fs.aPlaten.N_Re, 1e-6)

    iscale.set_scaling_factor(fs.aDrum.control_volume.energy_holdup, 1e-10)
    iscale.set_scaling_factor(fs.aDrum.control_volume.material_holdup, 1e-5)
    if m.dynamic:
        for t, c in fs.aDrum.control_volume.energy_accumulation_disc_eq.items():
            iscale.constraint_scaling_transform(c, 1e-4)

    iscale.set_scaling_factor(fs.aDowncomer.control_volume.energy_holdup, 1e-10)
    iscale.set_scaling_factor(fs.aDowncomer.control_volume.material_holdup, 1e-5)

    iscale.set_scaling_factor(fs.aRoof.control_volume.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.aRoof.control_volume.material_holdup, 1e-6)

    iscale.set_scaling_factor(fs.aPlaten.control_volume.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.aPlaten.control_volume.material_holdup, 1e-6)

    iscale.set_scaling_factor(fs.aPipe.control_volume.energy_holdup, 1e-9)
    iscale.set_scaling_factor(fs.aPipe.control_volume.material_holdup, 1e-5)

    iscale.set_scaling_factor(fs.aBoiler.out_heat, 1e-6)
    iscale.set_scaling_factor(fs.aBoiler.heat_total_ww, 1e-7)

    iscale.set_scaling_factor(fs.aECON.shell._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aECON.tube._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aECON.shell.enthalpy_flow_dx, 1e-7)
    iscale.set_scaling_factor(fs.aECON.tube.enthalpy_flow_dx, 1e-7)
    for t, c in fs.aECON.shell.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)
    for t, c in fs.aECON.tube.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)

    iscale.set_scaling_factor(fs.aPSH.shell._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aPSH.tube._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aPSH.shell.enthalpy_flow_dx, 1e-7)
    iscale.set_scaling_factor(fs.aPSH.tube.enthalpy_flow_dx, 1e-7)
    for t, c in fs.aPSH.shell.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)
    for t, c in fs.aPSH.tube.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)

    iscale.set_scaling_factor(fs.aRH1.shell._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aRH1.tube._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aRH1.shell.enthalpy_flow_dx, 1e-7)
    iscale.set_scaling_factor(fs.aRH1.tube.enthalpy_flow_dx, 1e-7)
    for t, c in fs.aRH1.shell.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)
    for t, c in fs.aRH1.tube.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)

    iscale.set_scaling_factor(fs.aRH2.shell._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aRH2.tube._enthalpy_flow, 1e-8)
    iscale.set_scaling_factor(fs.aRH2.shell.enthalpy_flow_dx, 1e-7)
    iscale.set_scaling_factor(fs.aRH2.tube.enthalpy_flow_dx, 1e-7)
    for t, c in fs.aRH2.shell.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)
    for t, c in fs.aRH2.tube.enthalpy_flow_dx_disc_eq.items():
        iscale.constraint_scaling_transform(c, 1e-7)

    # Set steam cycle scale factors
    fs = m.fs_main.fs_stc

    iscale.set_scaling_factor(fs.condenser.side_1.heat, 1e-9)
    iscale.set_scaling_factor(fs.condenser.side_2.heat, 1e-9)

    iscale.set_scaling_factor(fs.aux_condenser.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.aux_condenser.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.hotwell_tank.control_volume.energy_holdup, 1e-9)
    iscale.set_scaling_factor(fs.hotwell_tank.control_volume.material_holdup, 1e-6)

    iscale.set_scaling_factor(fs.fwh1.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh1.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh1.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh1.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh1.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh1.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.fwh2.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh2.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh2.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh2.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh2.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh2.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.fwh3.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh3.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh3.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh3.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh3.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh3.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.da_tank.control_volume.energy_holdup, 1e-10)
    iscale.set_scaling_factor(fs.da_tank.control_volume.material_holdup, 1e-6)

    iscale.set_scaling_factor(fs.fwh5.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh5.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh5.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh5.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh5.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh5.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.fwh6.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh6.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh6.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh6.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh6.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh6.condense.side_2.heat, 1e-7)

    # Calculate calculated scaling factors
    iscale.calculate_scaling_factors(m)


def add_overall_performance_expressions(m):
    @m.fs_main.Expression(m.fs_main.time,
        doc="Heat input to the steam cycle (W)")
    def boiler_heat(b, t):
        return (
            b.fs_stc.turb.inlet_split.mixed_state[t].enth_mol
            * b.fs_stc.turb.inlet_split.mixed_state[t].flow_mol
            - b.fs_stc.fwh6.desuperheat.tube.properties_out[t].enth_mol
            * b.fs_stc.fwh6.desuperheat.tube.properties_out[t].flow_mol
            + b.fs_stc.turb.ip_stages[1].control_volume.properties_in[t].enth_mol
            * b.fs_stc.turb.ip_stages[1].control_volume.properties_in[t].flow_mol
            - b.fs_stc.turb.hp_split[14].outlet_1.enth_mol[t]
            * b.fs_stc.turb.hp_split[14].outlet_1.flow_mol[t]
        )

    @m.fs_main.Expression(m.fs_main.time)
    def aux_power(b, t):
        steam_f = m.fs_main.fs_stc.turb.inlet_split.mixed_state[t].flow_mass*7.937
        air_f = m.fs_main.fs_blr.flowrate_pa_total[t]*7.937
        cw_f = m.fs_main.fs_stc.condenser.tube.properties_in[t].flow_mass/997*15850/1000
        steam_t = (m.fs_main.fs_blr.aPlaten.control_volume.properties_out[t].temperature - 273.15)*9/5 + 32
        steam_p = (m.fs_main.fs_blr.aPlaten.control_volume.properties_out[t].pressure - 79410)/6895
        air_p = m.fs_main.fs_blr.aAPH.side_2_inlet.pressure[t]/1000
        return (
            -7.743210e-2 * cw_f  +
            2.424004e-7 * steam_f**2 +
            1.718702e-6 * steam_f * air_f +
            5.301540e-11 * steam_f**3 +
            2.806236e-10 * steam_f**2 * steam_t +
            7.943112e-10 * steam_f**2 * air_f +
            5.106800e-10 * steam_f * steam_t *air_f +
            6.356823e-10 * steam_f * air_f**2 +
            -2.094381e-7 * steam_p * cw_f**2 +
            5.849115e-10 * steam_t**3 +
            1.049080e-11 * steam_t**2 * air_f +
            1.913389e-9 * steam_t * air_p**2 +
            15.19445
        )*1e6
    # Calculate the heat rate of the plant.  This doesn't account for
    # heat loss in the boiler, so actual plant efficiency would be lower.
    @m.fs_main.Expression(m.fs_main.time)
    def steam_cycle_eff(b, t):
        return -100 * (b.fs_stc.turb.power[t] + b.aux_power[t])/b.boiler_heat[t]

    # Calculate the heat rate of the plant.
    @m.fs_main.Expression(m.fs_main.time,
        doc="Heat rate based on gross power (BTU/MW)")
    def plant_heat_rate(b, t):
        return (
            -b.fs_blr.aBoiler.flowrate_coal_raw[t]
            * (1 - b.fs_blr.aBoiler.mf_H2O_coal_raw[t])
            * b.fs_blr.aBoiler.hhv_coal_dry
            * 3600 / (b.fs_stc.turb.power[t] + b.aux_power[t])
            * 1000 / 1055.05585)

    # Calculate the overall efficiency of the plant.
    @m.fs_main.Expression(m.fs_main.time,
        doc="Overall efficency based on gross power (%)")
    def overall_efficiency(b, t):
        return 3412 / (
            -b.fs_blr.aBoiler.flowrate_coal_raw[t]
            * (1 - b.fs_blr.aBoiler.mf_H2O_coal_raw[t])
            * b.fs_blr.aBoiler.hhv_coal_dry
            * 3600 / (b.fs_stc.turb.power[t] + b.aux_power[t])
            * 1000 / 1055.05585) * 100

    # Calculate the positive power output.
    @m.fs_main.Expression(m.fs_main.time, doc="Gross power (W)")
    def gross_power(b, t):
        return -b.fs_stc.turb.power[t]

    # Calculate the positive power output.
    @m.fs_main.Expression(m.fs_main.time, doc="Net power (W)")
    def net_power(b, t):
        return -b.fs_stc.turb.power[t] - + b.aux_power[t]

    @m.fs_main.Expression(m.fs_main.time)
    def nox_ppm(b, t):
        gross_power = b.gross_power[t]
        Raw_coal_mass_flow_rate = b.fs_blr.aBoiler.flowrate_coal_raw[t]
        ratio_coal = 33.06/Raw_coal_mass_flow_rate
        Moisture_mass_fraction_in_coal = b.fs_blr.aBoiler.mf_H2O_coal_raw[t]
        G005_F = b.fs_blr.aAPH.side_3.properties_out[t].flow_mol
        G005_T = b.fs_blr.aAPH.side_3.properties_out[t].temperature
        G006_F = b.fs_blr.Mixer_PA.mixed_state[t].flow_mol
        G006_T = b.fs_blr.Mixer_PA.mixed_state[t].temperature
        G006_P = b.fs_blr.Mixer_PA.mixed_state[t].pressure
        G009_T = b.fs_blr.aRH2.shell.properties[t,0].temperature
        O2_avg = b.fs_blr.aBoiler.fluegas_o2_pct_dry[t]
        B001_F = b.fs_blr.aECON.tube.properties[t,0].flow_mol
        B001_T = b.fs_blr.aECON.tube.properties[t,0].temperature
        B001_P = b.fs_blr.aECON.tube.properties[t,0].pressure
        G006_G005 = G006_F/G005_F
        B022_P = b.fs_blr.aRoof.control_volume.properties_in[t].pressure
        B022_T = b.fs_blr.aRoof.control_volume.properties_in[t].temperature
        return 20.4387283107588 * gross_power/120053666.9 - 19.537587252287 \
            * (G005_F*G009_T/578187.969804) - 7.03577808341245 \
            * (G005_T*G006_F/63239.8298584)**2 + 4.68722396236018 \
            * (G006_F*G006_T/45921.2044537)**2

    @m.fs_main.Expression(m.fs_main.time)
    def nox_ppm_2burner(b, t):
        G005_T = b.fs_blr.aAPH.side_3.properties_out[t].temperature
        G006_P = b.fs_blr.Mixer_PA.mixed_state[t].pressure
        B001_T = b.fs_blr.aECON.tube.properties[t,0].temperature
        return (
            1.18452388314462 * G005_T +
            0.0017106112236888 * B001_T**2 -
            0.0000214714471385713 * G005_T*G006_P)

    @m.fs_main.Expression(m.fs_main.time)
    def nox_ppm_3burner(b, t):
        gross_power = b.gross_power[t]
        Raw_coal_mass_flow_rate = b.fs_blr.aBoiler.flowrate_coal_raw[t]
        Moisture_mass_fraction_in_coal = b.fs_blr.aBoiler.mf_H2O_coal_raw[t]
        G005_F = b.fs_blr.aAPH.side_3.properties_out[t].flow_mol
        G005_T = b.fs_blr.aAPH.side_3.properties_out[t].temperature
        G006_F = b.fs_blr.Mixer_PA.mixed_state[t].flow_mol
        G006_T = b.fs_blr.Mixer_PA.mixed_state[t].temperature
        G006_P = b.fs_blr.Mixer_PA.mixed_state[t].pressure
        G009_T = b.fs_blr.aRH2.shell.properties[t,0].temperature
        O2_avg = b.fs_blr.aBoiler.fluegas_o2_pct_dry[t]
        B001_F = b.fs_blr.aECON.tube.properties[t,0].flow_mol
        B001_T = b.fs_blr.aECON.tube.properties[t,0].temperature
        B001_P = b.fs_blr.aECON.tube.properties[t,0].pressure
        B022_P = b.fs_blr.aRoof.control_volume.properties_in[t].pressure
        B022_T = b.fs_blr.aRoof.control_volume.properties_in[t].temperature
        return (
            -46.4111344907845 * O2_avg +
            1.25317043722935E-06 * gross_power -
            12.5119173504801 * Raw_coal_mass_flow_rate +
            364.685331694782
        )



def main_steady(load_state=None, save_state=None):
    m = get_model(dynamic=False, load_state=load_state, save_state=save_state)
    return m

def input_profile(t, x0):
    #calculate the user input x as a function of time
    #x0 is the initial value
    #Load cycling of 5% ramp rate
    if t<60:
       x = x0
    elif t<660:
       x = x0*(1-(t-60)/600*0.5)
    elif t<4260:
       x = x0*0.5
    elif t<4860:
       x = x0*(0.5+(t-4260)/600*0.5)
    else: #hold for 1200 sec to 6060 sec
       x = x0
    return x

def main_dyn():
    start_time = wall_clock.time()
    # declare dictionary for the data to plot
    plot_data = {}
    plot_data['time'] = []
    plot_data['coal_flow'] = []
    plot_data['bfpt_opening'] = []
    plot_data['gross_power'] = []
    plot_data['ww_heat'] = []
    plot_data['fegt'] = []
    plot_data['drum_level'] = []
    plot_data['feed_water_flow_sp'] = []
    plot_data['drum_master_ctrl_op'] = []
    plot_data['feed_water_flow'] = []
    plot_data['spray_flow'] = []
    plot_data['main_steam_flow'] = []
    plot_data['rh_steam_flow'] = []
    plot_data['bfpt_flow'] = []
    plot_data['main_steam_temp'] = []
    plot_data['rh_steam_temp'] = []
    plot_data['fw_pres'] = []
    plot_data['drum_pres'] = []
    plot_data['main_steam_pres'] = []
    plot_data['rh_steam_pres'] = []
    plot_data['hw_tank_level'] = []
    plot_data['da_tank_level'] = []
    plot_data['fwh2_level'] = []
    plot_data['fwh3_level'] = []
    plot_data['fwh5_level'] = []
    plot_data['fwh6_level'] = []
    plot_data['makeup_valve_opening'] = []
    plot_data['cond_valve_opening'] = []
    plot_data['fwh2_valve_opening'] = []
    plot_data['fwh3_valve_opening'] = []
    plot_data['fwh5_valve_opening'] = []
    plot_data['fwh6_valve_opening'] = []
    plot_data['spray_valve_opening'] = []
    plot_data['tube_temp_rh2'] = []
    plot_data['t_fg_econ_exit'] = []
    plot_data['t_fg_aph_exit'] = []
    plot_data['throttle_opening'] = []
    plot_data['load_demand'] = []
    plot_data['sliding_pressure'] = []
    plot_data['steam_pressure_sp'] = []
    plot_data['gross_heat_rate'] = []
    plot_data['deaerator_pressure'] = []
    plot_data['SR'] = []
    plot_data['temp_econ_in'] = []
    plot_data['temp_econ_out'] = []
    plot_data['temp_econ_out_sat'] = []
    #plot_data['boiler_efficiency_heat'] = []
    #plot_data['boiler_efficiency_steam'] = []

    # steady-state model
    m_ss = get_model(dynamic=False)
    num_step = [2, 2, 2]
    step_size = [30, 60, 180]
    # 1st dynamic model with smallest time step size
    m_dyn0 = get_model(dynamic=True, time_set=[0, num_step[0]*step_size[0]], nstep=num_step[0])
    # 2nd dynamic model with intermediate time step size
    m_dyn1 = get_model(dynamic=True, time_set=[0, num_step[1]*step_size[1]], nstep=num_step[1])
    # 3rd dynamic model with largest time size
    m_dyn2 = get_model(dynamic=True, time_set=[0, num_step[2]*step_size[2]], nstep=num_step[2])
    # model type list, user input to specify the time duration
    itype_list = []
    for i in range(31): # to 1860 sec
        itype_list.append(0)
    # number of periods
    for i in range(20): # to 4260 sec
        itype_list.append(1)
    for i in range(30): # to 6060 sec
        itype_list.append(0)
    nperiod = len(itype_list)
    tstart = []
    model_list = []
    t = 0
    for i in range(nperiod):
        tstart.append(t)
        t += step_size[itype_list[i]]*num_step[itype_list[i]]
        if itype_list[i]==0:
            model_list.append(m_dyn0)
        elif itype_list[i]==1:
            model_list.append(m_dyn1)
        else:
            model_list.append(m_dyn2)
    # start first model
    m_dyn = model_list[0]
    copy_non_time_indexed_values(m_dyn.fs_main, m_ss.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
    for t in m_dyn.fs_main.config.time:
        copy_values_at_time(m_dyn.fs_main, m_ss.fs_main, t, 0.0, copy_fixed=True, outlvl=idaeslog.ERROR)
    t0 = m_dyn.fs_main.config.time.first()
    # reset bias of controller to current steady-state value, this makes both error and integral error to zero
    m_dyn.fs_main.fs_stc.fwh2_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh2_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.fwh3_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh3_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.fwh5_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh5_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.fwh6_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh6_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.makeup_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.makeup_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.da_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.cond_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.spray_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.spray_valve.valve_opening[t0].value
    opening = m_dyn.fs_main.fs_stc.spray_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error[:].value = pyo.value(m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error_ref[t0])
    opening = m_dyn.fs_main.fs_stc.makeup_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.makeup_ctrl.integral_of_error[:].value = pyo.value(m_dyn.fs_main.fs_stc.makeup_ctrl.integral_of_error_ref[t0])

    m_dyn.fs_main.fs_blr.aDrum.level[t0].value = 0.889
    m_dyn.fs_main.flow_level_ctrl_output[t0].value = m_dyn.fs_main.fs_stc.bfp.outlet.flow_mol[t0].value

    m_dyn.fs_main.drum_slave_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.bfp_turb_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_blr.aBoiler.flowrate_coal_raw[:].value = m_ss.fs_main.fs_blr.aBoiler.flowrate_coal_raw[t0].value

    m_dyn.fs_main.main_steam_pressure[:].value = m_ss.fs_main.fs_blr.aPlaten.outlet.pressure[t0].value/1e6
    m_dyn.fs_main.turbine_master_ctrl.mv_ref.value = m_ss.fs_main.fs_stc.turb.throttle_valve[1].valve_opening[t0].value
    m_dyn.fs_main.turbine_master_ctrl.setpoint[:].value = m_ss.fs_main.fs_stc.power_output[t0].value
    m_dyn.fs_main.boiler_master_ctrl.mv_ref.value = m_ss.fs_main.fs_blr.aBoiler.flowrate_coal_raw[t0].value
    m_dyn.fs_main.boiler_master_ctrl.setpoint[:].value = m_ss.fs_main.fs_blr.aPlaten.outlet.pressure[t0].value/1e6

    print('boiler_master_setpoint=', m_dyn.fs_main.boiler_master_ctrl.setpoint[0].value)
    print('sliding pressure=', pyo.value(m_dyn.fs_main.sliding_pressure[0]))
    solver = pyo.SolverFactory("ipopt")
    solver.options = {
            "tol": 1e-7,
            "linear_solver": "ma27",
            "max_iter": 20,
    }
    # copy non-time-indexed variables to all dynamic models
    if itype_list[0]==0:
        copy_non_time_indexed_values(m_dyn1.fs_main, m_dyn.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
        copy_non_time_indexed_values(m_dyn2.fs_main, m_dyn.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
    elif itype_list[0]==1:
        copy_non_time_indexed_values(m_dyn0.fs_main, m_dyn.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
        copy_non_time_indexed_values(m_dyn2.fs_main, m_dyn.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
    else:
        copy_non_time_indexed_values(m_dyn0.fs_main, m_dyn.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
        copy_non_time_indexed_values(m_dyn1.fs_main, m_dyn.fs_main, copy_fixed=True, outlvl=idaeslog.ERROR)
    dof = degrees_of_freedom(m_dyn)
    print('dof of full model', dof)
    # solving dynamic model at steady-state, this step could be skipped if all setup is good
    print('Solving dynamic model at steady-state...')
    results = solver.solve(m_dyn, tee=True)
    # start 1st simulation
    print("Solving for period number 0 from ", tstart[0], " sec.")
    ss_value = m_ss.fs_main.fs_stc.power_output[0].value
    m_dyn = run_dynamic(m_dyn, ss_value, tstart[0], plot_data, solver)

    # loop for remaining periods
    tlast = m_dyn.fs_main.config.time.last()
    m_prev = m_dyn
    for i in range(1,nperiod):
        m_dyn = model_list[i]
        for t in m_dyn.fs_main.config.time:
            if itype_list[i]!=itype_list[i-1] or t!=tlast:
                copy_values_at_time(m_dyn.fs_main, m_prev.fs_main, t, tlast, copy_fixed=True, outlvl=idaeslog.ERROR)
            print('windup=', m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error[t].value)
            p_only_with_ref = pyo.value(m_dyn.fs_main.fs_stc.spray_ctrl.mv_p_only_with_ref[t])
            if p_only_with_ref<pyo.value(m_dyn.fs_main.fs_stc.spray_ctrl.mv_lb):
                m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error[t].value = 0
                print('reachig lower bound. windup reset to 0')
            if p_only_with_ref>pyo.value(m_dyn.fs_main.fs_stc.spray_ctrl.mv_ub):
                m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error[t].value = 0
                print('reachig upper bound. windup reset to 0')
        print("Solving for period number ", i, "from ", tstart[i], " sec.")
        m_dyn = run_dynamic(m_dyn, ss_value, tstart[i], plot_data, solver)
        tlast = m_dyn.fs_main.config.time.last()
        m_prev = m_dyn
    end_time = wall_clock.time()
    time_used = end_time - start_time
    print("simulation time=", time_used)
    write_data_to_txt_file(plot_data)
    plot_results(plot_data)
    return m_dyn

def get_model(dynamic=True, load_state=None, save_state=None, time_set=None, nstep=None):
    if load_state is not None and not os.path.exists(load_state):
        # Want to load state but file doesn't exist, so warn and reinitialize
        _log.warning(f"Warning cannot load saved state {load_state}")
        load_state = None

    m = pyo.ConcreteModel()
    m.dynamic = dynamic
    m.init_dyn = False
    if time_set is None:
        time_set = [0,20,200]
    if nstep is None:
        nstep = 5
    if m.dynamic:
        m.fs_main = FlowsheetBlock(default={"dynamic": True, "time_set": time_set})
    else:
        m.fs_main = FlowsheetBlock(default={"dynamic": False})
    # Add property packages to flowsheet library
    m.fs_main.prop_water = iapws95.Iapws95ParameterBlock()
    m.fs_main.prop_gas = FlueGasParameterBlock()
    m.fs_main.fs_blr = FlowsheetBlock()
    m.fs_main.fs_stc = FlowsheetBlock()
    m = blr.add_unit_models(m)
    m = stc.add_unit_models(m)
    if m.dynamic:
        # extra variables required by controllers
        # master level control output, desired feed water flow rate at bfp outlet
        m.fs_main.flow_level_ctrl_output = pyo.Var(
            m.fs_main.config.time,
            initialize=10000,
            doc="mole flow rate of feed water demand from drum level master controller"
        )
        # boiler master pv main steam pressure in MPa
        m.fs_main.main_steam_pressure = pyo.Var(
            m.fs_main.config.time,
            initialize=13,
            doc="main steam pressure in MPa for boiler master controller"
        )

        # PID controllers
        # master of cascading level controller
        m.fs_main.drum_master_ctrl = PIDController(default={"pv":m.fs_main.fs_blr.aDrum.level,
                                  "mv":m.fs_main.flow_level_ctrl_output,
                                  "type": 'PI'})
        # slave of cascading level controller
        m.fs_main.drum_slave_ctrl = PIDController(default={"pv":m.fs_main.fs_stc.bfp.outlet.flow_mol,
                                  "mv":m.fs_main.fs_stc.bfp_turb_valve.valve_opening,
                                  "type": 'PI',
                                  "bounded_output": False})
        # turbine master PID controller to control power output in MW by manipulating throttling valve
        m.fs_main.turbine_master_ctrl = PIDController(default={"pv":m.fs_main.fs_stc.power_output,
                                  "mv":m.fs_main.fs_stc.turb.throttle_valve[1].valve_opening,
                                  "type": 'PI'})
        # boiler master PID controller to control main steam pressure in MPa by manipulating coal feed rate
        m.fs_main.boiler_master_ctrl = PIDController(default={"pv":m.fs_main.main_steam_pressure,
                                  "mv":m.fs_main.fs_blr.aBoiler.flowrate_coal_raw,
                                  "type": 'PI'})

        m.discretizer = pyo.TransformationFactory('dae.finite_difference')
        m.discretizer.apply_to(m,
                           nfe=nstep,
                           wrt=m.fs_main.config.time,
                           scheme="BACKWARD")

        # desired sliding pressure in MPa as a function of power demand in MW
        @m.fs_main.Expression(m.fs_main.config.time, doc="Sliding pressure as a function of power output")
        def sliding_pressure(b, t):
            return 12.511952+(12.511952-9.396)/(249.234-96.8)*(b.turbine_master_ctrl.setpoint[t]-249.234)

        # main steam pressure in MPa
        @m.fs_main.Constraint(m.fs_main.config.time, doc="main steam pressure in MPa")
        def main_steam_pressure_eqn(b, t):
            return b.main_steam_pressure[t] == 1e-6*b.fs_blr.aPlaten.outlet.pressure[t]

        # Constraint for setpoint of the slave controller of the three-element drum level controller
        @m.fs_main.Constraint(m.fs_main.config.time, doc="Set point of drum level slave control")
        def drum_level_control_setpoint_eqn(b, t):
            return b.drum_slave_ctrl.setpoint[t] == b.flow_level_ctrl_output[t] + \
                   b.fs_blr.aPlaten.outlet.flow_mol[t] + \
                   b.fs_blr.blowdown_split.FW_Blowdown.flow_mol[t]  #revised to add steam flow only

        # Constraint for setpoint of boiler master
        @m.fs_main.Constraint(m.fs_main.config.time, doc="Set point of boiler master")
        def boiler_master_setpoint_eqn(b, t):
            return b.boiler_master_ctrl.setpoint[t] == 0.02*(b.turbine_master_ctrl.setpoint[t] - b.fs_stc.power_output[t]) + b.sliding_pressure[t]

        # Constraint for setpoint of boiler master
        @m.fs_main.Constraint(m.fs_main.config.time, doc="dry O2 in flue gas in dynamic mode")
        def dry_o2_in_flue_gas_dyn_eqn(b, t):
            return b.fs_blr.aBoiler.fluegas_o2_pct_dry[t] == 0.05*(b.fs_stc.spray_ctrl.setpoint[t] - b.fs_stc.temperature_main_steam[t]) - \
            0.0007652*b.fs_blr.aBoiler.flowrate_coal_raw[t]**3 + \
            0.06744*b.fs_blr.aBoiler.flowrate_coal_raw[t]**2 - 1.9815*b.fs_blr.aBoiler.flowrate_coal_raw[t] + 22.275

        # controller inputs
        # for master level controller
        m.fs_main.drum_master_ctrl.gain_p.fix(40000) #increased from 5000
        m.fs_main.drum_master_ctrl.gain_i.fix(100)
        m.fs_main.drum_master_ctrl.setpoint.fix(0.889)
        m.fs_main.drum_master_ctrl.mv_ref.fix(0)	#revised to 0
        # for slave level controller, note the setpoint is defined by the constraint
        m.fs_main.drum_slave_ctrl.gain_p.fix(2e-2)  # increased from 1e-2
        m.fs_main.drum_slave_ctrl.gain_i.fix(2e-4)  # increased from 1e-4
        m.fs_main.drum_slave_ctrl.mv_ref.fix(0.5)
        # for turbine master controller, note the setpoint is the power demand
        m.fs_main.turbine_master_ctrl.gain_p.fix(5e-4) #changed from 2e-3
        m.fs_main.turbine_master_ctrl.gain_i.fix(5e-4) #changed from 2e-3
        m.fs_main.turbine_master_ctrl.mv_ref.fix(0.6)
        # for boiler master controller, note the setpoint is specified by the constraint
        m.fs_main.boiler_master_ctrl.gain_p.fix(10)
        m.fs_main.boiler_master_ctrl.gain_i.fix(0.25)
        m.fs_main.boiler_master_ctrl.mv_ref.fix(29.0)

        t0 = m.fs_main.config.time.first()
        m.fs_main.drum_master_ctrl.integral_of_error[t0].fix(0)
        m.fs_main.drum_slave_ctrl.integral_of_error[t0].fix(0)
        m.fs_main.turbine_master_ctrl.integral_of_error[t0].fix(0)
        m.fs_main.boiler_master_ctrl.integral_of_error[t0].fix(0)

    blr.set_arcs_and_constraints(m)
    blr.set_inputs(m)
    stc.set_arcs_and_constraints(m)
    stc.set_inputs(m)
    # Now that the mole is discreteized set and calculate scaling factors

    set_scaling_factors(m)
    add_overall_performance_expressions(m)

    # Add performance measures
    if load_state is None:
        blr.initialize(m)
        stc.initialize(m)

    optarg={"tol":5e-7,"linear_solver":"ma27","max_iter":50}
    solver = pyo.SolverFactory("ipopt")
    solver.options = optarg

    _log.info("Bring models closer together...")
    m.fs_main.fs_blr.flow_mol_steam_rh_eqn.deactivate()
    # Hook the boiler to the steam cycle.
    m.fs_main.S001 = Arc(
        source=m.fs_main.fs_blr.aPlaten.outlet, destination=m.fs_main.fs_stc.turb.inlet_split.inlet
    )
    m.fs_main.S005 = Arc(
        source=m.fs_main.fs_stc.turb.hp_split[14].outlet_1, destination=m.fs_main.fs_blr.aRH1.tube_inlet
    )
    m.fs_main.S009 = Arc(
        source=m.fs_main.fs_blr.aRH2.tube_outlet, destination=m.fs_main.fs_stc.turb.ip_stages[1].inlet
    )
    m.fs_main.S042 = Arc(
        source=m.fs_main.fs_stc.fwh6.desuperheat.outlet_2, destination=m.fs_main.fs_blr.aECON.tube_inlet
    )
    m.fs_main.B006 = Arc(
        source=m.fs_main.fs_stc.spray_valve.outlet, destination=m.fs_main.fs_blr.Attemp.Water_inlet
    )
    pyo.TransformationFactory('network.expand_arcs').apply_to(m.fs_main)
    # unfix all connected streams
    m.fs_main.fs_stc.turb.inlet_split.inlet.unfix()
    m.fs_main.fs_stc.turb.hp_split[14].outlet_1.unfix()
    m.fs_main.fs_blr.aRH1.tube_inlet.unfix()
    m.fs_main.fs_stc.turb.ip_stages[1].inlet.unfix()
    m.fs_main.fs_blr.aECON.tube_inlet.unfix()
    m.fs_main.fs_blr.Attemp.Water_inlet.unfix()
    m.fs_main.fs_stc.spray_valve.outlet.unfix() #outlet pressure fixed on steam cycle sub-flowsheet
    # deactivate constraints on steam cycle flowsheet
    m.fs_main.fs_stc.fw_flow_constraint.deactivate()
    m.fs_main.fs_stc.turb.constraint_reheat_flow.deactivate()
    m.fs_main.fs_blr.aBoiler.flowrate_coal_raw.unfix() # steam circulation and coal flow are linked

    if m.dynamic==False:
        if load_state is None:
            m.fs_main.fs_stc.spray_valve.valve_opening.unfix()
            m.fs_main.fs_stc.temperature_main_steam.fix(810)
            _log.info("Solve connected models...")
            print("Degrees of freedom = {}".format(degrees_of_freedom(m)))
            assert degrees_of_freedom(m) == 0
            res = solver.solve(m, tee=True)
            _log.info("Solved: {}".format(idaeslog.condition(res)))
            # increase load to around 250 MW
            _log.info("Increase coal feed rate to 32.5...")
            m.fs_main.fs_stc.bfp.outlet.pressure.unfix()
            m.fs_main.fs_stc.turb.throttle_valve[1].valve_opening.fix(0.9)
            m.fs_main.fs_blr.aBoiler.flowrate_coal_raw.fix(32.5)
            res = solver.solve(m, tee=True)

        if not load_state is None:
            # the fwh heat transfer coefficient correlations are added in the
            # initialization, so if we are skipping the init, we have to add
            # them here.
            stc._add_heat_transfer_correlation(m.fs_main.fs_stc)
            ms.from_json(m, fname=load_state)

        if save_state is not None:
            ms.to_json(m, fname=save_state)

    else:
        m.fs_main.fs_blr.dry_o2_in_flue_gas_eqn.deactivate()
        t0 = m.fs_main.config.time.first()
        m.fs_main.fs_stc.fwh2.condense.level[t0].fix()
        m.fs_main.fs_stc.fwh3.condense.level[t0].fix()
        m.fs_main.fs_stc.fwh5.condense.level[t0].fix()
        m.fs_main.fs_stc.fwh6.condense.level[t0].fix()
        m.fs_main.fs_stc.hotwell_tank.level[t0].fix()
        m.fs_main.fs_stc.da_tank.level[t0].fix()
        m.fs_main.fs_stc.temperature_main_steam[t0].unfix()
        m.fs_main.fs_stc.spray_valve.valve_opening[t0].fix()
        m.fs_main.fs_blr.aDrum.level.unfix()
        m.fs_main.fs_blr.aDrum.level[t0].fix()
        m.fs_main.flow_level_ctrl_output.unfix()
        m.fs_main.flow_level_ctrl_output[t0].fix()
        m.fs_main.fs_stc.bfp.outlet.pressure.unfix()
        m.fs_main.fs_blr.aBoiler.flowrate_coal_raw.unfix()
        m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[t0].fix()
        m.fs_main.fs_stc.turb.throttle_valve[1].valve_opening.unfix()
        m.fs_main.fs_stc.turb.throttle_valve[1].valve_opening[t0].fix()
        m.fs_main.turbine_master_ctrl.setpoint.fix()

    return m


def run_dynamic(m, x0, t0, pd, solver):
    # set input profile
    i = 0
    for t in m.fs_main.config.time:
        power_demand = input_profile(t0+t, x0)
        m.fs_main.turbine_master_ctrl.setpoint[t].value = power_demand
        print ('Point i, t, power demand=', i, t+t0, power_demand)
        i += 1
    df = degrees_of_freedom(m)
    print ("***************degree of freedom = ", df, "********************")
    '''
    solver = pyo.SolverFactory("ipopt")
    solver.options = {
            "tol": 1e-7,
            "linear_solver": "ma27",
            "max_iter": 60,
    }
    '''
    # Initialize by time element
    #initialize_by_time_element(m.fs_main, m.fs_main.config.time, solver=solver, outlvl=idaeslog.DEBUG)
    #print('dof after integrating:', degrees_of_freedom(m.fs_main))
    results = solver.solve(m, tee=True)

    # Print results
    print(results)
    print()


    # append results
    for t in m.fs_main.config.time:
        if t==0 and len(pd['time'])>0:
            continue
        pd['time'].append(t+t0)
        pd['coal_flow'].append(m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[t].value)
        pd['bfpt_opening'].append(m.fs_main.fs_stc.bfp_turb_valve.valve_opening[t].value)
        pd['gross_power'].append(pyo.value(m.fs_main.fs_stc.power_output[t]))
        pd['ww_heat'].append(m.fs_main.fs_blr.aBoiler.heat_total_ww[t].value/1e6)
        pd['fegt'].append(m.fs_main.fs_blr.aBoiler.side_1_outlet.temperature[t].value)
        pd['drum_level'].append(m.fs_main.fs_blr.aDrum.level[t].value)
        pd['feed_water_flow_sp'].append(m.fs_main.drum_slave_ctrl.setpoint[t].value/1000)
        pd['drum_master_ctrl_op'].append(m.fs_main.flow_level_ctrl_output[t].value/1000)
        pd['feed_water_flow'].append(m.fs_main.fs_blr.aECON.tube_inlet.flow_mol[t].value/1000)
        pd['main_steam_flow'].append(m.fs_main.fs_blr.aPlaten.outlet.flow_mol[t].value/1000)
        pd['rh_steam_flow'].append(m.fs_main.fs_blr.aRH2.tube_outlet.flow_mol[t].value/1000)
        pd['bfpt_flow'].append(m.fs_main.fs_stc.bfp_turb_valve.outlet.flow_mol[t].value)
        pd['spray_flow'].append(m.fs_main.fs_stc.spray_valve.outlet.flow_mol[t].value)
        pd['main_steam_temp'].append(m.fs_main.fs_stc.temperature_main_steam[t].value)
        pd['rh_steam_temp'].append(pyo.value(m.fs_main.fs_blr.aRH2.tube.properties[t,0].temperature))
        pd['fw_pres'].append(m.fs_main.fs_stc.bfp.outlet.pressure[t].value/1e6)
        pd['drum_pres'].append(m.fs_main.fs_blr.aDrum.inlet.pressure[t].value/1e6)
        pd['main_steam_pres'].append(m.fs_main.main_steam_pressure[t].value)
        pd['rh_steam_pres'].append(m.fs_main.fs_blr.aRH2.tube_outlet.pressure[t].value/1e6)
        pd['hw_tank_level'].append(m.fs_main.fs_stc.hotwell_tank.level[t].value)
        pd['da_tank_level'].append(m.fs_main.fs_stc.da_tank.level[t].value)
        pd['fwh2_level'].append(m.fs_main.fs_stc.fwh2.condense.level[t].value)
        pd['fwh3_level'].append(m.fs_main.fs_stc.fwh3.condense.level[t].value)
        pd['fwh5_level'].append(m.fs_main.fs_stc.fwh5.condense.level[t].value)
        pd['fwh6_level'].append(m.fs_main.fs_stc.fwh6.condense.level[t].value)
        pd['makeup_valve_opening'].append(m.fs_main.fs_stc.makeup_valve.valve_opening[t].value)
        pd['cond_valve_opening'].append(m.fs_main.fs_stc.cond_valve.valve_opening[t].value)
        pd['fwh2_valve_opening'].append(m.fs_main.fs_stc.fwh2_valve.valve_opening[t].value)
        pd['fwh3_valve_opening'].append(m.fs_main.fs_stc.fwh3_valve.valve_opening[t].value)
        pd['fwh5_valve_opening'].append(m.fs_main.fs_stc.fwh5_valve.valve_opening[t].value)
        pd['fwh6_valve_opening'].append(m.fs_main.fs_stc.fwh6_valve.valve_opening[t].value)
        pd['spray_valve_opening'].append(m.fs_main.fs_stc.spray_valve.valve_opening[t].value)
        pd['tube_temp_rh2'].append(m.fs_main.fs_blr.aRH2.temp_wall_shell[t,0].value)
        pd['t_fg_econ_exit'].append(m.fs_main.fs_blr.aECON.shell_outlet.temperature[t].value)
        pd['t_fg_aph_exit'].append(m.fs_main.fs_blr.aAPH.side_1_outlet.temperature[t].value)
        pd['throttle_opening'].append(m.fs_main.fs_stc.turb.throttle_valve[1].valve_opening[t].value)
        pd['load_demand'].append(m.fs_main.turbine_master_ctrl.setpoint[t].value)
        pd['sliding_pressure'].append(pyo.value(m.fs_main.sliding_pressure[t]))
        pd['steam_pressure_sp'].append(m.fs_main.boiler_master_ctrl.setpoint[t].value)
        pd['gross_heat_rate'].append(pyo.value(m.fs_main.plant_heat_rate[t]))
        pd['deaerator_pressure'].append(m.fs_main.fs_stc.da_tank.inlet.pressure[t].value/1e6)
        pd['SR'].append(m.fs_main.fs_blr.aBoiler.SR[t].value)
        pd['temp_econ_in'].append(pyo.value(m.fs_main.fs_blr.aECON.tube.properties[t,1].temperature))
        pd['temp_econ_out'].append(pyo.value(m.fs_main.fs_blr.aECON.tube.properties[t,0].temperature))
        pd['temp_econ_out_sat'].append(pyo.value(m.fs_main.fs_blr.aECON.tube.properties[t,0].temperature_sat))
        #pd['boiler_efficiency_heat'].append(pyo.value(m.fs_main.fs_blr.boiler_efficiency_heat[t]))
        #pd['boiler_efficiency_steam'].append(pyo.value(m.fs_main.fs_blr.boiler_efficiency_steam[t]))

    return m

def plot_results(pd):
    # ploting responses
    plt.figure(1)
    plt.plot(pd['time'], pd['coal_flow'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Coal Flow Rate [kg/s]")
    plt.show(block=False)

    plt.figure(2)
    plt.plot(pd['time'], pd['bfpt_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("BFPT Valve Opening")
    plt.show(block=False)

    plt.figure(3)
    plt.plot(pd['time'], pd['gross_power'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Gross Power Output [MW]")
    plt.show(block=False)

    plt.figure(4)
    plt.plot(pd['time'], pd['ww_heat'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Waterwall Heat [MW]")
    plt.show(block=False)

    plt.figure(5)
    plt.plot(pd['time'], pd['fegt'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FEGT [K]")
    plt.show(block=False)

    plt.figure(6)
    plt.plot(pd['time'], pd['drum_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Drum Level [m]")
    plt.show(block=False)

    plt.figure(7)
    plt.plot(pd['time'], pd['feed_water_flow'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Feed Water Flow [kmol/s]")
    plt.show(block=False)

    plt.figure(8)
    plt.plot(pd['time'], pd['main_steam_flow'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Main Steam Flow [kmol/s]")
    plt.show(block=False)

    plt.figure(9)
    plt.plot(pd['time'], pd['rh_steam_flow'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("RH Steam Flow [kmol/s]")
    plt.show(block=False)

    plt.figure(10)
    plt.plot(pd['time'], pd['bfpt_flow'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("BFPT Flow [mol/s]")
    plt.show(block=False)

    plt.figure(11)
    plt.plot(pd['time'], pd['spray_flow'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Water Spray Flow [mol/s]")
    plt.show(block=False)

    plt.figure(12)
    plt.plot(pd['time'], pd['main_steam_temp'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Main Steam temperature [K]")
    plt.show(block=False)

    plt.figure(13)
    plt.plot(pd['time'], pd['rh_steam_temp'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("RH Steam temperature [K]")
    plt.show(block=False)

    plt.figure(14)
    plt.plot(pd['time'], pd['fw_pres'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Feed Water Pressure [MPa]")
    plt.show(block=False)

    plt.figure(15)
    plt.plot(pd['time'], pd['drum_pres'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Drum Pressure [MPa]")
    plt.show(block=False)

    plt.figure(16)
    plt.plot(pd['time'], pd['main_steam_pres'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Main Steam Pressure [MPa]")
    plt.show(block=False)

    plt.figure(17)
    plt.plot(pd['time'], pd['rh_steam_pres'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("RH Steam Pressure [MPa]")
    plt.show(block=False)

    plt.figure(18)
    plt.plot(pd['time'], pd['hw_tank_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Hotwell Tank Level [m]")
    plt.show(block=False)

    plt.figure(19)
    plt.plot(pd['time'], pd['da_tank_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("DA Tank Level [m]")
    plt.show(block=False)

    plt.figure(20)
    plt.plot(pd['time'], pd['fwh2_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH2 Level [m]")
    plt.show(block=False)

    plt.figure(21)
    plt.plot(pd['time'], pd['fwh3_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH3 Level [m]")
    plt.show(block=False)

    plt.figure(22)
    plt.plot(pd['time'], pd['fwh5_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH5 Level [m]")
    plt.show(block=False)

    plt.figure(23)
    plt.plot(pd['time'], pd['fwh6_level'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH6 Level [m]")
    plt.show(block=False)

    plt.figure(24)
    plt.plot(pd['time'], pd['makeup_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Makeup Valve Opening")
    plt.show(block=False)

    plt.figure(25)
    plt.plot(pd['time'], pd['cond_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Condensate Valve Opening")
    plt.show(block=False)

    plt.figure(26)
    plt.plot(pd['time'], pd['fwh2_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH2 Valve Opening")
    plt.show(block=False)

    plt.figure(27)
    plt.plot(pd['time'], pd['fwh3_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH3 Valve Opening")
    plt.show(block=False)

    plt.figure(28)
    plt.plot(pd['time'], pd['fwh5_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH5 Valve Opening")
    plt.show(block=False)

    plt.figure(29)
    plt.plot(pd['time'], pd['fwh6_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("FWH6 Valve Opening")
    plt.show(block=False)

    plt.figure(30)
    plt.plot(pd['time'], pd['spray_valve_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Spray Valve Opening")
    plt.show(block=False)

    plt.figure(31)
    plt.plot(pd['time'], pd['tube_temp_rh2'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("RH Tube temperature [K]")
    plt.show(block=False)

    plt.figure(32)
    plt.plot(pd['time'], pd['t_fg_econ_exit'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Flue Gas T at Econ Exit [K]")
    plt.show(block=False)

    plt.figure(33)
    plt.plot(pd['time'], pd['t_fg_aph_exit'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Flue Gas T at APH Exit [K]")
    plt.show(block=False)

    plt.figure(34)
    plt.plot(pd['time'], pd['throttle_opening'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Throttle Valve Opening")
    plt.show(block=False)

    plt.figure(35)
    plt.plot(pd['time'], pd['load_demand'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Load Demand [MW]")
    plt.show(block=False)

    plt.figure(36)
    plt.plot(pd['time'], pd['sliding_pressure'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Desired Sliding Pressure [MPa]")
    plt.show(block=False)

    plt.figure(37)
    plt.plot(pd['time'], pd['gross_heat_rate'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Gross Heat Rate [BTU/kW-hr]")
    plt.show(block=False)

    plt.figure(38)
    plt.plot(pd['time'], pd['deaerator_pressure'])
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("DA pressure [MPa]")
    plt.show(block=True)

def write_data_to_txt_file(plot_data):
    # write data to file
    ntime = len(plot_data['time'])
    ncount = len(plot_data)
    icount = 0
    with open("dyn_results.txt","w") as fout:
        # write headings
        for k in plot_data:
            icount += 1
            fout.write(k)
            if icount<ncount:
                fout.write("\t")
            else:
                fout.write("\n")
        # write values
        for i in range(ntime):
            icount = 0
            for k in plot_data:
                icount += 1
                fout.write(str(plot_data[k][i]))
                if icount<ncount:
                    fout.write("\t")
                else:
                    fout.write("\n")

if __name__ == "__main__":
    m_dyn = main_dyn()
    #m_ss = main_steady()
