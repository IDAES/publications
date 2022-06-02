import csv
import pandas as pd
import numpy as np
from plant_dyn import main_steady
import report_steady as rpt
import data_util as data_module
import pyomo.environ as pyo
from idaes.generic_models.properties import iapws95
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
import idaes.core.util.model_serializer as ms
from idaes.core.util.model_statistics import large_residuals_set

meth = "normal" #normal, slide, or fixed

_log = idaeslog.getLogger(__name__)

if __name__ == "__main__":
    solver = pyo.SolverFactory("ipopt")
    m = main_steady(load_state="initial_steady.json.gz")
    strip_bounds = pyo.TransformationFactory("contrib.strip_var_bounds")
    strip_bounds.apply_to(m, reversible=False)
    # Start from the highest output case, this should already have the
    # estimated parameters loaded in from the validation run.
    ms.from_json(m, fname=f"results/state4_26.json.gz")

    _log.info("Make sure initial model setup is okay.")
    solver.solve(m, tee=True)

    # Add constraints for caol flow dependent operating variables, PA, O2, fheat_ww
    pa = m.fs_main.fs_blr.flow_pa_kg_per_s
    o2pct = m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry
    fww = m.fs_main.fs_blr.aBoiler.fheat_ww
    coal_flow = m.fs_main.fs_blr.aBoiler.flowrate_coal_raw

    @m.fs_main.Constraint(m.fs_main.time)
    def pa_eqn(b, t):
        return pa[t] == 1.6569*coal_flow[t] + 16.581
    pa.unfix()

    @m.fs_main.Constraint(m.fs_main.time)
    def o2pct_3burner_eqn(b, t):
        return o2pct[t] == 3.0
    @m.fs_main.Constraint(m.fs_main.time)
    def o2pct_2burner_eqn(b, t):
        return o2pct[t] == -0.5117*coal_flow[t] + 12.193
    m.fs_main.o2pct_2burner_eqn.deactivate()
    o2pct.unfix()
    #o2pct.setlb(3)
    #o2pct.setub(8)

    @m.fs_main.Constraint(m.fs_main.time)
    def fww_2burner_constraint(b, t):
        return fww[t] == (0.0101*coal_flow[t] + 0.859) * 1.0
        #return fww[t] == 1.0
    @m.fs_main.Constraint(m.fs_main.time)
    def fww_3burner_constraint(b, t):
        return fww[t] == (-0.0074*coal_flow[t] + 1.178) * 1.0
        #return fww[t] == 1.0
    m.fs_main.fww_2burner_constraint.deactivate()
    fww.unfix()
    #fww.setlb(0.94)
    #fww.setub(1.06)
    #m.main_steam_temp_constraint = pyo.Constraint(expr=m.fs_main.fs_stc.turb.inlet_split.mixed_state[0].temperature <= 815)
    #m.reheat_temp_constraint = pyo.Constraint(expr=m.fs_main.fs_stc.turb.ip_stages[1].control_volume.properties_in[0].temperature <= 815)

    @m.fs_main.Objective()
    def obj(b):
        return (
            (m.fs_main.fs_stc.turb.inlet_split.mixed_state[0].temperature - 815)**2 +
            (m.fs_main.fs_stc.turb.ip_stages[1].control_volume.properties_in[0].temperature - 815)**2
        )
    main_pressure = m.fs_main.fs_stc.turb.inlet_split.mixed_state[0].pressure
    throttle_pressure = m.fs_main.fs_stc.turb.throttle_valve[1].control_volume.properties_out[0].pressure


    _log.info("Initial solve with operating constraints.")
    solver.solve(m, tee=True)
    solver.options["max_iter"] = 50
    gps = np.linspace(1e7, 4.0e6, 50)

    col = [
        "status",
        "gross_power",
        "eff",
        "main_steam_pressure",
        "main_steam_pressure (MPa)",
        "main_steam_temperature",
        "fheat_ww",
        "o2pct",
        "fheat_platen",
        "gross_power",
        "gross_power (MW)",
        "turbine_inlet_pressure",
        "turbine_inlet_pressure (MPa)",
        "reheat_steam_temperature",
        "coal_flow",
        "net_power",
        "net_power (MW)",
        "boost_suction_T",
        "boost_suction_Tsat",
        "boost_suction_P",
        "boost_suction_Psat",
        "boost_suction_Fmass",
        "bfp_suction_T",
        "bfp_suction_Tsat",
        "bfp_suction_P",
        "bfp_suction_Psat",
        "bfp_suction_Fmass",
        "econ_outlet_T",
        "econ_outlet_Tsat",
        "last_stage_in_x",
        "last_stage_out_x",
        "nox_ppm_2b",
        "nox_ppm_3b",
        "fegt",
    ]
    # Data frame to tabulate results
    df_val = pd.DataFrame(columns=col, index=gps)

    strip_bounds = pyo.TransformationFactory("contrib.strip_var_bounds")
    strip_bounds.apply_to(m, reversible=False)
    m.fs_main.fs_stc.turb.throttle_valve[1].pressure_flow_equation.deactivate()
    for p in gps:
        if p < 6.2e6: # Should be about where we switch into 2 burner mode
            m.fs_main.o2pct_2burner_eqn.activate()
            m.fs_main.o2pct_3burner_eqn.deactivate()
            m.fs_main.fww_2burner_constraint.activate()
            m.fs_main.fww_3burner_constraint.deactivate()

        throttle_pressure.fix(p)
        if meth == "normal":
            main_pressure.fix(0.4541*p + 7.44e6)
            #m.fs_main.fs_blr.aBoiler.fheat_ww.fix(0.98)
        elif meth == "fixed":
            main_pressure.fix(1e7*1.27)
            #m.fs_main.fww_2burner_eqn.deactivate()
            #m.fs_main.fww_3burner_eqn.deactivate()
            #fww.unfix()
            #m.fs_main.fs_stc.temperature_main_steam[0].fix()
        elif meth == "slide":
            main_pressure.fix(p*1.1)
        else:
            raise Exception("Say what?!")

        _log.info(f"Throttle pressure {p/1e6} MPa")
        _log.info(f"Main pressure {pyo.value(main_pressure/1e6)} MPa")
        try:
            res = solver.solve(m, tee=True)
            res = idaeslog.condition(res)
        except:
            res = "fail"


        df_val.loc[p]["status"] = res
        df_val.loc[p]["eff"] = pyo.value(m.fs_main.overall_efficiency[0])
        df_val.loc[p]["main_steam_pressure"] = pyo.value(main_pressure)
        df_val.loc[p]["main_steam_pressure (MPa)"] = pyo.value(main_pressure/1e6)
        df_val.loc[p]["main_steam_temperature"] = pyo.value(m.fs_main.fs_stc.turb.inlet_split.mixed_state[0].temperature)
        df_val.loc[p]["fheat_ww"] = pyo.value(m.fs_main.fs_blr.aBoiler.fheat_ww[0])
        df_val.loc[p]["o2pct"] = pyo.value(o2pct[0])
        df_val.loc[p]["fheat_platen"] = pyo.value(m.fs_main.fs_blr.aBoiler.fheat_platen[0])
        df_val.loc[p]["gross_power"] = pyo.value(m.fs_main.gross_power[0])
        df_val.loc[p]["gross_power (MW)"] = pyo.value(m.fs_main.gross_power[0]/1e6)
        df_val.loc[p]["net_power"] = pyo.value(m.fs_main.net_power[0])
        df_val.loc[p]["net_power (MW)"] = pyo.value(m.fs_main.net_power[0]/1e6)
        df_val.loc[p]["turbine_inlet_pressure"] = pyo.value(throttle_pressure)
        df_val.loc[p]["turbine_inlet_pressure (MPa)"] = pyo.value(throttle_pressure/1e6)
        df_val.loc[p]["reheat_steam_temperature"] = pyo.value(m.fs_main.fs_stc.turb.ip_stages[1].control_volume.properties_in[0].temperature)
        df_val.loc[p]["coal_flow"] = pyo.value(m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[0])
        df_val.loc[p]["boost_suction_T"] = pyo.value(m.fs_main.fs_stc.booster.control_volume.properties_in[0].temperature)
        df_val.loc[p]["boost_suction_Tsat"] = pyo.value(m.fs_main.fs_stc.booster.control_volume.properties_in[0].temperature_sat)
        df_val.loc[p]["boost_suction_P"] = pyo.value(m.fs_main.fs_stc.booster.control_volume.properties_in[0].pressure)
        df_val.loc[p]["boost_suction_Psat"] = pyo.value(m.fs_main.fs_stc.booster.control_volume.properties_in[0].pressure_sat)
        df_val.loc[p]["boost_suction_Fmass"] = pyo.value(m.fs_main.fs_stc.booster.control_volume.properties_in[0].flow_mass)
        df_val.loc[p]["bfp_suction_T"] = pyo.value(m.fs_main.fs_stc.bfp.control_volume.properties_in[0].temperature)
        df_val.loc[p]["bfp_suction_Tsat"] = pyo.value(m.fs_main.fs_stc.bfp.control_volume.properties_in[0].temperature_sat)
        df_val.loc[p]["bfp_suction_P"] = pyo.value(m.fs_main.fs_stc.bfp.control_volume.properties_in[0].pressure)
        df_val.loc[p]["bfp_suction_Psat"] = pyo.value(m.fs_main.fs_stc.bfp.control_volume.properties_in[0].pressure_sat)
        df_val.loc[p]["bfp_suction_Fmass"] = pyo.value(m.fs_main.fs_stc.bfp.control_volume.properties_in[0].flow_mass)
        df_val.loc[p]["econ_outlet_T"] = pyo.value(m.fs_main.fs_blr.aECON.tube.properties[0,1].temperature)
        df_val.loc[p]["econ_outlet_Tsat"] = pyo.value(m.fs_main.fs_blr.aECON.tube.properties[0,1].temperature_sat)
        df_val.loc[p]["last_stage_in_x"] = pyo.value(m.fs_main.fs_stc.turb.outlet_stage.control_volume.properties_in[0].vapor_frac)
        df_val.loc[p]["last_stage_out_x"] = pyo.value(m.fs_main.fs_stc.turb.outlet_stage.control_volume.properties_out[0].vapor_frac)
        df_val.loc[p]["nox_ppm_2b"] = pyo.value(m.fs_main.nox_ppm_2burner[0])
        df_val.loc[p]["nox_ppm_3b"] = pyo.value(m.fs_main.nox_ppm_3burner[0])
        df_val.loc[p]["fegt"] = pyo.value(m.fs_main.fs_blr.aBoiler.side_1_outlet.temperature[0])
        df_val.to_csv(f"results/slide_{meth}x.csv")
