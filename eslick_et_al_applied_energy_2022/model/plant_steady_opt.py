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


_log = idaeslog.getLogger(__name__)

if __name__ == "__main__":
    solver = pyo.SolverFactory("ipopt")
    m = main_steady(load_state="initial_steady.json.gz")
    # Start from the highest output case, this should already have the
    # estimated parameters loaded in from the validation run.
    ms.from_json(m, fname=f"results/state4_26.json.gz")
    strip_bounds = pyo.TransformationFactory("contrib.strip_var_bounds")
    strip_bounds.apply_to(m, reversible=False)
    _log.info("Make sure initial model setup is okay.")
    solver.solve(m, tee=True)

    m.obj_eff = pyo.Objective(expr=-m.fs_main.overall_efficiency[0])

    # Add constraints for caol flow dependent operating variables, PA, O2, fheat_ww
    pa = m.fs_main.fs_blr.flow_pa_kg_per_s
    o2pct = m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry
    fww = m.fs_main.fs_blr.aBoiler.fheat_ww
    coal_flow = m.fs_main.fs_blr.aBoiler.flowrate_coal_raw
    main_pressure = m.fs_main.fs_stc.turb.inlet_split.mixed_state[0].pressure
    throttle_pressure = m.fs_main.fs_stc.turb.throttle_valve[1].control_volume.properties_out[0].pressure

    m.set_pow = pyo.Var()
    m.set_pow.fix(220e6)
    m.pow_constraint = pyo.Constraint(expr=m.fs_main.gross_power[0]==m.set_pow)

    plus_t = 20
    m.throttle_pressure_constraint = pyo.Constraint(expr=main_pressure >= 1.1*throttle_pressure)
    m.main_steam_temp_constraint = pyo.Constraint(expr=m.fs_main.fs_stc.turb.inlet_split.mixed_state[0].temperature <= 814 + plus_t)
    m.reheat_temp_constraint = pyo.Constraint(expr=m.fs_main.fs_stc.turb.ip_stages[1].control_volume.properties_in[0].temperature <= 818 + plus_t)

    m.main_pressure_expr = pyo.Expression(expr=0.4541*throttle_pressure + 7.44e6)

    @m.fs_main.Constraint(m.fs_main.time)
    def pa_eqn(b, t):
        return pa[t] == 1.6569*coal_flow[t] + 16.581

    @m.fs_main.Expression(m.fs_main.time)
    def o2pct_3burner_expr(b, t):
        return (3.0 - o2pct[t])**2
    @m.fs_main.Expression(m.fs_main.time)
    def o2pct_2burner_expr(b, t):
        return (-0.5117*coal_flow[t] + 12.193 - o2pct[t])**2

    fww_dev = 0.10
    @m.fs_main.Expression(m.fs_main.time)
    def fww_2burner_expr(b, t):
        return (0.0101*coal_flow[t] + 0.859 - fww[t])**2
    @m.fs_main.Expression(m.fs_main.time)
    def fww_3burner_expr(b, t):
        return (-0.0074*coal_flow[t] + 1.178 - fww[t])**2
    @m.fs_main.Constraint(m.fs_main.time)
    def fww_2burner_constraint(b, t):
        return m.fs_main.fww_2burner_expr[0] <= fww_dev**2
    @m.fs_main.Constraint(m.fs_main.time)
    def fww_3burner_constraint(b, t):
        return m.fs_main.fww_3burner_expr[0] <= fww_dev**2
    m.fs_main.fww_2burner_constraint.deactivate()

    m.fs_main.thk_slag = pyo.Var(initialize=0.001, bounds=(0.001, 0.003))

    for i in m.fs_main.fs_blr.ww_zones:
        m.fs_main.fs_blr.Waterwalls[i].thk_slag.unfix()
    @m.fs_main.Constraint(m.fs_main.fs_blr.ww_zones)
    def slag_thickness_constraint(b, i):
        return b.fs_blr.Waterwalls[i].thk_slag[0] == m.fs_main.thk_slag

    eps = 1/10.0
    #m.obj_eff_3 = pyo.Objective(expr=-m.fs_main.overall_efficiency[0]*100 + eps*m.fs_main.o2pct_3burner_expr[0] + eps*m.fs_main.fww_3burner_expr[0] + eps*(m.fs_main.thk_slag - 0.001)**2/0.001**2 + eps*(m.main_pressure_expr - main_pressure)**2/1e7**2)
    #m.obj_eff_2 = pyo.Objective(expr=-m.fs_main.overall_efficiency[0]*100 + eps*m.fs_main.o2pct_2burner_expr[0] + eps*m.fs_main.fww_2burner_expr[0] + eps*(m.fs_main.thk_slag - 0.001)**2/0.001**2 + eps*(m.main_pressure_expr - main_pressure)**2/1e7**2)

    #m.obj_eff.deactivate()
    #m.obj_eff_2.deactivate()

    m.fs_main.thk_slag.fix()
    pa.unfix()
    fww.unfix()
    o2pct.unfix()
    main_pressure.unfix()
    throttle_pressure.unfix()
    main_pressure.setlb(3.3e6)
    main_pressure.setub(1.3e7)
    throttle_pressure.setlb(3.0e6)
    throttle_pressure.setub(1.1e7)
    o2pct[0].setlb(3.0)
    o2pct[0].setub(8.0)
    fww[0].setlb(0.94)
    fww[0].setub(1.06)

    _log.info("Initial solve with operating constraints.")
    solver.solve(m, tee=True)
    solver.options["max_iter"] = 60
    gps = np.linspace(220e6, 80e6, 50)
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
        "throttle_pressure",
        "throttle_pressure (MPa)",
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
        "slag_thickness",
        "baseline_eff",
        "baseline_improvement",
        "last_stage_in_x",
        "last_stage_out_x",
        "nox_ppm_2b",
        "nox_ppm_3b",
        "fegt",
    ]
    # Data frame to tabulate results
    df_val = pd.DataFrame(columns=col, index=gps)

    m.fs_main.fs_stc.turb.throttle_valve[1].pressure_flow_equation.deactivate()
    for p in gps:
        m.set_pow.fix(p)
        if p < 140e6:
            #m.obj_eff_3.deactivate()
            #m.obj_eff_2.activate()
            m.fs_main.fww_2burner_constraint.activate()
            m.fs_main.fww_3burner_constraint.deactivate()

        _log.info(f"Gross power {p/1e6} MW")
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
        df_val.loc[p]["throttle_pressure"] = pyo.value(throttle_pressure)
        df_val.loc[p]["throttle_pressure (MPa)"] = pyo.value(throttle_pressure/1e6)
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
        df_val.loc[p]["slag_thickness"] = pyo.value(m.fs_main.thk_slag)
        df_val.loc[p]["last_stage_in_x"] = pyo.value(m.fs_main.fs_stc.turb.outlet_stage.control_volume.properties_in[0].vapor_frac)
        df_val.loc[p]["last_stage_out_x"] = pyo.value(m.fs_main.fs_stc.turb.outlet_stage.control_volume.properties_out[0].vapor_frac)
        df_val.loc[p]["nox_ppm_2b"] = pyo.value(m.fs_main.nox_ppm_2burner[0])
        df_val.loc[p]["nox_ppm_3b"] = pyo.value(m.fs_main.nox_ppm_3burner[0])
        df_val.loc[p]["fegt"] = pyo.value(m.fs_main.fs_blr.aBoiler.side_1_outlet.temperature[0])

        df_val.to_csv(f"opt_{int(fww_dev*100+0.1)}_{int(plus_t)}.csv")
