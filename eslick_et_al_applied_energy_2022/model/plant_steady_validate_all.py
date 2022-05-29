""" Run validation for all data points.  This reads the parameters estimated in
the pest_big.py script from results/est.csv.
"""

import csv
import pandas as pd
from plant_dyn import main_steady
import report_steady as rpt
import data_util as data_module
import pyomo.environ as pyo
from idaes.generic_models.properties import iapws95
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
import idaes.core.util.model_serializer as ms

_log = idaeslog.getLogger(__name__)

def set_input_from_data(m):
    m.kq["paper_mill_flow"].fix(0)
    m.kq["paper_mill_split"].unfix()
    m.kq["coal_flow"].unfix()
    m.kq["main_steam_pressure"].fix(m.data_param["S833"].value)
    m.kq["throttle_opening"].unfix()
    m.kq["throttle_pressure"].fix(m.data_param["S839C"].value)
    m.fs_main.fs_stc.turb.throttle_valve[1].pressure_flow_equation.deactivate()

    m.kq["cw_flow"].fix(m.data_param["CWCOND_MOL"].value)
    m.kq["cw_temperature"].fix(m.data_param["CW213A"].value)
    m.kq["cw_enth"].unfix()

    m.kq["cw_aux_flow"].fix(m.data_param["CWAUXCOND_MOL"].value)
    m.kq["cw_aux_temperature"].fix(m.data_param["CW313"].value)
    m.kq["cw_aux_enth"].unfix()

    m.fs_main.fs_blr.dry_o2_in_flue_gas_eqn.deactivate()
    m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry[0].fix(m.data_param["BG3506"].value)

    m.fs_main.fs_blr.pa_to_coal_ratio_eqn.deactivate()
    m.fs_main.fs_blr.flow_pa_kg_per_s[0].fix(m.data_param["1FUE-TOTPA-FLOW"].value)

    m.fs_main.fs_blr.aAPH.side_2_inlet.pressure[0].fix(m.data_param["BF2001"].value)
    m.fs_main.fs_blr.aAPH.side_3_inlet.pressure[0].fix(m.data_param["BA1003"].value)
    m.fs_main.fs_blr.aAPH.side_2_inlet.temperature[0].fix(m.data_param["BF2002"].value)
    m.fs_main.fs_blr.aAPH.side_3_inlet.temperature[0].fix(m.data_param["BA2007"].value)

    m.fs_main.fs_stc.cond_pump.cond_pump_curve_constraint.activate()
    m.fs_main.fs_stc.cond_pump.control_volume.properties_out[0].pressure.unfix()
    #m.fs_main.fs_stc.cond_pump.control_volume.properties_out[0].pressure.fix(
    #    m.data_param["C51"].value)

    m.fs_main.fs_blr.aBoiler.SR_lf.fix(1.0)


    m.kq["main_steam_temperature"].unfix()
    m.kq["hot_reheat_temperature"].unfix()
    m.kq["spray_valve_opening"].unfix()
    m.fs_main.fs_blr.Attemp.Water_inlet_state[0].flow_mol.fix(m.data_param["FW463"].value/0.018015)

    m.fs_main.fs_stc.S036b_expanded.pressure_equality.deactivate()
    m.fs_main.fs_stc.booster_static_head_eqn.activate()

def set_params(m, case):
    _log.info("Setting estimated parameters.")
    m.fs_main.fs_stc.fwh5.desuperheat.overall_heat_transfer_coefficient[0.0].value = 500
    m.fs_main.fs_stc.fwh6.desuperheat.overall_heat_transfer_coefficient[0.0].value = 500
    m.fs_main.fs_blr.aBoiler.fheat_platen[0.0].fix()
    m.fs_main.fs_blr.fheat_platen_eqn.deactivate()
    m.fs_main.fs_blr.aBoiler.fheat_ww[0.0].fix()
    m.fs_main.fs_blr.fheat_ww_eqn.deactivate()
    m.fs_main.fs_stc.cond_pump.cond_pump_curve_constraint.activate()
    m.fs_main.fs_stc.cond_pump.control_volume.properties_out[0].pressure.unfix()
    m.fs_main.fs_stc.S036b_expanded.pressure_equality.deactivate()
    m.fs_main.fs_stc.booster_static_head_eqn.activate()

    two_burner_case = 676
    three_burner_case = 1670
    burner_dep_params = [
    #    "m.fs_main.fs_blr.a_fheat_ww",
    #    "m.fs_main.fs_blr.b_fheat_ww",
    #    "m.fs_main.fs_blr.c_fheat_ww",
    #    "m.fs_main.fs_blr.a_fheat_platen",
    #    "m.fs_main.fs_blr.b_fheat_platen",
    #    "m.fs_main.fs_blr.c_fheat_platen",
    ]
    independent = [
        #"m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw",
        #"m.fs_main.fs_blr.aBoiler.fheat_ww",
        #"m.fs_main.fs_blr.aBoiler.fheat_platen",
    ]
    # for now fix moisture at 18% and use the rest of the estimated params
    #m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw.fix(0.175)
    with open('results/est.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            s = row[0].split(".", 1)
            row[0] = s[1]
            if row[0] in independent:
                if s[0] == f"case[{case}]":
                    # independent for each case
                    pass
                else:
                    # not for this case
                    continue
            elif row[0] in burner_dep_params:
                if s[0] == f"case[{two_burner_case}]" and m.fs_main.fs_blr.num_mills.value == 2:
                    # two burner specific parameters`
                    pass
                elif s[0] == f"case[{three_burner_case}]" and m.fs_main.fs_blr.num_mills.value == 3:
                    # three burner specific parameters
                    pass
                else:
                    continue
            elif s[0] == f"case[{three_burner_case}]":
                # cases where all parameters == three burner case
                pass
            else:
                continue
            # if the parameter is for this case read it.
            if row[1] is not None and row[1]:
                eval(f"{row[0]}[{row[1]}]").value = float(row[2])
            else:
                eval(f"{row[0]}").value = float(row[2])

    # For this where we are running on all points, need to set fheat_ww by
    # some operating policy derived from the data.
    m.fs_main.fs_blr.aBoiler.fheat_platen.fix(0.98)
    m.fs_main.fs_blr.aBoiler.fheat_ww.unfix()

    #if m.fs_main.fs_blr.num_mills.value == 2:
    #    m.fs_main.fheat_ww_2_eqn.activate()
    #    m.fs_main.fheat_ww_3_eqn.deactivate()
    #else:
    m.fs_main.fheat_ww_3_eqn.deactivate()
    m.fs_main.fheat_ww_2_eqn.deactivate()
    m.fs_main.fs_blr.aBoiler.fheat_ww[0].setlb(0.94)
    m.fs_main.fs_blr.aBoiler.fheat_ww[0].setub(1.06)
    m.fs_main.fs_blr.aBoiler.fheat_ww[0].unfix()
    m.obj_datarec.deactivate()

if __name__ == "__main__":
    solver = pyo.SolverFactory("ipopt")
    # Using the monkey patched solver so solver settings are there

    m = main_steady(
        load_state="initial_steady.json.gz",
        save_state="initial_steady.json.gz")

    @m.fs_main.Constraint(m.fs_main.time)
    def fheat_ww_2_eqn(b, t):
        return b.fs_blr.aBoiler.fheat_ww[t] == 0.0101*b.fs_blr.aBoiler.flowrate_coal_raw[t] + 0.859

    @m.fs_main.Constraint(m.fs_main.time)
    def fheat_ww_3_eqn(b, t):
        return b.fs_blr.aBoiler.fheat_ww[t] == -0.0074*b.fs_blr.aBoiler.flowrate_coal_raw[t] + 1.178

    m.fs_main.fheat_ww_2_eqn.deactivate()
    m.fs_main.fheat_ww_3_eqn.deactivate()


    # Read in plant data, bin it, and calculate standard deviations
    df, df_meta, bin_stdev = data_module.read_data(model=m)

    # Make a nice dictionary of refernces for quantiies I want easy access to
    data_module.tag_key_quantities(model=m)

    # Add model parameters to hold a data for a specific point in time
    data_module.add_data_params(model=m, df=df, df_meta=df_meta)

    # Add an objective function based on the error reletive to stdev
    data_module.add_data_match_obj(model=m, df_meta=df_meta, bin_stdev=bin_stdev)
    #m.obj_datarec.deactivate()
    #m.obj_weight["1JT66801S"] = 30
    #m.obj_weight["1FWS-FWFLW-A"] = 20
    #m.obj_weight["DGS3"] = 6

    m.obj_mst = pyo.Objective(
        expr=m.err_stdev["S831"]**2 + m.err_stdev["S634"]**2/2.0)

    col = [
        "status",
        "overall_efficiency",
        "primary_air",
        "fg_o2_dry_pct",
        "fheat_ww",
        "fheat_platen",
        "coal_moisture_pct",
        "coal_hhv",
        "gross_power",
        "Raw_coal_mass_flow_rate",
        "Moisture_mass_fraction_in_coal",
        "FG06_Fmass",
        "FG06_T",
        "FG06_P",
        "FG06_yN2",
        "FG06_yO2",
        "FG06_yNO",
        "FG06_yCO2",
        "FG06_yH2O",
        "FG06_ySO2",
    ]
    for tag in m.data_param:
        col.append(tag + "_data")
        col.append(tag + "_model")

    val_cases = range(len(df.index))
    df_val = pd.DataFrame(columns=col, index=val_cases)

    data_module.set_data_params(m, df, index_index=0)
    set_input_from_data(m)
    set_params(m, 0)
    strip_bounds = pyo.TransformationFactory("contrib.strip_var_bounds")
    strip_bounds.apply_to(m, reversible=False)

    n = 0
    ms.from_json(m, fname=f"results/state4_{n}.json.gz")
    set_params(m, n)
    solver.solve(m, tee=True, linear_eliminate=False)
    sd_latest = ms.to_json(m, return_dict=True)

    for i in val_cases:
        data_module.set_data_params(m, df, index_index=i)
        set_input_from_data(m)
        set_params(m, i)

        print(f"Starting run for {i}, {df.iloc[i]['1LDC-GROSS-MW']/1e6} MW")
        for q in m.kq:
            for j in m.kq[q]:
                print(f"{q}[{j}] {pyo.value(m.kq[q][j])}")

        solver.options["max_iter"] = 100
        try:
            res = solver.solve(m, tee=True, linear_eliminate=False)
            res = idaeslog.condition(res)
        except:
            res = "fail"

        if not res.startswith("optimal"):
            ms.from_json(m, sd=sd_one_back)
            data_module.set_data_params(m, df, index_index=i)
            set_input_from_data(m)
            set_params(m, i)
            try:
                res = solver.solve(m, tee=True, linear_eliminate=False)
                res = idaeslog.condition(res)
            except:
                res = "fail"
        else:
            sd_one_back = sd_latest
            sd_latest = ms.to_json(m, return_dict=True)

        sd = rpt.create_stream_dict(m)
        sdf = rpt.stream_table(sd, fname=f"results/streams{i}.csv")
        tag, tag_format = rpt.create_tags(m, sd)
        rpt.svg_output(tag, tag_format, n)
        rpt.turbine_table(m, fname=f"results/turbine_table{i}.csv")

        df_val.loc[i]["status"] = res
        df_val.loc[i]["overall_efficiency"] = pyo.value(m.fs_main.overall_efficiency[0])
        df_val.loc[i]["primary_air"] = pyo.value(m.fs_main.fs_blr.flow_pa_kg_per_s[0])
        df_val.loc[i]["fg_o2_dry_pct"] = pyo.value(m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry[0])
        df_val.loc[i]["fheat_ww"] = pyo.value(m.fs_main.fs_blr.aBoiler.fheat_ww[0])
        df_val.loc[i]["fheat_platen"] = pyo.value(m.fs_main.fs_blr.aBoiler.fheat_platen[0])
        df_val.loc[i]["coal_moisture_pct"] = pyo.value(m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0.0])
        df_val.loc[i]["coal_hhv"] = pyo.value(m.fs_main.fs_blr.aBoiler.hhv_coal_dry)
        df_val.loc[i]["gross_power"] = pyo.value(m.fs_main.gross_power[0])
        df_val.loc[i]["Raw_coal_mass_flow_rate"] = pyo.value(m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[0])
        df_val.loc[i]["Moisture_mass_fraction_in_coal"] = pyo.value(m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0])
        df_val.loc[i]["FG06_Fmass"] = pyo.value(tag["FG06_Fmass"])
        df_val.loc[i]["FG06_T"] = pyo.value(tag["FG06_T"])
        df_val.loc[i]["FG06_P"] = pyo.value(tag["FG06_P"])
        df_val.loc[i]["FG06_yN2"] = pyo.value(tag["FG06_yN2"])
        df_val.loc[i]["FG06_yO2"] = pyo.value(tag["FG06_yO2"])
        df_val.loc[i]["FG06_yNO"] = pyo.value(tag["FG06_yNO"])
        df_val.loc[i]["FG06_yCO2"] = pyo.value(tag["FG06_yCO2"])
        df_val.loc[i]["FG06_yH2O"] = pyo.value(tag["FG06_yH2O"])
        df_val.loc[i]["FG06_ySO2"] = pyo.value(tag["FG06_ySO2"])

        for tag in m.data_param:
            df_val.loc[i][tag + "_data"] = m.data_param[tag].value
            df_val.loc[i][tag + "_model"] = pyo.value(df_meta[tag]["reference"][0])

        ms.to_json(m, fname=f"results/statex_{i}.json.gz")

        #try:
        #    ms.from_json(m, fname=f"results/state4_{i}.json.gz")
        #except:
        #    # IF the file doesn't exist fine, just start from previous
        #    pass
        df_val.to_csv("results/validate_all_for_S.csv")
