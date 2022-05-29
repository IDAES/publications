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
        "m.fs_main.fs_blr.aBoiler.fheat_ww",
        "m.fs_main.fs_blr.aBoiler.fheat_platen",
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

if __name__ == "__main__":
    solver = pyo.SolverFactory("ipopt")
    # Using the monkey patched solver so solver settings are there

    m = main_steady(
        load_state="initial_steady.json.gz",
        save_state="initial_steady.json.gz")


    # Read in plant data, bin it, and calculate standard deviations
    df, df_meta, bin_stdev = data_module.read_data(model=m)

    # Make a nice dictionary of refernces for quantiies I want easy access to
    data_module.tag_key_quantities(model=m)

    # Add model parameters to hold a data for a specific point in time
    data_module.add_data_params(model=m, df=df, df_meta=df_meta)

    # Add an objective function based on the error reletive to stdev
    data_module.add_data_match_obj(model=m, df_meta=df_meta, bin_stdev=bin_stdev)
    #m.obj_datarec.deactivate()
    m.obj_weight["1JT66801S"] = 30
    m.obj_weight["1FWS-FWFLW-A"] = 20
    m.obj_weight["DGS3"] = 6

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
        "net_power",
        "Raw_coal_mass_flow_rate",
        "Moisture_mass_fraction_in_coal",
        "G005_F",
        "G005_T",
        "G006_F",
        "G006_T",
        "G006_P",
        "G009_T",
        "O2_avg",
        "B001_F",
        "B001_T",
        "B001_P",
        "B022_P",
        "B022_T",
    ]
    for tag in m.data_param:
        col.append(tag + "_data")
        col.append(tag + "_model")

    val_cases = [
        1670, 1536, 1462, 1368, 1280, 1205, 1139, 1048, 951, 784, 676, 544, 508,
        460, 437, 402, 370, 328, 299, 280, 229, 163, 110, 47, 26]
    df_val = pd.DataFrame(columns=col, index=val_cases)
    indexes = df.iloc[val_cases,]
    indexes = pd.DataFrame(indexes.index)
    indexes.loc[:,"case"] = val_cases
    indexes.to_csv("val_times.csv")

    data_module.set_data_params(m, df, index_index=0)
    set_input_from_data(m)
    set_params(m, 0)
    strip_bounds = pyo.TransformationFactory("contrib.strip_var_bounds")
    strip_bounds.apply_to(m, reversible=False)

    n = 26
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

        df_val.loc[i]["status"] = res
        df_val.loc[i]["overall_efficiency"] = pyo.value(m.fs_main.overall_efficiency[0])
        df_val.loc[i]["primary_air"] = pyo.value(m.fs_main.fs_blr.flow_pa_kg_per_s[0])
        df_val.loc[i]["fg_o2_dry_pct"] = pyo.value(m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry[0])
        df_val.loc[i]["fheat_ww"] = pyo.value(m.fs_main.fs_blr.aBoiler.fheat_ww[0])
        df_val.loc[i]["fheat_platen"] = pyo.value(m.fs_main.fs_blr.aBoiler.fheat_platen[0])
        df_val.loc[i]["coal_moisture_pct"] = pyo.value(m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0.0])
        df_val.loc[i]["coal_hhv"] = pyo.value(m.fs_main.fs_blr.aBoiler.hhv_coal_dry)
        df_val.loc[i]["gross_power"] = pyo.value(m.fs_main.gross_power[0])
        df_val.loc[i]["net_power"] = pyo.value(m.fs_main.net_power[0])
        df_val.loc[i]["Raw_coal_mass_flow_rate"] = pyo.value(m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[0])
        df_val.loc[i]["Moisture_mass_fraction_in_coal"] = pyo.value(m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0])
        df_val.loc[i]["G005_F"] = pyo.value(m.fs_main.fs_blr.aAPH.side_3.properties_out[0].flow_mol)
        df_val.loc[i]["G005_T"] = pyo.value(m.fs_main.fs_blr.aAPH.side_3.properties_out[0].temperature)
        df_val.loc[i]["G006_F"] = pyo.value(m.fs_main.fs_blr.Mixer_PA.mixed_state[0].flow_mol)
        df_val.loc[i]["G006_T"] = pyo.value(m.fs_main.fs_blr.Mixer_PA.mixed_state[0].temperature)
        df_val.loc[i]["G006_P"] = pyo.value(m.fs_main.fs_blr.Mixer_PA.mixed_state[0].pressure)
        df_val.loc[i]["O2_avg"] = pyo.value(m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry[0])
        df_val.loc[i]["G009_T"] = pyo.value(m.fs_main.fs_blr.aRH2.shell.properties[0,0].temperature)
        df_val.loc[i]["B001_F"] = pyo.value(m.fs_main.fs_blr.aECON.tube.properties[0,0].flow_mol)
        df_val.loc[i]["B001_T"] = pyo.value(m.fs_main.fs_blr.aECON.tube.properties[0,0].temperature)
        df_val.loc[i]["B001_P"] = pyo.value(m.fs_main.fs_blr.aECON.tube.properties[0,0].pressure)
        df_val.loc[i]["B022_P"] = pyo.value(m.fs_main.fs_blr.aRoof.control_volume.properties_in[0].pressure)
        df_val.loc[i]["B022_T"] = pyo.value(m.fs_main.fs_blr.aRoof.control_volume.properties_in[0].temperature)
        for tag in m.data_param:
            df_val.loc[i][tag + "_data"] = m.data_param[tag].value
            df_val.loc[i][tag + "_model"] = pyo.value(df_meta[tag]["reference"][0])

        ms.to_json(m, fname=f"results/state4_{i}.json.gz")

        sd = rpt.create_stream_dict(m)
        sdf = rpt.stream_table(sd, fname=f"results/streams{i}.csv")
        tag, tag_format = rpt.create_tags(m, sd)
        rpt.svg_output(tag, tag_format, n)
        rpt.turbine_table(m, fname=f"results/turbine_table{i}.csv")

        #try:
        #    ms.from_json(m, fname=f"results/state4_{i}.json.gz")
        #except:
        #    # IF the file doesn't exist fine, just start from previous
        #    pass
        df_val.to_csv("results/validate.csv")
