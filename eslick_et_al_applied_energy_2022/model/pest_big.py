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
import idaes.dmf.model_data as mdata


_log = idaeslog.getLogger(__name__)

def get_params(s, case_indexes, intramodel_indexes, group_by, independent, pl):
    gr = {}
    for i in case_indexes:
        if group_by:
            x = eval("main_model.case[i].m." + group_by).value
        elif independent:
            x = i
        else:
            x = None
        if x in gr:
            l = gr[x]
            first_case_in_group = False
        else:
            gr[x] = []
            l = gr[x]
            first_case_in_group = True
        if intramodel_indexes is None:
            l.append(eval("main_model.case[i].m." + s))
        else:
            for j in intramodel_indexes:
                l.append(eval("main_model.case[i].m." + s.format(j)))
        if first_case_in_group:
            pl.extend(l)
    return [x for x in gr.values()]

def unfix_params(
    m,
    s,
    case_indexes,
    intramodel_indexes=None,
    bounds=(None, None),
    mbounds=None,
    constraint=None,
    group_by=None,
    pl=[],
    fix=None,
    independent=False):
    g = get_params(s, case_indexes, intramodel_indexes, group_by, independent, pl)
    for l in g:
        for v in l:
            if fix is None:
                v.unfix()
            else:
                v.fix(fix)
            if mbounds is not None:
                v.setlb(v.value*mbounds[0])
                v.setub(v.value*mbounds[1])
            else:
                v.setlb(bounds[0])
                v.setub(bounds[1])
    if constraint is not None and fix is None:
        for j, l in enumerate(g):
            def rfunc(b, i):
                if i == 0:
                    return pyo.Constraint.Skip
                else:
                    return l[i] == l[0]
            setattr(m, constraint + str(j), pyo.Constraint(range(len(l)), rule=rfunc))

def set_weight(tag, value, case_indexes):
    for i in case_indexes:
        main_model.case[i].m.obj_weight[tag].value = value

if __name__ == "__main__":
    solver = pyo.SolverFactory("ipopt")
    # Using the monkey patched solver so solver settings are there

    case_indexes = [1670, 1536, 1462, 1368, 1280, 1205, 1139, 1048, 951, 784, 676, 544, 508, 460, 437, 402, 370, 328, 299, 280, 229, 163, 110, 47, 26]
    main_model = pyo.ConcreteModel()
    main_model.case = pyo.Block(case_indexes)
    df = None
    for i in case_indexes:
        main_model.case[i].m = main_steady(
            load_state="initial_steady.json.gz")
        if df is None:
            df, df_meta, bin_stdev = data_module.read_data(model=main_model.case[i].m)
        mdata.update_metadata_model_references(main_model.case[i].m, df_meta)
        data_module.tag_key_quantities(model=main_model.case[i].m)
        data_module.add_data_params(model=main_model.case[i].m, df=df, df_meta=df_meta)
        data_module.add_data_match_obj(model=main_model.case[i].m, df_meta=df_meta, bin_stdev=bin_stdev)
        #load data and model inputs already solved
        ms.from_json(main_model.case[i].m, fname=f"results/state4_{i}.json.gz", root_name="unknown")
        main_model.case[i].m.obj_datarec.deactivate()
    #make sure the whole thing solves
    main_model.obj = pyo.Objective(expr=sum(main_model.case[i].m.obj_expr for i in case_indexes)/len(case_indexes)/100)
    strip_bounds = pyo.TransformationFactory("contrib.strip_var_bounds")
    strip_bounds.apply_to(main_model, reversible=False)
    for v, sv in iscale.badly_scaled_var_generator(
        main_model,
        small=0,
        large=100
    ):
        iscale.set_scaling_factor(v, 1/pyo.value(v))
    iscale.calculate_scaling_factors(main_model)
    solver.solve(main_model, linear_eliminate=True, tee=True)

    set_weight("1JT66801S", 30, case_indexes)
    set_weight("1FWS-FWFLW-A", 20, case_indexes)
    set_weight("DGS3", 0.5, case_indexes)
    set_weight("S634", 2, case_indexes)
    set_weight("S831", 5, case_indexes)

    drop_tags = [
        "1FWS-STMFLW-A",
        "1TCMVACPT2",
        "S11",
        "1OPT-NOXPPM",
    ]
    for tag in drop_tags:
        set_weight(tag, 0, case_indexes)

    pl = []

    unfix_params(
        main_model,
        s="fs_main.fs_blr.blowdown_frac",
        case_indexes=case_indexes,
        bounds=(0.002, 0.02),
        constraint="bd_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0.0]",
        case_indexes=case_indexes,
        bounds=(0.14, 0.20),
        constraint="coal_mc_constraint",
        #independent=True,
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aAPH.frac_heatloss",
        case_indexes=case_indexes,
        bounds=(0.05, 0.15),
        constraint="aph_frac_heatloss_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aBoiler.hhv_coal_dry",
        case_indexes=case_indexes,
        mbounds=(2.40e+007, 2.5e+007),
        constraint="hhv_constraint",
        pl=pl,
        fix=2.48579e+007,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.outlet_stage.design_exhaust_flow_vol",
        case_indexes=case_indexes,
        mbounds=(0.70, 1.05),
        constraint="outlet_des_flow_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.outlet_stage.eff_dry",
        case_indexes=case_indexes,
        bounds=(0.8, 0.92),
        constraint="outlet_dry_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.outlet_stage.flow_coeff",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="outlet_flow_coeff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.inlet_stage[1].eff_nozzle",
        case_indexes=case_indexes,
        bounds=(0.86, 0.94),
        constraint="inlet_nozzel_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.inlet_stage[1].flow_coeff[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="inlet_flow_coeff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.hp_stages[{}].efficiency_isentropic[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        bounds=(0.9, 0.94),
        constraint="hp_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.ip_stages[{}].efficiency_isentropic[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[1,2,4,5,6],
        bounds=(0.9, 0.94),
        constraint="ip1_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.ip_stages[{}].efficiency_isentropic[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[7,8,9],
        bounds=(0.9, 0.92),
        constraint="ip7_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.lp_stages[{}].efficiency_isentropic[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[1,2],
        bounds=(0.70, 0.91),
        constraint="lp1_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.lp_stages[{}].efficiency_isentropic[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[3,4],
        bounds=(0.70, 0.90),
        constraint="lp3_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.lp_stages[5].efficiency_isentropic[0]",
        case_indexes=case_indexes,
        bounds=(0.70, 0.89),
        constraint="lp5_eff_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.hp_stages[{}].ratioP[0]",
        case_indexes=case_indexes,
        bounds=(0.85, 0.96),
        intramodel_indexes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        constraint="hp_pressure_ratio_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.ip_stages[{}].ratioP[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[1,2,4,5,6],
        bounds=(0.80, 0.94),
        constraint="ip1_pressure_ratio_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.ip_stages[{}].ratioP[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[7,8,9],
        bounds=(0.80, 0.94),
        constraint="ip7_pressure_ratio_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.lp_stages[{}].ratioP[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[1,2],
        mbounds=(0.9, 1.1),
        constraint="lp1_pressure_ratio_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.lp_stages[{}].ratioP[0]",
        case_indexes=case_indexes,
        intramodel_indexes=[3,4],
        mbounds=(0.9, 1.1),
        constraint="lp3_pressure_ratio_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.turb.lp_stages[5].ratioP[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="lp5_pressure_ratio_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.cond_pump.pump_a",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.10),
        constraint="a_cond_pump_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.cond_pump.pump_b",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.10),
        constraint="b_cond_pump_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.cond_pump.pump_c",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.10),
        constraint="c_cond_pump_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.cond_pump.pump_d",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.30),
        constraint="d_cond_pump_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.booster_static_head",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.10),
        constraint="booster_static_head_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.booster.booster_a",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.10),
        constraint="a_booster_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.booster.booster_b",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.1),
        constraint="b_booster_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.booster.booster_c",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.10),
        constraint="c_booster_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.booster.booster_d",
        case_indexes=case_indexes,
        mbounds=(0.90, 1.10),
        constraint="d_booster_constraint",
        pl=pl)
    #unfix_params(
    #    main_model,
    #    s="fs_main.fs_blr.aBoiler.SR_lf",
    #    case_indexes=case_indexes,
    #    bounds=(0.95, 1.05),
    #    #constraint="sr_lf_constraint",
    #    independent=True,
    #    pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aBoiler.fheat_ww[0]",
        case_indexes=case_indexes,
        bounds=(0.94, 1.06),
        #constraint="fheat_ww_constraint",
        independent=True,
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aBoiler.fheat_platen[0]",
        case_indexes=case_indexes,
        bounds=(0.94, 1.06),
        #constraint="fheat_platen_constraint",
        fix=0.98,
        independent=True,
        pl=pl)
    """
    unfix_params(
        main_model,
        s="fs_main.fs_blr.a_fheat_ww",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.2),
        group_by="fs_main.fs_blr.num_mills",
        constraint="a_fheat_ww_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.b_fheat_ww",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.2),
        group_by="fs_main.fs_blr.num_mills",
        constraint="b_fheat_ww_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.c_fheat_ww",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.2),
        group_by="fs_main.fs_blr.num_mills",
        constraint="c_fheat_ww_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.a_fheat_platen",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.2),
        group_by="fs_main.fs_blr.num_mills",
        constraint="a_fheat_platen_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.b_fheat_platen",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.2),
        group_by="fs_main.fs_blr.num_mills",
        constraint="b_fheat_platen_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.c_fheat_platen",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.2),
        group_by="fs_main.fs_blr.num_mills",
        constraint="c_fheat_platen_constraint",
        pl=pl)
    """
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh2.cooling.overall_heat_transfer_coefficient[0.0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh2_coolingU_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh3.cooling.overall_heat_transfer_coefficient[0.0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh3_coolingU_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh5.cooling.overall_heat_transfer_coefficient[0.0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh5_coolingU_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh6.cooling.overall_heat_transfer_coefficient[0.0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh6_coolingU_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh1.condense.U0[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh1_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh2.condense.U0[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh2_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh3.condense.U0[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh3_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh5.condense.U0[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh5_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.fwh6.condense.U0[0]",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="fwh6_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.condenser.overall_heat_transfer_coefficient[0]",
        case_indexes=case_indexes,
        bounds=(7000., 1.1e5),
        constraint="condenser_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_stc.aux_condenser.overall_heat_transfer_coefficient[0]",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.25),
        constraint="aux_condenser_U0_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aAPH.side_1.deltaP[0.0]",
        case_indexes=case_indexes,
        mbounds=(1.1, 0.9),  # This parameter is negative
        constraint="aph1_dp_constraint",
        pl=pl)
    # in the model dp2 = dp3, accoriding to the comment this is because there
    # is no data to estimate one of them
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aAPH.side_3.deltaP[0.0]",
        case_indexes=case_indexes,
        mbounds=(1.1, 0.9),  # This parameter is negative
        constraint="aph3_dp_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aph2_ua_a",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.25),
        constraint="aph2_ua_a_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aph2_ua_b",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.25),
        constraint="aph2_ua_b_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aph2_ua_c",
        case_indexes=case_indexes,
        mbounds=(0.8, 1.25),
        constraint="aph2_ua_c_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aph3_ua_a",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="aph3_ua_a_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aph3_ua_b",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="aph3_ua_b_constraint",
        pl=pl)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.fcorrection_htc_shell",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="econ_fcor_htc_shell_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.fcorrection_htc_tube",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="econ_fcor_htc_tube_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.fcorrection_dp_shell",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="econ_fcor_dp_shell_constraint",
        pl=pl,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.fcorrection_dp_tube",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="econ_fcor_dp_tube_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.fcorrection_htc_shell",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="psh_fcor_htc_shell_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.fcorrection_htc_tube",
        case_indexes=case_indexes,
        mbounds=(0.9, 1.1),
        constraint="psh_fcor_htc_tube_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.fcorrection_dp_shell",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.2),
        constraint="psh_fcor_dp_shell_constraint",
        pl=pl,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.fcorrection_dp_tube",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.2),
        constraint="psh_fcor_dp_tube_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.fcorrection_htc_shell",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.2),
        constraint="rh1_fcor_htc_shell_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.fcorrection_htc_tube",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.2),
        constraint="rh1_fcor_htc_tube_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.fcorrection_dp_shell",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.2),
        constraint="rh1_fcor_dp_shell_constraint",
        pl=pl,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.fcorrection_dp_tube",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.2),
        constraint="rh1_fcor_dp_tube_constraint",
        pl=pl,
        fix=1,)
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.Nu_tube_a",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="rh1_Nu_tube_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.Nu_tube_b",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="rh1_Nu_tube_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.Nu_tube_c",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="rh1_Nu_tube_c_constraint",
        pl=pl,
        fix=0.4,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.Nu_shell_a",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="rh1_Nu_shell_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.Nu_shell_b",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="rh1_Nu_shell_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.Nu_shell_c",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="rh1_Nu_shell_c_constraint",
        pl=pl,
        fix=1.0/3.0,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.ff_tube_a",
        case_indexes=case_indexes,
        bounds=(0.21, 0.25),
        constraint="rh1_ff_tube_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aRH1.ff_tube_b",
        case_indexes=case_indexes,
        bounds=(0.68, 0.75),
        constraint="rh1_ff_tube_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.Nu_tube_a",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="econ_Nu_tube_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.Nu_tube_b",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="econ_Nu_tube_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.Nu_tube_c",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="econ_Nu_tube_c_constraint",
        pl=pl,
        fix=0.4,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.Nu_shell_a",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="econ_Nu_shell_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.Nu_shell_b",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="econ_Nu_shell_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.Nu_shell_c",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="econ_Nu_shell_c_constraint",
        pl=pl,
        fix=1.0/3.0,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.ff_tube_a",
        case_indexes=case_indexes,
        bounds=(0.09, 0.12),
        constraint="econ_ff_tube_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aECON.ff_tube_b",
        case_indexes=case_indexes,
        bounds=(0.56, 0.70),
        constraint="econ_ff_tube_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.Nu_tube_a",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="psh_Nu_tube_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.Nu_tube_b",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="psh_Nu_tube_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.Nu_tube_c",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="psh_Nu_tube_c_constraint",
        pl=pl,
        fix=0.4,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.Nu_shell_a",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="psh_Nu_shell_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.Nu_shell_b",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="psh_Nu_shell_b_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.Nu_shell_c",
        case_indexes=case_indexes,
        mbounds=(0.85, 1.15),
        constraint="psh_Nu_shell_c_constraint",
        pl=pl,
        fix=1.0/3.0,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.ff_tube_a",
        case_indexes=case_indexes,
        bounds=(0.085, 0.2),
        constraint="psh_ff_tube_a_constraint",
        pl=pl,
    )
    unfix_params(
        main_model,
        s="fs_main.fs_blr.aPSH.ff_tube_b",
        case_indexes=case_indexes,
        bounds=(0.725, 0.85),
        constraint="psh_ff_tube_b_constraint",
        pl=pl,
    )

    solver.options["max_iter"] = 65
    solver.solve(main_model, linear_eliminate=True, tee=True)

    for v in pl:
        print(f"{v}: {v.value} [{v.lb}, {v.ub}], fixed = {v.fixed}")

    with open("results/est.csv", "w", newline="") as f:
        w = csv.writer(f)
        for v in pl:
            w.writerow([str(v.parent_component()), v.index(), v.value, v.lb, v.ub])
