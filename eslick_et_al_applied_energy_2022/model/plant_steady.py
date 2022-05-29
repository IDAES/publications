from plant_dyn import main_steady
import pyomo.environ as pyo
import report_steady as rpt
import json
import data_util as data_module
import pandas as pd
import idaes.core.util.scaling as iscale

if __name__ == "__main__":
    m = main_steady(
        load_state="initial_steady.json.gz",
        save_state="initial_steady.json.gz")

    solver = pyo.SolverFactory("ipopt")
    iscale.calculate_scaling_factors(m)
    solver.solve(m, tee=True)


    sd = rpt.create_stream_dict(m)
    sdf = rpt.stream_table(sd, fname="results/streams.csv")
    tag, tag_format = rpt.create_tags(m, sd)
    rpt.svg_output(tag, tag_format)
    rpt.turbine_table(m, fname="results/turbine_table.csv")

    df, df_meta, bin_stdev = data_module.read_data(model=m)
    dfs = pd.DataFrame()
    df.to_csv("binned.csv")
    for b in bin_stdev:
        dfs[f"{85+b*5} to {90+b*5} MW"] = bin_stdev[b]
    dfs.to_csv("stdev.csv")
