from plant_dyn import main_steady
import pyomo.environ as pyo
import report_steady as rpt
import idaes.dmf.model_data as md
import data_util as data_module
import numpy as np
import pandas as pd
import idaes.dmf.model_data as mdata
import idaes.core.util.scaling as iscale

if __name__ == "__main__":
    m = main_steady(
        load_state="initial_steady.json.gz",
        save_state="initial_steady.json.gz")

    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=True)
    df, df_meta, bin_stdev = data_module.read_data(model=m)
    data_module.add_data_params(model=m, df=df, df_meta=df_meta)

    bins = set(df["bin_number"])

    tags = set(m.data_param.keys())
    tags -= {"PAPERFLOW_MOL"}
    tags = list(tags)
    bin_mean = data_module.bin_mean(df, bin_no="bin_number")
    #for tag in tags:
    #    for bin in bins:
    #        print(f"{tag}[{bin}]: {bin_stdev[bin][tag]}")
    rep_index = []
    rep_loc = []
    for bin in bins:
        df_sub = df[df["bin_number"]==bin]
        mean_vec = np.array([bin_mean[bin][tag] for tag in tags])
        stdev_vec = np.array([bin_stdev[bin][tag] for tag in tags])
        row_vec = []
        for i in df_sub.index:
            row_vec.append(np.array(df_sub.loc[i][tags])/stdev_vec)
        low_g = None
        for n, i in enumerate(df_sub.index):
            geo = sum(np.sqrt(np.sum(np.square(row_vec[n] - row_vec[n2]))) for n2, i2 in enumerate(df_sub.index))
            if low_g is None or geo < low_g:
                low_g = geo
                low_i = i
            print(f"{i}, {geo}")
        rep_index.append(low_i)
        rep_loc.append(df.index.get_loc(low_i))
        print(low_i)
        print(low_g)

    # Fix rep_index here to keep it from shifting once I settle on it for the plot book
    rep_index = [df.index[i] for i in [1670, 1536, 1462, 1368, 1280, 1205, 1139, 1048, 951, 784, 676, 544, 508, 460, 437, 402, 370, 328, 299, 280, 229, 163, 110, 47, 26]]
    res_df = pd.read_csv("results/validate.csv", index_col=0)

    data_module.data_plot_book(
        df,
        bin_nom="bin_nominal_mw",
        file="data_plot_book.pdf",
        tmp_dir="tmp_plots",
        xlabel="Gross Power (MW)",
        metadata=df_meta,
        cols=None,
        skip_cols=[],
        point_indexs=rep_index,
        results_df=res_df)

    print(rep_loc)
