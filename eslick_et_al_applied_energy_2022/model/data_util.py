import os
import numpy as np
import pandas as pd
import csv
import pyomo.environ as pyo
import idaes.dmf.model_data as mdata
import idaes.logger as idaeslog
import seaborn as sns
from PyPDF2 import PdfFileMerger
import matplotlib.pyplot as plt

_log = idaeslog.getLogger(__name__)

def deactivate_constraints_for_data_rec(model):
    pass

def read_data(model, metadata="data/meta_data.csv", data="data/data_small.csv"):

    _log.info("Reading data...")
    df, df_meta = mdata.read_data(
        model=model,
        csv_file=data,
        csv_file_metadata=metadata,
        ambient_pressure="BAROMETRIC-PRESS",
        ambient_pressure_unit="inHg",
        unit_system="mks",
    )
    # drop tags

    # FWH1 Shell Pressure
    df.drop("S105", axis=1, inplace=True)
    del df_meta["S105"]
    # FWH2 Extraction Temperature
    #df.drop("S302", axis=1, inplace=True)
    #del df_meta["S302"]
    # FWH2 Extraction Temperature
    #df.drop("S202", axis=1, inplace=True)
    #del df_meta["S202"]

    # for now drop attemperator flow
    #df.drop("FW463", axis=1, inplace=True)
    #del df_meta["FW463"]

    df.drop("BG3503", axis=1, inplace=True)
    del df_meta["BG3503"]

    df.drop("BG3505", axis=1, inplace=True)
    del df_meta["BG3505"]

    # Second hot reheat pressure measure (don't double weight)
    df.drop("S640", axis=1, inplace=True)
    del df_meta["S640"]

    df.drop("S202", axis=1, inplace=True)
    del df_meta["S202"]

    # want the data for number of mills in service, but don't want it in
    # objective and such
    del df_meta["num_mills"]
    # Calculated tags can go here

    # keep paper mill and attemperator flow nonnegative.
    df["5C0002FLOW"] = np.maximum(df["5C0002FLOW"], 0)
    #df["FW463"] += 0.265
    #df["FW463"] = np.maximum(df["FW463"], 0)

    # Average the O2 readings
    df["BG3506"] = (df["BG3506A"] + df["BG3506B"])/2.0

    df.drop("BG3506A", axis=1, inplace=True)
    df.drop("BG3506B", axis=1, inplace=True)
    del df_meta["BG3506A"]
    del df_meta["BG3506B"]


    # Add the two main aux cooling water volumetric flows to get total mol flow
    df["CWCOND_MOL"] = (df["CW119"] + df["CW129"])*997000/18.015
    df["CWAUXCOND_MOL"] = df["CW317-COND"]*997000/18.015

    df["PAPERFLOW_MOL"] = np.sqrt(df["5C0002FLOW"])*24.29*0.4816*1000/2.2046/3600/0.018015

    df["gross_mw"] = df["1JT66801S"]*1e-6
    df_meta["CWCOND_MOL"] = {
        "description": "Condenser cooling water mole flow",
        "reference_string": "m.fs_main.fs_stc.condenser.inlet_2.flow_mol[:]",
        "units": "mol/s",
    }
    df_meta["CWAUXCOND_MOL"] = {
        "description": "Aux condenser cooling water mole flow",
        "reference_string": "m.fs_main.fs_stc.aux_condenser.inlet_2.flow_mol[:]",
        "units": "mol/s",
    }
    df_meta["PAPERFLOW_MOL"] = {
        "description": "Paper mill mole flow",
        "reference_string": "m.fs_main.fs_stc.turb.hp_split[14].outlet_3.flow_mol[:]",
        "units": "mol/s",
    }
    df_meta["gross_mw"] = {
        "description": "Gross power in MW",
        "reference_string": None,
        "units": "MW",
    }
    df_meta["BG3506"] = {
        "description": "Dry Flue gas O2 fraction",
        "reference_string": "m.fs_main.fs_blr.aBoiler.fluegas_o2_pct_dry[:]",
        "units": "None",
    }
    mdata.update_metadata_model_references(model, df_meta)

    # bin the data and calculate the standard deviation
    mdata.bin_data(
        df,
        bin_by="gross_mw",
        bin_no="bin_number",
        bin_nom="bin_nominal_mw",
        bin_size=5,
        min_value=85,
        max_value=235)

    bin_stdev = mdata.bin_stdev(df, bin_no="bin_number")

    # drop data from bins where there isn't enough data to calculate stdev
    idx = df.index[~df["bin_number"].isin(bin_stdev)]
    df.drop(idx, axis=0, inplace=True)

    _log.info("Done reading data")
    return df, df_meta, bin_stdev


def tag_key_quantities(model):
    kq = model.kq = {}

    kq["paper_mill_split"] = pyo.Reference(
        model.fs_main.fs_stc.turb.hp_split[14].split_fraction[:, "outlet_3"])

    kq["paper_mill_flow"] = pyo.Reference(
        model.fs_main.fs_stc.turb.hp_split[14].outlet_3.flow_mol[:])

    kq["main_steam_pressure"] = pyo.Reference(
        model.fs_main.fs_stc.turb.inlet_split.mixed_state[:].pressure)

    kq["coal_flow"] = pyo.Reference(
        model.fs_main.fs_blr.aBoiler.flowrate_coal_raw[:])

    kq["throttle_pressure"] = pyo.Reference(
        model.fs_main.fs_stc.turb.throttle_valve[1].control_volume.properties_out[:].pressure)

    kq["throttle_opening"] = pyo.Reference(
        model.fs_main.fs_stc.turb.throttle_valve[1].valve_opening[:])

    kq["cw_flow"] = pyo.Reference(
        model.fs_main.fs_stc.condenser.tube.properties_in[:].flow_mol)

    kq["cw_enth"] = pyo.Reference(
        model.fs_main.fs_stc.condenser.tube.properties_in[:].enth_mol)

    kq["cw_temperature"] = pyo.Reference(
        model.fs_main.fs_stc.cw_temperature[:])

    kq["cw_aux_flow"] = pyo.Reference(
        model.fs_main.fs_stc.aux_condenser.tube.properties_in[:].flow_mol)

    kq["cw_aux_enth"] = pyo.Reference(
        model.fs_main.fs_stc.aux_condenser.tube.properties_in[:].enth_mol)

    kq["cw_aux_temperature"] = pyo.Reference(
        model.fs_main.fs_stc.cw_aux_temperature[:])

    kq["bfpt_valve_opening"] = pyo.Reference(
        model.fs_main.fs_stc.bfp_turb_valve.valve_opening[:])

    kq["main_steam_temperature"] = pyo.Reference(
        model.fs_main.fs_stc.temperature_main_steam[:])

    kq["hot_reheat_temperature"] = pyo.Reference(
        model.fs_main.fs_stc.temperature_hot_reheat[:])

    kq["spray_valve_opening"] = pyo.Reference(
        model.fs_main.fs_stc.spray_valve.valve_opening[:])

    kq["moisture_content"] = pyo.Reference(
        model.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[:])

    kq["condenser_pressure"] = pyo.Reference(
        model.fs_main.fs_stc.condenser.shell.properties_out[:].pressure)

    kq["aux_condenser_pressure"] = pyo.Reference(
        model.fs_main.fs_stc.aux_condenser.shell.properties_out[:].pressure)

    kq["num_mills"] = pyo.Reference(
        model.fs_main.fs_blr.num_mills)

def add_data_params(model, df, df_meta):
    data_tags = []
    for t, d in df_meta.items():
        if d.get("reference", None) is not None:
            data_tags.append(t)
    idict = {}
    for t in data_tags:
        idict[t] = float(df.iloc[0][t])
    model.data_param = pyo.Param(data_tags, mutable=True, initialize=idict)
    model.data_bin = int(df.iloc[0]["bin_number"])

def set_data_params(model, df, index=None, index_index=None):
    if index is not None:
        model.data_bin = int(df.loc[index]["bin_number"])
        model.fs_main.fs_blr.num_mills = int(df.loc[index]["num_mills"])
        for tag in model.data_param:
            model.data_param[tag] = float(df.loc[index][tag])
    if index_index is not None:
        model.data_bin = int(df.iloc[index_index]["bin_number"])
        model.fs_main.fs_blr.num_mills = int(df.iloc[index_index]["num_mills"])
        for tag in model.data_param:
            model.data_param[tag] = float(df.iloc[index_index][tag])


def add_data_match_obj(model, df_meta, bin_stdev):
    @model.Expression(model.data_param.index_set())
    def err_abs(b, tag):
        return  df_meta[tag]["reference"][0] - model.data_param[tag]

    @model.Expression(model.data_param.index_set())
    def err_rel(b, tag):
        return  model.err_abs[tag]/model.data_param[tag]

    @model.Expression(model.data_param.index_set())
    def err_pct(b, tag):
        return  model.err_rel[tag]*100

    @model.Expression(model.data_param.index_set())
    def err_stdev(b, tag):
        return  model.err_abs[tag]/bin_stdev[model.data_bin][tag]

    model.obj_weight = pyo.Param(
        model.data_param.index_set(), initialize=1, mutable=True)


    n = len(model.data_param.index_set())
    model.obj_datarec = pyo.Objective(
        expr=pyo.sqrt(sum(model.obj_weight[t]*(model.err_stdev[t])**2 for t in model.data_param)/n))
    model.obj_expr = pyo.Expression(
        expr=pyo.sqrt(sum(model.obj_weight[t]*(model.err_stdev[t])**2 for t in model.data_param)/n))


def drop_unused(
    metafile="data/meta_data.csv",
    infile="data/data.csv",
    outfile="data/data_small.csv",
    time_stamp_file="data/test_set.csv"):

    df = pd.read_csv(infile, parse_dates=True, index_col=0)

    tags = []
    with open(metafile, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            tag = line[0].strip()
            tags.append(tag)

    if time_stamp_file is not None:
        rows = set(df.index)
        #print(rows)
        test_set = pd.read_csv(time_stamp_file, parse_dates=True, index_col="time")
        test_set=set(test_set.index)
        drop_rows = rows - test_set
        df.drop(drop_rows, axis=0, inplace=True)

    df.sort_values("1JT66801S", ascending=False, inplace=True)
    cols = set(df.columns)
    tags = set(tags)
    drop_tags = cols - tags
    df.drop(drop_tags, axis=1, inplace=True)

    df.to_csv(outfile)

def bin_stdev(df, bin_no, min_data=4):
    """
    Calculate the standard deviation for each column in each bin.

    Args:
        df (pandas.DataFrame): pandas data frame that is a bin number column
        bin_no (str): Column to group by, usually contains bin number
        min_data (int): Minimum number of data points requitred to calculate
            standard deviation for a bin (default=4)

    Returns:
        dict: key is the bin number and the value is a pandas.Serries with column
            standard deviations
    """
    nos = np.unique(df[bin_no])
    res = {}
    for i in nos:
        idx = df.index[df[bin_no] == i]
        if len(idx) < min_data:
            continue
        df2 = df.loc[idx]
        res[i] = df2.std(axis=0)
    return res


def bin_mean(df, bin_no, min_data=4):
    """
    Calculate the standard deviation for each column in each bin.

    Args:
        df (pandas.DataFrame): pandas data frame that is a bin number column
        bin_no (str): Column to group by, usually contains bin number
        min_data (int): Minimum number of data points requitred to calculate
            standard deviation for a bin (default=4)

    Returns:
        dict: key is the bin number and the value is a pandas.Serries with column
            standard deviations
    """
    nos = np.unique(df[bin_no])
    res = {}
    for i in nos:
        idx = df.index[df[bin_no] == i]
        if len(idx) < min_data:
            continue
        df2 = df.loc[idx]
        res[i] = df2.mean(axis=0)
    return res

def data_plot_book(
        df,
        bin_nom,
        file="data_plot_book.pdf",
        tmp_dir="tmp_plots",
        xlabel=None,
        metadata=None,
        cols=None,
        skip_cols=[],
        point_indexs=[],
        results_df=None,):
    """
    Make box and whisker plots from process data based on bins from the
    bin_data() function.

    Args:
        df: data frame
        bin_nom: bin mid-point value column
        file: path for generated pdf
        tmp_dir: a directory to store temporary plots in
        xlabel: Label for x axis
        metadata: tag meta data dictionary

    Return:
        None

    """
    if sns is None:
        _log.error(
            "Plotting data requires the 'seaborn' and 'PyPDF2' packages. "
            "Install the required packages before using the data_book() function. "
            "Plot terminated."
        )
        return

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    pdfs = []
    flierprops = dict(markerfacecolor='0.5', markersize=2, marker="o", linestyle='none')
    f = plt.figure(figsize=(16, 9))
    if cols is None:
        cols = sorted(df.columns)
    else:
        cols = sorted(cols)

    f = plt.figure(figsize=(16, 9))
    ax = sns.countplot(x=bin_nom, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fname = os.path.join(tmp_dir, "plot_hist.pdf")
    f.savefig(fname, bbox_inches='tight')
    pdfs.append(fname)
    plt.close(f)

    if results_df is not None:
        print(results_df.index)
        for i in results_df.index.values:
            print(i)
            results_df.loc[i, bin_nom] = df.iloc[i][bin_nom]
    print(results_df[bin_nom])

    for i, col in enumerate(cols):
        if col in skip_cols:
            continue
        f = plt.figure(i, figsize=(16, 9))
        ax = sns.boxplot(x=df[bin_nom], y=df[col], flierprops=flierprops)
        pts = df.loc[point_indexs]
        sns.stripplot(x=bin_nom, y=col, data=pts, size=10, color="black", jitter=False)
        if results_df is not None and col + "_model" in results_df.columns:
            print(col)
            sns.stripplot(x=bin_nom, y=col+"_model", data=results_df, size=10, color="black", jitter=False, marker="s")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if metadata is not None:
            md = metadata.get(col, {})
            yl = "{} {} [{}]".format(
                col,
                md.get("description", ""),
                md.get("units", "none")
            )
        else:
            yl = col
        fname = os.path.join(tmp_dir, f"plot_{i}.pdf")
        ax.set(xlabel=xlabel, ylabel=yl)
        f.savefig(fname, bbox_inches='tight')
        pdfs.append(fname)
        plt.close(f)

    # Combine pdfs into one multi-page document
    writer = PdfFileMerger()
    for pdf in pdfs:
        writer.append(pdf)
    writer.write(file)
    _log.info(f"Plot written to {file}.")
