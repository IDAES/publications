import os
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

def dicopt_solve(m, tee=True, solver_options=None):
    if solver_options is None:
        solver_options = {}
    if not os.path.exists('temp'):
        os.makedirs('temp')
    with open(os.path.join("temp", "dicopt.opt"), "w") as f:
        for k, v in solver_options.items():
            f.write(f"{k} {v} \n")
    return pyo.SolverFactory("gams").solve(
        m, 
        tee=tee, 
        keepfiles=True, 
        solver="dicopt", 
        tmpdir='temp', 
        add_options=['gams_model.optfile=1;']
    )


def print_simple_model_summary(m):
    print(f"Objective = {pyo.value(-m.obj)}")
    try:
        print(f"Time in hydrogen mode = {pyo.value(sum(m.bin_hydrogen[i] for i in m.bin_hydrogen))/len(m.bin_off):.2%}")
    except AttributeError:
        pass
    try:
        print(f"Time in power mode = {pyo.value(sum(m.bin_power[i] for i in m.bin_power))/len(m.bin_off):.2%}")
    except AttributeError:
        pass
    print(f"Time in off mode = {pyo.value(sum(m.bin_off[i] for i in m.bin_off))/len(m.bin_off):.2%}")


def get_dataframe():
    return pd.DataFrame(
        columns=[
            "lmp ($/MWh)",
            "profit ($/hr)",
            "net_power (MW)",
            "h_prod (kg/s)",
            "ng_price ($/million BTU)",
            "h2_price ($/kg)",
            "el_revenue ($/hr)",
            "el_cost ($/hr)",
            "ng_cost ($/hr)",
            "h2_revenue ($/hr)",
            "other_cost ($/hr)",
            "fixed_cost ($/hr)",
            "mode_power_only",
            "mode_hydrogen_only",
            "mode_hydrogen",
            "mode_off",
            "start_cost",
            "stop_cost",
        ]
    )
    

def add_data_row(df, model, i, mode):
    idx = len(df.index)
    
    ng_price = pyo.value(model.ng_price)
    h2_price = pyo.value(model.h2_price)
    start_cost = pyo.value(model.start_cost)
    stop_cost = pyo.value(model.stop_cost)
    
    try: 
        el_price = pyo.value(model.pblk[i].el_price)
    except AttributeError:
        el_price = pyo.value(model.hblk[i].el_price)
    
    df.loc[idx, "lmp ($/MWh)"] = el_price
    df.loc[idx, "ng_price ($/million BTU)"] = ng_price
    df.loc[idx, "h2_price ($/kg)"] = h2_price
    df.loc[idx, "mode_power_only"] = 0
    df.loc[idx, "mode_hydrogen_only"] = 0 
    df.loc[idx, "mode_hydrogen"] = 0
    df.loc[idx, "mode_off"] = 0
    df.loc[idx, "start_cost"] = 0
    df.loc[idx, "stop_cost"] = 0
    if hasattr(model, "start"):
        if pyo.value(model.start[i]) > 0.99:
            df.loc[idx, "start_cost"] = start_cost
    if hasattr(model, "stop"):
        if pyo.value(model.stop[i]) > 0.99:
             df.loc[idx, "stop_cost"] = stop_cost
    
    if mode == "off":
        df.loc[idx, "mode_off"] = 1
        df.loc[idx, "h_prod (kg/s)"] = 0
        df.loc[idx, "net_power (MW)"] = 0
        df.loc[idx, "el_revenue ($/hr)"] = 0
        df.loc[idx, "el_cost ($/hr)"] = 0
        df.loc[idx, "ng_cost ($/hr)"] = 0
        df.loc[idx, "h2_revenue ($/hr)"] = 0
        df.loc[idx, "other_cost ($/hr)"] = 0
        try: 
            df.loc[idx, "fixed_cost ($/hr)"] = pyo.value(model.pblk[i].fixed_costs)
            df.loc[idx, "profit ($/hr)"] = pyo.value(-model.pblk[i].fixed_costs)
        except AttributeError:
            df.loc[idx, "fixed_cost ($/hr)"] = pyo.value(-model.hblk[i].fixed_costs)
            df.loc[idx, "profit ($/hr)"] = pyo.value(-model.hblk[i].fixed_costs)
    elif mode == "hydrogen":
        m = model.hblk[i]
        df.loc[idx, "profit ($/hr)"] = pyo.value(m.profit)
        df.loc[idx, "other_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["other_var_cost"])
        df.loc[idx, "fixed_cost ($/hr)"] = pyo.value(m.fixed_costs)
        df.loc[idx, "mode_hydrogen"] = 1
        df.loc[idx, "net_power (MW)"] = pyo.value(m.net_power)
        df.loc[idx, "h_prod (kg/s)"] = pyo.value(m.h_prod)
        df.loc[idx, "el_revenue ($/hr)"] = pyo.value(m.net_power*m.el_price)
        df.loc[idx, "el_cost ($/hr)"] = pyo.value(-m.net_power*m.el_price)
        df.loc[idx, "h2_revenue ($/hr)"] = pyo.value(m.h_prod*m.h2_price)
        df.loc[idx, "ng_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["fuel_var_cost"])
    elif mode == "hydrogen_only":
        m = model.hblk[i]
        df.loc[idx, "profit ($/hr)"] = pyo.value(m.profit)
        df.loc[idx, "other_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["other_var_cost"])
        df.loc[idx, "fixed_cost ($/hr)"] = pyo.value(m.fixed_costs)
        df.loc[idx, "mode_hydrogen_only"] = 1
        df.loc[idx, "net_power (MW)"] = pyo.value(-m.surrogate.alamo_expression["elec_var_cost"]/30.0)
        df.loc[idx, "h_prod (kg/s)"] = pyo.value(m.h_prod)
        df.loc[idx, "el_revenue ($/hr)"] = pyo.value(-m.surrogate.alamo_expression["elec_var_cost"]/30.0*m.el_price)
        df.loc[idx, "el_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["elec_var_cost"]/30.0*m.el_price)
        df.loc[idx, "h2_revenue ($/hr)"] = pyo.value(m.h_prod*m.h2_price)
        df.loc[idx, "ng_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["fuel_var_cost"])
    elif mode == "power_only":
        m = model.pblk[i]
        df.loc[idx, "profit ($/hr)"] = pyo.value(m.profit)
        df.loc[idx, "other_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["other_var_cost"])
        df.loc[idx, "fixed_cost ($/hr)"] = pyo.value(m.fixed_costs)
        df.loc[idx, "mode_power_only"] = 1
        df.loc[idx, "net_power (MW)"] = pyo.value(m.net_power)
        df.loc[idx, "h_prod (kg/s)"] = 0
        df.loc[idx, "el_revenue ($/hr)"] = pyo.value(m.net_power*m.el_price)
        df.loc[idx, "el_cost ($/hr)"] = pyo.value(-m.net_power*m.el_price)
        df.loc[idx, "h2_revenue ($/hr)"] = 0
        df.loc[idx, "ng_cost ($/hr)"] = pyo.value(m.surrogate.alamo_expression["fuel_var_cost"])


def write_result_csv(m, model_name, data_set, path="../market_results/"):
    if not os.path.exists(path):
        os.mkdir(path)
    
    fname = os.path.join(path, f"{data_set}_{model_name}.csv")
    df = get_dataframe()
    for i in m.bin_off:
        if pyo.value(m.bin_off[i] > 0.99):
            mode = "off"
        else:
            if hasattr(m, "bin_power") and pyo.value(m.bin_power[i] > 0.99):
                mode = "power_only"
            elif hasattr(m, "bin_hydrogen") and pyo.value(m.bin_hydrogen[i] > 0.99):
                if "elec_var_cost" in m.hblk[i].surrogate.alamo_expression:
                    mode = "hydrogen_only"
                else:
                    mode = "hydrogen"
        add_data_row(df, m, i, mode)

    df.to_csv(fname)
    return df


def print_summary(df, model, data_set):
    n = len(df["profit ($/hr)"])
    profit = (sum(df["profit ($/hr)"]) - sum(df["start_cost"]) - sum(df["stop_cost"]))/n*365*24/1e6
    total_power = sum(df["net_power (MW)"])/n*365*24/1e6
    total_h2 = sum(df["h_prod (kg/s)"])/n*365*24/1e6*3600
    pct_time_off = sum(df["mode_off"])/n * 100
    pct_time_power_only = sum(df["mode_power_only"])/n * 100
    pct_time_hydrogen = sum(df["mode_hydrogen"])/n * 100
    pct_time_hydrogen_only = sum(df["mode_hydrogen_only"])/n * 100

    print(f"Model: {model}")
    print(f"LMP Data: {data_set}")
    print(f"Annual Profit = {profit} Million $/yr)")
    print(f"Annual Power = {total_power} Million MWh")
    print(f"Annual Hydrogen = {total_h2} Million kg")
    print(f"Time power only = {pct_time_power_only}%")
    print(f"Time hydrogen/power = {pct_time_hydrogen}%")
    print(f"Time hydrogen only = {pct_time_hydrogen_only}%")
    print(f"Time off = {pct_time_off}%")


def profit_hist(dat, range=(None, None), nbins=20):
    if range[0] is None:
        range = (min(dat/1000), range[1])
    if range[1] is None:
        range = (range[0], max(dat/1000))
    plt.hist(dat/1000, nbins, weights=np.ones(len(dat)) / len(dat), range=range)
    plt.xlabel("Net profit ($1000/hr)")
    plt.ylabel("Portion of time in bin")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()
    
def power_hist(dat, range=(None, None), nbins=20):
    if range[0] is None:
        range = (min(dat), range[1])
    if range[1] is None:
        range = (range[0], max(dat))
    plt.hist(dat, nbins, weights=np.ones(len(dat)) / len(dat), range=range)
    plt.xlabel("Net Power (MW)")
    plt.ylabel("Portion of time in bin")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

def h2_hist(dat, range=(None, None), nbins=20):
    if range[0] is None:
        range = (min(dat), range[1])
    if range[1] is None:
        range = (range[0], max(dat))
    plt.hist(dat, 20, weights=np.ones(len(dat)) / len(dat), range=range)
    plt.xlabel("Hydrogen Production (kg/hr)")
    plt.ylabel("Portion of time in bin")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()
    
def lmp_hist(dat, range=(None, None), nbins=20):
    if range[0] is None:
        range = (min(dat), range[1])
    if range[1] is None:
        range = (range[0], max(dat))
    plt.hist(dat, 20, weights=np.ones(len(dat)) / len(dat), range=range)
    plt.xlabel("LMP ($/MWh)")
    plt.ylabel("Portion of time in bin")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))


def strip_chart_power_and_hydrogen(df, start_day=125, ndays=20):
    fig = plt.figure(figsize=(18,4))
    ax1 = host_subplot(111, axes_class=AA.Axes)
    ax1.set_ylabel("LMP ($/MWh)")
    plt.subplots_adjust(right=0.75)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax2.axis["right"] = ax3.get_grid_helper().new_fixed_axis(loc="right", axes=ax2,offset=(0, 0))
    ax2.set_ylabel("Power (MW)")
    ax3.axis["right"] = ax3.get_grid_helper().new_fixed_axis(loc="right", axes=ax3,offset=(70, 0))
    ax3.set_ylabel("Hydrogen (kg/s)")
    ax3.set_ylim(-0.25, 5.25)

    days = ndays
    day1 = start_day

    s1 = ax1.plot([i/24 for i in range(days*24)], df["lmp ($/MWh)"][day1*24:days*24 + day1*24], label="LMP", color="black")
    s2 = ax2.plot([i/24 for i in range(days*24)], df["net_power (MW)"][day1*24:days*24+day1*24], label="power", color="g")
    s3 = ax3.plot([i/24 for i in range(days*24)], df["h_prod (kg/s)"][day1*24:days*24+day1*24], label="power", color="r")
    plt.xlabel("Day")
    s = s1 + s2 + s3
    ax1.legend(s, ["LMP", "power", "hydrogen"], loc=(0.01, 0.01))
    plt.show()
    return fig

def strip_chart_power(df, start_day=125, ndays=20):
    fig, ax1 = plt.subplots(figsize=(18,4))
    ax1.set_ylabel("LMP ($/MWh)")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Power (MW)")

    days = ndays
    day1 = start_day

    s1 = ax1.plot([i/24 for i in range(days*24)], df["lmp ($/MWh)"][day1*24:days*24 + day1*24], label="LMP", color="black")
    s2 = ax2.plot([i/24 for i in range(days*24)], df["net_power (MW)"][day1*24:days*24+day1*24], label="power", color="g")
    plt.xlabel("Day")
    s = s1 + s2
    ax1.legend(s, ["LMP", "power"], loc=(0.01, 0.01))
    plt.show()
    return fig

def strip_chart_hydrogen(df, start_day=125, ndays=20):
    fig, ax1 = plt.subplots(figsize=(18,4))
    ax1.set_ylabel("LMP ($/MWh)")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Hydrogen (kg/s)")

    days = ndays
    day1 = start_day

    s1 = ax1.plot([i/24 for i in range(days*24)], df["lmp ($/MWh)"][day1*24:days*24 + day1*24], label="LMP", color="black")
    s2 = ax2.plot([i/24 for i in range(days*24)], df["h_prod (kg/s)"][day1*24:days*24+day1*24], label="hydrogen", color="g")
    plt.xlabel("Day")
    s = s1 + s2
    ax1.legend(s, ["LMP", "hydrogen"], loc=(0.01, 0.01))
    plt.show()
    return fig