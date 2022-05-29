import pandas as pd
import matplotlib.pyplot as plt
import csv
from PyPDF2 import PdfFileMerger

def get_meta(filename):
    metadata = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            tag = line[0].strip()
            metadata[tag] = {
                "reference_string": line[1].strip(),
                "reference": None,
                "description": line[2].strip(),
                "units": line[3].strip(),
                }
    return metadata

pretty_labels = {
    "1JT66801S": ("Selected Power (MW)", 1e-6),
    "1FWS-STMFLW-A": ("Main Steam Flow (kg/s)", 1),
    "1FWS-FWFLW-A": ("Main Feedwater Flow (kg/s)", 1),
    "1LDC-GROSS-MW": ("Gross Power (MW)", 1e-6),
    "S11": ("Condenser Pressure kPa", 1e-3),
    "C19": ("Condenser Temperature (K)", 1),
    "C105": ("FWH1 Outlet Temperature (K)", 1),
    "C205": ("FWH2 Outlet Temperature (K)", 1),
    "C305": ("FWH3 Outlet Temperature (K)", 1),
    "C305": ("FWH3 Outlet Temperature (K)", 1),
    "C411": ("Deaerator Drain Temperature (K)", 1),
    "C51": ("Condesate Pump Outlet Pressure kPa", 1e-3),
    "CW213A": ("Condenser CW Inlet Temperature (K)", 1),
    "CW216A": ("Condenser CW Outlet Temperature (K)", 1),
    "CW313": ("Aux. Condenser CW Inlet Temperature (K)", 1),
    "CW316": ("Aux. Condenser CW Outlet Temperature (K)", 1),
    "CW317-COND": ("Aux. Condenser Volumetric Flow (m^3/s)", 1),
    "D207": ("FWH2 Drips Temperature (K)", 1),
    "D607": ("FWH2 Drips Temperature (K)", 1),
    "DGS3": ("Coal Flow (kg/s)", 1),
    "FW401": ("FW Booster Pump Suction Temperature (K)", 1),
    "FW402": ("FW Booster Pump Suction Pressure (kPa)", 1e-3),
    "FW413": ("FW Booster Pump Discharge Pressure (kPa)", 1e-3),
    "FW451": ("Feedwater Pressure (kPa)", 1e-3),
    "FW452": ("Feedwater Temperature (K)", 1),
    "FW463": ("Attemperator Water Flow (kg/s)", 1),
    "FW606": ("FWH6 Outlet Temperature (K)", 1),
    "FW611": ("Economizer Inlet Pressure (kPa)", 1e-3),
    "PERFTEST-D508": ("FWH5 Drain Temperature (K)", 1),
    "PERFTEST-D608": ("FWH6 Drain Temperature (K)", 1),
    "PERFTEST-FW503": ("FWH5 Inlet Temperature (K)", 1),
    "PERFTEST-FW603": ("FWH6 Inlet Temperature (K)", 1),
    "PERFTEST-S633": ("Hot Reheat Pressure (kPa)", 1e-3),
    "PERFTEST-S501": ("FWH5 Extracted Steam Pressure (kPa)", 1e-3),
    "S105": ("FWH1 Shell Pressure (kPa)", 1e-3),
    "S202": ("FWH2 Extracted Steam Temperature (K)", 1),
    "S212": ("FWH2 Extracted Steam Pressure (kPa)", 1e-3),
    "S302": ("FWH3 Extracted Steam Temperature (K)", 1),
    "S312": ("FWH3 Extracted Steam Pressure (kPa)", 1e-3),
    "S412": ("Deaerator Pressure (kPa)", 1e-3),
    "S422": ("Inlet Steam BFP Turbine Pressure (kPa)", 1e-3),
    "S423": ("Inlet Steam BFP Turbine Temperature (K)", 1),
    "S613": ("Cold Reheat Pressure (kPa)", 1e-3),
    "S624": ("Cold Reheat Temperature A (K)", 1),
    "S625": ("Cold Reheat Temperature B (K)", 1),
    "S633": ("Hot Reheat Pressure (kPa)", 1e-3),
    "S634": ("Hot Reheat Temperature (K)", 1),
    "S640": ("Hot Reheat Pressure (kPa)", 1e-3),
    "S831": ("Main Steam Temperature (K)", 1),
    "S833": ("Main Steam Pressure (kPa)", 1e-3),
    "S834": ("Turbine Feed Steam Temperature (K)", 1),
    "S839C": ("Turbine Feed Steam Pressure (kPa)", 1e-3),
    "BG3506A": ("Flue Gas O2 A (%)", 1),
    "BG3506B": ("Flue Gas O2 B (%)", 1),
    "BA1T": ("Total Air Flow (kg/s)", 1),
    "1FUE-TOTPA-FLOW": ("Primary Air Flow (kg/s)", 1),
    "S801": ("Drum Pressure (kPa)", 1e-3),
    "S804": ("Drum Temperature (K)", 1),
    "S811": ("Primary Superheater Outlet Temperature (K)", 1),
    "BA2052": ("Air Heater 1A Gas Inlet Temperature (K)", 1),
    "BA2552": ("Air Heater 1B Gas Inlet Temperature (K)", 1),
    "BA2051": ("Air Heater 1A Gas Inlet Pressure (kPa)", 1e-3),
    "BA2551": ("Air Heater 1B Gas Inlet Pressure (kPa)", 1e-3),
    "BA2054": ("Air Heater 1A Gas Outlet Temperature (K)", 1),
    "BA2554": ("Air Heater 1B Gas Outlet Temperature (K)", 1),
    "BA2053": ("Air Heater 1A Gas Outlet Pressure (kPa)", 1e-3),
    "BA2553": ("Air Heater 1B Gas Outlet Pressure (kPa)", 1e-3),
    "BF2002": ("Air Heater 1A PA Inlet Temperature (K)", 1),
    "BF2502": ("Air Heater 1B PA Inlet Temperature (K)", 1),
    "BF2001": ("Air Heater 1A PA Outlet Pressure (kPa)", 1e-3),
    "BF2501": ("Air Heater 1B PA Outlet Pressure (kPa)", 1e-3),
    "BF2003": ("Air Heater 1A PA Outlet Temperature (K)", 1),
    "BF2503": ("Air Heater 1B PA Outlet Temperature (K)", 1),
    "BA2007": ("Air Heater 1A SA Inlet Temperature (K)", 1),
    "BA2507": ("Air Heater 1B SA Inlet Temperature (K)", 1),
    "BA1003": ("FD Fan Outlet Pressure (kPa)", 1e-3),
    "BA1503": ("FD Fan 1B Gas Outlet Pressure (kPa)", 1e-3),
    "BA2004": ("Air Heater 1A SA Outlet Temperature (K)", 1),
    "BA2504": ("Air Heater 1A SA Outlet Temperature (K)", 1),
    "BA2003": ("Air Heater 1B SA Outlet Pressure (kPa)", 1e-3),
    "BA2503": ("Air Heater 1B SA Outlet Pressure (kPa)", 1e-3),
    "BG3502": ("Superheater Gas Outlet Pressure (kPa)", 1e-3),
    "BG3503": ("Reheater Gas Outlet Pressure (kPa)", 1e-3),
    "BG3505": ("Economizer Gas Inlet Pressure (kPa)", 1e-3),
    "FW711": ("Economizer Water Outlet Temperature A (K)", 1),
    "FW712": ("Economizer Water Outlet Temperature B (K)", 1),
}

if __name__ == "__main__":
    df = pd.read_csv("results/validate.csv")
    df_meta = get_meta("data/meta_data.csv")

    df.drop(df.index[df["status"] != "optimal - Optimal Solution Found"], axis=0, inplace=True)
    plots = []
    rows = []
    for col in df.columns:
        if col.endswith("_data"):
            tag = col[:-5]
        else:
            continue
        if not tag + "_model" in df.columns:
            continue

        f = plt.figure(0, figsize=(9, 9))
        plt.rcParams.update({'font.size': 22})
        if tag in df_meta:
            title = df_meta[tag]["description"]
        else:
            title = "No Description"
        pretty = pretty_labels.get(tag, [tag])[0]
        sc = pretty_labels.get(tag, [1, 1])[1]

        try:
            nrmse = ((df[tag+"_data"] - df[tag+"_model"])**2).mean()**0.5 / df[tag+"_data"].mean()
            rows.append([pretty, tag, nrmse, nrmse*100])
        except:
            pass

        ax = plt.scatter(
            x=df[tag+"_data"]*sc,
            y=df[tag+"_model"]*sc,
            c="black",)
        plt.xlabel(pretty + " Data")
        plt.ylabel(pretty + " Model")
        #plt.title(title)
        plt.plot(
            [min(df[tag+"_data"])*sc, max(df[tag+"_data"])*sc],
            [min(df[tag+"_data"])*sc, max(df[tag+"_data"])*sc],
            c="green",
            label="0% Error")
        plt.plot(
            [min(df[tag+"_data"])*sc, max(df[tag+"_data"])*sc],
            [min(df[tag+"_data"])*1.025*sc, max(df[tag+"_data"])*1.025*sc],
            c="blue",
            linestyle="--",
            label="2.5% Error")
        plt.plot(
            [min(df[tag+"_data"])*sc, max(df[tag+"_data"])*sc],
            [min(df[tag+"_data"])*0.975*sc, max(df[tag+"_data"])*0.975*sc],
            c="blue",
            linestyle="--")
        plt.plot(
            [min(df[tag+"_data"])*sc, max(df[tag+"_data"])*sc],
            [min(df[tag+"_data"])*1.05*sc, max(df[tag+"_data"])*1.05*sc],
            c="red",
            label="5% Error",
            linestyle="--",)
        plt.plot(
            [min(df[tag+"_data"])*sc, max(df[tag+"_data"])*sc],
            [min(df[tag+"_data"])*0.95*sc, max(df[tag+"_data"])*0.95*sc],
            c="red",
            linestyle="--",)
        plt.tight_layout()
        plots.append(f"tmp\plt_pair_{tag}.pdf")
        #plt.subplots_adjust(left=0.15)
        plt.legend()
        plt.savefig(plots[-1])
        plt.close()
        f = plt.figure(0, figsize=(9, 9))
        ax = plt.scatter(
            x=df["1LDC-GROSS-MW_data"]*1e-6,
            y=df[tag+"_model"]*sc,
            c="blue",
            label="Model")
        plt.scatter(
            x=df["1LDC-GROSS-MW_data"]*1e-6,
            y=df[tag+"_data"]*sc,
            c="green",
            label="Data")
        plt.xlabel("Gross Power Data (MW)")
        plt.ylabel(pretty)
        plt.title(title)
        plt.tight_layout()
        plots.append(f"tmp\plt_load_{tag}.pdf")
        plt.legend()
        plt.savefig(plots[-1])
        plt.close()

    writer = PdfFileMerger()
    for pdf in plots:
        writer.append(pdf)
    writer.write("val_plots.pdf")

    with open("val_nrmse.csv", "w", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
