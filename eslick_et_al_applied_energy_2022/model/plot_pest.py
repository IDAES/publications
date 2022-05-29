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

if __name__ == "__main__":
    df = pd.read_csv("results\pest2.csv")
    df_meta = get_meta("data\meta_data.csv")

    df.drop(df.index[df["status"] != "optimal - Optimal Solution Found"], axis=0, inplace=True)
    plots = []
    for col in df.columns:
        if col.endswith("_data"):
            tag = col[:-5]
        else:
            continue
        if not tag + "_model" in df.columns:
            continue
        f = plt.figure(0, figsize=(9, 9))
        if tag in df_meta:
            title = df_meta[tag]["description"]
        else:
            title = "No Description"
        ax = df.plot(
            x=tag+"_data",
            y=tag+"_model",
            kind="scatter",
            xlabel=tag + " Data",
            ylabel=tag + " Model",
            title=title,
            c="black",)
        df_temp = pd.DataFrame()
        df_temp["data"] = [min(df[tag+"_data"]), max(df[tag+"_data"])]
        df_temp["model"] = [min(df[tag+"_data"]), max(df[tag+"_data"])]
        df_temp.plot(
            x="data",
            y="model",
            ax=ax,
            c="green",
            xlabel=tag + " Data",
            ylabel=tag + " Model",
            legend=False,)
        plots.append(f"tmp\plt_{tag}.pdf")
        plt.savefig(plots[-1])
        plt.close()

    for col in df.columns:
        if col.endswith("_data") or col.endswith("_model"):
            continue
        if col == "time" or col == "status":
            continue
        f = plt.figure(0, figsize=(9, 9))
        ax = df.plot(
            x="1LDC-GROSS-MW_data",
            y=col,
            kind="scatter",
            xlabel="Gross Power (W)",
            ylabel=col,
            c="black",)
        median = df[col].median()
        mean = df[col].mean()
        print(f"{col}, mean={mean}, median={median}")
        df_temp["mean"] = [mean, mean]
        df_temp["median"] = [median, median]
        df_temp["pow"] = [min(df["1LDC-GROSS-MW_data"]), max(df["1LDC-GROSS-MW_data"])]
        df_temp.plot(
            x="pow",
            y="mean",
            c ="green",
            ax=ax
        )
        df_temp.plot(
            x="pow",
            y="median",
            c ="blue",
            ax=ax
        )
        plots.append(f"tmp\plt_{col}.pdf")
        plt.savefig(plots[-1])
        plt.close()

    writer = PdfFileMerger()
    for pdf in plots:
        writer.append(pdf)
    writer.write("pest_plots.pdf")
