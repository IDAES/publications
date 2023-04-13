import pandas as pd
import numpy as np
import json
import os
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def get_model_data(path="../data/"):
    """Get data for models, such as description and fixed costs.
    """
    model_data_file = os.path.join(path, "model_data.json")
    with open(model_data_file, "r") as f:
        model_data = json.load(f)
    for model in model_data:
        omh = model_data[model]["fixed_om"] * 1e6 / 365 / 24
        caph = model_data[model]["fixed_cap"] * 1e6 / 365 / 24
        model_data[model]["fixed_om_hourly"] = omh
        model_data[model]["fixed_cap_hourly"] = caph
    return model_data


class DataObject(object):
    def __init__(self, path="../data/"):
        self.data, self.metadata = self.read_data(path=path)

    @staticmethod
    def read_data(path="../data/"):
        data_file = os.path.join(path, "lmp_data.csv")
        metadata_file = os.path.join(path, "lmp_metadata.json")
        model_data_file = os.path.join(path, "model_data.json")

        data = pd.read_csv(data_file)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        for col in metadata:
            metadata[col]["ndata"] = int(data[col].count())
            metadata[col]["lmp_mean"] = float(data[col].mean())
            metadata[col]["lmp_median"] = float(data[col].median())

        return data, metadata

    def histogram(self, col, fname=None, range=(None, None), nbins=40, dpi=160):
        fig, ax = plt.subplots()
        dat = self.data[col]
        if range[0] is None:
            range = (min(dat), range[1])
        if range[1] is None:
            range = (range[0], max(dat))
        plt.hist(dat, nbins, weights=np.ones(len(dat)) / len(dat), range=range)
        plt.xlabel("LMP ($/MWh)")
        plt.ylabel("Time in bin")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        if fname:
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        else:
            plt.show()


    def smr_cost(self, col, co2_tsm=False, capacity_factor=1.0, capacity=5.0, capacity_scaling_exponent=0.8):
        """ Calculate SMR H2 cost based on the electricity and natural
        gas prices from the LMP data.

        Source:	https://netl.doe.gov/projects/files/ComparisonofCommercialStateofArtFossilBasedHydrogenProductionTechnologies_041222.pdf
        See:	~Exhibit 3-39
        """
        data = self.data
        metadata = self.metadata

        original_electricity_price = 71.7 # $/MWhr
        original_fuel_price = 4.42 # $/MBTU
        original_capacity_factor = 0.9 
        original_capacity = 5.59 # kg/s
        original_fixed_cost = 0.150 # $/kg
        original_capital_cost = 0.330 # $/kg
        original_fuel_cost = 0.822 # $/kg
        original_electricity_cost = 0.145 # $/kg
        original_other_variable_cost = 0.097 # $/kg
        original_total = 1.544 # $/kg
        original_tsm = 0.1 # $/kg
        original_total_with_tsm = 1.644 # $/kg

        fuel_price = metadata[col]["ng_price"]
        electricity_price = metadata[col]["lmp_mean"]

        fixed_scale = (original_capacity_factor/capacity_factor
            * original_capacity/capacity
            * (capacity/original_capacity)**capacity_scaling_exponent
        )
        fixed_cost = original_fixed_cost * fixed_scale
        capital_cost = original_capital_cost * fixed_scale
        fuel_cost = original_fuel_cost * fuel_price / original_fuel_price
        electricity_cost = original_electricity_cost * electricity_price / original_electricity_price
        other_variable_cost = original_other_variable_cost

        total = fixed_cost + capital_cost + fuel_cost + electricity_cost + other_variable_cost

        tsm = original_tsm   

        if co2_tsm:
            total += tsm

        return total # $/kg H2


    def atr_cost(self, col, co2_tsm=False, capacity_factor=1.0, capacity=5.0, capacity_scaling_exponent=0.8):
        """ Calculate ATR H2 cost based on the electricity and natural
        gas prices from the LMP data.

        Source:	https://netl.doe.gov/projects/files/ComparisonofCommercialStateofArtFossilBasedHydrogenProductionTechnologies_041222.pdf	
        See:	~Exhibit 3-49	
        """
        data = self.data
        metadata = self.metadata

        original_electricity_price = 71.7 # $/MWhr
        original_fuel_price = 4.42 # $/MBTU
        original_capacity_factor = 0.9 
        original_capacity = 7.60 # kg/s
        original_fixed_cost = 0.110 # $/kg
        original_capital_cost = 0.260 # $/kg
        original_fuel_cost = 0.772 # $/kg
        original_electricity_cost = 0.287 # $/kg
        original_other_variable_cost = 0.070 # $/kg
        original_total = 1.500 # $/kg
        original_tsm = 0.09 # $/kg
        original_total_with_tsm = 1.59 # $/kg

        fuel_price = metadata[col]["ng_price"]
        electricity_price = metadata[col]["lmp_mean"]

        fixed_scale = (original_capacity_factor/capacity_factor
            * original_capacity/capacity
            * (capacity/original_capacity)**capacity_scaling_exponent
        )
        fixed_cost = original_fixed_cost * fixed_scale
        capital_cost = original_capital_cost * fixed_scale
        fuel_cost = original_fuel_cost * fuel_price / original_fuel_price
        electricity_cost = original_electricity_cost * electricity_price / original_electricity_price
        other_variable_cost = original_other_variable_cost

        total = fixed_cost + capital_cost + fuel_cost + electricity_cost + other_variable_cost

        tsm = original_tsm   

        if co2_tsm:
            total += tsm

        return total # $/kg H2

# Just for Testing
if __name__ == "__main__":
    data = DataObject()
    print(json.dumps(data.metadata, indent=4))
    print(f"SRM Cost: {data.smr_cost('WinterNYTaxCapRes_2030')} $/kg")
    print(f"ATR Cost: {data.atr_cost('WinterNYTaxCapRes_2030')} $/kg")
    data.histogram(fname="test_junk.png", col='WinterNYTaxCapRes_2030')
