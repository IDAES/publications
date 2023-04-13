import os
import pandas as pd
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt

import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core.surrogate.alamopy import AlamoTrainer
from idaes.core.surrogate.sampling import split_training_validation
from idaes.core.surrogate.metrics import compute_fit_metrics
from idaes.core.surrogate.plotting.sm_plotter import surrogate_residual, surrogate_parity
from idaes.core.surrogate.alamopy import AlamoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock

from .data import get_model_data, DataObject

class SurrogateProcessModels(object):
    _xltabs = {
        "model0": {
            "power_only":"ngcc",
            "hydrogen": None,
            "hydrogen_only": None,
        },
        "model1":{
            "power_only": "sofc",
            "hydrogen": None,
            "hydrogen_only": None,
        },
        "model3":{
            "power_only": "ngcc",
            "hydrogen": "ngcc_soec",
            "hydrogen_only": None,
        },
        "model4": {
            "power_only": "sofc",
            "hydrogen_only": "rsofc_soec",
            "hydrogen": None,
        },
        "model5": {
            "power_only": "sofc_soec_power_only", 
            "hydrogen": "sofc_soec",
            "hydrogen_only": None,

        },
        "model6": {
            "hydrogen": None,
            "hydrogen_only": "soec",
            "power_only": None,
        },
    }

    def __init__(self, path="../data/", saved_surrogate_path="../saved_surrogate_models"):
        self.process_data_path = os.path.join(path, "process_model_data.xlsx")
        self.saved_surrogate_path = saved_surrogate_path
        if not os.path.exists(self.saved_surrogate_path):
            os.mkdir(self.saved_surrogate_path)
        self.models = {}
        for model in self._xltabs:
            self.models[model] = {}
            for mode in self._xltabs[model]:
                self.models[model][mode] = None

    def big_m_tables(self, model, data, ng_price, h2_price, ng_price_tax, data_set):
        model_data = get_model_data()
        mtables_min = {}
        mtables_max = {}
        max_el_price = max(data.data[data_set])
        min_el_price = min(data.data[data_set])
        el_prices = list(np.linspace(min_el_price, max_el_price, 2000))
        solver_obj = pyo.SolverFactory("ipopt")
        mtables_min[model] = {}
        mtables_max[model] = {}
        for mode in model_data[model]["modes"]:
            mtables_min[model][mode] = [] 
            mtables_max[model][mode] = [] 
            m = self.single_point_model(model, mode, obj=False)
            m.obj_max = pyo.Objective(expr=-m.profit_no_fixed)
            m.obj_min = pyo.Objective(expr=m.profit_no_fixed)
            for el_price in el_prices:
                try:
                    m.ng_price.fix(ng_price)
                except AttributeError:
                    pass
                try:
                    m.ng_price_tax.fix(ng_price_tax)
                except AttributeError:
                    pass
                m.el_price.fix(el_price)
                try:
                    m.h2_price.fix(h2_price)
                except AttributeError:
                    pass
                m.obj_min.deactivate()
                m.obj_max.activate()
                res = solver_obj.solve(m)
                mtables_max[model][mode].append(pyo.value(m.profit_no_fixed))
                m.obj_max.deactivate()
                m.obj_min.activate()
                res = solver_obj.solve(m)
                mtables_min[model][mode].append(pyo.value(m.profit_no_fixed))
        return el_prices, mtables_min, mtables_max

    def single_point_model(self, model, mode, el_price=71.70, ng_price=4.42, h2_price=2.0, ng_price_tax=0.0, blk=None, obj=True):
        model_data = get_model_data()
        if blk is not None:
            m = blk
        else:
            m = pyo.ConcreteModel(f"{model}_{mode}")
        #m.profit = pyo.Var(initialize=0.0)
        m.fixed_costs = pyo.Var()
        m.fixed_costs.fix(model_data[model]["fixed_om_hourly"] + model_data[model]["fixed_cap_hourly"])
        m.surrogate = SurrogateBlock()
        if mode == "power_only":
            m.ng_price = pyo.Var(doc="Natural gas price $/million BTU")
            m.ng_price_tax = pyo.Var(doc="Added natural gas price for CO2 tax $/million BTU")
            m.el_price = pyo.Var(doc="Electricity price $/MWhr")
            m.el_price.fix(el_price)
            m.ng_price.fix(ng_price)
            m.ng_price_tax.fix(ng_price_tax)
            m.net_power = pyo.Var(initialize=650) # net power mw
            #m.fuel_cost = pyo.Var()
            #m.other_cost = pyo.Var()
            #m.total_cost = pyo.Var()
            m.surrogate.build_model(
                self.models[model][mode]["model"],
                input_vars=[m.net_power],
                #output_vars=[m.total_cost, m.fuel_cost, m.other_cost],
                as_expression=True
            )
            m.profit = pyo.Expression(expr=
                m.el_price*m.net_power - 
                m.fixed_costs - 
                (m.ng_price/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                m.surrogate.alamo_expression["other_var_cost"]
            )
            m.profit_no_fixed = pyo.Expression(expr=
                m.el_price*m.net_power - 
                (m.ng_price/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                m.surrogate.alamo_expression["other_var_cost"]
            )
            m.net_power.setub(model_data[model]["power_ub"])
            m.net_power.setlb(model_data[model]["power_lb"])
            if obj:
                m.obj = pyo.Objective(expr=-m.profit)
            return m
        if mode == "hydrogen":
            #m.fuel_cost = pyo.Var()
            #m.other_cost = pyo.Var()
            #m.total_cost = pyo.Var()
            m.ng_price = pyo.Var(doc="Natural gas price $/million BTU")
            m.ng_price_tax = pyo.Var(doc="Added natural gas price for CO2 tax $/million BTU")
            m.el_price = pyo.Var(doc="Electricity price $/MWhr")
            m.h2_price = pyo.Var(doc="Hydrogen price $/kg")
            m.el_price.fix(el_price)
            m.ng_price.fix(ng_price)
            m.h2_price.fix(h2_price)
            m.net_power = pyo.Var(initialize=-60) # net power mw
            m.h_prod = pyo.Var(initialize=3.0)
            m.surrogate.build_model(
                self.models[model][mode]["model"],
                input_vars=[m.net_power, m.h_prod],
                #output_vars=[m.total_cost, m.fuel_cost, m.other_cost]
                as_expression=True
            )
            m.profit = pyo.Expression(expr=
                (
                    m.el_price * m.net_power + 
                    m.h2_price * m.h_prod * 3600 - 
                    m.fixed_costs - 
                    (m.ng_price/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                    m.surrogate.alamo_expression["other_var_cost"]
                )
            )
            m.profit_no_fixed = pyo.Expression(expr=
                (
                    m.el_price * m.net_power + 
                    m.h2_price * m.h_prod * 3600 - 
                    (m.ng_price/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                    m.surrogate.alamo_expression["other_var_cost"]
                )
            )
            ineq = self.models[model][mode]["linear_hprod_ineq"]
            n_ineq = len(ineq)
            @m.Constraint(pyo.RangeSet(n_ineq))
            def hprod_ineq(b, i):
                a = ineq[i - 1]["slope"]
                b = ineq[i - 1]["intercept"]
                o = ineq[i - 1]["op"]
                if o == ">=":
                    return m.net_power >= a * m.h_prod + b                   
                else:
                    return m.net_power <= a * m.h_prod + b
            m.net_power.setub(model_data[model]["power_with_h2_ub"])
            m.net_power.setlb(model_data[model]["power_with_h2_lb"])
            m.h_prod.setub(model_data[model]["hydrogen_ub"])
            m.h_prod.setlb(model_data[model]["hydrogen_lb"])
            if obj:
                m.obj = pyo.Objective(expr=-m.profit)
            return m
        if mode == "hydrogen_only":
            #m.fuel_cost = pyo.Var()
            #m.other_cost = pyo.Var()
            #m.total_cost = pyo.Var()
            #m.electricity_cost = pyo.Var()
            m.ng_price = pyo.Var(doc="Natural gas price $/million BTU")
            m.ng_price_tax = pyo.Var(doc="Added natural gas price for CO2 tax $/million BTU")
            m.el_price = pyo.Var(doc="Electricity price $/MWhr")
            m.h2_price = pyo.Var(doc="Hydrogen price $/kg")
            m.el_price.fix(el_price)
            m.ng_price.fix(ng_price)
            m.h2_price.fix(h2_price)
            m.h_prod = pyo.Var(initialize=1)
            m.surrogate.build_model(
                self.models[model][mode]["model"],
                input_vars=[m.h_prod],
                #output_vars=[m.total_cost, m.electricity_cost, m.fuel_cost, m.other_cost]
                as_expression=True
            )
            m.profit = pyo.Expression(expr=
                (
                    m.h2_price * m.h_prod * 3600 - 
                    m.fixed_costs - 
                    (m.ng_price/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                    (m.ng_price_tax/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                    (m.el_price/30.0)*m.surrogate.alamo_expression["elec_var_cost"] -
                    m.surrogate.alamo_expression["other_var_cost"]
                )
            )
            m.profit_no_fixed = pyo.Expression(expr=
                (
                    m.h2_price * m.h_prod * 3600 - 
                    (m.ng_price/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                    (m.ng_price_tax/4.42)*m.surrogate.alamo_expression["fuel_var_cost"] - 
                    (m.el_price/30.0)*m.surrogate.alamo_expression["elec_var_cost"] -
                    m.surrogate.alamo_expression["other_var_cost"]
                )
            )
            m.h_prod.setub(model_data[model]["hydrogen_ub"])
            m.h_prod.setlb(model_data[model]["hydrogen_lb"])
            if obj:
                m.obj = pyo.Objective(expr=-m.profit)
            return m

        raise RuntimeError(f"No single point model defined for {model} {mode}")

    def full_model(self, model, data_set, h2_price=2.0):
        modes = {
            "model0":("power_only",),
            "model1":("power_only",),
            "model3":("power_only", "hydrogen"),
            "model4":("power_only", "hydrogen_only"),
            "model5":("power_only", "hydrogen"),
            "model6":("hydrogen_only",),
        }
        m = pyo.ConcreteModel()

        data = DataObject()
        ng_price = m.ng_price = pyo.Param(initialize=data.metadata[data_set]["ng_price"])
        h2_price = m.h2_price = pyo.Param(initialize=h2_price)
        model_data = get_model_data()
        ng_price_tax = m.ng_price_tax = pyo.Param(initialize=data.metadata[data_set][model_data[model]["co2_tax"]])
        lmps = data.data[data_set].dropna()
        m.fixed_costs = pyo.Var()
        m.fixed_costs.fix(model_data[model]["fixed_om_hourly"] + model_data[model]["fixed_cap_hourly"])

        m.Mp_hi = pyo.Param(pyo.RangeSet(len(lmps)), initialize=1e5, mutable=True)
        m.Mh_hi = pyo.Param(pyo.RangeSet(len(lmps)), initialize=1e2, mutable=True)
        m.Mp_lo = pyo.Param(pyo.RangeSet(len(lmps)), initialize=1e5, mutable=True)
        m.Mh_lo = pyo.Param(pyo.RangeSet(len(lmps)), initialize=1e2, mutable=True)

        min_on = m.min_on = pyo.Param(initialize=model_data[model]["min_on"])
        min_off = m.min_off = pyo.Param(initialize=model_data[model]["min_off"])
        start_cost = m.start_cost = pyo.Param(initialize=model_data[model]["start_cost"])
        stop_cost = m.stop_cost = pyo.Param(initialize=model_data[model]["stop_cost"])

        # There are basically 3 possible model configurations
        # power only, hydrogen only, or power and hydrogen. 
        # Hydrogen mode can be hydrogen only or power and hydrogen,
        # but not both. So systems have at most two "on" modes.

        elp, prof_min, prof_max = self.big_m_tables(model, data, ng_price, h2_price, ng_price_tax, data_set)
        if "power_only" in modes[model]:
            m.mp_profit = pyo.Var(pyo.RangeSet(len(lmps)))
            m.pblk = pyo.Block(pyo.RangeSet(len(lmps)))
            for i, b in m.pblk.items():
                self.single_point_model(model, "power_only", ng_price=ng_price, blk=b, obj=False)
                b.ng_price.fix(ng_price)
                b.ng_price_tax.fix(ng_price_tax)
                b.el_price.fix(lmps[i-1])
                f1 = interpolate.interp1d(elp, prof_max[model]["power_only"])
                f2 = interpolate.interp1d(elp, prof_min[model]["power_only"])
                m.Mp_hi[i] = f1(lmps[i-1]) + 0.2*abs(f1(lmps[i-1]))
                m.Mp_lo[i] = f2(lmps[i-1]) - 0.2*abs(f2(lmps[i-1]))
                # add a little space if the Ms happen to be near 0
                if abs(pyo.value(m.Mp_hi[i])) < 4000:
                    m.Mp_hi[i] = 4000
                if abs(pyo.value(m.Mp_lo[i])) < 4000:
                    m.Mp_lo[i] = -4000

            m.bin_power = pyo.Var(pyo.RangeSet(len(lmps)), initialize=1, within=pyo.Binary)
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def p_M1(b, t):
                return m.Mp_lo[t]*(1 - m.bin_power[t])/1000 <= (m.pblk[t].profit_no_fixed - m.mp_profit[t])/1000
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def p_M2(b, t):
                return m.Mp_hi[t]*(1 - m.bin_power[t])/1000 >= (m.pblk[t].profit_no_fixed - m.mp_profit[t])/1000
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def p_M3(b, t):
                return m.Mp_hi[t]*m.bin_power[t]/1000 >= m.mp_profit[t]/1000
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def p_M4(b, t):
                return m.Mp_lo[t]*m.bin_power[t]/1000 <= m.mp_profit[t]/1000
            #@m.Constraint(pyo.RangeSet(len(lmps)))
            #def p_M5(b, t):
            #    return 2000*m.bin_power[t] >= m.pblk[t].net_power - model_data[model]["power_lb"]

        if "hydrogen" in modes[model]: # hydrogen and power
            hmode = "hydrogen"
        elif "hydrogen_only" in modes[model]: # just hydrogen
            hmode = "hydrogen_only"
        else:
            hmode = None
        
        if hmode is not None:
            m.mh_profit = pyo.Var(pyo.RangeSet(len(lmps)))
            m.hblk = pyo.Block(pyo.RangeSet(len(lmps)))
            for i, b in m.hblk.items():
                self.single_point_model(model, hmode, ng_price=ng_price, blk=b, obj=False)
                b.ng_price.fix(ng_price)
                b.h2_price.fix(h2_price)
                b.ng_price_tax.fix(ng_price_tax)
                b.el_price.fix(lmps[i-1])
                f1 = interpolate.interp1d(elp, prof_max[model][hmode])
                f2 = interpolate.interp1d(elp, prof_min[model][hmode])
                m.Mh_hi[i] = f1(lmps[i-1]) + 0.15*abs(f1(lmps[i-1]))
                m.Mh_lo[i] = f2(lmps[i-1]) - 0.15*abs(f2(lmps[i-1]))
                # add a little space if the Ms happen to be near 0
                if abs(pyo.value(m.Mh_hi[i])) < 4000:
                    m.Mh_hi[i] = 4000
                if abs(pyo.value(m.Mh_lo[i])) < 4000:
                    m.Mh_lo[i] = -4000

            m.bin_hydrogen = pyo.Var(pyo.RangeSet(len(lmps)), initialize=0, within=pyo.Binary)
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def h_M1(b, t):
                return m.Mh_lo[t]*(1 - m.bin_hydrogen[t])/1000.0 <= (m.hblk[t].profit_no_fixed - m.mh_profit[t])/1000.0
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def h_M2(b, t):
                return m.Mh_hi[t]*(1 - m.bin_hydrogen[t])/1000.0 >= (m.hblk[t].profit_no_fixed - m.mh_profit[t])/1000.0
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def h_M3(b, t):
                return m.Mh_hi[t]*m.bin_hydrogen[t]/1000.0 >= m.mh_profit[t]/1000.0
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def h_M4(b, t):
                return m.Mh_lo[t]*m.bin_hydrogen[t]/1000.0 <= m.mh_profit[t]/1000.0
            #@m.Constraint(pyo.RangeSet(len(lmps)))
            #def h_M5(b, t):
            #    return 20*m.bin_hydrogen[t] >= m.hblk[t].h_prod - model_data[model]["hydrogen_lb"]

        m.bin_off = pyo.Var(pyo.RangeSet(len(lmps)), initialize=0, within=pyo.Binary)

        @m.Constraint(pyo.RangeSet(len(lmps)))
        def one_mode_constraint(b, i):
            if len(modes[model]) == 2: # all modes
                return 1 == m.bin_off[i] + m.bin_power[i] + m.bin_hydrogen[i]
            if hmode is not None: # hydrogen mode only 
                return 1 == m.bin_off[i] + m.bin_hydrogen[i]
            # Power only
            return 1 == m.bin_off[i] + m.bin_power[i]

        m.start = pyo.Var(pyo.RangeSet(len(lmps)), initialize=0, within=pyo.Binary)
        m.stop = pyo.Var(pyo.RangeSet(len(lmps)), initialize=0, within=pyo.Binary)
        m.initial_off = pyo.Var(initialize=0, within=pyo.Binary)
        m.initial_off.fix()

        @m.Constraint(pyo.RangeSet(len(lmps)))
        def start_constraint(b, i):
            if i == 1:
                return m.start[i] >= (1 - m.bin_off[i]) - (1 - m.initial_off)
            return m.start[i] >= (1 - m.bin_off[i]) - (1 - m.bin_off[i - 1])

        @m.Constraint(pyo.RangeSet(len(lmps)))
        def stop_constraint(b, i):
            if i == 1:
                return m.stop[i] >= m.bin_off[i] - m.initial_off
            return m.stop[i] >= m.bin_off[i] - m.bin_off[i - 1]

        if min_off > 0:
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def min_down_constraint(b, i):
                if i > min_off:
                    return sum(m.start[j] for j in range(i - min_off + 1, i + 1)) <= m.bin_off[i-min_off]
                return pyo.Constraint.Skip

        if min_on > 0:
            @m.Constraint(pyo.RangeSet(len(lmps)))
            def min_up_constraint(b, i):
                if i > min_on:
                    return sum(m.stop[j] for j in range(i - min_on + 1, i + 1)) <= (1 - m.bin_off[i-min_on])
                return pyo.Constraint.Skip

        if modes[model] == ("power_only", "hydrogen"):
            m.obj = pyo.Objective(expr=-0.01*(
                    sum(m.mp_profit[i] for i in m.pblk)
                    + sum(m.mh_profit[i] for i in m.hblk)
                    - start_cost * sum(m.start[i] for i in m.pblk)
                    - stop_cost * sum(m.stop[i] for i in m.pblk)
                )/len(lmps) #+ m.fixed_costs
            )
        elif modes[model] == ("power_only", "hydrogen_only"):
            m.obj = pyo.Objective(expr=-0.01*(
                    sum(m.mp_profit[i] for i in m.pblk)
                    + sum(m.mh_profit[i] for i in m.hblk)
                    - start_cost * sum(m.start[i] for i in m.pblk)
                    - stop_cost * sum(m.stop[i] for i in m.pblk)
                )/len(lmps) #+ m.fixed_costs
            )
        elif modes[model] == ("hydrogen_only",):
            m.obj = pyo.Objective(expr=-0.01*(
                    sum(m.mh_profit[i] for i in m.hblk)
                    - start_cost * sum(m.start[i] for i in m.hblk)
                    - stop_cost * sum(m.stop[i] for i in m.hblk)
                )/len(lmps) #+ m.fixed_costs
            )
        elif modes[model] == ("power_only",):
            m.obj = pyo.Objective(expr=-0.01*(
                    sum(m.mp_profit[i] for i in m.pblk)
                    - start_cost * sum(m.start[i] for i in m.pblk)
                    - stop_cost * sum(m.stop[i] for i in m.pblk)
                )/len(lmps) #+ m.fixed_costs
            )

        if modes[model] == ("power_only", "hydrogen"):
            m.obj_bilin = pyo.Objective(expr=-0.01*(
                    sum(m.bin_power[i] * m.pblk[i].profit_no_fixed for i in m.pblk)
                    + sum(m.bin_hydrogen[i] * m.hblk[i].profit_no_fixed for i in m.hblk)
                    - start_cost * sum(m.start[i] for i in m.pblk)
                    - stop_cost * sum(m.stop[i] for i in m.pblk)
                )/len(lmps) #+ m.fixed_costs
            )
        elif modes[model] == ("power_only", "hydrogen_only"):
            m.obj_bilin = pyo.Objective(expr=-0.01*(
                    sum(m.bin_power[i] * m.pblk[i].profit_no_fixed for i in m.pblk)
                    + sum(m.bin_hydrogen[i] * m.hblk[i].profit_no_fixed for i in m.hblk)
                    - start_cost * sum(m.start[i] for i in m.pblk)
                    - stop_cost * sum(m.stop[i] for i in m.pblk)
                )/len(lmps) #+ m.fixed_costs
            )
        elif modes[model] == ("hydrogen_only",):
            m.obj_bilin = pyo.Objective(expr=-0.01*(
                    sum(m.bin_hydrogen[i] * m.hblk[i].profit_no_fixed for i in m.hblk)
                    - start_cost * sum(m.start[i] for i in m.hblk)
                    - stop_cost * sum(m.stop[i] for i in m.hblk)
                )/len(lmps) #+ m.fixed_costs
            )
        elif modes[model] == ("power_only",):
            m.obj_bilin = pyo.Objective(expr=-0.01*(
                    sum(m.bin_power[i] * m.pblk[i].profit_no_fixed for i in m.pblk)
                    - start_cost * sum(m.start[i] for i in m.pblk)
                    - stop_cost * sum(m.stop[i] for i in m.pblk)
                )/len(lmps) #+ m.fixed_costs
            )
        m.obj_bilin.deactivate()
            
        return m


    def generate_surrogate_models(self, force=False):
        for model in self._xltabs:
            for mode in self._xltabs[model]:
                if self._xltabs[model][mode] is not None:
                    self._generate_surrogate_case(model, mode, force=force)
        self.model3_ineq_bounds()
        self.model5_ineq_bounds()

    def _get_trainer(self, input_labels, output_labels, df_training):
        # Create ALAMO trainer object
        trainer = AlamoTrainer(
            input_labels=input_labels, 
            output_labels=output_labels,
            training_dataframe=df_training,
        )
        
        # Set ALAMO options
        trainer.config.linfcns = True
        trainer.config.constant = True
        """
        trainer.config.monomialpower = [2, 3]
        trainer.config.multi2power = [1, 2, 3]
        trainer.config.ratiopower = [1, 2]
        trainer.config.maxtime = 2500.0
        trainer.config.maxterms = 8, 8, 8, 8
        """

        trainer.config.monomialpower = [2, 3]
        trainer.config.multi2power = [1,2]
        #trainer.config.ratiopower = [1, 2]
        trainer.config.maxtime = 2500.0
        trainer.config.maxterms = 8, 8, 8, 8

        return trainer

    def _generate_surrogate_case(self, model, mode, force=False, parity=False):
        fname=self.process_data_path

        # Get the data tab to read from spreadsheet
        try:
            tab = self._xltabs[model][mode]
        except KeyError:
            if model not in self._xltabs:
                raise KeyError(f"Invalid model: {model}")
            if mode not in self._xltabs[model]:
                raise KeyError(f"Invalid mode: {mode}")
            raise

        if tab is None:
            raise RuntimeError(f"Model {model} has no {mode} mode.")

        # Put data from tab into a data frame
        df = pd.read_excel(fname, sheet_name=tab)

        if mode == "power_only":
            input_labels=["net_power"]
            output_labels=["total_var_cost", "fuel_var_cost", "other_var_cost"]
        elif mode == "hydrogen":
            input_labels=["net_power", "h_prod"]
            output_labels=["total_var_cost", "fuel_var_cost", "other_var_cost"]
        elif mode == "hydrogen_only":
            input_labels=["h_prod"]
            output_labels=["total_var_cost", "elec_var_cost", "fuel_var_cost", "other_var_cost"]
        else:
            raise RuntimeError(f"Mode {mode} is not a recognized mode.")

        # surrogate name
        sname = f"{model}_{mode}"

        # Training and Validation Data
        n_data = df[input_labels[0]].size

        # Split the training and validation data
        df_training, df_validation = split_training_validation(df, 0.8, seed=n_data)

        # surrogate file
        spath = os.path.join(self.saved_surrogate_path, f"{sname}_alamo_surrogate.json") 
        # parity plot filet
        pppath = os.path.join(self.saved_surrogate_path, f"{sname}_parity_plots.pdf") 

        if not os.path.exists(spath) or force:
            trainer = self._get_trainer(input_labels, output_labels, df_training)
            # Train surrogate (this will call to Alamo through AlamoPy)
            success, alamo_surrogate, msg = trainer.train_surrogate()
            metrics = compute_fit_metrics(alamo_surrogate, df_validation)
            if parity:
                surrogate_parity(alamo_surrogate, df_training, filename=pppath)
            # Save the alamo surrogate
            alamo_surrogate.save_to_file(spath, overwrite=True)
        else:
            # Load the surrogate object
            alamo_surrogate = AlamoSurrogate.load_from_file(spath)
            # Display the metrics of fit
            metrics = compute_fit_metrics(alamo_surrogate, df_validation)
            if parity:
                surrogate_parity(alamo_surrogate, df_training, filename=pppath)
        
        self.models[model][mode] = {
            "model": alamo_surrogate,
            "metrics": metrics,
            "data": df,
        }

    def display_metrics(self, model, mode):
        metrics = self.models[model][mode]["metrics"]
        model = self.models[model][mode]["model"]
        for o in metrics:
            print(f"\n{o}")
            for m in metrics[o]:
                print(f"    {m} {metrics[o][m]}")
            print(f"   {model._surrogate_expressions[o]}")

    def model3_ineq_bounds(self, show_plot=False):
        clusters = []
        df = self.models["model3"]["hydrogen"]["data"]

        for r in df["h_prod"].index:
            u = False
            for c in clusters:
                if (df.loc[r, "h_prod"] - df.loc[c[0], "h_prod"]) ** 2 < 1e-2:
                    c.append(r)
                    u = True
                    break
            if not u:
                clusters.append([r])

        df_bounds = pd.DataFrame(columns=["h_prod", "net_power_min", "net_power_max"])
        for c in clusters:
            d = df.loc[c,:]
            df_bounds.loc[len(df_bounds.index)] = [
                d["h_prod"].mean(),
                d["net_power"].min(),
                d["net_power"].max(),
            ]
        df_bounds.sort_values(by="h_prod", inplace=True, ignore_index=True)
        lastv = df_bounds.loc[0, "net_power_min"]
        r_prev = 0
        for r in df_bounds.index[1:]:
            if df_bounds.loc[r, "net_power_min"] > lastv:
                break_point = r_prev
                break
            r_prev = r
            lastv = df_bounds.loc[r, "net_power_min"]

        res1 = scipy.stats.linregress(df_bounds["h_prod"], df_bounds["net_power_max"])
        res2 = scipy.stats.linregress(df_bounds["h_prod"][0:break_point], df_bounds["net_power_min"][0:break_point])
        res3 = scipy.stats.linregress(df_bounds["h_prod"][break_point:], df_bounds["net_power_min"][break_point:])

        print(f"res1 r^2 = {res1.rvalue**2}")
        print(f"res2 r^2 = {res2.rvalue**2}")
        print(f"res3 r^2 = {res3.rvalue**2}")
        
        xsplit = (res2.intercept - res3.intercept)/(res3.slope - res2.slope)

        try:
            assert 5*res1.slope + res1.intercept >= 5*res3.slope + res3.intercept
            assert 5*res1.slope + res1.intercept >= 5*res2.slope + res2.intercept
        except:
            # If you get this we may need to revisit this and make some minor adjustment
            # to the bounds.  As long as this passes, I won't worry about it. 
            raise AssertionError("The model3 bounds cut out 5 kg/s hydrogen production")

        if show_plot:
            plt.scatter(df_bounds["h_prod"], df_bounds["net_power_max"])
            plt.scatter(df_bounds["h_prod"][0:break_point], df_bounds["net_power_min"][0:break_point])
            plt.scatter(df_bounds["h_prod"][break_point:], df_bounds["net_power_min"][break_point:])

            def pltline(x, slope=None, intercept=None, y=None):
                if y is not None:
                    plt.plot((x,x), y)
                else:
                    y = intercept + slope * x
                    plt.plot(x, y)
            pltline(x=np.array([1, 5]), slope=res1.slope, intercept=res1.intercept)
            pltline(x=np.array([1, xsplit]), slope=res2.slope, intercept=res2.intercept)
            pltline(x=np.array([xsplit, 5]), slope=res3.slope, intercept=res3.intercept)
            pltline(x=5, y=(max(df_bounds["net_power_max"]), min(df_bounds["net_power_min"])))
            pltline(x=1, y=(max(df_bounds["net_power_max"]), min(df_bounds["net_power_min"])))
            plt.xlabel("Hydrogen Production (kg/s)")
            plt.ylabel("Net Power (MW)")
            plt.savefig(f"../saved_surrogate_models/bound_model3.png", dpi=320, bbox_inches="tight")
            plt.show()

        res = [
            {"op": "<=", "slope":res1.slope, "intercept":res1.intercept},
            {"op": ">=", "slope":res2.slope, "intercept":res2.intercept},
            {"op": ">=", "slope":res3.slope, "intercept":res3.intercept},
        ]

        self.models["model3"]["hydrogen"]["linear_hprod_ineq"] = res


    def model5_ineq_bounds(self, show_plot=False):
        clusters = []
        df = self.models["model5"]["hydrogen"]["data"]

        for r in df["h_prod"].index:
            u = False
            for c in clusters:
                if (df.loc[r, "h_prod"] - df.loc[c[0], "h_prod"]) ** 2 < 1e-2:
                    c.append(r)
                    u = True
                    break
            if not u:
                clusters.append([r])

        df_bounds = pd.DataFrame(columns=["h_prod", "net_power_min", "net_power_max"])
        for c in clusters:
            d = df.loc[c,:]
            df_bounds.loc[len(df_bounds.index)] = [
                d["h_prod"].mean(),
                d["net_power"].min(),
                d["net_power"].max(),
            ]

        res1 = scipy.stats.linregress(df_bounds["h_prod"], df_bounds["net_power_max"])
        res2 = scipy.stats.linregress(df_bounds["h_prod"], df_bounds["net_power_min"])

        print(f"res1 r^2 = {res1.rvalue**2}")
        print(f"res2 r^2 = {res2.rvalue**2}")

        if show_plot:
            plt.scatter(df_bounds["h_prod"], df_bounds["net_power_max"])
            plt.scatter(df_bounds["h_prod"], df_bounds["net_power_min"])

            def pltline(x, slope=None, intercept=None, y=None):
                if y is not None:
                    plt.plot((x,x), y)
                else:
                    y = intercept + slope * x
                    plt.plot(x, y)

            pltline(x=np.array([1, 5]), slope=res1.slope, intercept=res1.intercept)
            pltline(x=np.array([1, 5]), slope=res2.slope, intercept=res2.intercept)
            pltline(x=5, y=(max(df_bounds["net_power_max"]), min(df_bounds["net_power_min"])))
            pltline(x=1, y=(max(df_bounds["net_power_max"]), min(df_bounds["net_power_min"])))
            plt.xlabel("Hydrogen Production (kg/s)")
            plt.ylabel("Net Power (MW)")
            plt.savefig(f"../saved_surrogate_models/bound_model5.png", dpi=320, bbox_inches="tight")
            plt.show()

        res = [
            {"op": "<=", "slope":res1.slope, "intercept":res1.intercept},
            {"op": ">=", "slope":res2.slope, "intercept":res2.intercept},
        ]

        self.models["model5"]["hydrogen"]["linear_hprod_ineq"] = res





# Just for Testing
if __name__ == "__main__":
    surrogate_models = SurrogateProcessModels()
    surrogate_models.generate_surrogate_models()

    for model, mode in [
        ("model0", "power_only"),
        ("model1", "power_only"),
        ("model3", "power_only"),
        ("model3", "hydrogen"),
        ("model4", "power_only"),
        ("model4", "hydrogen_only"),
        ("model5", "power_only"),
        ("model5", "hydrogen"),
        ("model6", "hydrogen_only"),
    ]:
        print("\n\n")
        print("###########################################################################")
        print(f"Surrogate for {model} {mode}")
        print("###########################################################################")
        surrogate_models.display_metrics(model, mode)

    mode = "hydrogen"
    for model in ["model3", "model5"]:
        print("\n\n")
        print("###########################################################################")
        print(f"Surrogate for {model}")
        print("###########################################################################")
        for ineq in surrogate_models.models[model][mode]["linear_hprod_ineq"]:
            print(f'Net Power {ineq["op"]} {ineq["slope"]} * h_prod + {ineq["intercept"]}')
