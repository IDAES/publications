# imports
import math
import numpy as np
import pandas as pd
import idaes
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from linear_regression import read_raw_data

class IESSolution:
    def __init__(self, m = None, csv_file = None, case_object = None):
        '''
        Initialize IESSolution class with solve model properties

        Arguments:
            m: solved IES model
            csv_file: string, location of results csv file to load into object
            case_object: case object to identify case number and modes
                Note: not needed for m input, only needed for csv input

        Returns:
            None
        '''

        if m is not None and csv_file is None:
            # store model indicator booleans
            self.POWER = pyo.value(m.POWER)
            self.HYDROGEN = pyo.value(m.HYDROGEN)
            self.COPRODUCTION = pyo.value(m.COPRODUCTION)
            self.OPERATIONS = pyo.value(m.OPERATIONS)
            self.DESIGN = pyo.value(m.DESIGN)
            self.STARTUPS = pyo.value(m.STARTUPS)
            self.STORAGE = pyo.value(m.STORAGE)
            self.type = pyo.value(m.type)

            # go through different combinations of booleans
            if self.POWER and not self.HYDROGEN:
                # only off and power
                # create empty lists to store data

                # horizon - hrs
                self.horizon = []
                # LMP - $/MWh
                self.lmp = []
                # operation mode
                self.mode = []
                # off mode binary
                self.OFF_binary = []
                # power system dispatch - MW
                self.p_dispatch = []
                #power binary
                self.POWER_binary = []
                # fuel and variable costs
                self.fuel_cost_hourly = []
                self.var_cost_hourly = []
                self.power_cost_hourly = []
                self.power_revenue_hourly = []

                for i in m.HOUR:
                    # append lists with values from solved model
                    self.horizon.append(i)
                    self.lmp.append(pyo.value(m.power_price[i]))
                    if self.type == 'GDP':
                        self.OFF_binary.append(pyo.value(m.OFF_mode[i].binary_indicator_var))
                        self.POWER_binary.append(pyo.value(m.POWER_mode[i].binary_indicator_var))
                    elif self.type == 'MINLP':
                        self.OFF_binary.append(pyo.value(m.OFF_mode[i]))
                        self.POWER_binary.append(pyo.value(m.POWER_mode[i]))
                    self.power_revenue_hourly.append(pyo.value(m.power_price[i] * m.pt[i])/1E6)
                    # if storage - use, grid_power instead of pt
                    if m.STORAGE:
                        self.p_dispatch.append(pyo.value(m.grid_power[i]))
                    else:
                        self.p_dispatch.append(pyo.value(m.pt[i]))
                    # check mode and append mode list
                    if math.isclose(self.OFF_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('OFF')
                    elif math.isclose(self.POWER_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('POWER')
                    else:
                        self.mode.append('INVALID BINARY')

                    self.fuel_cost_hourly.append(pyo.value(m.fuel_cost_exp2[self.mode[i], i]))
                    self.var_cost_hourly.append(pyo.value(m.var_cost_exp[self.mode[i], i]))
                    self.power_cost_hourly.append(pyo.value(m.power_cost_exp[self.mode[i], i]))

                # plant capacity - MW
                self.plant_nameplate = pyo.value(m.P)
                # plant minimum output - MW
                self.pmin = pyo.value(m.P*m.turndown)
                # fixed cost per hour
                self.fixed_cost = pyo.value(m.fixed_cost_exp)

            # hydrogen only
            elif self.HYDROGEN and not self.POWER:
                # horizon - hrs
                self.horizon = []
                # LMP - $/MWh
                self.lmp = []
                # operation mode
                self.mode = []
                # off mode binary
                self.OFF_binary = []
                # hydrogen mode binary
                self.HYDROGEN_binary = []
                # hydrogen dispatch - kg/s
                self.h_dispatch = []
                # hydrogen price - $/kg
                self.h2_price = []
                # fuel and variable costs
                self.fuel_cost_hourly = []
                self.var_cost_hourly = []
                self.power_cost_hourly = []
                self.hydrogen_revenue_hourly = []

                for i in m.HOUR:
                    # append lists with values from solved model
                    self.horizon.append(i)
                    self.lmp.append(pyo.value(m.power_price[i]))
                    if self.type == 'GDP':
                        self.OFF_binary.append(pyo.value(m.OFF_mode[i].binary_indicator_var))
                        self.HYDROGEN_binary.append(pyo.value(m.HYDROGEN_mode[i].binary_indicator_var))
                    elif self.type == 'MINLP':
                        self.OFF_binary.append(pyo.value(m.OFF_mode[i]))
                        self.HYDROGEN_binary.append(pyo.value(m.HYDROGEN_mode[i]))
                    self.h_dispatch.append(pyo.value(m.ht[i]))
                    self.h2_price.append(pyo.value(m.h2_price[i]))
                    self.hydrogen_revenue_hourly.append(pyo.value(m.ht[i] * m.h2_price[i] * 3600)/1E6)
                    # check mode and append mode list
                    if math.isclose(self.OFF_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('OFF')
                    elif math.isclose(self.HYDROGEN_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('HYDROGEN')
                    else:
                        self.mode.append('INVALID BINARY')

                    self.fuel_cost_hourly.append(pyo.value(m.fuel_cost_exp2[self.mode[i], i]))
                    self.var_cost_hourly.append(pyo.value(m.var_cost_exp[self.mode[i], i]))
                    self.power_cost_hourly.append(pyo.value(m.power_cost[self.mode[i], i]))

                # hydrogen plant capacity - kg/s
                self.h_nameplate = pyo.value(m.H)
                # minimum hydrogen output - kg/s
                self.hmin = pyo.value(m.H*m.h2_turndown)
                # fixed cost per hour
                self.fixed_cost = pyo.value(m.fixed_cost_exp)

            # power and hydrogen
            elif self.HYDROGEN and self.POWER:
                # horizon - hrs
                self.horizon = []
                # LMP - $/MWh
                self.lmp = []
                # operation mode
                self.mode = []
                # off mode binary
                self.OFF_binary = []
                # power mode binary
                self.POWER_binary = []
                # power dispatch - MW
                self.p_dispatch = []
                # hydrogen mode binary
                self.HYDROGEN_binary = []
                # hydrogen dispatch - kg/s
                self.h_dispatch = []
                # hydrogen price - $/kg
                self.h2_price = []
                # coproduction mode binary
                self.COPRODUCTION_binary = []
                # fuel and variable costs
                self.fuel_cost_hourly = []
                self.var_cost_hourly = []
                self.power_cost_hourly = []
                self.power_revenue_hourly = []
                self.hydrogen_revenue_hourly = []

                for i in m.HOUR:
                    # append lists with values from solved model
                    self.horizon.append(i)
                    self.lmp.append(pyo.value(m.power_price[i]))
                    if self.type == 'GDP':
                        self.OFF_binary.append(pyo.value(m.OFF_mode[i].binary_indicator_var))
                        self.HYDROGEN_binary.append(pyo.value(m.HYDROGEN_mode[i].binary_indicator_var))
                        self.POWER_binary.append(pyo.value(m.POWER_mode[i].binary_indicator_var))
                    elif self.type == 'MINLP':
                        self.OFF_binary.append(pyo.value(m.OFF_mode[i]))
                        self.HYDROGEN_binary.append(pyo.value(m.HYDROGEN_mode[i]))
                        self.POWER_binary.append(pyo.value(m.POWER_mode[i]))
                    self.h_dispatch.append(pyo.value(m.ht[i]))
                    self.h2_price.append(pyo.value(m.h2_price[i]))
                    self.power_revenue_hourly.append(pyo.value(m.power_price[i] * m.pt[i])/1E6)
                    self.hydrogen_revenue_hourly.append(pyo.value(m.ht[i] * m.h2_price[i] * 3600)/1E6)
                    if m.COPRODUCTION:
                        if self.type == 'GDP':
                            self.COPRODUCTION_binary.append(pyo.value(m.COPRODUCTION_mode[i].binary_indicator_var))
                        elif self.type == 'MINLP':
                            self.COPRODUCTION_binary.append(pyo.value(m.COPRODUCTION_mode[i]))
                    else:
                        # no coproduction - set to zero
                        self.COPRODUCTION_binary.append(0)
                    # if storage - use, grid_power instead of pt
                    if m.STORAGE:
                        self.p_dispatch.append(pyo.value(m.grid_power[i]))
                    else:
                        self.p_dispatch.append(pyo.value(m.pt[i]))
                    # check mode and append mode list
                    if math.isclose(self.OFF_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('OFF')
                    elif math.isclose(self.HYDROGEN_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('HYDROGEN')
                    elif math.isclose(self.POWER_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('POWER')
                    elif math.isclose(self.COPRODUCTION_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('COPRODUCTION')
                    else:
                        self.mode.append('INVALID BINARY')

                    self.fuel_cost_hourly.append(pyo.value(m.fuel_cost_exp2[self.mode[i], i]))
                    self.var_cost_hourly.append(pyo.value(m.var_cost_exp[self.mode[i], i]))
                    self.power_cost_hourly.append(pyo.value(m.power_cost[self.mode[i], i]))

                # plant capacity - MW
                self.plant_nameplate = pyo.value(m.P)
                # plant minimum output - MW
                self.pmin = pyo.value(m.P*m.turndown)
                # hydrogen plant capacity - kg/s
                self.h_nameplate = pyo.value(m.H)
                # minimum hydrogen output - kg/s
                self.hmin = pyo.value(m.H*m.h2_turndown)
                # fixed cost per hour
                self.fixed_cost = pyo.value(m.fixed_cost_exp)

            # storage only
            if m.STORAGE:
                # battery state of charge - MWh
                self.soc = []
                # charging rate - MW
                self.charge = []
                # discharging rate - MW
                self.discharge = []

                for i in m.HOUR:
                    # append lists
                    self.soc.append(pyo.value(m.soc[i]))
                    self.charge.append(pyo.value(m.charge[i]))
                    self.discharge.append(pyo.value(m.discharge[i]))

                # battery capacity - MWh
                self.battery_size = pyo.value(m.E)

            # starup counting 
            if m.STARTUPS:
                # startup binary
                self.SU_binary = []
                # shutdown binary
                self.SD_binary = []
                # startup costs hourly - $/hr
                self.startup_cost_hourly = []
                # shutdown costs hourly - $/hr 
                self.shutdown_cost_hourly = []

                for i in m.HOUR:
                    # append lists 
                    self.SU_binary.append(pyo.value(m.su[i]))
                    self.SD_binary.append(pyo.value(m.sd[i]))
                    self.startup_cost_hourly.append(pyo.value(m.startup_cost_exp[i]))
                    self.shutdown_cost_hourly.append(pyo.value(m.shutdown_cost_exp[i]))
            
            # store cost expressions
            self.total_cost = pyo.value(m.total_cost_exp)
            self.fixed_cost = pyo.value(m.fixed_cost_sum)
            self.var_cost = pyo.value(m.var_cost_sum)
            self.fuel_cost = pyo.value(m.fuel_cost_sum)
            self.power_cost = pyo.value(m.power_cost_sum)
            self.startup_cost = pyo.value(m.startup_cost_sum)
            self.shutdown_cost = pyo.value(m.shutdown_cost_sum)
            self.revenue = pyo.value(m.hydrogen_revenue_exp + m.power_revenue_sum_exp)
            self.h2_revenue = pyo.value(m.hydrogen_revenue_exp)
            self.power_revenue = pyo.value(m.power_revenue_sum_exp)
            self.battery_revenue = pyo.value(m.battery_revenue_exp)
            self.profit = pyo.value(m.obj)

        elif m is None and csv_file is not None:
            # assert case_number must be present if the csv_file is being used for solution input 
            assert case_object is not None, 'Please include case_number for csv_file input.'

            # mode booleans 
            self.POWER = case_object.POWER_mode
            self.HYDROGEN = case_object.HYDROGEN_mode
            self.COPRODUCTION = case_object.COPRODUCTION_mode

            # assume in operations mode, counting startups, no storage
            self.OPERATIONS = True
            self.DEISIGN = False 
            self.STARTUPS = True 
            self.STORAGE = case_object.CHARGE_mode
            self.type = 'csv_file'

            # read the csv file 
            results = pd.read_csv(csv_file)

            # go through different combinations of booleans
            if self.POWER and not self.HYDROGEN:
                # only off and power
                # create empty lists to store data

                # horizon - hrs
                self.horizon = results.index.values.tolist()
                # LMP - $/MWh
                self.lmp = results['lmp ($/MWh)'].to_list()
                # operation mode
                self.mode = []
                # off mode binary
                self.OFF_binary = results['mode_off'].to_list()
                # power system dispatch - MW
                self.p_dispatch = results['net_power (MW)'].to_list()
                #power binary
                self.POWER_binary = results['mode_power_only'].to_list()
                # fuel and variable costs
                self.fuel_cost_hourly = results['ng_cost ($/hr)'].to_list()
                self.var_cost_hourly = results['other_cost ($/hr)'].to_list()
                self.power_cost_hourly = results['el_cost ($/hr)'].to_list()
                self.power_revenue_hourly = results['el_revenue ($/hr)'].to_list()
                self.profit_hourly = results['profit ($/hr)'].to_list()

                for i in self.horizon:
                    
                    # check mode and append mode list
                    if math.isclose(self.OFF_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('OFF')
                    elif math.isclose(self.POWER_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('POWER')
                    else:
                        self.mode.append('INVALID BINARY')

                # plant capacity - MW
                self.plant_nameplate = case_object.PowerSpecs.up_bound_nameplate
                # plant minimum output - MW
                self.pmin = case_object.PowerSpecs.up_bound_nameplate * case_object.PowerSpecs.turndown_limit
                # fixed cost per hour
                self.fixed_cost_hourly = results['fixed_cost ($/hr)'].to_list()

            # hydrogen only
            elif self.HYDROGEN and not self.POWER:
                # horizon - hrs
                self.horizon = results.index.values.tolist()
                # LMP - $/MWh
                self.lmp = results['lmp ($/MWh)'].to_list()
                # operation mode
                self.mode = []
                # off mode binary
                self.OFF_binary = results['mode_off'].to_list()
                # hydrogen mode binary
                self.HYDROGEN_binary = results['mode_hydrogen_only'].to_list()
                # net power dispatch - buying from grid the entire time (MW)
                self.p_dispatch = results['net_power (MW)'].to_list()
                # hydrogen dispatch - kg/s
                self.h_dispatch = results['h_prod (kg/s)'].to_list()
                # hydrogen price - $/kg
                self.h2_price = results['h2_price ($/kg)'].to_list()
                # fuel and variable costs
                self.fuel_cost_hourly = results['ng_cost ($/hr)'].to_list()
                self.var_cost_hourly = results['other_cost ($/hr)'].to_list()
                self.power_cost_hourly = results['el_cost ($/hr)'].to_list()
                self.hydrogen_revenue_hourly = results['h2_revenue ($/hr)'].to_list()
                self.profit_hourly = results['profit ($/hr)'].to_list()

                for i in self.horizon:
                    
                    # check mode and append mode list
                    if math.isclose(self.OFF_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('OFF')
                    elif math.isclose(self.HYDROGEN_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('HYDROGEN')
                    else:
                        self.mode.append('INVALID BINARY')

                # hydrogen plant capacity - kg/s
                self.h_nameplate = case_object.HydrogenSpecs.up_bound_nameplate
                # minimum hydrogen output - kg/s
                self.hmin = case_object.HydrogenSpecs.up_bound_nameplate * case_object.HydrogenSpecs.turndown_limit
                # fixed cost per hour
                self.fixed_cost_hourly = results['fixed_cost ($/hr)'].to_list()

            # power and hydrogen
            elif self.HYDROGEN and self.POWER:
                # horizon - hrs
                self.horizon = results.index.values.tolist()
                # LMP - $/MWh
                self.lmp = results['lmp ($/MWh)'].tolist()
                # operation mode
                self.mode = []
                # off mode binary
                self.OFF_binary = results['mode_off'].to_list()
                # power mode binary
                self.POWER_binary = results['mode_power_only'].to_list()
                # power dispatch - MW
                self.p_dispatch = results['net_power (MW)'].to_list()
                # hydrogen mode binary
                self.HYDROGEN_binary = results['mode_hydrogen_only'].to_list()
                # hydrogen dispatch - kg/s
                self.h_dispatch = results['h_prod (kg/s)'].to_list()
                # hydrogen price - $/kg
                self.h2_price = results['h2_price ($/kg)'].to_list()
                # coproduction mode binary
                self.COPRODUCTION_binary = results['mode_hydrogen'].to_list()
                # fuel and variable costs
                self.fuel_cost_hourly = results['ng_cost ($/hr)'].to_list()
                self.var_cost_hourly = results['other_cost ($/hr)'].to_list()
                self.power_cost_hourly = results['el_cost ($/hr)'].to_list()
                self.hydrogen_revenue_hourly = results['h2_revenue ($/hr)'].to_list()
                self.power_revenue_hourly = results['el_revenue ($/hr)'].to_list()
                self.profit_hourly = results['profit ($/hr)'].to_list()

                for i in self.horizon:
                    
                    # check mode and append mode list
                    if math.isclose(self.OFF_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('OFF')
                    elif math.isclose(self.HYDROGEN_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('HYDROGEN')
                    elif math.isclose(self.POWER_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('POWER')
                    elif math.isclose(self.COPRODUCTION_binary[i], 1, abs_tol = 1e-9):
                        self.mode.append('COPRODUCTION')
                    else:
                        self.mode.append('INVALID BINARY')

                # plant capacity - MW
                self.plant_nameplate = case_object.PowerSpecs.up_bound_nameplate
                # plant minimum output - MW
                self.pmin = case_object.PowerSpecs.up_bound_nameplate * case_object.PowerSpecs.turndown_limit
                # hydrogen plant capacity - kg/s
                self.h_nameplate = case_object.HydrogenSpecs.up_bound_nameplate
                # minimum hydrogen output - kg/s
                self.hmin = case_object.HydrogenSpecs.up_bound_nameplate * case_object.HydrogenSpecs.turndown_limit
                # fixed cost per hour
                self.fixed_cost_hourly = results['fixed_cost ($/hr)'].to_list()

            # storage only
            if self.STORAGE:
                # battery state of charge - MWh
                self.soc = []
                # charging rate - MW
                self.charge = []
                # discharging rate - MW
                self.discharge = []

                for i in m.HOUR:
                    # append lists
                    self.soc.append(pyo.value(m.soc[i]))
                    self.charge.append(pyo.value(m.charge[i]))
                    self.discharge.append(pyo.value(m.discharge[i]))

                # battery capacity - MWh
                self.battery_size = pyo.value(m.E)

            # starup counting 
            if self.STARTUPS:
                # startup costs hourly - $/hr
                self.startup_cost_hourly = results['start_cost'].to_list()
                # shutdown costs hourly - $/hr 
                self.shutdown_cost_hourly = results['stop_cost'].to_list()
            
            # store cost expressions
            self.fixed_cost = sum(self.fixed_cost_hourly)/1e6
            self.var_cost = sum(self.var_cost_hourly)/1e6
            self.fuel_cost = sum(self.fuel_cost_hourly)/1e6
            self.power_cost = sum(self.power_cost_hourly)/1e6
            self.startup_cost = sum(self.startup_cost_hourly)/1e6
            self.shutdown_cost = sum(self.shutdown_cost_hourly)/1e6
            self.total_cost = self.fixed_cost + self.var_cost + self.fuel_cost + self.power_cost + self.startup_cost + self.shutdown_cost
            if self.HYDROGEN:
                self.h2_revenue = sum(self.hydrogen_revenue_hourly)/1e6
            else:
                self.h2_revenue = 0.0
            if self.POWER:
                self.power_revenue = sum(self.power_revenue_hourly)/1e6
            else:
                self.power_revenue = 0.0
            self.battery_revenue = 0.0
            self.revenue = self.h2_revenue + self.power_revenue
            self.profit = sum(self.profit_hourly)/1e6

        return

    def summarize(self):
        '''
        Generates a print statement with model results.

        Arguments:
            None

        Returns:
            None
        '''
        print('PLANT SUMMARY:')
        if self.OPERATIONS:
            print('The problem was solved in OPERATIONS mode: plant capacities were fixed and the schedule was optimized.')
            if self.POWER:
                print('The power plant capacity is', self.plant_nameplate, 'MW.')
                print('The power plant capacity factor is', sum(self.p_dispatch)/(len(self.horizon)*self.plant_nameplate))
            if self.HYDROGEN:
                print('The hydrogen plant capacity is', self.h_nameplate, 'kg/s.')
                print('The hydrogen plant capacity factor is', (sum(self.h_dispatch)*3600)/(len(self.horizon)*self.h_nameplate*3600 ))
            if self.STORAGE:
                print('The battery capacity is', self.battery_size, 'MW')
        else:
            print('The problem was solved in DESIGN mode: plant capacities and schedule were both optimized.')
            if self.POWER:
                print('The optimal power plant capacity is', self.plant_nameplate, 'MW.')
            if self.HYDROGEN:
                print('The optimal hydrogen plant capacity is', self.h_nameplate, 'kg/hr.')
            if self.STORAGE:
                print('The optimal battery capacity is', self.battery_size, 'MW.')
        print('ECONOMIC SUMMARY:')
        print('The annual costs (variable + fixed) for the plant was M$', self.total_cost)
        print('The annual fixed costs for the plant were M$', self.fixed_cost)
        print('The annual variable costs for the plant were M$', self.var_cost)
        print('The annual plant revenue was M$', self.revenue)
        print('The plant profits for the year were M$', self.profit)
        pass

    def return_mode_statistics(self):
        '''
        Prints percentage of time at each operation mode

        Arguments:
            None

        Returns:
            None
        '''
        # count modes
        off = self.mode.count('OFF')
        power = self.mode.count('POWER')
        hydrogen = self.mode.count('HYDROGEN')
        coproduction = self.mode.count('COPRODUCTION')

        # initialize lists
        if self.POWER:
            pmax = 0
            pmin = 0
            buying = 0
            other = 0
            for i in range(len(self.p_dispatch)):
                if self.p_dispatch[i] != 0.0 and self.p_dispatch[i] >= self.plant_nameplate - 1:
                    pmax += 1
                elif self.p_dispatch[i] != 0 and self.pmin <= self.p_dispatch[i] and self.p_dispatch[i] <= (self.pmin + 1):
                    pmin += 1
                elif self.p_dispatch[i] >= self.pmin + 1.0:
                    other += 1
                elif self.p_dispatch[i] < 0.0:
                    buying += 1

        # repeat same process for hydrogen
        if self.HYDROGEN:
            hmax = 0
            hmin = 0
            otherh = 0
            for i in range(len(self.h_dispatch)):
                if self.h_dispatch[i] != 0.0 and self.h_dispatch[i] >= self.h_nameplate - 1:
                    hmax += 1
                elif self.h_dispatch[i] != 0 and self.hmin <= self.h_dispatch[i] and self.h_dispatch[i] <= (self.hmin + 1):
                    hmin += 1
                elif self.h_dispatch[i] >= self.hmin + 1.0:
                    otherh += 1


        print(round((off/len(self.horizon)*100), 4), '% of hours off.')
        print(round((power/len(self.horizon)*100), 4), '% of hours in power mode.')
        print(round((hydrogen/len(self.horizon)*100), 4), '% of hours in hydrogen mode.')
        print(round((coproduction/len(self.horizon)*100), 4), '% of hours in coproduction mode.')
        if self.POWER:
            print(round((pmax/len(self.horizon)*100), 4), '% of hours at pmax.')
            print(round((pmin/len(self.horizon)*100), 4), '% of hours at pmin.')
            print(round((other/len(self.horizon)*100), 4), '% of hours at intermediate power output. ')
            print(round((buying/len(self.horizon)*100), 4), '% of buying power from the grid. ')
        if self.HYDROGEN:
            print(round((hmax/len(self.horizon)*100), 4), '% of hours at hmax.')
            print(round((hmin/len(self.horizon)*100), 4), '% of hours at hmin.')
            print(round((otherh/len(self.horizon)*100), 4), '% of hours at intermediate hydrogen output. ')

        return

    def save_mode_data(self, fdest = None):
        '''
        Saves a csv file containing modes of model at each time step

        Arguments:
            fdest: string, destination and filename of csv file containing mode data
                default = None, only returns dataframe, doesn't save csv

        Returns:
            allmodes: pd.DataFrame, dataframe containing all mode information
        '''

        # create dictionary
        mode_dict = {'Hour': self.horizon, 'OFF': self.OFF_binary}

        # add power if included
        if self.POWER:
            mode_dict['POWER'] = self.POWER_binary
        # add hydrogen if included
        if self.HYDROGEN:
            mode_dict['HYDROGEN'] = self.HYDROGEN_binary
        # add coproduction if included
        if self.COPRODUCTION:
            mode_dict['COPRODUCTION'] = self.COPRODUCTION_binary

        # convert dict to dataframe
        allmodes = pd.DataFrame.from_dict(mode_dict)

        # save as csv if fdest is included
        if fdest is not None:
            allmodes.to_csv(fdest)

        return allmodes


    def save_economic_results(self, fdest):
        '''
        Saves CSV file containing economic results of model

        Arguments:
            fdest: string, name and location to save results

        Returns:
            None
        '''
        if self.OPERATIONS:
            mode = 'OPERATION'
        else:
            mode = 'DESIGN'


        if self.HYDROGEN and self.POWER:
            econ_results_dict = {'Problem Mode': mode, 'Power Capacity (MW)': self.plant_nameplate, \
            'Hydrogen Capacity (kg/s)': self.h_nameplate, \
            'Power Plant Capacity factor': (sum(self.p_dispatch)/(len(self.horizon)*self.plant_nameplate)), \
            'Hydrogen Plant Capacity Factory': (sum(self.h_dispatch)*3600)/(len(self.horizon)*self.h_nameplate*3600), \
            'Fixed Cost (M$/yr)': self.fixed_cost, 'Fuel Cost (M$/yr)': self.fuel_cost, 'Power Cost (M$/yr)': self.power_cost, \
            'Variable Cost (M$/yr)': self.var_cost, 'Power Revenue (M$/yr)': self.power_revenue, \
            'Hydrogen Revenue (M$/yr)': self.h2_revenue, 'Profit (M$/yr)': self.profit}
            time_series_dict = {'Hour': self.horizon, 'LMP ($/MWh)': self.lmp, 'Hydrogen Price ($/kg)': self.h2_price, \
            'Operation Mode': self.mode, 'Net Power to Grid (MW)': self.p_dispatch, 'Hydrogen Production (kg/s)': self.h_dispatch, \
            'Fuel Cost (M$/hr)': self.fuel_cost_hourly, 'Other Variable Cost (M$/hr)': self.var_cost_hourly, 'Power Cost (M$/hr)': self.power_cost_hourly, \
            'Fixed Cost (M$/hr)': np.ones(len(self.horizon))*(self.fixed_cost/len(self.horizon)), \
            'Power Revenue (M$/hr)': self.power_revenue_hourly, 'Hydrogen Revenue (M$/hr)': self.hydrogen_revenue_hourly}
        elif self.POWER and not self.HYDROGEN:
            econ_results_dict = {'Problem Mode': mode, 'Power Capacity (MW)': self.plant_nameplate, \
            'Power Plant Capacity factor': (sum(self.p_dispatch)/(len(self.horizon)*self.plant_nameplate)), \
            'Fixed Cost (M$/yr)': self.fixed_cost, 'Fuel Cost (M$/yr)': self.fuel_cost, 'Power Cost (M$/yr)': self.power_cost,\
            'Variable Cost (M$/yr)': self.var_cost, 'Power Revenue (M$/yr)': self.power_revenue, \
            'Profit (M$/yr)': self.profit}
            time_series_dict = {'Hour': self.horizon, 'LMP ($/MWh)': self.lmp, \
            'Operation Mode': self.mode, 'Net Power to Grid (MW)': self.p_dispatch, \
            'Fuel Cost (M$/hr)': self.fuel_cost_hourly, 'Other Variable Cost (M$/hr)': self.var_cost_hourly, 'Power Cost (M$/hr)': self.power_cost_hourly, \
            'Fixed Cost (M$/hr)': np.ones(len(self.horizon))*(self.fixed_cost/len(self.horizon)), \
            'Power Revenue (M$/hr)': self.power_revenue_hourly}
        elif self.HYDROGEN and not self.POWER:
            econ_results_dict = {'Problem Mode': mode, \
            'Hydrogen Capacity (kg/s)': self.h_nameplate, \
            'Hydrogen Plant Capacity Factory': (sum(self.h_dispatch)*3600)/(len(self.horizon)*self.h_nameplate*3600), \
            'Fixed Cost (M$/yr)': self.fixed_cost, 'Fuel Cost (M$/yr)': self.fuel_cost, 'Power Cost (M$/yr)': self.power_cost,\
            'Variable Cost (M$/yr)': self.var_cost, 'Power Revenue (M$/yr)': self.power_revenue, \
            'Hydrogen Revenue (M$/yr)': self.h2_revenue, 'Profit (M$/yr)': self.profit}
            time_series_dict = {'Hour': self.horizon, 'LMP ($/MWh)': self.lmp, 'Hydrogen Price ($/kg)': self.h2_price, \
            'Operation Mode': self.mode, 'Hydrogen Production (kg/s)': self.h_dispatch, \
            'Fuel Cost (M$/hr)': self.fuel_cost_hourly, 'Other Variable Cost (M$/hr)': self.var_cost_hourly, 'Power Cost (M$/hr)': self.power_cost_hourly, \
            'Fixed Cost (M$/hr)': np.ones(len(self.horizon))*(self.fixed_cost/len(self.horizon)), \
            'Hydrogen Revenue (M$/hr)': self.hydrogen_revenue_hourly}

        # turn dict to dataframe
        econ_results = pd.DataFrame.from_dict(econ_results_dict, orient = 'index')
        time_series_results = pd.DataFrame.from_dict(time_series_dict, orient = 'columns')

        # create excel writer object
        writer = pd.ExcelWriter(fdest)

        # save to excel file at fdest
        econ_results.to_excel(writer, sheet_name = 'Economic Results')
        time_series_results.to_excel(writer, sheet_name = 'Time Series Results')

        writer.save()

        return

    def plot_power(self,start_hour, end_hour, fdest = None):
        '''
        Plot power system dispatch
        Note: More detailed version is under plot_operation() method

        Arguments:
            start_hour: int, hour to begin plotting at
            end_hour: int, hour to end plot at
            fdest: string, name and destination to save file at
                default = None, no file saved, plot is only printed

        Returns:
            None
        '''
        plt.figure(figsize = (9,7))
        plt.plot(self.horizon[start_hour:end_hour], self.p_dispatch[start_hour:end_hour],'r*-', label = 'Power Dispatch')
        plt.title('Power Dispatch')
        plt.xlabel('Hour')
        plt.ylabel('Power Output (MWh)')
        plt.grid(True)
        plt.legend(loc = 'best')
        if fdest is not None:
            plt.savefig(fdest)
        plt.show()
        return

    def plot_h2(self, start_hour, end_hour, fdest = None):
        '''
        Plot hydrogen system dispatch
        Note: More detailed version is under plot_operation() method

        Arguments:
            start_hour: int, hour to begin plotting at
            end_hour: int, hour to end plot at
            fdest: string, name and destination to save file at
                default = None, no file saved, plot is only printed

        Returns:
            None
        '''
        assert self.HYDROGEN, 'Model does not include Hydrogen production, this result cannot be plotted.'
        plt.figure(figsize = (9,7))
        plt.plot(self.horizon[start_hour:end_hour], self.h_dispatch[start_hour:end_hour], 'b*-', label = 'Hydrogen Dispatch')
        plt.title('Hydrogen Dispatch from hour', start_hour, 'to hour', end_hour)
        plt.xlabel('Hour')
        plt.ylabel('Hydrogen Output (kg/hr)')
        plt.grid(True)
        plt.legend(loc = 'best')
        if fdest is not None:
            plt.savefig(fdest)
        plt.show()
        return

    def plot_dispatch_histogram(self, fdest = None):
        '''
        Plot power system dispatch historgram

        Arguments:
            fdest: string, name and destination to save file at
                default = None, no file saved, plot is only printed

        Returns:
            None
        '''
        # bin data
        histogram_bins = np.histogram(self.p_dispatch)
        # create plot
        fig, ax = plt.subplots(figsize=(6,6))
        plt.hist(self.p_dispatch,bins = histogram_bins[1], weights = np.ones(len(self.p_dispatch))/len(self.p_dispatch), color = 'r')

        # axis labels and title
        plt.title('Power Dispatch', fontsize=20, fontweight = 'bold')
        plt.xlabel('Power Output (MW)', fontsize = 20, fontweight = 'bold')
        plt.ylabel('Frequency', fontsize = 20, fontweight = 'bold')

        # set axes limits
        ax.set_xlim(xmin = 0.0, xmax = self.plant_nameplate+100)
        ax.set_ylim(ymin = 0.0, ymax = 1.0)

        # axes ticks and tick labels
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
        plt.tick_params(direction='in', top = True, right = True)
        plt.minorticks_on()
        plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        # turn grid on
        plt.grid(True)

        # save figure if destination was chosen
        if fdest is not None:
            plt.savefig(fdest, dpi = 300, bbox_inches = 'tight')

        plt.show()
        return

    def plot_h2_dispatch_histogram(self, fdest = None):
        '''
        Plot hydrogen system dispatch historgram

        Arguments:
            fdest: string, name and destination to save file at
                default = None, no file saved, plot is only printed

        Returns:
            None
        '''
        # assert system must have hydrogen to produce this plot
        assert self.HYDROGEN, 'IES does not produce hydrogen.'

        # bin data
        histogram_bins = np.histogram(self.h_dispatch)
        # create plot
        fig, ax = plt.subplots(figsize=(6,6))
        plt.hist(self.h_dispatch, bins = histogram_bins[1], weights = np.ones(len(self.h_dispatch))/len(self.h_dispatch), color = 'm')

        # axis labels and title
        plt.title('Hydrogen Dispatch', fontsize=20, fontweight = 'bold')
        plt.xlabel('Hydrogen Output (MW)', fontsize = 20, fontweight = 'bold')
        plt.ylabel('Frequency', fontsize = 20, fontweight = 'bold')

        # set axes limits
        ax.set_xlim(xmin = 0.0)
        ax.set_ylim(ymin = 0.0, ymax = 1.0)

        # axes ticks and tick labels
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
        plt.tick_params(direction='in', top = True, right = True)
        plt.minorticks_on()
        plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        # turn grid on
        plt.grid(True)

        # save figure if destination was chosen
        if fdest is not None:
            plt.savefig(fdest, dpi = 300, bbox_inches = 'tight')

        plt.show()
        return

    def plot_3d_histogram(self, fdest = None):
        '''
        Plots 3d coproduction dispatch histogram

        Arguments:
            soln: IESSolution object initialized with solved model

        Returns:
            none
        '''
        # assert model must include hydrogen
        assert self.HYDROGEN, 'Model does not include Hydrogen production, this result cannot be plotted.'

        # get data into arrays
        p_dispatch = np.array(self.p_dispatch)
        h_dispatch = np.array(self.h_dispatch)

        # plot figure
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        xedges = np.linspace(0, self.plant_nameplate+0.1)
        yedges = np.linspace(0, self.h_nameplate+0.1)
        hist, xedges, yedges = np.histogram2d(p_dispatch, h_dispatch, bins = (xedges, yedges))

        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = (xedges[1] - xedges[0])*0.8
        dy = (yedges[1] - yedges[0])*0.8
        dz = hist.ravel()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        # set title and axis labels
        plt.title('Optimal Operation', fontsize=20, fontweight = 'bold')
        ax.set_xlabel('Power Production (MW)', fontsize=15, fontweight = 'bold')
        ax.set_ylabel('H$_{2}$ Production (kg/s)', fontsize=15, fontweight = 'bold')
        ax.set_zlabel('Frequency', fontsize = 15, fontweight = 'bold')
        ax.tick_params(labelsize = 10)

        # save figure
        if fdest is not None:
            plt.savefig(fdest, dpi = 300, bbox_inches = 'tight')
        plt.show()
        return

    def plot_operation(self, start_end = None, title_pow = None, title_h2 = None, fdest = None, fdest_h2 = None):
        '''
        Plots dispatch and price for entire planning horizon.
        If system is power only, one plot will be printed. If coproduction, one plot for power
        and one plot for hydrogen will be printed.

        Arguments:
            soln: IESSolution object initialized with solved model
            start_end: list, start and end hours to plot.
                default = None: The entire planning horizon is plotted
            title_pow: string, title of plot
                default = None: title set to 'Power Operation'
            title_h2: string, title of hydrogen operation plot
                default = None: title set to 'Hydrogen Operation'
            fdest: string, name and location of saved power operation plot
                default = None: plot not saved, only displayed
            fdest_h2: string, name and location of saved hydrogen operation plot
                default = None: plot not saved, only displayed

        Returns:
            none
        '''
        if start_end is None:
            start_end = [0,len(self.lmp)]

        # plot dispatch over entire horizon
        # make figure
        fig, ax = plt.subplots(figsize=(20,9))

        # if horizon being plotted is less than one month - plot dispatch as a line
        if self.POWER:
            if (start_end[1]-start_end[0]) <= 744:
                plt.plot(self.horizon[start_end[0]:start_end[1]], self.p_dispatch[start_end[0]:start_end[1]], 'r.-', markersize = 8, linewidth = 3)
            else: # plot dispatch as points
                plt.scatter(self.horizon[start_end[0]:start_end[1]], self.p_dispatch[start_end[0]:start_end[1]], facecolors='none',edgecolors='r', linewidths = 3)
            #ax.set_ylim(ymax = self.plant_nameplate+50)
        else: # hydrogen
            if (start_end[1]-start_end[0]) <= 744:
                plt.plot(self.horizon[start_end[0]:start_end[1]], self.h_dispatch[start_end[0]:start_end[1]], 'm.-', markersize = 8, linewidth = 3)
            else:
                plt.scatter(self.horizon[start_end[0]:start_end[1]], self.h_dispatch[start_end[0]:start_end[1]], facecolors='none',edgecolors='m', linewidths = 3)
            #ax.set_ylim(ymin = 0.0, ymax = self.h_nameplate+.1)
        # set title and axis labels
        if title_pow is None:
            plt.title('Operation', fontsize=20, fontweight = 'bold')
        else:
            plt.title(title_pow, fontsize = 20, fontweight = 'bold')
        plt.xlabel('Hour', fontsize=30, fontweight = 'bold')
        if self.POWER:
            plt.ylabel('Power Output (MW)', color = 'r', fontsize=30, fontweight = 'bold')
        else: # hydrogen
            plt.ylabel('H$_{2}$ Output (kg/s)', color = 'm', fontsize=30, fontweight = 'bold')
        plt.grid(True)


        # major and minor ticks
        plt.tick_params(direction='in', top = True, right = True)
        plt.minorticks_on()
        plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)
        # make a plot with different y-axis using second axis object
        ax2=ax.twinx()
        ax2.scatter(self.horizon[start_end[0]:start_end[1]], self.lmp[start_end[0]:start_end[1]], facecolors='none',edgecolors='b', linewidths = 3)
        # second y axis label
        ax2.set_ylabel("LMP ($/MWh)",color="blue",fontsize=30, fontweight = 'bold')

        # major and minor ticks
        for label in (ax.get_xticklabels() + ax.get_yticklabels()+ax2.get_yticklabels()):
            label.set_fontsize(25)
            label.set_fontweight('bold')
        plt.tick_params(direction='in', top = True, right = True)
        plt.minorticks_on()
        plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

        # save figure
        if fdest is not None:
            plt.savefig((fdest+ '.png'), dpi = 300, bbox_inches = 'tight')
            plt.savefig((fdest + '.pdf'), dpi = 300, bbox_inches = 'tight')
        plt.show()

        # plot hydrogen dispatch if model includes hydrogen
        if self.HYDROGEN:
            # make plot
            fig, ax = plt.subplots(figsize=(20,9))
            if (start_end[1]-start_end[0]) <= 744:
                plt.plot(self.horizon[start_end[0]:start_end[1]], self.h_dispatch[start_end[0]:start_end[1]], 'm.-', markersize = 8, linewidth = 3)
                #ax.set_ylim(ymax = self.h_nameplate+.1)
            else:
                plt.scatter(self.horizon[start_end[0]:start_end[1]], self.h_dispatch[start_end[0]:start_end[1]], facecolors='none',edgecolors='m', linewidths = 3)
                #ax.set_ylim(ymax = self.h_nameplate+.1)

            # set title and axis labels
            if title_h2 is None:
                plt.title('H$_{2}$ Operation', fontsize=20, fontweight = 'bold')
            else:
                plt.title(title_h2, fontsize = 20, fontweight = 'bold')
            plt.xlabel('Hour', fontsize=30, fontweight = 'bold')
            plt.ylabel('H$_{2}$ Output (kg/s)', color = 'm', fontsize=30, fontweight = 'bold')
            plt.grid(True)

            # major and minor ticks
            plt.tick_params(direction='in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # make a plot with different y-axis using second axis object
            ax2=ax.twinx()
            ax2.scatter(self.horizon[start_end[0]:start_end[1]], self.h2_price[start_end[0]:start_end[1]], facecolors='none',edgecolors='g', linewidths = 3)
            ax2.set_ylabel(r'Price ($/kg)',color="g",fontsize=30, fontweight = 'bold')

            # major and minor ticks
            for label in (ax.get_xticklabels() + ax.get_yticklabels()+ax2.get_yticklabels()):
                label.set_fontsize(25)
                label.set_fontweight('bold')
            plt.tick_params(direction='in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # save plot
            if fdest_h2 is not None:
                plt.savefig((fdest_h2 + '.png'), dpi = 300, bbox_inches = 'tight')
                plt.savefig((fdest_h2 + '.pdf'), dpi = 300, bbox_inches = 'tight')
            plt.show()

        return

    def plot_dispatch_scatter(self, fdest = None):

        '''
        Creates scatter plot of system dispatch with varying LMP

        Arguments:
            fdest: str, destination and file name of plot

        Returns:
        '''
        fig, ax = plt.subplots(figsize = (6,6))
        if self.POWER:
            plt.scatter(self.lmp, self.p_dispatch, facecolors='none',edgecolors='r', linewidths = 3, label = 'Net Power')
            plt.ylabel('Net Power (MW)', fontsize = 20, fontweight = 'bold', color = 'r')


            # titles and axes
            plt.title('Plant Dispatch vs LMP', fontsize = 20, fontweight = 'bold')
            plt.xlabel('LMP ($/MWh)', fontsize = 20, fontweight = 'bold')
            plt.grid(True)

            # major and minor ticks
            plt.tick_params(direction='in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)

            # if hydrogen, make twin axes for hydrogen production
            if self.HYDROGEN:
                # make a plot with different y-axis using second axis object
                ax2=ax.twinx()
                ax2.scatter(self.lmp, self.h_dispatch, facecolors = 'none', edgecolors = 'magenta', linewidths = 3, label = r'H$_{2}$ Output')
                ax2.set_ylabel(r'H$_{2}$ Output (kg/s)', fontsize = 20, fontweight = 'bold', color = 'magenta')
                # major and minor ticks
                for label in (ax.get_xticklabels() + ax.get_yticklabels()+ax2.get_yticklabels()):
                    label.set_fontsize(15)
                plt.tick_params(direction='in', top = True, right = True)
                plt.minorticks_on()
                plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # save figure
            if fdest is not None:
                plt.savefig(fdest, dpi = 300, bbox_inches = 'tight')

            plt.show()

        else:
            ax.scatter(self.lmp, self.h_dispatch, facecolors = 'none', edgecolors = 'magenta', linewidths = 3, label = r'H$_{2}$ Output')
            ax.set_ylabel(r'H$_{2}$ Output (kg/s)', fontsize = 20, fontweight = 'bold', color = 'magenta')
            #ax.set_ylim(ymin = 0.0)

            # titles and axes
            plt.title('Plant Dispatch vs LMP', fontsize = 20, fontweight = 'bold')
            plt.xlabel('LMP ($/MWh)', fontsize = 20, fontweight = 'bold')
            plt.grid(True)

            # major and minor ticks
            plt.tick_params(direction='in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)

            # save figure
            if fdest is not None:
                plt.savefig(fdest, dpi = 300, bbox_inches = 'tight')

            plt.show()

    def plot_lmpsignal_hist(self, signal_name, fdest = None):
        '''
        Plots histogram of LMP signal distribution 

        Arguments:
            fdest: string, location and file name to save the plot 
                default = None, do not save a copy 

        Returns:
            None 
        '''
        
        # create a bar chart 
        fig, ax = plt.subplots(figsize = (6,6))
        plt.hist(self.lmp, bins = 20, color = 'b')
        
        # title 
        plt.title(signal_name, fontsize = 20, fontweight = 'bold')
        
        # labels 
        plt.xlabel('Locational Marginal Price ($/MWh)', fontsize = 15, fontweight = 'bold')
        plt.ylabel('Frequency', fontsize = 15, fontweight = 'bold')
        
        # make y axis a percentage 
        ax.yaxis.set_major_formatter(PercentFormatter(xmax = len(self.lmp)))
        
        # set axes limits 
        # ax.set_xlim([0,200])
        # ax.set_ylim([0, len(lmp)])
        
        # fix y labels to be multiples of 10
        # plt.yticks(ticks = np.linspace(0,len(lmp), 11))
        
        
        # major and minor ticks 
        ax.tick_params(direction='in', top = True, right = True)
        ax.minorticks_on()
        ax.tick_params(which = 'minor', direction = 'in', top = True, bottom = True, right = True, left = True)
        
        # label size
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)
            label.set_fontweight('bold')
        
        #grid
        ax.set_axisbelow(True)
        plt.grid()
        
        # add textbox with info
        formatted_data = read_raw_data('formatted_raw_data.csv')

        # convert hydrogen price to string 
        # Convert the float to an integer by multiplying it by 10 to the power of the number of decimal places, then rounding it to the nearest integer.
        int_num = round(self.h2_price[0] * (10 ** len(str(int(self.h2_price[0])))), 0)

        # Convert the integer to a string
        string_h2 = str(int_num)
        
        med = formatted_data.at[(signal_name + '_{0}{1}').format(string_h2[0], string_h2[1]), '50th Percentile ($/MWh)']
        ng = formatted_data.at[(signal_name + '_{0}{1}').format(string_h2[0], string_h2[1]), 'Natural Gas Price ($/MMBtu)']

        #number of points over 200
        count = 0
        for i in self.lmp:
            if i >=200:
                count += 1
        
        textstring = 'Median LMP = ${0}/MWh \nNatural Gas Price = ${1}/MMBtu'.format(round(med,2), ng)

        plt.text(x = 50, y = 0.24*len(self.lmp), s = textstring, fontsize = 12, fontweight = 'bold', backgroundcolor = 'white', \
                bbox = dict(edgecolor ='black', facecolor = 'white'))
        
        # save figure
        if fdest is not None:
            plt.savefig((fdest + '.png'), dpi = 300, bbox_inches = 'tight')
            plt.savefig((fdest + '.pdf'), dpi = 300, bbox_inches = 'tight')

        plt.show()
        
        return
