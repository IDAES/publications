"""
Economics Calculations
MEA Solvent Based Carbon Capture System integrated with NGCC Power Plant 
(point source of flue gas)
"""

import idaes
from math import pi
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models_extra.power_generation.costing.power_plant_capcost import (
    QGESSCosting,
    QGESSCostingData,
    )
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core.solvers import get_solver
import os
from pyomo.common.fileutils import this_file_dir
import csv


def get_ngcc_solvent_cost(self): 
# -----------------------------------------------------------------------------
    # Create a main Costing block
    self.costing = QGESSCosting()
# -----------------------------------------------------------------------------
    
    # ---------------- TPC Calculation for NGCC and CCS -----------------------
    # Note: the numbering of the costing sub-blocks are kept the same 
    # as in the original costing file used in the Aspen-based high capture 
    # optimization runs ("MEA_NGCC_integrated_cost_module.py"). But because the 
    # code has been revised for the IDAES flowsheet, some costing sub-blocks 
    # have become redundant and are deleted. This sometimes results in seemingly  
    # inconsistent costing sub-block numbering.
        
    import json

    directory = this_file_dir()  # access the path to the folder containing this script

    with open(os.path.join(directory, "FEEDEquipmentDetailsFinal.json"), "r") as file:
        FEEDEquipmentDetailsFinal = json.load(file)
        
    pyo.units.load_definitions_from_strings(["USD_2021 = 500/708 * USD_CE500"])
    
    CE_index_year = "2018"
    CE_index_units = getattr(pyo.units, "MUSD_" + CE_index_year)    
    
    ###########################################################################
    ######################## NGCC with Capture ################################
    # For NGCC with capture, it is considered that steam for the stripper 
    # reboiler duty is extracted from the IP-LP crossover. The fuel to the 
    # NGCC is considered fixed, while its power output decreases as a function
    # of steam extracted for the capture system's solvent regeneration.
    
    # Costing sub-block b1, b3, b5, b6, are all function of process variables/
    # inputs that are not expected to change (e.g., fuel, flue gas flowrate). 
    # A parameter representing the value of these costs is defined as 
    # NGCC_TPC_fixed_B31B (B31B is the reference case where the process 
    # variables/inputs that are not expected to change come from).

    self.costing.NGCC_TPC_fixed_B31B = pyo.Param(
        initialize=193650607.861172, mutable=True, units=pyo.units.USD_2018
    )  
    
    # Additional process & design variables of the combined NGCC and carbon 
    # capture system (e.g., LP cooling tower duty, CCS auxiliary load, plant 
    # power, etc.) need to be calculated to be used as arguments to the costing
    # sub-blocks b2, b4, b7, b8, b9, b10, b11, b12, b13, b14, b15, and b16. 
    # After the costs from these sub-blocks are computed, an expression is 
    # defined summing the costs calculated with these sub-blocks 
    # (self.costing.NGCC_TPC_varying)).
    
    # Low pressure steam condenser duty (NGCC) as a function of fraction of 
    # steam extraction. Surrogate model developed through linear regression.
    self.costing.LP_condenser_duty = pyo.Expression(
        expr=((1327.03 - 
              (1163.4 * self.costing_setup.IP_LP_crossover_steam_fraction[0]))
              *pyo.units.MBtu/pyo.units.hr)
        ) # MMBtu/hr, units from surrogate model
    
    # Cooling tower duty calculation
    # Aggregate cooling duty requirement in CCS, CO2 compression
    # Negative sign is used to adjust the sign convention and get a positive value
    self.costing.cooling_duty_CCS_compr = pyo.Expression(
        expr=((pyo.units.convert(-self.costing_setup.dcc_duty[0], 
                                to_units=pyo.units.W) + 
              pyo.units.convert(-self.stripper_section.condenser.heat_duty[0],
                                to_units=pyo.units.W)) * 
              self.number_trains +
              self.costing_setup.CO2_compressor_auxiliary_load[0])
        ) # W
    
    # Auxilliary cooling system requirement
    self.costing.aux_cooling_load = pyo.Param(
        initialize=146, mutable=True, units=pyo.units.GJ/pyo.units.hr) # GJ/hr
    
    # Cooling tower duty
    self.costing.cooling_tower_duty = pyo.Expression(
        expr=(
            pyo.units.convert(self.costing.cooling_duty_CCS_compr, 
                              to_units=pyo.units.MBtu/pyo.units.hr) +
            pyo.units.convert(self.costing.aux_cooling_load, 
                              to_units=pyo.units.MBtu/pyo.units.hr) +
            pyo.units.convert(self.costing.LP_condenser_duty, 
                              to_units=pyo.units.MBtu/pyo.units.hr)
            )
        ) # MMBtu/hr
    
    # Circulating water flowrate calculation
    
    # Difference between circulating water return and supply temperatures
    # (cooling tower design specification, cooling water inlet temperature is 
    # 16 deg C and outlet is 27 deg C)
    self.costing.dTCW = pyo.Param(initialize=11, 
                                  mutable=True, 
                                  units=pyo.units.K) # K
    # Average cp of water between 16 deg C and 27 deg C
    self.costing.cpwater_avg = pyo.Param(initialize=4.18147, 
                                          mutable=True, 
                                          units=pyo.units.kJ/pyo.units.kg/pyo.units.K) # kJ/kg.K
    # Density of water
    self.costing.rhowater = pyo.Param(initialize=996.38, 
                                      mutable=True, 
                                      units=pyo.units.kg/pyo.units.m**3) # kg/m3
    # Circulating water flowrate
    self.costing.cir_water_flowrate = pyo.Expression(
        expr=pyo.units.convert(
            self.costing.cooling_tower_duty /
            (self.costing.dTCW * self.costing.cpwater_avg * 
              self.costing.rhowater),
            to_units=pyo.units.gal/pyo.units.min)) # gpm
    
    
    # Raw water withdrawal cost based on stripper condenser duty
    # Accounts with Raw water withdrawal as the reference/scaling parameter
    # Exhibit 5-14
    self.costing.evap_losses = pyo.Expression(
        expr=0.008 * self.costing.cir_water_flowrate 
        * self.costing.dTCW / 5.5 / pyo.units.K) # gpm
    self.costing.drift_losses = pyo.Expression(
        expr=0.00001 * self.costing.cir_water_flowrate) # gpm
    self.costing.CC = pyo.Param(initialize=4,
                                mutable=True,
                                units=pyo.units.dimensionless)
    self.costing.blowdown_losses = pyo.Expression(
        expr=self.costing.evap_losses / (self.costing.CC - 1)) # gpm
    
    self.costing.raw_water_withdrawal = pyo.Expression(
        expr=self.costing.evap_losses 
        + self.costing.drift_losses 
        + self.costing.blowdown_losses) # gpm
    
    RW_withdraw_accounts = ['3.2', '3.4', '3.5', '9.5', '14.6']
    self.b2 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b2,
            RW_withdraw_accounts,
            self.costing.raw_water_withdrawal,
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with process water discharge as the reference/scaling parameter
    # Exhibit 5-14
    
    # Process water discharge
    self.costing.process_water_discharge = pyo.Expression(
        expr=0.35 * self.costing.raw_water_withdrawal
        ) # gpm
    
    PW_discharge_accounts = ['3.7']
    self.b4 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b4,
            PW_discharge_accounts,
            self.costing.process_water_discharge, # gpm
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with gas flow to stack as the reference/scaling parameter
    # Exhibit 5-8, stream 4
    Stack_flow_gas_accounts = ['7.3', '7.4', '7.5']
    self.b8 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
            self.b8,
            Stack_flow_gas_accounts,
            pyo.units.convert(self.costing_setup.stackgas_flowrate[0], 
                              to_units=pyo.units.ft**3/pyo.units.min),
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with steam turbine gross power as the reference/scaling
    # parameter
    # Exhibit 5-9
    
    # Low pressure steam turbine power before generator losses

    self.costing.LPST_power = pyo.Expression(
        expr=((122.03 - (108.43 * 
                          self.costing_setup.IP_LP_crossover_steam_fraction[0])) * pyo.units.MW)
    )  # MWe, units from surrogate model
    # High pressure steam turbine power before generator losses
    
    self.costing.HPST_power = pyo.Param(
        initialize=56.24, mutable=True, units=pyo.units.MW
    )  # MWe
    # Intermediate pressure steam turbine power before generator losses
    
    self.costing.IPST_power = pyo.Param(
        initialize=87.39, mutable=True, units=pyo.units.MW
    )  # MWe
    # Steam turbine efficiency
    
    self.costing.turbine_eff = pyo.Param(
        initialize=0.987186, mutable=True, units=pyo.units.dimensionless
    )
    # Aggregate power from steam turbine after generator losses
    
    self.costing.ST_gross_power = pyo.Expression(
        expr=self.costing.turbine_eff
        * (
            pyo.units.convert(self.costing.LPST_power, to_units=pyo.units.kW)
            + pyo.units.convert(self.costing.IPST_power, to_units=pyo.units.kW)
            + pyo.units.convert(self.costing.HPST_power, to_units=pyo.units.kW)
        )
    )  # kW

    Steam_turbine_gross_power_accounts = ['8.1', '8.2', '8.5', '14.3']
    self.b9 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b9,
            Steam_turbine_gross_power_accounts,
            self.costing.ST_gross_power,
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with LP condenser duty as the reference/scaling parameter
    # Exhibit 5-9
    Condenser_duty_accounts = ['8.3']
    self.b10 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b10,
            Condenser_duty_accounts,
            self.costing.LP_condenser_duty,
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with cooling tower duty as the reference/scaling parameter
    # Exhibit 5-16
    Cooling_tower_accounts = ['9.1']
    self.b11 = pyo.Block()

    # Function of cooling tower duty in MMBtu/hr (includes condenser, acid gas
    # removal, and other cooling loads)

    QGESSCostingData.get_PP_costing(
            self.b11,
            Cooling_tower_accounts,
            self.costing.cooling_tower_duty,  # MMBtu/hr
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with circulating water flowrate as the reference/scaling
    # parameter
    Circ_water_accounts = ['9.2', '9.3', '9.4', '9.6', '9.7', '14.5']
    self.b12 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b12,
            Circ_water_accounts,
            self.costing.cir_water_flowrate, # gpm
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with total plant gross power as the reference/scaling parameter
    # Exhibit 5-9

    # Power from combustion turbine after generator losses 
    # (same as gas turbine power?)
    self.costing.CT_gross_power = pyo.Expression(
        expr=pyo.units.convert(
            (469.56*pyo.units.MW),
            to_units=pyo.units.kW)
        ) # kW
    
    # Plant gross power   
    self.costing.plant_gross_power = pyo.Expression(
        expr=pyo.units.convert(self.costing.CT_gross_power +
                               self.costing.ST_gross_power,
                               to_units=pyo.units.kW)
        ) # kW
        
    plant_gross_power_accounts = ['11.1', '11.7', '11.9', '13.1', '13.2',
                                  '13.3', '14.4', '14.7', '14.8', '14.9',
                                  '14.10']
    self.b13 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b13,
            plant_gross_power_accounts,
            self.costing.plant_gross_power,
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with auxilliary load as the reference/scaling parameter
    # Exhibit 5-9
    
    # Total auxilliary load calculation

    # Circulating water flowrate case B31B
    self.costing.circ_water_flow_ref = pyo.Param(
        initialize=223629.5473, mutable=True, units=pyo.units.gal / pyo.units.min
    )  # gpm
    
    # Circulating water pumps power requirement case B31B
    self.costing.circ_water_pumps_load_ref = pyo.Param(
        initialize=4580, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Circulating water pumps power requirement - scaled
    self.costing.circ_water_pumps_load = pyo.Expression(
        expr=self.costing.circ_water_pumps_load_ref
        * (self.costing.cir_water_flowrate / self.costing.circ_water_flow_ref)
    )
    
    # Combustion turbine auxilliary load
    self.costing.CT_aux_load = pyo.Param(
        initialize=1020, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Condensate pumps auxilliary load
    self.costing.condensate_pumps_aux_load = pyo.Param(
        initialize=170, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Cooling tower fans power requirement case B31B
    self.costing.cool_tower_fans_load_ref = pyo.Param(
        initialize=2370, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Cooling tower fans power requirement - scaled
    self.costing.cool_tower_fans_load = pyo.Expression(
        expr=self.costing.cool_tower_fans_load_ref
        * (self.costing.cir_water_flowrate / self.costing.circ_water_flow_ref)
    )
    
    # CO2 capture system auxilliary load
    
    # Absorber wash water pump net power requirement (kW)
    # Assume it scales linearly with flue gas flowrate
    self.costing.FG_flowrate_per_train_ref = pyo.Param(
        initialize=930909, mutable=True, units=pyo.units.ft**3 / pyo.units.min
    )  # acfm
    
    self.costing.WWashLoadRef = pyo.Param(
        initialize=71, mutable=True, units=pyo.units.kW
    )  # kW
    
    self.costing.WWashLoad = pyo.Expression(
        expr=pyo.units.convert(
            (pyo.units.convert(self.costing_setup.fluegas_flowrate[0], 
                               to_units=pyo.units.ft**3/pyo.units.min) / 
             self.number_trains)
            * self.costing.WWashLoadRef
            / self.costing.FG_flowrate_per_train_ref,
            to_units=pyo.units.kW,
        )
    )

    # Stripper reflux pump net power requirement (kW)
    # Assume it scales linearly with CO2 capture flowrate
    
    self.costing.CO2CaptureRate_ref = pyo.Param(
        initialize=290676, mutable=True, units=pyo.units.kg / pyo.units.hr
    )  # kg/hr
    self.costing.Stripperrefluxload_ref = pyo.Param(
        initialize=21.277, mutable=True, units=pyo.units.kW
    )  # kW
    self.costing.Stripperrefluxload = pyo.Expression(
        expr=pyo.units.convert(
            (pyo.units.convert(self.costing_setup.CO2_capture_rate[0], 
                               to_units=pyo.units.lb/pyo.units.hr) / 
             self.number_trains)
            * (self.costing.Stripperrefluxload_ref / self.costing.CO2CaptureRate_ref),
            to_units=pyo.units.kW,
        )
    )
    
    # Lean solvent pump 2 (after the solvent makeup is added) net power requirement (kW)
    # Assume it scales linearly with lean solvent flowrate
    
    self.costing.LSflow_ref = pyo.Param(
        initialize=21166, mutable=True, units=pyo.units.gal / pyo.units.min
    )  # gpm
    self.costing.LSpumpload_ref = pyo.Param(
        initialize=1071.35, mutable=True, units=pyo.units.kW
    )  # kW
    self.costing.LSpumpload = pyo.Expression(
        expr=pyo.units.convert(
            (pyo.units.convert(self.stripper_section.lean_solvent_flow_vol[0], 
                               to_units=pyo.units.m**3/pyo.units.s)) * 
            self.costing.LSpumpload_ref / self.costing.LSflow_ref,
            to_units=pyo.units.kW,
        )
    )
    
    self.costing.AuxLoad_CCS = pyo.Expression(
        expr=pyo.units.convert(
            self.number_trains
            * (
                pyo.units.convert(self.costing_setup.flue_gas_blower_load[0], 
                                  to_units=pyo.units.kW) +
                self.costing.WWashLoad +
                pyo.units.convert(self.absorber_section.rich_solvent_pump.work_mechanical[0], 
                                  to_units=pyo.units.kW) +
                pyo.units.convert(self.absorber_section.lean_solvent_pump.work_mechanical[0], 
                                  to_units=pyo.units.kW) +
                pyo.units.convert(self.costing_setup.dcc_pump_load[0], 
                                  to_units=pyo.units.kW) +
                self.costing.Stripperrefluxload +
                self.costing.LSpumpload
            ),
            to_units=pyo.units.kW,
        )
    )  # kW
    
    # CO2 compression system auxilliary load
    self.costing.AuxLoad_compr = pyo.Expression(
        expr=pyo.units.convert(self.costing_setup.CO2_compressor_auxiliary_load[0], 
                               to_units=pyo.units.kW)
    )  # kW
    
    # Feedwater pumps load
    self.costing.feedwater_pumps_load = pyo.Param(
        initialize=4830, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Groundwater pumps load
    # Groundwater pumps power requirement case B31B
    self.costing.groundwater_pumps_load_ref = pyo.Param(
        initialize=430, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Raw water withdrawal case B31B
    self.costing.raw_water_withdrawal_ref = pyo.Param(
        initialize=4773, mutable=True, units=pyo.units.gal / pyo.units.min
    )  # gpm
    self.costing.groundwater_pumps_load = pyo.Expression(
        expr=self.costing.groundwater_pumps_load_ref
        * self.costing.raw_water_withdrawal
        / self.costing.raw_water_withdrawal_ref
    )  # kWe
    
    # SCR load
    self.costing.SCR_load = pyo.Param(
        initialize=2, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Steam turbine auxilliaries
    self.costing.ST_aux_load = pyo.Param(
        initialize=200, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Transformer losses
    self.costing.transformer_losses = pyo.Param(
        initialize=2200, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Miscellanous load
    self.costing.misc_loads = pyo.Param(
        initialize=570, mutable=True, units=pyo.units.kW
    )  # kWe
    
    # Total auxilliary load
    self.costing.auxiliary_load = pyo.Expression(
        expr=pyo.units.convert(
            self.costing.circ_water_pumps_load
            + self.costing.CT_aux_load
            + self.costing.condensate_pumps_aux_load
            + self.costing.cool_tower_fans_load
            + self.costing.AuxLoad_CCS
            + self.costing.AuxLoad_compr
            + self.costing.feedwater_pumps_load
            + self.costing.groundwater_pumps_load
            + self.costing.SCR_load
            + self.costing.ST_aux_load
            + self.costing.transformer_losses
            + self.costing.misc_loads,
            to_units=pyo.units.kW,
        )
    )  # kW

    auxilliary_load_accounts = ['11.2', '11.3', '11.4', '11.5', '11.6', '12.1',
                                '12.2', '12.3', '12.4', '12.5', '12.6', '12.7',
                                '12.8', '12.9']
    self.b14 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b14,
            auxilliary_load_accounts,
            self.costing.auxiliary_load, # kW
            6,
            CE_index_year=CE_index_year,
        )
    
    # Accounts with STG,CTG output as the reference/scaling parameter
    # Case B31A Account 9 - Pg 501 rev 4 baseline report

    # STG CTG output

    self.costing.ctg_op = pyo.Param(
        initialize=540000, mutable=True, units=pyo.units.kW
    )  # kW, same for case B31A and case B31B
    self.costing.stg_op_ref = pyo.Param(
        initialize=200, mutable=True, units=pyo.units.MW
    )  # MW, case B31B
    self.costing.ST_gross_power_ref = pyo.Param(
        initialize=213, mutable=True, units=pyo.units.MW
    )  # MW, case B31B
    self.costing.stg_op = pyo.Expression(
        expr=pyo.units.convert(
            self.costing.ST_gross_power
            * (self.costing.stg_op_ref / self.costing.ST_gross_power_ref),
            to_units=pyo.units.kW,
        )
    )  # kW
    self.costing.stg_ctg_op = pyo.Expression(
        expr=pyo.units.convert(
            self.costing.ctg_op + self.costing.stg_op, to_units=pyo.units.kW
        )
    )  # kW

    stg_ctg_accounts = ['11.8']
    self.b15 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b15,
            stg_ctg_accounts,
            self.costing.stg_ctg_op,
            6,
            CE_index_year=CE_index_year,
        )

    # Accounts with gas turbine power as the reference/scaling parameter
    # Exhibit 5-9

    gasturbine_accounts = ['14.1']
    self.b16 = pyo.Block()

    QGESSCostingData.get_PP_costing(
            self.b16,
            gasturbine_accounts,
            self.costing.CT_gross_power, # gas turbine power,
            6,
            CE_index_year=CE_index_year,
        )
    
    # NGCC TPC components that are function of fraction of steam extracted (thus
    # stripper reboiler duty), stripper condenser duty, lean and rich pumps 
    # works, stack gas flowrate and composition, lean solvent flow rate.
    
    self.costing.NGCC_TPC_varying = pyo.Expression(
        expr=
        (pyo.units.convert(
            sum(self.b2.total_plant_cost[ac] for ac in RW_withdraw_accounts) +
            sum(self.b4.total_plant_cost[ac] for ac in PW_discharge_accounts) +
            sum(self.b8.total_plant_cost[ac] for ac in Stack_flow_gas_accounts) +
            sum(self.b9.total_plant_cost[ac] for ac in Steam_turbine_gross_power_accounts) +
            sum(self.b10.total_plant_cost[ac] for ac in Condenser_duty_accounts) +
            sum(self.b11.total_plant_cost[ac] for ac in Cooling_tower_accounts) +
            sum(self.b12.total_plant_cost[ac] for ac in Circ_water_accounts) +
            sum(self.b13.total_plant_cost[ac] for ac in plant_gross_power_accounts) +
            sum(self.b14.total_plant_cost[ac] for ac in auxilliary_load_accounts) +
            sum(self.b15.total_plant_cost[ac] for ac in stg_ctg_accounts) +
            sum(self.b16.total_plant_cost[ac] for ac in gasturbine_accounts),
            to_units=getattr(pyunits, "USD_" + CE_index_year)  # $
            )
            )
        )
    
    # The team decided to use the B31A case NGCC TPC value from the NETL 
    # baseline report.    
    self.costing.NGCC_TPC = pyo.Param(
        initialize=601239000, mutable=True, units=pyo.units.USD_2018
    )
    
    # =========================================================================
    # ============ Carbon Capture System Costs ================================
    # ***Accounts with carbon capture system units***
    
    # Flue Gas Blower 
    fg_blower_accounts = ["1.3"]
    self.b17 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b17,
        fg_blower_accounts,
        pyo.units.convert(self.costing_setup.flue_gas_blower_load[0.0], 
                          to_units=pyo.units.hp),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
        )
    
    self.costing.FG_blower_cost = pyo.Expression(
        expr=sum(self.b17.total_plant_cost[ac] for ac in fg_blower_accounts)
        )
    
    # The scaled parameter that needs to be given as input for the DCC related
    # accounts is the flue gas flowrate per train 
    
    # DCC Column
    DCC_column_accounts = ["1.6"]
    self.b18 = pyo.Block()
    
    
    QGESSCostingData.get_PP_costing(
        self.b18,
        DCC_column_accounts,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0] /
                          self.number_trains, to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.DCC_column_cost = pyo.Expression(
        expr=sum(self.b18.total_plant_cost[ac] for ac in DCC_column_accounts)
        )
    
    # DCC Column Packing
    DCC_column_packing_accounts = ["1.9"]
    self.b19 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b19,
        DCC_column_packing_accounts,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0] /
                          self.number_trains, to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.DCC_packing_cost = pyo.Expression(
        expr=sum(self.b19.total_plant_cost[ac] for ac in DCC_column_packing_accounts)
        )
    
    # DCC Pump
    DCC_pump_accounts = ["1.13"]
    self.b20 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b20,
        DCC_pump_accounts,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0] /
                          self.number_trains, to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.DCC_pump_cost = pyo.Expression(
        expr=sum(self.b20.total_plant_cost[ac] for ac in DCC_pump_accounts)
        )

    # DCC Cooler    
    DCC_cooler_accounts = ["1.16"]
    self.b21 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b21,
        DCC_cooler_accounts,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0] /
                          self.number_trains, to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.DCC_cooler_cost = pyo.Expression(
        expr=sum(self.b21.total_plant_cost[ac] for ac in DCC_cooler_accounts)
        )
    
    # Absorber volume
    # Column volume calculation, obtain absorber volume in m3
    
    # Height of absorber [m], fixed at (sub)flowsheet level
    absorber_height = self.absorber_section.absorber.length_column
    # Diameter of absorber [m], fixed at (sub)flowsheet level
    absorber_diameter = self.absorber_section.absorber.diameter_column
    # Packing height of the wash section in absorber column [m], 
    # assumed value, fixed at (sub)flowsheet level 
    wash_section_packing_height = self.costing_setup.wash_section_packing_height
    
    # Absorber + water wash volume
    self.costing.absorber_and_WW_volume = pyo.Expression(
        expr=pyo.units.convert(
            pi*(1.25*(absorber_height + wash_section_packing_height) 
                + (absorber_height + wash_section_packing_height)*0.75/10)
            * absorber_diameter**2/4,
            to_units=pyo.units.m**3
            )
        ) # m3
    
    # Absorber with heads volume
    self.costing.absorber_volume_column_withheads = pyo.Expression(
        expr=pyo.units.convert(
            pi * (absorber_diameter ** 2 * absorber_height) / 4
            + pi / 6 * absorber_diameter ** 3,
            to_units=pyo.units.m**3
            ), 
        doc="Volume of column with heads (cubic m)",
        )
    
    absorber_accounts = ["2.3"]
    self.b22 = pyo.Block() 

    QGESSCostingData.get_PP_costing(
        self.b22,
        absorber_accounts,
        self.costing.absorber_and_WW_volume, # with Water Wash
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
        )
        
    self.costing.absorber_column_cost = pyo.Expression(
        expr=sum(self.b22.total_plant_cost[ac] for ac in absorber_accounts)
        )
    
    # Absorber packing
    # Packing volume calculation, obtain absorber packing volume in m3
    # With water wash
    self.costing.absorber_and_WW_packing_volume = pyo.Expression(
        expr=pyo.units.convert(
            pi*(absorber_height + wash_section_packing_height)
            *(absorber_diameter**2)/4,
            to_units=pyo.units.m**3
            )
        ) # m3
    
    # Without water wash
    self.costing.absorber_packing_volume = pyo.Expression(
        expr=pyo.units.convert(
            pi*absorber_height*(absorber_diameter**2)/4,
            to_units=pyo.units.m**3
            )
        ) # m3
    
    absorber_packing_accounts = ["3.3"]
    self.b23 = pyo.Block()  
    
    QGESSCostingData.get_PP_costing(
        self.b23,
        absorber_packing_accounts,
        self.costing.absorber_and_WW_packing_volume, # with Water Wash
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.absorber_packing_cost = pyo.Expression(
        expr=sum(self.b23.total_plant_cost[ac] for ac in absorber_packing_accounts)
        )
    
    # blocks 24-27 are left out as they are intercooling blocks and the  IDAES 
    # absorber model used in this study does not have intercooling
    
    # WW Recirculation Pump
    ww_pump_accounts = ["6.3"]
    self.b28 = pyo.Block()
    
    # Assume L/G ratio in the wash column is equivalent to 1
    # Assume 1 kg water is equivalent to 1 litre
    # Obtain wash water rated flow in gpm, considering 10 % margin
    self.costing.ww_rated_flow = 1.1 * pyo.units.convert(
        self.absorber_section.stack_gas_flow_mass[0] * pyo.units.L / pyo.units.kg,
        to_units=pyo.units.gal / pyo.units.min,
    )  # gpm

    QGESSCostingData.get_PP_costing(
        self.b28,
        ww_pump_accounts,
        self.costing.ww_rated_flow,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.ww_pump_cost = pyo.Expression(
    expr=sum(self.b28.total_plant_cost[ac] for ac in ww_pump_accounts)
        )
    
    # WW Cooler
    
    # Wash water cooler heat transfer surface area (m2)
    # Assume it scales linearly with flue gas flowrate
    
    self.costing.wwcooler_htsa_ref = pyo.Param(
        initialize=20, mutable=True, units=pyo.units.m**2
    )  # m2
    self.costing.wwcooler_htsa = pyo.Expression(
        expr=pyo.units.convert(
            (
                pyo.units.convert(
                    self.costing_setup.fluegas_flowrate[0],
                    to_units=pyo.units.ft**3 / pyo.units.min,
                )
                / self.number_trains
            )
            * self.costing.wwcooler_htsa_ref
            / self.costing.FG_flowrate_per_train_ref,
            to_units=pyo.units.m**2,
        )
    )  
    
    ww_cooler_accounts = ["7.3"]
    self.b29 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b29,
        ww_cooler_accounts,
        self.costing.wwcooler_htsa,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.ww_cooler_cost = pyo.Expression(
    expr=sum(self.b29.total_plant_cost[ac] for ac in ww_cooler_accounts)
    )
    
    # Rich solvent pump
    rich_solvent_pump_account = ["8.3"]
    self.b30 = pyo.Block()
    
    # Consider 10 % margin when computing the rich solvent rated Fflowrate
    # based on the flowsheet rich solvent volumetric flowrate (defined in the 
    # absorber section subflowsheet)

    QGESSCostingData.get_PP_costing(
        self.b30,
        rich_solvent_pump_account,
        1.1*pyo.units.convert(
            self.absorber_section.rich_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.rich_solvent_pump_cost = pyo.Expression(
        expr=sum(self.b30.total_plant_cost[ac] for ac in rich_solvent_pump_account)
        )
    
    # Lean rich heat exchanger
    lean_rich_hex_accounts = ["9.3"]
    self.b31 = pyo.Block()    
    
    QGESSCostingData.get_PP_costing(
        self.b31,
        lean_rich_hex_accounts,
        pyo.units.convert(self.absorber_section.lean_rich_heat_exchanger.area, 
                          to_units=pyo.units.m**2),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.lean_rich_hex_cost = pyo.Expression(
        expr=sum(self.b31.total_plant_cost[ac] for ac in lean_rich_hex_accounts)
        )
    
    # Stripper volume
    # Column volume calculation, obtain stripper volume in m3
    
    # Height of stripper [m], fixed at (sub)flowsheet level
    stripper_height = self.stripper_section.stripper.length_column
    # Diameter of stripper [m], fixed at (sub)flowsheet level
    stripper_diameter = self.stripper_section.stripper.diameter_column
    
    self.costing.stripper_volume = pyo.Expression(
        expr=pyo.units.convert(
            pi*(1.25*stripper_height + stripper_height*0.75/10)
            *stripper_diameter**2/4,
            to_units=pyo.units.m**3)
        )# m3
    
    self.costing.stripper_volume_column_withheads = pyo.Expression(
        expr=(
            pi * (stripper_diameter ** 2 * stripper_height) / 4
            + pi / 6 * stripper_diameter ** 3
        ),
        doc="Volume of column with heads (cubic m)",
    )
    
    stripper_accounts = ["10.3"]
    self.b32 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b32,
        stripper_accounts,
        self.costing.stripper_volume,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
        
    self.costing.stripper_column_cost = pyo.Expression(
        expr=sum(self.b32.total_plant_cost[ac] for ac in stripper_accounts)
        )
    
    # Stripper packing
    # Packing volume calculation, obtain stripper packing volume in m3
    stripper_packing_accounts = ["11.3"]
    self.b33 = pyo.Block()
    
    # Obtain stripper packing volume in m3
    self.costing.stripper_packing_volume = pyo.Expression(
        expr=pyo.units.convert(
            pi*stripper_height*(stripper_diameter**2)/4,
            to_units=pyo.units.m**3)
        )

    QGESSCostingData.get_PP_costing(
        self.b33,
        stripper_packing_accounts,
        self.costing.stripper_packing_volume,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )

    self.costing.stripper_packing_cost = pyo.Expression(
        expr=sum(self.b33.total_plant_cost[ac] for ac in stripper_packing_accounts)
        )

    # Stripper condenser - related costs
    
    # Stripper condenser cost based on area
    stripper_condenser_accounts = ["12.3"]
    self.b34 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b34,
        stripper_condenser_accounts,
        pyo.units.convert(self.stripper_section.condenser.area, 
                          to_units=pyo.units.m**2),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.stripper_condenser_cost = pyo.Expression(
        expr=sum(self.b34.total_plant_cost[ac] for ac in stripper_condenser_accounts)
        )
    
    # Stripper resboiler - related costs
    # Stripper reboiler cost based on area 
    stripper_reboiler_accounts = ["13.3"]
    self.b35 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b35,
        stripper_reboiler_accounts,
        pyo.units.convert(self.stripper_section.reboiler.area, 
                          to_units=pyo.units.m**2),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.stripper_reboiler_cost = pyo.Expression(
        expr=sum(self.b35.total_plant_cost[ac] for ac in stripper_reboiler_accounts)
        )
    
    # Stripper reflux drum
    stripper_reflux_drum_accounts = ["14.3"]
    self.b36 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b36,
        stripper_reflux_drum_accounts,
        (pyo.units.convert(self.costing_setup.CO2_capture_rate[0], 
                                to_units=pyo.units.kg/pyo.units.hr) / 
              self.number_trains.value),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.stripper_reflux_drum_cost = pyo.Expression(
        expr=sum(self.b36.total_plant_cost[ac] for ac in stripper_reflux_drum_accounts)
        )
    
    # Lean solvent pump
    lean_solvent_pump_account = ["15.3"]
    self.b37 = pyo.Block()
    # Consider 10 % margin when computing the lean solvent rated flowrate, 
    # based on the flowsheet lean solvent volumetric flowrate (defined in the 
    # stripper section subflowsheet)
    
    QGESSCostingData.get_PP_costing(
        self.b37,
        lean_solvent_pump_account,
        1.1*pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.lean_solvent_pump_cost = pyo.Expression(
        expr=sum(self.b37.total_plant_cost[ac] for ac in lean_solvent_pump_account)
        )
    
    # block 38 is skipped (lean solvent cooling related block) as the IDAES 
    # carbon capture flowsheet does not have lean solvent cooling
    # TODO: add lean solvent cooling costs as option
    
    # block 39 is skipped (solvent stripper reclaimer related block) as it is 
    # a function of solvent makeup and the IDAES carbon capture flowsheet does 
    # not have solvent makeup (MEA is considered non-volatile for now)
    # TODO: figure out whether we need to include solvent stripper reclaimer costs 
    
    # Solvent filtration
    solvent_filtration_account = ["18.3"]
    self.b40 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b40,
        solvent_filtration_account,
        pyo.units.convert(self.stripper_section.lean_solvent_flow_vol[0], 
                          to_units=pyo.units.gal/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.solvent_filtration_cost = pyo.Expression(
        expr=sum(self.b40.total_plant_cost[ac] for ac in solvent_filtration_account)
        )
    
    # Solvent Storage Tank - multiplying by number of trains because this cost
    # will be added to CCS_TPC * number_trains
    solvent_storage_tank_account = ["19.3"]
    self.b41 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b41,
        solvent_storage_tank_account,
        pyo.units.convert(self.stripper_section.lean_solvent_flow_vol[0], 
                          to_units=pyo.units.gal/pyo.units.min) *
              self.number_trains.value,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.solvent_storage_tank_cost = pyo.Expression(
        expr=sum(self.b41.total_plant_cost[ac] for ac in solvent_storage_tank_account)
        )
    
    # Stripper reflux pump
    stripper_reflux_pump_account = ["50.3"]
    self.b42 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b42,
        stripper_reflux_pump_account,
        (pyo.units.convert(self.costing_setup.CO2_capture_rate[0], 
                                to_units=pyo.units.kg/pyo.units.hr) / 
              self.number_trains.value),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.stripper_reflux_pump_cost = pyo.Expression(
        expr=sum(self.b42.total_plant_cost[ac] for ac in stripper_reflux_pump_account)
        )
    
    # block 43 is skipped - it costs a second lean solvent pump, function of CO2 
    # capture rate 
    
    # Solvent Sump
    Solvent_Sump_account = ["51.1"]
    self.b44 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b44,
        Solvent_Sump_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Solvent_Sump_cost = pyo.Expression(
        expr=sum(self.b44.total_plant_cost[ac] for ac in Solvent_Sump_account)
    )
    
    # Solvent Sump Pump
    Solvent_Sump_pump_account = ["52.1"]
    self.b45 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b45,
        Solvent_Sump_pump_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Solvent_Sump_pump_cost = pyo.Expression(
        expr=sum(self.b45.total_plant_cost[ac] for ac in Solvent_Sump_pump_account)
    )
    
    # Solvent Sump Filter
    Solvent_Sump_Filter_account = ["53.1"]
    self.b46 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b46,
        Solvent_Sump_Filter_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Solvent_Sump_Filter_cost = pyo.Expression(
        expr=sum(self.b46.total_plant_cost[ac] for ac in Solvent_Sump_Filter_account)
    )
     
    # Solvent Sump Pit Pump
    Solvent_Sump_Pit_Pump_account = ["54.1"]
    self.b47 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b47,
        Solvent_Sump_Pit_Pump_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )

    self.costing.Solvent_Sump_Pit_Pump_cost = pyo.Expression(
        expr=sum(self.b47.total_plant_cost[ac] for ac in Solvent_Sump_Pit_Pump_account)
    )
       
    # Pre Scrubber Pump
    # The scaled parameter that needs to be given as input for the pre scrubber
    # related accounts is the flue gas flowrate per train measured in ft3/min
    
    Pre_Scrubber_Pump_account = ["55.1"]
    self.b48 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b48,
        Pre_Scrubber_Pump_account,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0] /
                          self.number_trains, to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Pre_Scrubber_Pump_cost = pyo.Expression(
        expr=sum(self.b48.total_plant_cost[ac] for ac in Pre_Scrubber_Pump_account)
    )    
    
    # Reboiler condensate pot
    reboiler_condensate_pot_account = ["56.1"]
    self.b49 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b49,
        reboiler_condensate_pot_account,
        pyo.units.convert(self.stripper_section.reboiler.heat_duty[0], 
                          to_units=pyo.units.MW),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.reboiler_condensate_pot_cost = pyo.Expression(
        expr=sum(self.b49.total_plant_cost[ac] for ac in reboiler_condensate_pot_account)
        )
    
    # DCC Water Filter
    # The scaled parameter that needs to be given as input for the DCC related
    # accounts is the flue gas flowrate per train measured in ft3/min
    
    DCC_Water_Filter_account = ["57.1"]
    self.b50 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b50,
        DCC_Water_Filter_account,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0] /
                          self.number_trains, to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.DCC_Water_Filter_cost = pyo.Expression(
        expr=sum(self.b50.total_plant_cost[ac] for ac in DCC_Water_Filter_account)
    )

    # Corrosion Inhib Package
    Corrosion_Inhib_Package_account = ["58.1"]
    self.b51 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b51,
        Corrosion_Inhib_Package_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )


    self.costing.Corrosion_Inhib_Package_cost = pyo.Expression(
        expr=sum(self.b51.total_plant_cost[ac] for ac in Corrosion_Inhib_Package_account)
    )
    
    # Antifoam Feed Package
    Antifoam_Feed_Package_account = ["59.1"]
    self.b52 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b52,
        Antifoam_Feed_Package_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0], 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Antifoam_Feed_Package_cost = pyo.Expression(
        expr=sum(self.b52.total_plant_cost[ac] for ac in Antifoam_Feed_Package_account)
    )
    
    # NaOH Makeup Pump
    # NaOH is most likely used to remove the HCl in the syngas scrubber (HCl is
    # reacted with sodium hydroxide (NaOH) to form sodium chloride (NaCl) - as
    # per the NETL report rev 4a
    
    # The scaled parameter that needs to be given as argument to blocks 53 & 55
    # is the total flue gas flowrate measured in ft3/min
    
    NaOH_Makeup_Pump_account = ["60.1"]
    self.b53 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b53,
        NaOH_Makeup_Pump_account,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0], 
                          to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )

    self.costing.NaOH_Makeup_Pump_cost = pyo.Expression(
        expr=sum(self.b53.total_plant_cost[ac] for ac in NaOH_Makeup_Pump_account)
        )
    
    # block 54 is the cost of Solvent Makeup Pump, hence skipped (MEA is considered
    # to be nonvolatile)
    
    # NaOH Storage Tank
    NaOH_Storage_Tank_account = ["62.1"]
    self.b55 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b55,
        NaOH_Storage_Tank_account,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0], 
                          to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.NaOH_Storage_Tank_cost = pyo.Expression(
        expr=sum(self.b55.total_plant_cost[ac] for ac in NaOH_Storage_Tank_account)
        )
    
    # Foundations
    Foundations_accounts = ["34.1"]
    self.b67 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b67,
        Foundations_accounts,
        pyo.units.convert(self.costing_setup.fluegas_flowrate[0], 
                          to_units=pyo.units.ft**3/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Foundations_cost = pyo.Expression(
        expr=sum(self.b67.total_plant_cost[ac] for ac in Foundations_accounts)
        )
    
    # Interconnecting Piping
    Interconnecting_Piping_account = ["35.1"]
    self.b68 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b68,
        Interconnecting_Piping_account,
        pyo.units.convert(
            self.stripper_section.lean_solvent_flow_vol[0] * self.number_trains, 
            to_units=pyo.units.gal/pyo.units.min), # gpm
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )    
    
    self.costing.Interconnecting_Piping_cost = pyo.Expression(
        expr=sum(self.b68.total_plant_cost[ac] for ac in Interconnecting_Piping_account)
        )
    
    # Additional Line Items
    
    # Station Service Equipment
    
    Station_Service_Equipment_account = ["22.1"]
    self.b56 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b56,
        Station_Service_Equipment_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Station_Service_Equipment_cost = pyo.Expression(
        expr=sum(self.b56.total_plant_cost[ac] for ac in Station_Service_Equipment_account)
    )
    
    # Switchgear & Motor Control
    
    Switchgear_Motor_Control_account = ["23.1"]
    self.b57 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b57,
        Switchgear_Motor_Control_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Switchgear_Motor_Control_cost = pyo.Expression(
        expr=sum(self.b57.total_plant_cost[ac] for ac in Switchgear_Motor_Control_account)
    )
    
    # Conduit & Cable Tray
    
    Conduit_Cable_Tray_account = ["24.1"]
    self.b58 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b58,
        Conduit_Cable_Tray_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Conduit_Cable_Tray_cost = pyo.Expression(
        expr=sum(self.b58.total_plant_cost[ac] for ac in Conduit_Cable_Tray_account)
    )
    
    # Wire & Cable
    
    Wire_Cable_account = ["25.1"]
    self.b59 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b59,
        Wire_Cable_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    self.costing.Wire_Cable_cost = pyo.Expression(
        expr=sum(self.b59.total_plant_cost[ac] for ac in Wire_Cable_account)
    )
    
    # Main Power Transformers
    
    Main_Power_Transformers_account = ["26.1"]
    self.b60 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b60,
        Main_Power_Transformers_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Main_Power_Transformers_cost = pyo.Expression(
        expr=sum(self.b60.total_plant_cost[ac] for ac in Main_Power_Transformers_account)
    )
    
    # Electrical Foundations
    
    Electrical_Foundations_account = ["27.1"]
    self.b61 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b61,
        Electrical_Foundations_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Electrical_Foundations_cost = pyo.Expression(
        expr=sum(self.b61.total_plant_cost[ac] for ac in Electrical_Foundations_account)
    )
    
    # Control Boards, Panels & Racks
    
    Control_Boards_Panels_Racks_account = ["28.1"]
    self.b62 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b62,
        Control_Boards_Panels_Racks_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Control_Boards_Panels_Racks_cost = pyo.Expression(
        expr=sum(
            self.b62.total_plant_cost[ac] for ac in Control_Boards_Panels_Racks_account
        )
    )
    
    # Distributed Control System Equipment
    
    Distributed_Control_System_Equipment_account = ["29.1"]
    self.b63 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b63,
        Distributed_Control_System_Equipment_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Distributed_Control_System_Equipment_cost = pyo.Expression(
        expr=sum(
            self.b63.total_plant_cost[ac]
            for ac in Distributed_Control_System_Equipment_account
        )
    )
    
    # Instrument Wiring & Tubing
    
    Instrument_Wiring_Tubing_account = ["30.1"]
    self.b64 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b64,
        Instrument_Wiring_Tubing_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Instrument_Wiring_Tubing_cost = pyo.Expression(
        expr=sum(self.b64.total_plant_cost[ac] for ac in Instrument_Wiring_Tubing_account)
    )
    
    # Other Instrumentation & Controls Equipment
    
    Other_Instrumentation_Controls_Equipment_account = ["31.1"]
    self.b65 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b65,
        Other_Instrumentation_Controls_Equipment_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Other_Instrumentation_Controls_Equipment_cost = pyo.Expression(
        expr=sum(
            self.b65.total_plant_cost[ac]
            for ac in Other_Instrumentation_Controls_Equipment_account
        )
    )
    # Other Buildings & Structures
    
    Other_Buildings_Structures_account = ["33.1"]
    self.b66 = pyo.Block()
    
    QGESSCostingData.get_PP_costing(
        self.b66,
        Other_Buildings_Structures_account,
        self.costing.AuxLoad_CCS,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.Other_Buildings_Structures_cost = pyo.Expression(
        expr=sum(self.b66.total_plant_cost[ac] for ac in Other_Buildings_Structures_account)
    )
        

    # ######################## Economic Metrics ###############################
    
    # TPC of carbon capture system, one train
    
    self.costing.CCS_TPC = pyo.Expression(
        expr=(
            pyo.units.convert(
                self.costing.FG_blower_cost
                + self.costing.DCC_column_cost
                + self.costing.DCC_packing_cost
                + self.costing.DCC_pump_cost
                + self.costing.DCC_cooler_cost
                + self.costing.absorber_column_cost
                + self.costing.absorber_packing_cost
                + self.costing.ww_pump_cost
                + self.costing.ww_cooler_cost
                + self.costing.rich_solvent_pump_cost
                + self.costing.lean_rich_hex_cost
                + self.costing.stripper_column_cost
                + self.costing.stripper_packing_cost
                + self.costing.stripper_condenser_cost
                + self.costing.stripper_reboiler_cost
                + self.costing.stripper_reflux_drum_cost
                + self.costing.lean_solvent_pump_cost
                # + self.costing.solvent_stripper_reclaimer_cost
                + self.costing.solvent_filtration_cost
                + self.costing.stripper_reflux_pump_cost
                # # + self.costing.LS_Pump2_cost
                + self.costing.Solvent_Sump_cost
                + self.costing.Solvent_Sump_pump_cost
                + self.costing.Solvent_Sump_Filter_cost
                + self.costing.Solvent_Sump_Pit_Pump_cost
                + self.costing.Pre_Scrubber_Pump_cost
                + self.costing.reboiler_condensate_pot_cost
                + self.costing.DCC_Water_Filter_cost
                + self.costing.Corrosion_Inhib_Package_cost
                + self.costing.Antifoam_Feed_Package_cost,
                to_units=getattr(pyunits, "USD_" + CE_index_year),  # $
            )
        )
    )
        
    # Removal System Equipment TPC
    self.costing.RemovalSystemEquipment_TPC = pyo.Expression(
        expr=(
            (self.number_trains * self.costing.CCS_TPC)
            + pyo.units.convert(
                self.costing.solvent_storage_tank_cost
                + self.costing.NaOH_Makeup_Pump_cost
                + self.costing.NaOH_Storage_Tank_cost
                + self.costing.Foundations_cost
                + self.costing.Interconnecting_Piping_cost,
                to_units=getattr(pyunits, "USD_" + CE_index_year),
            )
        )
    )  # $    
    
    # Site Improvements
    
    self.costing.CO2CapTPC_ref = pyo.Param(
        initialize=337408894.9, mutable=True, units=pyo.units.USD_2018
    )  # 2018 $
    
    self.costing.Site_Improvements_BEC_ref = pyo.Param(
        initialize=4287211.38811096, mutable=True, units=pyo.units.USD_2021
    )  # 2021 $
    
    self.costing.Site_Improvements_BEC = pyo.Expression(
        expr=pyo.units.convert(
            self.costing.Site_Improvements_BEC_ref, to_units=pyo.units.USD_2018
        )
        * (
            pyo.units.convert(
                self.costing.RemovalSystemEquipment_TPC, to_units=pyo.units.USD_2018
            )
            / self.costing.CO2CapTPC_ref
        )
        ** 0.2
    )  # $ 2018
    
    self.costing.Site_Improvements_EPCC = pyo.Expression(
        expr=0.175 * self.costing.Site_Improvements_BEC
    )
    
    self.costing.Site_Improvements_ProcCont = pyo.Expression(
        expr=0.17 * self.costing.Site_Improvements_BEC
    )
    
    self.costing.Site_Improvements_ProjCont = pyo.Expression(
        expr=0.175
        * (
            self.costing.Site_Improvements_BEC
            + self.costing.Site_Improvements_EPCC
            + self.costing.Site_Improvements_ProcCont
        )
    )
    
    self.costing.Site_Improvements_cost = pyo.Expression(
        expr=(
            pyunits.convert(
                self.costing.Site_Improvements_BEC
                + self.costing.Site_Improvements_EPCC
                + self.costing.Site_Improvements_ProcCont
                + self.costing.Site_Improvements_ProjCont,
                to_units=pyo.units.MUSD_2018,
            )
        )
    )  # MM USD 2018
    
    self.costing.RemovalSystem_TPC = pyo.Expression(
        expr=self.costing.RemovalSystemEquipment_TPC
        + pyo.units.convert(
            self.costing.Station_Service_Equipment_cost
            + self.costing.Switchgear_Motor_Control_cost
            + self.costing.Conduit_Cable_Tray_cost
            + self.costing.Wire_Cable_cost
            + self.costing.Main_Power_Transformers_cost
            + self.costing.Electrical_Foundations_cost
            + self.costing.Control_Boards_Panels_Racks_cost
            + self.costing.Distributed_Control_System_Equipment_cost
            + self.costing.Instrument_Wiring_Tubing_cost
            + self.costing.Other_Instrumentation_Controls_Equipment_cost
            + self.costing.Other_Buildings_Structures_cost
            + self.costing.Site_Improvements_cost,
            to_units=getattr(pyunits, "USD_" + CE_index_year),
        )  # $
    )    
        
    
    # MEA solvent initial fill cost
    self.costing.solvent_cost = pyo.Param(
        initialize=2.09,
        units=getattr(pyunits, "USD_" + CE_index_year)/pyo.units.kg) # $/kg
    
    self.costing.RemovalSystem_Equip_Adjust = pyo.Expression(
        expr=pyo.units.convert(self.costing_setup.solvent_fill_init[0]
                               * self.costing.solvent_cost,
                               to_units=getattr(pyunits, "MUSD_" + CE_index_year)
                               )
        )
    
    # References: (B31B) (Exhibit 5-31, Rev 4 baseline report),
    # Exhibit 3-32 QGESS Capital Cost Scaling
    # Account 5.1 - CO2 Removal System
    
    self.costing.CC5_1 = pyo.Expression(
        expr=self.costing.RemovalSystem_TPC + self.costing.RemovalSystem_Equip_Adjust
    )
    
    # Account 5.4 CO2 Compression & Drying
    
    self.costing.CompAuxLoad_Base = pyo.Param(
        initialize=22918.0318, mutable=True, units=pyo.units.hp
    )  # hp
    self.costing.CC5_4 = pyo.Expression(
        expr=59674000
        * pyo.units.USD_2018
        * (
            pyo.units.convert(
                self.costing_setup.CO2_compressor_auxiliary_load[0], to_units=pyo.units.hp
            )
            / self.costing.CompAuxLoad_Base
        )
        ** 0.41
    )
    
    # Account 5.5 CO2 Compressor Aftercooler - assume its cost scales with CO2 capture flowrate
    
    self.costing.CO2capflow_Base = pyo.Param(
        initialize=223619.8061, mutable=True, units=pyo.units.kg / pyo.units.hr
    )  # kg/hr
    self.costing.CC5_5 = pyo.Expression(
        expr=498000
        * pyo.units.USD_2018
        * (
            pyo.units.convert(
                self.costing_setup.CO2_capture_rate[0], to_units=pyo.units.kg / pyo.units.hr
            )
            / self.costing.CO2capflow_Base
        )
        ** 0.6
    )
    # Account 5.12 Gas Cleanup Foundations
    
    self.costing.Reference_Capture = pyo.Param(
        initialize=493482.814, mutable=True, units=pyo.units.lb / pyo.units.hr
    )  # lb/hr
    self.costing.CC5_12 = pyo.Expression(
        expr=1145000
        * pyo.units.USD_2018
        * (self.costing_setup.CO2_capture_rate[0] / self.costing.Reference_Capture) ** 0.79
    )
    
    # No of trains are included in the removal system
    
    self.costing.FG_Cleanup_TPC = pyo.Expression(
        expr= self.costing.CC5_1
        + self.costing.CC5_4
        + self.costing.CC5_5
        + self.costing.CC5_12
    )
    
    # TPC of the integrated system

    self.costing.TPC = pyo.Var(
        initialize = 500, 
        units=CE_index_units)
    
    @self.costing.Constraint()
    def TPC_eqn(b):
        return 1e-3 * b.TPC == 1e-3 * 1e-6 * (b.NGCC_TPC + b.FG_Cleanup_TPC)

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # O&M cost calculation for NGCC with capture
    
    # LCOE Calculation components
    
    LCOE_costcomp_units = getattr(pyunits, "USD_" + CE_index_year)
    LCOE_units = getattr(pyunits, "USD_" + CE_index_year)/(pyunits.MW * pyunits.hr)
    CO2Cost_units = getattr(pyunits, "USD_" + CE_index_year)/(pyunits.tonne)
    
    # The levelized cost of electricity (LCOE_nocap_B31A), CO2 emissions (CO2_emission_nocap_B31A), 
    # net power (plant_net_power_nocap_B31A) of the NGCC plant without capture are 
    # taken from the NETL COST AND PERFORMANCE BASELINE FOR FOSSIL ENERGY PLANTS 
    # VOLUME 1: BITUMINOUS COAL AND NATURAL GAS TO ELECTRICITY, revision 4a, 
    # and are defined as parameters.
    
    self.costing.LCOE_nocap_B31A = pyo.Param(
        initialize=43.3, mutable=True, 
        units=pyo.units.USD_2018 / pyo.units.MW / pyo.units.hr  
    ) # $/MWh
    
    # From Exhibit 5-8. Case B31A stream table, NGCC without capture, the flue
    # gas flow rate (stream #4) is 8658430 lb/h and contains 4.08 % CO2
    self.costing.CO2_emission_nocap_B31A = pyo.Param(
        initialize=8658430*0.0408, mutable=True, 
        units=pyo.units.lb/pyo.units.hr 
    ) # lb/h
    
    self.costing.plant_net_power_nocap_B31A = pyo.Param(
        initialize=727, mutable=True, 
        units=pyo.units.MW  
    ) # MW
    
    self.costing.CO2_emission_cap = pyo.Expression(
        expr=pyo.units.convert(
            self.number_trains * 
            self.absorber_section.stack_gas_flow_mass_CO2[0],
            to_units=pyo.units.lb/pyo.units.hr,
        )
    )  # lb/h
    
    # Calculate Plant Net Power
    self.costing.plant_net_power = pyo.Var(
        initialize = 500, 
        units=pyo.units.MW)
    
    # self.costing.plant_net_power.fix()
    
    @self.costing.Constraint()
    def plant_net_power_eqn(b):
        return (1e-2 * b.plant_net_power == 
                1e-2 * 1e-3 * (b.parent_block().costing.plant_gross_power - 
                               b.parent_block().costing.auxiliary_load)
                )
    
    calculate_variable_from_constraint(
        self.costing.plant_net_power, self.costing.plant_net_power_eqn
    )
    
    # Define variables that need to be given as arguments in the second costing block
    # They are defined as variables because they need to be an indexed component    
    
    self.costing.natural_gas_rate = pyo.Var(
        self.time,
        initialize=4500,
        units=pyo.units.MBTU / pyo.units.hr)
    
    @self.costing.Constraint(self.time)
    def natural_gas_rate_eqn(b, t):
        return (1e-3 * b.natural_gas_rate[t] == 1e-3 * pyo.units.convert(
            b.parent_block().costing_setup.fuelgas_flowrate[t]
            *22500 * pyo.units.BTU / pyo.units.lb,  # heating value of 22500 BTU/lb
            to_units=pyo.units.MBTU / pyo.units.hr,
        )
    )
     
    for t in self.time:
        calculate_variable_from_constraint( 
            self.costing.natural_gas_rate[t], self.costing.natural_gas_rate_eqn[t]
            )
        
    
    self.costing.water_rate = pyo.Var(
            self.time,
            initialize=4500, 
            units=pyo.units.gal / pyo.units.min)
    
    @self.costing.Constraint(self.time)
    def water_rate_eqn(b, t):
        return (1e-3 * b.water_rate[t] == 1e-3 * pyo.units.convert(
                    b.raw_water_withdrawal, 
                    to_units=pyo.units.gal / pyo.units.min
                    )
        )
       
    for t in self.time:
        calculate_variable_from_constraint( 
            self.costing.water_rate[t], self.costing.water_rate_eqn[t]
            )
        
        
    self.costing.water_treatment_chemicals_rate = pyo.Var(
        self.time,
        initialize=10, 
        units=pyo.units.ton / pyo.units.d)
    
    @self.costing.Constraint(self.time)
    def water_treatment_chemicals_rate_eqn(b, t):
        return (b.water_treatment_chemicals_rate[t] == pyo.units.convert(
                b.raw_water_withdrawal
                / (4773 * pyo.units.gal / pyo.units.min)  # 4773 gpm is the BB ref value
                * 10.2
                * pyo.units.ton
                / pyo.units.d,  # 10.2 ton/day is the BB ref value
                to_units=pyo.units.ton / pyo.units.d,
            )
        )
    
    for t in self.time:
        calculate_variable_from_constraint( 
            self.costing.water_treatment_chemicals_rate[t], 
            self.costing.water_treatment_chemicals_rate_eqn[t]
            )
        
    self.costing.solvent_rate = pyo.Var(
        self.time,
        initialize=100, 
        units=pyo.units.kg / pyo.units.hr)
    self.costing.solvent_rate.fix(0)
    
    self.costing.SCR_catalyst_rate = pyo.Var(
        self.time, initialize=3.10, units=pyo.units.ft**3 / pyo.units.d
    )
    self.costing.SCR_catalyst_rate.fix(3.1)
    
    self.costing.ammonia_rate = pyo.Var(
        self.time, initialize=3.5, units=pyo.units.ton / pyo.units.d
    )
    self.costing.ammonia_rate.fix(3.5)
    
    
    self.costing.triethylene_glycol_rate = pyo.Var(
        self.time,
        initialize=300,
        units=pyo.units.gal / pyo.units.d)
        
    @self.costing.Constraint(self.time)
    def triethylene_glycol_rate_eqn(b, t):
        return (1e-2 * b.triethylene_glycol_rate[t] == 1e-2 * pyo.units.convert(
                b.parent_block().costing_setup.CO2_capture_rate[t]
                / (493588 * pyo.units.lb / pyo.units.hr)  # 493588 lb/hr is the BB ref value
                * 394
                * pyo.units.gal
                / pyo.units.d,  # 394 gal/day is the BB ref value
                to_units=pyo.units.gal / pyo.units.d,
            )
        )
    
    for t in self.time:
        calculate_variable_from_constraint( 
            self.costing.triethylene_glycol_rate[t], self.costing.triethylene_glycol_rate_eqn[t]
        )
        
    
    self.costing.SCR_catalyst_waste_rate = pyo.Var(
        self.time, initialize=3.10, units=pyo.units.ft**3 / pyo.units.d
    )
    self.costing.SCR_catalyst_waste_rate.fix(3.1)
    
    
    self.costing.triethylene_glycol_waste_rate = pyo.Var(
        self.time,
        initialize=300,
        units=pyo.units.gal / pyo.units.d)
        
    @self.costing.Constraint(self.time)
    def triethylene_glycol_waste_rate_eqn(b, t):
        return (1e-2 * b.triethylene_glycol_waste_rate[t] == 1e-2 * pyo.units.convert(
                b.parent_block().costing_setup.CO2_capture_rate[t]
                / (493588 * pyo.units.lb / pyo.units.hr)  # 493588 lb/hr is the BB ref value
                * 394
                * pyo.units.gal
                / pyo.units.d,  # 394 gal/day is the BB ref value
                to_units=pyo.units.gal / pyo.units.d,
            )
        )
    
    for t in self.time:
        calculate_variable_from_constraint( 
            self.costing.triethylene_glycol_waste_rate[t], self.costing.triethylene_glycol_waste_rate_eqn[t]
        )
        
        
    self.costing.amine_purification_unit_waste_rate = pyo.Var(
        self.time, initialize=6.11, units=pyo.units.ton / pyo.units.d
    )
    self.costing.amine_purification_unit_waste_rate.fix(6.11)
    
    
    self.costing.thermal_reclaimer_unit_waste_rate = pyo.Var(
        self.time, initialize=0.543, units=pyo.units.ton / pyo.units.d
    )
    self.costing.thermal_reclaimer_unit_waste_rate.fix(0.543)


    rates = [
        self.costing.natural_gas_rate,
        self.costing.water_rate,
        self.costing.water_treatment_chemicals_rate,       #
        self.costing.solvent_rate,                         #
        self.costing.SCR_catalyst_rate,                    #
        self.costing.ammonia_rate,                         #
        self.costing.triethylene_glycol_rate,
        self.costing.SCR_catalyst_waste_rate,
        self.costing.triethylene_glycol_waste_rate,
        self.costing.amine_purification_unit_waste_rate,    #
        self.costing.thermal_reclaimer_unit_waste_rate,     #
    ]  
    
    # Create another costing block (for O&M cost calculation for NGCC with capture?)
    
    self.ngcccap = QGESSCosting()    
    
    self.ngcccap.build_process_costs(
        total_plant_cost=self.costing.TPC, # CE_index_units  # NGCC with capture TPC
        nameplate_capacity=pyo.units.convert(self.costing.plant_gross_power, pyo.units.MW),
        labor_rate=38.5,
        labor_burden=30,
        operators_per_shift=6.3,
        tech=6,
        land_cost=3000 * 100 * 1e-6,  # units=CE_index_units,
        net_power=pyo.units.convert(self.costing.plant_net_power, pyo.units.MW),
        fixed_OM=True,
        variable_OM=True,
        resources=[
            "natural_gas", 
            "water", 
            "water_treatment_chemicals",
            "solvent", 
            "SCR_catalyst", 
            "ammonia", 
            "triethylene_glycol",
            "SCR_catalyst_waste", 
            "triethylene_glycol_waste",
            "amine_purification_unit waste", 
            "thermal_reclaimer_unit_waste"],
        rates=rates,
        prices={"solvent": 2.09 * pyunits.USD_2018 / pyunits.kg},
        fuel="natural_gas",
        chemicals=[
            "water_treatment_chemicals",
            "solvent",
            "SCR_catalyst",
            "ammonia",
            "triethylene_glycol",
        ],
        chemicals_inventory=[
            "water_treatment_chemicals",
            "solvent",
            "ammonia",
            "triethylene_glycol",
        ],
        waste=[
            "SCR_catalyst_waste",
            "triethylene_glycol_waste",
            "amine_purification_unit waste",
            "thermal_reclaimer_unit_waste",
        ],
        transport_cost=10 * 1e-6,  # units=CE_index_units/pyo.units.tonne
        tonne_CO2_capture=pyo.units.convert(
            self.costing_setup.CO2_capture_rate[0] * 8760 * pyo.units.hr / pyo.units.year,
            to_units=pyo.units.tonne / pyo.units.year,
        ),
        CE_index_year=CE_index_year,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       
    # Computing LCOE with capture
    
    # Obtain LCOE components (NGCC with capture)
    
    self.costing.TOC_fuel_supply_twomonths = pyo.Expression(
        expr=(self.ngcccap.fuel_cost_OC * (2 / 2.25) * pyunits.year)
    )
    
    self.costing.chemicals_non_init_fill = [
        "water_treatment_chemicals",
        "ammonia",
        "triethylene_glycol",
        "solvent",
    ]
    
    self.costing.TOC_chemicals_non_init_fill = pyo.Expression(
        expr=sum(
            self.ngcccap.variable_operating_costs[0, i]
            for i in self.costing.chemicals_non_init_fill
        )
        * 1
        * pyunits.year
        / 2
    )
    
    
    self.costing.capital_cost = pyo.Expression(
        expr=self.ngcccap.annualized_cost
        + 0.0707
        * 1.093
        * (
            -self.costing.TOC_fuel_supply_twomonths
            - self.costing.TOC_chemicals_non_init_fill
        )
    )  # $MM/year
    
    
    def get_capital_lcoe(b):
        return (
            self.costing.capital_cost
            / (
                self.ngcccap.capacity_factor
                * self.costing.plant_net_power
                * 8760
                * pyunits.hr * 1e-6)
        ) # LCOE_units (USD_2018/MWh)
    
    
    self.costing.capital_lcoe = pyo.Expression(rule=get_capital_lcoe) # LCOE_units (MUSD_2018/MWh)
    
    
    def get_fixed_lcoe(b):
        return pyunits.convert(
            self.ngcccap.total_fixed_OM_cost
            / (
                self.ngcccap.capacity_factor
                * self.costing.plant_net_power
                * 8760
                * pyunits.hr
            ),
            LCOE_units,
        )
    
    
    self.costing.fixed_lcoe = pyo.Expression(rule=get_fixed_lcoe)
    
    
    def get_variable_lcoe(b):
        return pyunits.convert(
            (
                (self.ngcccap.total_variable_OM_cost[0] * self.ngcccap.capacity_factor)
                / (
                    self.ngcccap.capacity_factor
                    * self.costing.plant_net_power
                    * 8760
                    * pyunits.hr
                )
            )
            * 1
            * pyunits.year,
            LCOE_units,
        )
    
    
    self.costing.variable_lcoe = pyo.Expression(rule=get_variable_lcoe)
    
    
    def get_transport_lcoe(b):
        return (
                self.ngcccap.transport_cost
                / (
                    self.ngcccap.capacity_factor
                    * self.costing.plant_net_power
                    * 8760
                    * pyunits.hr * 1e-6
                )
        ) #  LCOE_units
    
    
    self.costing.transport_lcoe = pyo.Expression(rule=get_transport_lcoe)
    
    self.costing.LCOE = pyo.Expression(
        expr=self.costing.capital_lcoe
        + self.costing.fixed_lcoe
        + self.costing.variable_lcoe
        + self.costing.transport_lcoe
    )

 
    # Cost of CO2 Capture
    # aka Levelized Cost of Capture (LCOC)

    def get_cost_of_capture(b):
        return (
            (b.LCOE - b.transport_lcoe - b.LCOE_nocap_B31A) * 
            b.plant_net_power / pyo.units.convert(
                b.parent_block().costing_setup.CO2_capture_rate[0],
                to_units=pyo.units.tonne / pyo.units.hour)
        ) # USD_2018/t
    
    self.costing.cost_of_capture = pyo.Expression(rule=get_cost_of_capture)

    
    def get_cost_CO2_avoided(b):
        return (
            (self.costing.LCOE - self.costing.LCOE_nocap_B31A)
            / (
                pyo.units.convert(self.costing.CO2_emission_nocap_B31A,
                to_units=pyo.units.tonne / pyo.units.hour)
                / self.costing.plant_net_power_nocap_B31A
                - pyo.units.convert(self.costing.CO2_emission_cap,
                to_units=pyo.units.tonne / pyo.units.hour) / self.costing.plant_net_power
            )
        ) # USD_2018/t CO2

    self.costing.cost_CO2_avoided = pyo.Expression(rule=get_cost_CO2_avoided)
    
    
    

