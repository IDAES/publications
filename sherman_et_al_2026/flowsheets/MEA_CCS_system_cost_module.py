"""
Economics Calculations
MEA Solvent Based Carbon Capture System (one train)
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
from idaes.core.solvers import get_solver
import os
from pyomo.common.fileutils import this_file_dir
import csv


def get_CO2_absorption_cost(self): 

    # Create a main Costing block
    self.costing = QGESSCosting()
    
    # ---------------- TEC Calculation for CCS -----------------------
    # TEC = Total Equipment Cost
    
    import json

    directory = this_file_dir()  # access the path to the folder containing this script

    with open(os.path.join(directory, "FEEDEquipmentDetailsFinal.json"), "r") as file:
        FEEDEquipmentDetailsFinal = json.load(file)
        
    pyo.units.load_definitions_from_strings(["USD_2021 = 500/708 * USD_CE500"])
    
    CE_index_year = "2018"
    CE_index_units = getattr(pyo.units, "MUSD_" + CE_index_year)
    
    
    # ***Accounts with carbon capture system units***
    
    ####################### Adsorber Section ##################################
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
    
    # For now, keeping the ngcc + MEA system block numbering (thus, instead of
    # starting with "b1", the numbering starts with "b22", and is out of order)
    absorber_accounts = ["2.3"]
    self.b22 = pyo.Block() 

    QGESSCostingData.get_PP_costing(
        self.b22,
        absorber_accounts,
        # self.costing.absorber_and_WW_volume, # with Water Wash
        self.costing.absorber_volume_column_withheads,
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
        self.costing.absorber_packing_volume,
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.absorber_packing_cost = pyo.Expression(
        expr=sum(self.b23.total_plant_cost[ac] for ac in absorber_packing_accounts)
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
    
    # Rich solvent pump
    rich_solvent_pump_account = ["8.3"]
    self.b30 = pyo.Block()
    
    # Consider 10 % margin when computing the rich solvent rated flowrate
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
    
    ####################### Stripper Section ##################################
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
        self.costing.stripper_volume_column_withheads,
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
    
    ########################## Solvent Costs ##################################
        
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
    
    # Solvent Storage Tank
    solvent_storage_tank_account = ["19.3"]
    self.b41 = pyo.Block()

    QGESSCostingData.get_PP_costing(
        self.b41,
        solvent_storage_tank_account,
        pyo.units.convert(self.stripper_section.lean_solvent_flow_vol[0], 
                          to_units=pyo.units.gal/pyo.units.min),
        6,
        CE_index_year=CE_index_year,
        additional_costing_params=FEEDEquipmentDetailsFinal,
        use_additional_costing_params=True,
    )
    
    self.costing.solvent_storage_tank_cost = pyo.Expression(
        expr=sum(self.b41.total_plant_cost[ac] for ac in solvent_storage_tank_account)
        )
    
    # MEA solvent initial fill cost
    self.costing.solvent_cost = pyo.Param(
        initialize=2.09,
        units=getattr(pyunits, "USD_" + CE_index_year)/pyo.units.kg) # $/kg
    
    self.costing.RemovalSystem_Equip_Adjust = pyo.Expression(
        expr=pyo.units.convert(self.costing_setup.solvent_fill_init[0]
                               *self.costing.solvent_cost
                               /self.number_trains.value,
                               to_units=getattr(pyunits, "MUSD_" + CE_index_year)
                               )
        )   # 1 train
    
    ######################## Economic Metrics #################################
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    # Carbon Capture System Total Equipment Cost (CCS TEC) components
    
    self.costing.CCS_TEC = pyo.Expression(
        expr=
        (pyo.units.convert(
            self.costing.absorber_column_cost +
            self.costing.absorber_packing_cost +
            self.costing.lean_rich_hex_cost +
            self.costing.rich_solvent_pump_cost +
            self.costing.lean_solvent_pump_cost +
            self.costing.stripper_column_cost +
            self.costing.stripper_packing_cost +
            self.costing.stripper_condenser_cost +
            self.costing.stripper_reboiler_cost +
            self.costing.reboiler_condensate_pot_cost +
            self.costing.stripper_reflux_drum_cost +
            self.costing.stripper_reflux_pump_cost +
            self.costing.solvent_storage_tank_cost,
            to_units=getattr(pyunits, "USD_" + CE_index_year)  # $
            )
            )
        )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Total Plant Cost (TPC)
    
    # The Total Plant Cost (TPC) accounts for the total equipment installed 
    # costs (TIC) plus the indirect cost (IDC) plus the balance of the plant 
    # cost (BPC). The IDC and BPC are considered to be 32 and 20% of TIC, 
    # respectively. The installed cost (TIC) of an equipment includes the 
    # equipment purchase cost (TEC) and the equipment installation cost (EIC).
    # EIC is considered to be 4 and 80% of the purchase costs TEC for general costing.CCS_TPC
    # equipment (columns, heat exchangers) and movers (compressors and vacuum 
    # pumps), respectively (Faruque et al., 2012))
    
    self.costing.CCS_EIC = pyo.Expression(
        expr=pyo.units.convert(0.04*(self.costing.absorber_column_cost +
                                     self.costing.stripper_column_cost +
                                     self.costing.lean_rich_hex_cost +
                                     self.costing.stripper_condenser_cost +
                                     self.costing.stripper_reboiler_cost +
                                     self.costing.reboiler_condensate_pot_cost +
                                     self.costing.stripper_reflux_drum_cost) +
                               0.8*(self.costing.rich_solvent_pump_cost + 
                                    self.costing.lean_solvent_pump_cost +
                                    self.costing.stripper_reflux_pump_cost),
                               to_units=getattr(pyunits, "USD_" + CE_index_year)
                               )
        )
    
    self.costing.CCS_TIC = pyo.Expression(
        expr=pyo.units.convert(self.costing.CCS_TEC + self.costing.CCS_EIC,
                                to_units=getattr(pyunits, "USD_" + CE_index_year)
                                )
        )
    
    self.costing.CCS_TPC = pyo.Expression(
        expr=pyo.units.convert(1.52*self.costing.CCS_TIC +
                               pyo.units.convert(self.costing.solvent_filtration_cost, 
                                                 to_units=getattr(pyunits, "USD_" + CE_index_year)) +
                               pyo.units.convert(self.costing.RemovalSystem_Equip_Adjust, 
                                                 to_units=getattr(pyunits, "USD_" + CE_index_year)),
                               to_units=getattr(pyunits, "USD_" + CE_index_year)
                               )
        )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Total Annual Maintainance Cost (AMC)
    
    # The Total Annual Maintainance Cost (AMC) is taken to be 5 % of TPC
    # Faruque et al., 2012
    self.costing.CCS_AMC = pyo.Expression(
        expr=pyo.units.convert(0.05 * self.costing.CCS_TPC,
                                to_units=getattr(pyunits, "USD_" + CE_index_year)
                                )
        )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Annualized Investment Cost (AIC)
    
    # The Annualized Investment Cost (AIC) it the sum of the AMC and TPC 
    # multiplied by the capital recovery factor (considered to be 0.0937 
    # (i = 8%, n = 25 yr))
    
    self.costing.CCS_AIC = pyo.Expression(
        expr=pyo.units.convert((0.0937 * self.costing.CCS_TPC +
                               self.costing.CCS_AMC) / pyo.units.year,
                                to_units=getattr(pyunits, "USD_" + CE_index_year) /
                                pyo.units.year
                                )
        )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    # Total Annual Operating Cost (AOC) components
    
    # Utility Costs
 
    # Assumption: electricity is available at $0.07/kWh
    self.costing.P0_electricity = pyo.Param(
        initialize=0.07, mutable=True, 
        units=pyo.units.USD_2018 / pyo.units.kW / pyo.units.hr  
    ) # $/kWh
    
    # Assumption: P0_steam = 14.5 $/tonne (Isenberg et al., 2020)
    self.costing.P0_steam = pyo.Param(
        initialize=14.5, mutable=True, 
        units=pyo.units.USD_2018 /pyo.units.tonne  
    )  # $/tonne
    
    # Assumption: P0_cooling_water = 0.0329 $/tonne (Isenberg et al., 2020)
    self.costing.P0_cooling_water = pyo.Param(
        initialize=0.0329, mutable=True, 
        units=pyo.units.USD_2018 /pyo.units.tonne
    )  # $/tonne
    
    # Assumption: P0_H2O_makeup = 0.04 $/tonne (Isenberg et al., 2020)
    self.costing.P0_H2O_makeup = pyo.Param(
        initialize=0.04, mutable=True, 
        units=pyo.units.USD_2018 /pyo.units.tonne
    )  # $/tonne
    
    self.costing.UC_rich_solvent_pump = pyo.Expression(
        expr=pyo.units.convert(self.costing.P0_electricity
                                *self.absorber_section.rich_solvent_pump.work_mechanical[0],
                                to_units=pyo.units.USD_2018/pyo.units.hr)
    )
    
    self.costing.UC_lean_solvent_pump = pyo.Expression(
        expr=pyo.units.convert(self.costing.P0_electricity
                                *self.absorber_section.lean_solvent_pump.work_mechanical[0],
                                to_units=pyo.units.USD_2018/pyo.units.hr)
    )
    
    self.costing.UC_reboiler = pyo.Expression(
        expr=pyo.units.convert(self.costing.P0_steam
                                *pyo.units.convert(self.stripper_section.reboiler.steam_flow_mol[0]
                                                   *self.stripper_section.liquid_properties.H2O.mw,
                                                   to_units=pyo.units.tonne/pyo.units.s), 
                                to_units=pyo.units.USD_2018/pyo.units.s) 
    )  
    
    self.costing.UC_condenser = pyo.Expression(
        expr=pyo.units.convert(self.costing.P0_cooling_water
                                *pyo.units.convert(self.stripper_section.condenser.cooling_flow_mol[0]
                                                   *self.stripper_section.liquid_properties.H2O.mw,
                                                   to_units=pyo.units.tonne/pyo.units.s),
                                to_units=pyo.units.USD_2018/pyo.units.s)
    )  
    
    self.costing.UC_H2O_makeup = pyo.Expression(
        expr=pyo.units.convert(self.costing.P0_H2O_makeup
                                *pyo.units.convert(self.makeup.flow_mol[0]
                                                   *self.absorber_section.liquid_properties.H2O.mw,
                                                   to_units=pyo.units.tonne/pyo.units.s),
                                to_units=pyo.units.USD_2018/pyo.units.s)
    )  
    
    # Annual Operating Cost (AOC), Faruque at al. 2012, Eq. (18)
    
    # Assumption: the plant operates 8760 hours in a year
    self.costing.CCS_AOC = pyo.Expression(
        expr=pyo.units.convert(8760*(            
            self.costing.UC_rich_solvent_pump + 
            self.costing.UC_lean_solvent_pump +
            pyo.units.convert(self.costing.UC_reboiler, 
                              to_units=pyo.units.USD_2018/pyo.units.hr) +
            pyo.units.convert(self.costing.UC_condenser, 
                              to_units=pyo.units.USD_2018/pyo.units.hr) +
            pyo.units.convert(self.costing.UC_H2O_makeup, 
                              to_units=pyo.units.USD_2018/pyo.units.hr)) *
            1 * pyo.units.hr / pyo.units.year,
            to_units=getattr(pyunits, "USD_" + CE_index_year) / pyo.units.year
            )
    )  
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Total Annualized Cost (TAC) measured in $/year and $/tonne of CO2 captured
    # Computed for one train  
    # Faruque at al. 2012, Eq. (1)
    
    self.costing.CCS_TAC = pyo.Expression(expr=self.costing.CCS_AIC + 
                                          self.costing.CCS_AOC)  
    
    # Total annualized cost per tonne of CO2 captured (indipendent of number of trains)
    # aka Levelized Cost of Capture (LCOC)
    self.costing.CCS_TAC_perCO2captured = pyo.Expression(
        expr=self.costing.CCS_TAC /
        pyo.units.convert(self.costing_setup.CO2_capture_rate[0]/self.number_trains.value, 
                          to_units=pyo.units.tonne/pyo.units.year))
    
    
    
    
    
    
    

